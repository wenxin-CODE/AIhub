"""SAR 影像分析节点。

对应方案 §4.2 的 SAR Agent 主干：
影像元数据 → 质量指标 → 诊断（斑点/条纹）→ 候选管线 → 选型 → 审批 → 执行 → 对比报告

关键设计（§4.3）：
- 诊断结论由**确定性指标**决定，不由 LLM 猜测。
- 候选管线都跑一遍，用**同一套指标**横评，选最优——
  而不是预先拍脑袋指定一个方法。
- 处理管线的执行同样走审批关口，审批前不落地。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agents.state import PlanningState
from schemas.models import ApprovalRequest
from services.approval import submit_approval
from tools import sar

logger = logging.getLogger(__name__)


def _diagnosis(metrics: dict[str, Any]) -> dict[str, Any]:
    """基于确定性指标给出诊断结论。

    阈值来自 SAR 工程经验值，此处明确列出而非藏在代码里，
    便于后续标定与解释。
    """
    cv = metrics.get("cv") or 0.0
    enl = metrics.get("enl")
    striping_strength = metrics.get("striping_strength")
    local_variance = metrics.get("local_variance") or 0.0

    # 斑点强度：CV 越高斑点越强；ENL 越低等效视数越少
    if cv > 0.6:
        speckle_level = "severe"
    elif cv > 0.35:
        speckle_level = "moderate"
    else:
        speckle_level = "mild"

    findings: list[str] = []
    if speckle_level == "severe":
        findings.append(f"斑点噪声严重（CV={cv:.3f}，ENL={enl if enl is None else round(enl, 2)}）")
    elif speckle_level == "moderate":
        findings.append(f"斑点噪声中等（CV={cv:.3f}）")
    else:
        findings.append(f"斑点噪声较弱（CV={cv:.3f}）")

    has_striping = False
    if striping_strength is not None and striping_strength > 0.25:
        has_striping = True
        direction = "水平" if metrics.get("striping_direction") == 0.0 else "垂直"
        findings.append(f"检测到{direction}条纹（突出度 {striping_strength}）")

    return {
        "speckle_level": speckle_level,
        "has_striping": has_striping,
        "cv": cv,
        "enl": enl,
        "local_variance": local_variance,
        "findings": findings,
    }


def node_sar_analyze(state: PlanningState, tracer: Any) -> PlanningState:
    """SAR 分析主节点：指标 → 诊断 → 候选管线 → 选型 → 审批。"""
    state["current_node"] = "sar_analyze"
    tracer.emit("node_start", node="sar_analyze", label="SAR 影像分析")

    image_path = state.get("image_path") or ""
    if not image_path:
        tracer.emit(
            "node_end",
            node="sar_analyze",
            label="SAR 影像分析",
            status="skipped",
            reason="未指定影像路径",
        )
        return state

    # 1) 元数据
    meta_res = tracer.tool(
        "read_image_metadata",
        lambda: sar.read_image_metadata(image_path),
        input_summary={"path": image_path},
    )
    if meta_res.status != "success":
        tracer.emit(
            "node_end",
            node="sar_analyze",
            label="SAR 影像分析",
            status="failed",
            reason=meta_res.reason,
        )
        state["errors"] = state.get("errors", []) + [
            {"node": "read_image_metadata", "reason": meta_res.reason}
        ]
        return state
    state["image_metadata"] = meta_res.data or {}

    # 2) 质量指标
    metric_res = tracer.tool(
        "calculate_image_metrics",
        lambda: sar.calculate_image_metrics(image_path),
        input_summary={"path": image_path},
    )
    if metric_res.status != "success":
        tracer.emit(
            "node_end",
            node="sar_analyze",
            label="SAR 影像分析",
            status="failed",
            reason=metric_res.reason,
        )
        state["errors"] = state.get("errors", []) + [
            {"node": "calculate_image_metrics", "reason": metric_res.reason}
        ]
        return state
    state["image_metrics"] = metric_res.data or {}

    # 3) 条纹检测（独立工具，检测与处理分离）
    strip_res = tracer.tool(
        "detect_striping",
        lambda: sar.detect_striping(image_path),
        input_summary={"path": image_path},
    )
    if strip_res.status == "success":
        state.setdefault("image_metadata", {})["striping"] = strip_res.data

    # 4) 诊断（确定性规则，不让 LLM 猜）
    diagnosis = _diagnosis(state["image_metrics"])
    state["diagnosis"] = diagnosis
    tracer.emit(
        "node_end",
        node="sar_analyze",
        label="SAR 影像分析",
        metrics=state["image_metrics"],
        diagnosis=diagnosis,
    )

    # 5) 候选管线 —— 按诊断结果决定候选集
    candidates: list[dict[str, Any]] = []
    if diagnosis["speckle_level"] in ("severe", "moderate"):
        candidates = [
            {"pipeline_id": "lee", "method": "lee", "filter_level": "L1_classical", "label": "Lee 滤波"},
            {"pipeline_id": "median", "method": "median", "filter_level": "L1_classical", "label": "中值滤波"},
            {"pipeline_id": "adaptive", "method": "adaptive", "filter_level": "L1_classical", "label": "自适应滤波"},
        ]
    else:
        candidates = [
            {"pipeline_id": "lee", "method": "lee", "filter_level": "L1_classical", "label": "Lee 滤波"},
        ]
    state["candidate_pipelines"] = candidates

    # 6) 横评选型：同一套指标跑每条管线，选最优
    best, rejected = _select_best_pipeline(image_path, candidates, tracer)
    state["rejected_pipelines"] = rejected
    if best is None:
        tracer.emit(
            "node_end",
            node="sar_analyze",
            label="SAR 影像分析",
            status="no_viable_pipeline",
            rejected=[r.get("reject_reason") for r in rejected],
        )
        return state

    state["selected_pipeline"] = best
    state["idempotency_key"] = f"pipeline:{state.get('task_id','')}:{best['pipeline_id']}"

    # 7) 审批关口 —— 处理管线同样需要人工批准
    approval_res = tracer.tool(
        "submit_approval",
        lambda: submit_approval(
            ApprovalRequest(
                target_type="sar_pipeline",
                target_id=best["pipeline_id"],
                plan_version=1,
                summary=(
                    f"SAR 处理管线 {best['label']}｜"
                    f"CV {best.get('before_cv')}→{best.get('after_cv')}｜"
                    f"ENL {best.get('before_enl')}→{best.get('after_enl')}"
                ),
                snapshot=best,
            )
        ),
        input_summary={"pipeline_id": best["pipeline_id"]},
    )
    if approval_res.data:
        state["approval_id"] = approval_res.data.get("approval_id")
        state["approval_status"] = "pending"
    state["awaiting_approval"] = True

    tracer.emit(
        "node_end",
        node="sar_analyze",
        label="SAR 影像分析",
        pipeline=best,
        approval_id=state.get("approval_id"),
        status="pending_approval",
    )
    return state


def _select_best_pipeline(
    image_path: str,
    candidates: list[dict[str, Any]],
    tracer: Any,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """逐条候选管线跑一遍，用统一指标横评选最优。

    选型标准：CV 降幅最大且均值无明显偏移者胜出。
    这保证"DL 优于经典基线"这类结论只能由实测数据得出。

    返回 (最优管线, 被拒管线列表)。
    被拒的管线也带原因——"为什么没选它"必须可解释，
    不能让 Agent 悄悄换掉一个方法。
    """
    baseline = sar.calculate_image_metrics(image_path)
    if baseline.status != "success":
        return None, []
    before = baseline.data or {}

    results: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for cand in candidates:
        out_path = (
            f"{image_path.rsplit('.', 1)[0]}__{cand['pipeline_id']}.png"
        )
        started = time.perf_counter()
        apply_res = sar.apply_speckle_filter(
            image_path, method=cand["method"], output_path=out_path
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        if apply_res.status != "success":
            tracer.emit(
                "tool_end",
                tool_name="apply_speckle_filter",
                status="failed",
                reason=apply_res.reason,
                elapsed_ms=elapsed,
                pipeline_id=cand["pipeline_id"],
            )
            rejected.append({**cand, "reject_reason": apply_res.reason})
            continue

        cmp_res = sar.compare_image_quality(image_path, out_path)
        if cmp_res.status != "success":
            tracer.emit(
                "tool_end",
                tool_name="apply_speckle_filter",
                status="failed",
                reason=cmp_res.reason,
                elapsed_ms=elapsed,
                pipeline_id=cand["pipeline_id"],
            )
            rejected.append({**cand, "reject_reason": cmp_res.reason})
            continue
        payload = cmp_res.data or {}

        # 均值偏移超过 5% 的管线视为不可接受（辐射强度失真）
        before_mean = payload["before"]["gray_mean"]
        after_mean = payload["after"]["gray_mean"]
        mean_shift = (
            abs(after_mean - before_mean) / before_mean if before_mean else 0.0
        )
        if mean_shift > 0.05:
            reason = f"均值偏移 {mean_shift:.1%} 超过 5% 阈值（辐射强度失真）"
            tracer.emit(
                "tool_end",
                tool_name="apply_speckle_filter",
                status="rejected",
                reason=reason,
                elapsed_ms=elapsed,
                pipeline_id=cand["pipeline_id"],
            )
            rejected.append(
                {
                    **cand,
                    "reject_reason": reason,
                    "mean_shift_pct": round(mean_shift * 100, 2),
                    "before_cv": payload["before"]["cv"],
                    "after_cv": payload["after"]["cv"],
                }
            )
            continue

        tracer.emit(
            "tool_end",
            tool_name="apply_speckle_filter",
            status="success",
            elapsed_ms=elapsed,
            pipeline_id=cand["pipeline_id"],
            output_path=out_path,
        )
        results.append(
            {
                **cand,
                "output_path": out_path,
                "before_cv": payload["before"]["cv"],
                "after_cv": payload["after"]["cv"],
                "before_enl": payload["before"]["enl"],
                "after_enl": payload["after"]["enl"],
                "cv_reduced_pct": payload["improvement"]["cv_reduced_pct"],
                "mean_shift_pct": round(mean_shift * 100, 2),
                "comparison": payload,
            }
        )

    if not results:
        return None, rejected

    # 排序：CV 降幅降序，同降幅时取 ENL 提升更大者
    results.sort(key=lambda r: (-r["cv_reduced_pct"], -(r["after_enl"] - r["before_enl"])))
    best = results[0]
    # 未入选的合格管线也要记原因（被更优管线压过）
    for r in results[1:]:
        rejected.append(
            {
                **r,
                "reject_reason": (
                    f"降噪效果弱于最优管线（CV 降幅 {r['cv_reduced_pct']:.1f}% "
                    f"vs {best['cv_reduced_pct']:.1f}%）"
                ),
            }
        )
    return best, rejected


def node_sar_execute(state: PlanningState, tracer: Any) -> PlanningState:
    """SAR 处理管线执行：审批通过才落地。"""
    state["current_node"] = "sar_execute"
    tracer.emit("node_start", node="sar_execute", label="SAR 管线执行")

    if state.get("approval_status") != "approved":
        tracer.emit(
            "node_end",
            node="sar_execute",
            label="SAR 管线执行",
            status="blocked",
            reason=f"审批状态 {state.get('approval_status')}，未获批准不执行",
        )
        return state

    pipeline = state.get("selected_pipeline") or {}
    state["execution_result"] = {
        "status": "succeeded",
        "pipeline_id": pipeline.get("pipeline_id"),
        "output_path": pipeline.get("output_path"),
        "note": "SAR 处理管线已执行",
    }
    tracer.emit(
        "node_end",
        node="sar_execute",
        label="SAR 管线执行",
        status="succeeded",
        pipeline_id=pipeline.get("pipeline_id"),
    )
    return state
