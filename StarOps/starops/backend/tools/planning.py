"""规划侧领域工具。

全部为确定性实现，不调用 LLM。返回统一的 ToolResult。
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

from database import query_all, utcnow
from schemas.models import ConstraintResult, PlanItem, ToolResult


def _elapsed(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# ---------------------------------------------------------------- 区域字典

#: 系统支持的可观测区域。**需求解析与调度共用同一份字典**——
#: LLM 不知道这些编码，靠 prompt 告知不可靠，故由确定性代码兜底反查。
AREAS: list[tuple[str, str]] = [
    ("CN-BJ", "北京"),
    ("CN-SH", "上海"),
    ("CN-GZ", "广州"),
    ("CN-SZ", "深圳"),
    ("CN-CD", "成都"),
    ("CN-XA", "西安"),
]

AREA_NAME_TO_ID = {name: aid for aid, name in AREAS}


def resolve_area_id(area_name: str | None, area_id: str | None = None) -> str | None:
    """由区域名反查区域编码（已给编码则原样返回）。

    真实 LLM 通常只能识别城市名，给不出 ``CN-BJ`` 这类编码；
    没有这一步，需求解析会因 area_id 缺失而卡在"待补充输入"。
    """
    if area_id:
        return area_id
    if not area_name:
        return None
    return AREA_NAME_TO_ID.get(area_name.strip())


# ---------------------------------------------------------------- 查询

def query_satellites(
    satellite_type: str = "any",
    max_resolution: float | None = None,
    imaging_mode: str | None = None,
) -> ToolResult:
    """查询卫星资源，按类型 / 分辨率 / 成像模式筛选。"""
    start = time.perf_counter()
    rows = query_all("satellite")
    if satellite_type and satellite_type != "any":
        rows = [r for r in rows if r["satellite_type"] == satellite_type]
    if max_resolution is not None:
        rows = [r for r in rows if r["resolution"] <= max_resolution]
    if imaging_mode:
        rows = [r for r in rows if imaging_mode in r["imaging_mode"]]
    return ToolResult.ok(
        data={"satellites": rows, "count": len(rows)},
        elapsed_ms=_elapsed(start),
        reason=None,
    )


def query_observation_windows(
    satellite_ids: list[str],
    area_id: str,
    time_start: str | None = None,
    time_end: str | None = None,
) -> ToolResult:
    """查询候选观测窗口，并计算数据质量标记。"""
    start = time.perf_counter()
    rows = query_all("observation_window")
    if satellite_ids:
        rows = [r for r in rows if r["satellite_id"] in satellite_ids]
    if area_id:
        rows = [r for r in rows if r["area_id"] == area_id]

    now = datetime.now(timezone.utc)
    enriched = []
    for r in rows:
        row = dict(r)
        end = _parse(r["end_time"])
        if end < now:
            row["quality"] = "EXPIRED"  # 窗口已过 → 硬性排除
        elif r["quality"] == "FRESH" and _parse(r["start_time"]) - now > timedelta(days=30):
            row["quality"] = "STALE"
        enriched.append(row)

    if time_start:
        enriched = [r for r in enriched if _parse(r["end_time"]) >= _parse(time_start)]
    if time_end:
        enriched = [r for r in enriched if _parse(r["start_time"]) <= _parse(time_end)]

    return ToolResult.ok(
        data={"windows": enriched, "count": len(enriched)},
        elapsed_ms=_elapsed(start),
    )


# ---------------------------------------------------------------- 硬约束校验

def build_candidate_plans(
    candidates: list[dict[str, Any]],
    windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把卫星与窗口配对成候选方案（satellite × window）。

    校验必须在"方案"粒度上做——只有配对后的方案才是真正可执行的单元。
    """
    win_by_sat: dict[str, list[dict[str, Any]]] = {}
    for w in windows:
        win_by_sat.setdefault(w["satellite_id"], []).append(w)

    plans: list[dict[str, Any]] = []
    for c in candidates:
        for w in win_by_sat.get(c["satellite_id"], []):
            # EXPIRED 窗口不构成可执行方案，但仍要留痕，由校验器记录为违规
            if w.get("quality") == "EXPIRED":
                continue
            plans.append(
                {
                    "plan_id": f"{c['satellite_id']}-{w['window_id'][-6:]}",
                    "satellite_id": c["satellite_id"],
                    "satellite_name": c["satellite_name"],
                    "satellite_type": c["satellite_type"],
                    "resolution": c["resolution"],
                    "window_id": w["window_id"],
                    "start_time": w["start_time"],
                    "end_time": w["end_time"],
                    "cloud_rate": w["cloud_rate"],
                    "incidence_angle": w["incidence_angle"],
                    "quality": w.get("quality", "FRESH"),
                }
            )
    return plans


def validate_constraints(
    requirements: dict[str, Any],
    candidates: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    max_cloud_rate: float = 0.3,
    max_incidence_angle: float = 45.0,
) -> ToolResult:
    """硬约束校验器——独立于 LLM 的纯代码判定。

    按方案粒度校验：每个候选方案独立过一遍规则，
    通过的进入 viable_plans，不通过的进入 rejected_plans 并附原因。

    整体 passed 的含义是"存在至少一个满足全部约束的方案"——
    部分方案被淘汰是正常的，不等于整体失败。
    """
    start = time.perf_counter()
    checked: list[str] = [
        "resolution",
        "satellite_type",
        "data_quality",
        "cloud_rate",
        "incidence_angle",
        "time_range",
    ]

    max_res = requirements.get("max_resolution")
    want_type = requirements.get("satellite_type")
    time_days = requirements.get("time_days")
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=int(time_days)) if time_days else None

    plans = build_candidate_plans(candidates, windows)
    viable: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    # 过期窗口单独留痕（它们不进入方案，但必须被报告，不能静默丢弃）
    expired_windows = [
        w for w in windows if w.get("quality") == "EXPIRED"
    ]

    for plan in plans:
        reasons: list[str] = []

        if max_res and plan["resolution"] > max_res:
            reasons.append(f"分辨率 {plan['resolution']}m 超出阈值 {max_res}m")
        if want_type and want_type != "any" and plan["satellite_type"] != want_type:
            reasons.append(f"类型 {plan['satellite_type']} 不满足 {want_type}")
        if plan.get("quality") == "SUSPECT":
            reasons.append("窗口数据可疑，需人工复核")
        if plan["cloud_rate"] > max_cloud_rate:
            reasons.append(f"云量 {plan['cloud_rate']} 超出上限 {max_cloud_rate}")
        if plan["incidence_angle"] > max_incidence_angle:
            reasons.append(f"入射角 {plan['incidence_angle']}° 超出上限 {max_incidence_angle}°")
        if horizon and _parse(plan["start_time"]) > horizon:
            reasons.append(f"开始时间超出 {time_days} 天范围")

        if reasons:
            rejected.append({"plan_id": plan["plan_id"], "reasons": reasons})
        else:
            viable.append(plan)

    # 整体违规清单：只列被淘汰的方案，不淹没在无关窗口里
    violations = [
        f"方案 {r['plan_id']}：{'；'.join(r['reasons'])}" for r in rejected
    ]
    violations += [
        f"窗口 {w['window_id']} 已过期，不可执行" for w in expired_windows
    ]

    # 若一个方案都没剩，说明确实无解
    no_solution = bool(plans) and not viable
    if expired_windows and not viable and not plans:
        no_solution = True

    result = {
        "passed": not no_solution,
        "violations": violations,
        "checked_rules": checked,
        "rule_version": "v1",
        "viable_plans": viable,
        "rejected_plans": rejected,
        "total_candidates": len(plans),
        "viable_count": len(viable),
        "rejected_count": len(rejected),
        "expired_count": len(expired_windows),
        "no_solution": no_solution,
    }
    return ToolResult.ok(
        data=result,
        elapsed_ms=_elapsed(start),
    )


# ---------------------------------------------------------------- 方案评分

DEFAULT_WEIGHTS = {
    "resolution": 0.35,
    "timeliness": 0.25,
    "cloud": 0.20,
    "incidence": 0.20,
}


def calculate_plan_score(
    candidates: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
    viable_plan_ids: list[str] | None = None,
) -> ToolResult:
    """多目标加权评分 + 排序。

    权重是业务规则不是训练目标——换客户改权重即可，不做学习排序。
    接口设计为可插拔：未来替换为 LTR 实现时上层无需改动。

    打分粒度必须是"方案"(satellite × window)而不是"卫星"：
    一颗卫星可能有多个候选窗口，按卫星聚合会让同一卫星的其它窗口
    方案被静默丢弃（曾导致可行方案丢失）。
    """
    start = time.perf_counter()
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    now = datetime.now(timezone.utc)

    # 与硬约束校验器共用同一套方案配对逻辑，保证 plan_id 可精确对齐
    plans = build_candidate_plans(candidates, windows)
    if viable_plan_ids is not None:
        allowed = set(viable_plan_ids)
        plans = [p for p in plans if p["plan_id"] in allowed]

    items: list[PlanItem] = []
    for plan in plans:
        res_score = 1.0 / (1.0 + plan["resolution"])  # 越小越好
        days = max(0.0, (_parse(plan["start_time"]) - now).total_seconds() / 86400.0)
        time_score = 1.0 / (1.0 + days)
        cloud_score = 1.0 - plan["cloud_rate"]
        inc_score = 1.0 - (plan["incidence_angle"] / 90.0)

        breakdown = {
            "resolution": round(res_score, 4),
            "timeliness": round(time_score, 4),
            "cloud": round(cloud_score, 4),
            "incidence": round(inc_score, 4),
        }
        score = sum(breakdown[k] * w.get(k, 0.0) for k in breakdown)

        reasons = [
            f"分辨率 {plan['resolution']}m",
            f"云量 {plan['cloud_rate']}",
            f"入射角 {plan['incidence_angle']}°",
            f"过境 {days:.1f} 天后",
        ]
        items.append(
            PlanItem(
                plan_id=plan["plan_id"],
                satellite_id=plan["satellite_id"],
                satellite_name=plan["satellite_name"],
                window_id=plan["window_id"],
                start_time=plan["start_time"],
                end_time=plan["end_time"],
                resolution=plan["resolution"],
                cloud_rate=plan["cloud_rate"],
                incidence_angle=plan["incidence_angle"],
                score=round(score, 4),
                score_breakdown=breakdown,
                reasons=reasons,
            )
        )

    items.sort(key=lambda x: x.score, reverse=True)
    return ToolResult.ok(
        data={"plans": [i.model_dump() for i in items], "weights": w, "count": len(items)},
        elapsed_ms=_elapsed(start),
    )


# ---------------------------------------------------------------- 提交 / 执行

def generate_task_report(
    selected_plan: dict[str, Any],
    alternatives: list[dict[str, Any]],
    constraint_result: dict[str, Any],
) -> ToolResult:
    """生成规划报告（模板化，不依赖 LLM）。"""
    start = time.perf_counter()
    lines = [
        "# 观测任务规划报告",
        "",
        f"**推荐方案**：{selected_plan.get('satellite_name')}（{selected_plan.get('satellite_id')}）",
        f"**观测窗口**：{selected_plan.get('start_time')} → {selected_plan.get('end_time')}",
        f"**综合评分**：{selected_plan.get('score')}",
        "",
        "## 评分构成",
    ]
    for k, v in (selected_plan.get("score_breakdown") or {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## 约束校验"]
    status = "通过" if constraint_result.get("passed") else "未通过"
    lines.append(f"状态：**{status}**")
    for v in constraint_result.get("violations", []):
        lines.append(f"- 违反项：{v}")
    if alternatives:
        lines += ["", "## 备选方案"]
        for alt in alternatives:
            lines.append(
                f"- {alt.get('satellite_name')}｜评分 {alt.get('score')}｜"
                f"分辨率 {alt.get('resolution')}m｜云量 {alt.get('cloud_rate')}"
            )
    lines += ["", "> 说明：本报告基于模拟卫星与模拟观测窗口数据生成。"]
    return ToolResult.ok(data={"report": "\n".join(lines)}, elapsed_ms=_elapsed(start))
