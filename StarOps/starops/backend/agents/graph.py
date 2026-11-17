"""LangGraph 工作流图。

第一阶段：完成意图路由 + 规划 Agent 主干（需求解析 → 查询 → 校验 → 评分 → 审批）。
SAR Agent 节点在第二阶段补齐，此处留接口。

设计原则（方案 §3.2）：
- 单一编排图优先，不做多 Agent 自由协作
- 生成与执行分离，审批关口不可绕过
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from agents.llm import llm_client
from agents.nodes import order_nodes, sar_nodes
from agents.state import PlanningState
from database import record_tool_execution
from schemas.models import ApprovalRequest
from services.approval import build_idempotency_key, submit_approval
from tools import planning

logger = logging.getLogger(__name__)

# 事件回调类型：用于 SSE 推送
EventSink = Callable[[dict[str, Any]], None]


class _Tracer:
    """节点级追踪器：记录耗时与工具调用，喂给 SSE。"""

    def __init__(self, state: PlanningState, sink: EventSink | None):
        self.state = state
        self.sink = sink
        self.trace_id = state.get("trace_id", "")

    def emit(self, event: str, **payload: Any) -> None:
        if not self.sink:
            return
        self.sink({"event": event, "node": self.state.get("current_node", ""), **payload})

    def tool(
        self,
        tool_name: str,
        fn: Callable[[], Any],
        input_summary: Any = None,
        tool_version: str = "v1",
    ) -> Any:
        start = time.perf_counter()
        self.emit("tool_start", tool_name=tool_name, input=input_summary)
        try:
            result = fn()
            elapsed = int((time.perf_counter() - start) * 1000)
            record_tool_execution(
                trace_id=self.trace_id,
                tool_name=tool_name,
                input_summary=input_summary,
                output_summary=getattr(result, "data", None),
                status=getattr(result, "status", "success"),
                duration_ms=elapsed,
                tool_version=tool_version,
            )
            self.emit(
                "tool_end",
                tool_name=tool_name,
                status=getattr(result, "status", "success"),
                elapsed_ms=elapsed,
                output=getattr(result, "data", None),
                reason=getattr(result, "reason", None),
            )
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed = int((time.perf_counter() - start) * 1000)
            record_tool_execution(
                trace_id=self.trace_id,
                tool_name=tool_name,
                input_summary=input_summary,
                output_summary={"error": str(exc)},
                status="failed",
                duration_ms=elapsed,
                tool_version=tool_version,
            )
            self.emit("tool_end", tool_name=tool_name, status="failed", elapsed_ms=elapsed, reason=str(exc))
            raise


# ---------------------------------------------------------------- 节点

def node_route(state: PlanningState, tracer: _Tracer) -> PlanningState:
    """意图识别与任务路由。

    按 task_type 分流：planning 走规划主干，sar 走影像分析主干，order 走订单调度主干。
    三条链在末尾汇入同一个审批关口（§5 安全闭环）。

    **调用方已显式指定 task_type 时不被覆盖**——API 层（/api/sar/analyze、
    /api/orders/run）已经把链路定死，路由再改写会导致请求落到错误的主干。
    """
    state["current_node"] = "route"
    tracer.emit("node_start", node="route", label="意图识别与任务路由")

    preset = state.get("task_type")
    req = llm_client.chat_json(f"用户请求：{state.get('user_request', '')}")
    task_type = req.get("task_type")
    if preset in {"planning", "sar", "order"}:
        # 调用方已指定链路：以调用方为准
        task_type = preset
    elif task_type not in {"planning", "sar", "order"}:
        # LLM 可能给出白名单外的名字（如 "observation" / "imagery"），
        # 也可能给 null。非法值一律走关键词兜底，绝不原样写进 state——
        # 否则 state["task_type"] 指向一条不存在的链路，路由静默失效。
        text = state.get("user_request", "").lower()
        if "sar" in text or "影像" in text or "雷达" in text:
            task_type = "sar"
        elif "订单" in text or "批量调度" in text or "订单调度" in text:
            task_type = "order"
        else:
            task_type = "planning"

    state["task_type"] = task_type
    state["task_requirements"] = {
        k: v for k, v in req.items() if k not in {"task_type", "missing_fields", "_mock", "_note"}
    }
    state["missing_fields"] = req.get("missing_fields", []) or []
    if req.get("_mock"):
        state.setdefault("messages", [])
        state["messages"].append(
            {"role": "system", "content": "LLM 未配置，需求由规则降级解析（非真实模型推理）"}
        )

    tracer.emit(
        "node_end",
        node="route",
        label="意图识别与任务路由",
        task_type=task_type,
    )
    return state


def node_parse_requirements(state: PlanningState, tracer: _Tracer) -> PlanningState:
    """需求解析：提取结构化约束，标记缺失项由用户补充。"""
    state["current_node"] = "parse_requirements"
    tracer.emit("node_start", node="parse_requirements", label="需求解析")

    req = state.get("task_requirements", {}) or {}
    missing = list(state.get("missing_fields", []) or [])

    # 区域编码兜底：真实 LLM 能识别城市名但给不出 CN-XX 编码。
    # 由确定性字典反查，避免因 area_id 缺失而卡在"待补充输入"。
    resolved_area = planning.resolve_area_id(req.get("area_name"), req.get("area_id"))
    if resolved_area and not req.get("area_id"):
        req["area_id"] = resolved_area
        state["task_requirements"] = req
        if "area_id" in missing:
            missing.remove("area_id")

    state["missing_fields"] = missing
    if missing:
        tracer.emit(
            "need_input",
            missing=missing,
            message=f"缺少必要参数：{', '.join(missing)}。Agent 应主动追问，不自行假设。",
        )
    tracer.emit("node_end", node="parse_requirements", label="需求解析", requirements=state.get("task_requirements"))
    return state


def node_query_satellites(state: PlanningState, tracer: _Tracer) -> PlanningState:
    """卫星资源查询。"""
    state["current_node"] = "query_satellites"
    tracer.emit("node_start", node="query_satellites", label="卫星资源查询")
    req = state.get("task_requirements", {})
    res = tracer.tool(
        "query_satellites",
        lambda: planning.query_satellites(
            satellite_type=req.get("satellite_type", "any"),
            max_resolution=req.get("max_resolution"),
            imaging_mode=req.get("imaging_mode"),
        ),
        input_summary={
            "satellite_type": req.get("satellite_type"),
            "max_resolution": req.get("max_resolution"),
        },
    )
    state["candidates"] = res.data.get("satellites", []) if res.data else []
    tracer.emit("node_end", node="query_satellites", label="卫星资源查询", count=len(state["candidates"]))
    return state


def node_query_windows(state: PlanningState, tracer: _Tracer) -> PlanningState:
    """观测窗口查询。"""
    state["current_node"] = "query_windows"
    tracer.emit("node_start", node="query_windows", label="观测窗口查询")
    req = state.get("task_requirements", {})
    sats = [c["satellite_id"] for c in state.get("candidates", [])]
    res = tracer.tool(
        "query_observation_windows",
        lambda: planning.query_observation_windows(
            satellite_ids=sats,
            area_id=req.get("area_id", ""),
        ),
        input_summary={"satellite_ids": sats, "area_id": req.get("area_id")},
    )
    state["windows"] = res.data.get("windows", []) if res.data else []
    state["data_quality"] = {
        w["window_id"]: w.get("quality", "FRESH") for w in state["windows"]
    }
    tracer.emit("node_end", node="query_windows", label="观测窗口查询", count=len(state["windows"]))
    return state


def node_validate(state: PlanningState, tracer: _Tracer) -> PlanningState:
    """硬约束校验（独立于 LLM）。"""
    state["current_node"] = "validate_constraints"
    tracer.emit("node_start", node="validate_constraints", label="硬约束校验")
    res = tracer.tool(
        "validate_constraints",
        lambda: planning.validate_constraints(
            requirements=state.get("task_requirements", {}),
            candidates=state.get("candidates", []),
            windows=state.get("windows", []),
        ),
        input_summary={
            "candidates": len(state.get("candidates", [])),
            "windows": len(state.get("windows", [])),
        },
    )
    state["constraint_result"] = res.data or {}
    tracer.emit(
        "node_end",
        node="validate_constraints",
        label="硬约束校验",
        passed=(res.data or {}).get("passed"),
        violations=(res.data or {}).get("violations", []),
        viable_count=(res.data or {}).get("viable_count", 0),
        rejected_count=(res.data or {}).get("rejected_count", 0),
        no_solution=(res.data or {}).get("no_solution", False),
    )
    return state


def node_score(state: PlanningState, tracer: _Tracer) -> PlanningState:
    """方案评分与排序。"""
    state["current_node"] = "score_plans"
    tracer.emit("node_start", node="score_plans", label="方案评分与排序")
    # 只对通过硬约束校验的方案打分，避免把违规方案排到前面
    viable = (state.get("constraint_result") or {}).get("viable_plans") or []
    res = tracer.tool(
        "calculate_plan_score",
        lambda: planning.calculate_plan_score(
            candidates=state.get("candidates", []),
            windows=state.get("windows", []),
            viable_plan_ids=[p["plan_id"] for p in viable] if viable else None,
        ),
        input_summary={
            "candidates": len(state.get("candidates", [])),
            "viable_plans": len(viable),
        },
    )
    state["scored_plans"] = res.data.get("plans", []) if res.data else []
    tracer.emit("node_end", node="score_plans", label="方案评分与排序", count=len(state["scored_plans"]))
    return state


def node_report(state: PlanningState, tracer: _Tracer) -> PlanningState:
    """结果生成：产出待审批方案与报告。"""
    state["current_node"] = "generate_report"
    tracer.emit("node_start", node="generate_report", label="结果生成")

    plans = state.get("scored_plans", [])
    if not plans:
        tracer.emit("node_end", node="generate_report", label="结果生成", report=None)
        return state

    best = plans[0]
    state["selected_plan"] = best
    state["idempotency_key"] = build_idempotency_key(
        state.get("task_id", ""), state.get("plan_version", 1), state.get("task_requirements", {})
    )

    res = tracer.tool(
        "generate_task_report",
        lambda: planning.generate_task_report(
            selected_plan=best,
            alternatives=plans[1:],
            constraint_result=state.get("constraint_result", {}),
        ),
        input_summary={"plan_id": best.get("plan_id")},
    )
    report = res.data.get("report") if res.data else None
    tracer.emit("node_end", node="generate_report", label="结果生成", plan=best, report=report)
    return state


def node_approval_gate(state: PlanningState, tracer: _Tracer) -> PlanningState:
    """人工审批关口。

    第一阶段只创建审批单并挂起，等待人工在界面操作后恢复。
    """
    state["current_node"] = "approval_gate"
    tracer.emit("node_start", node="approval_gate", label="人工审批关口")

    best = state.get("selected_plan")
    if not best:
        state["approval_status"] = "not_required"
        tracer.emit("node_end", node="approval_gate", label="人工审批关口", skipped=True)
        return state

    res = tracer.tool(
        "submit_approval",
        lambda: submit_approval(
            ApprovalRequest(
                target_type="observation_plan",
                target_id=best.get("plan_id", ""),
                plan_version=state.get("plan_version", 1),
                summary=f"观测方案 {best.get('satellite_name')}｜评分 {best.get('score')}",
                snapshot=best,
            )
        ),
        input_summary={"plan_id": best.get("plan_id")},
    )
    if res.data:
        state["approval_id"] = res.data.get("approval_id")
        state["approval_status"] = "pending"
    state["awaiting_approval"] = True
    tracer.emit(
        "node_end",
        node="approval_gate",
        label="人工审批关口",
        approval_id=state.get("approval_id"),
        status="pending",
    )
    return state


def node_execute(state: PlanningState, tracer: _Tracer) -> PlanningState:
    """仿真执行（第一阶段为占位，第二阶段接幂等提交）。

    执行前强制校验审批状态——未审批绝不执行。
    """
    state["current_node"] = "execute"
    tracer.emit("node_start", node="execute", label="仿真执行")
    if state.get("approval_status") != "approved":
        tracer.emit(
            "node_end",
            node="execute",
            label="仿真执行",
            status="blocked",
            reason=f"审批状态 {state.get('approval_status')}，未获批准不执行",
        )
        return state
    state["execution_result"] = {"status": "succeeded", "note": "第一阶段占位执行"}
    tracer.emit("node_end", node="execute", label="仿真执行", status="succeeded")
    return state


class _NullTracer:
    """无事件订阅时的空追踪器（图可用于离线/测试调用）。"""

    def emit(self, event: str, **payload: Any) -> None:
        return None

    def tool(
        self,
        tool_name: str,
        fn: Callable[[], Any],
        input_summary: Any = None,
        tool_version: str = "v1",
    ) -> Any:
        return fn()


def _wrap(fn: Callable[[PlanningState, _Tracer], PlanningState], tracer: _Tracer) -> Callable[[PlanningState], PlanningState]:
    """把 (state, tracer) 签名的节点适配成 LangGraph 要求的 (state) 签名。

    LangGraph 只用一个参数调用节点，tracer 必须由闭包捕获。
    """

    def node(state: PlanningState) -> PlanningState:
        return fn(state, tracer)

    node.__name__ = getattr(fn, "__name__", "node")
    return node


# ---------------------------------------------------------------- 图定义

def _route_by_type(state: PlanningState) -> str:
    """条件边：根据 task_type 决定进入哪条主干。"""
    t = state.get("task_type")
    if t == "sar":
        return "sar_analyze"
    if t == "order":
        return "order_schedule"
    return "parse_requirements"


def build_graph(tracer: "_Tracer | None" = None):
    """构建 LangGraph 状态图。

    若 langgraph 未安装，返回 None，由 API 层回退到内置顺序执行器，
    保证在无依赖环境下也能跑通。

    tracer 通过闭包注入：LangGraph 只用单参数调用节点，
    而本项目的节点签名是 (state, tracer)。
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        logger.warning("langgraph 未安装，使用内置顺序执行器")
        return None

    tr = tracer or _NullTracer()

    g = StateGraph(PlanningState)
    g.add_node("route", _wrap(node_route, tr))
    g.add_node("parse_requirements", _wrap(node_parse_requirements, tr))
    g.add_node("query_satellites", _wrap(node_query_satellites, tr))
    g.add_node("query_windows", _wrap(node_query_windows, tr))
    g.add_node("validate", _wrap(node_validate, tr))
    g.add_node("score", _wrap(node_score, tr))
    g.add_node("report", _wrap(node_report, tr))
    g.add_node("approval_gate", _wrap(node_approval_gate, tr))
    g.add_node("execute", _wrap(node_execute, tr))
    g.add_node("sar_analyze", _wrap(sar_nodes.node_sar_analyze, tr))
    g.add_node("sar_execute", _wrap(sar_nodes.node_sar_execute, tr))
    g.add_node("order_schedule", _wrap(order_nodes.node_order_schedule, tr))

    g.add_edge(START, "route")
    # 条件分流：SAR 走影像分析链，订单走 TSLC 订单调度链，规划走规划链
    g.add_conditional_edges(
        "route",
        _route_by_type,
        {
            "parse_requirements": "parse_requirements",
            "sar_analyze": "sar_analyze",
            "order_schedule": "order_schedule",
        },
    )
    g.add_edge("parse_requirements", "query_satellites")
    g.add_edge("query_satellites", "query_windows")
    g.add_edge("query_windows", "validate")
    g.add_edge("validate", "score")
    g.add_edge("score", "report")
    g.add_edge("report", "approval_gate")
    g.add_edge("approval_gate", END)  # 审批关口挂起

    # SAR 链：分析（含选型 + 审批单创建）→ 挂起等待人工审批
    g.add_edge("sar_analyze", END)

    # 订单链：TSLC 三阶段调度 → 内部已创建审批单 → 挂起等待人工审批
    g.add_edge("order_schedule", END)

    g.add_edge("execute", END)
    g.add_edge("sar_execute", END)

    return g


# 内置顺序执行器：langgraph 缺失时的降级路径
BUILTIN_ORDER = [
    node_route,
    node_parse_requirements,
    node_query_satellites,
    node_query_windows,
    node_validate,
    node_score,
    node_report,
    node_approval_gate,
]

BUILTIN_ORDER_SAR = [
    node_route,
    sar_nodes.node_sar_analyze,
]

BUILTIN_ORDER_ORDER = [
    node_route,
    order_nodes.node_order_schedule,
]


def run_builtin(state: PlanningState, tracer: _Tracer) -> PlanningState:
    """按内置顺序执行节点（langgraph 缺失时的降级路径）。

    同样按 task_type 分流，保证三条链行为一致。
    """
    t = state.get("task_type")
    if t == "sar":
        order = BUILTIN_ORDER_SAR
    elif t == "order":
        order = BUILTIN_ORDER_ORDER
    else:
        order = BUILTIN_ORDER
    for fn in order:
        state = fn(state, tracer)
    return state
