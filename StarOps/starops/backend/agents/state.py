"""LangGraph 状态定义。

对应方案 §3.3。用结构化 State 承载任务上下文，不把信息全塞进对话历史。

合并语义说明（重要）
--------------------
所有节点都返回**完整 state**，因此这里一律采用「后写覆盖」（last-write-wins），
而不是追加型 reducer：

- 追加型 reducer（Annotated[List, merge_list]）会让 LangGraph 把每个节点的
  完整返回值再追加一遍，导致 candidates / schedule 等列表出现重复项。
- 改为覆盖语义后，LangGraph 路径与内置顺序执行器（run_builtin）行为完全
  一致——后者本来就是 `state = fn(state, tracer)` 的整份替换。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict  # type: ignore


class PlanningState(TypedDict, total=False):
    """双 Agent 共享的任务状态。"""

    # ---- 基础 ----
    task_id: str
    trace_id: str
    user_request: str
    task_type: str  # planning | sar | order
    messages: List[Any]
    errors: List[Dict[str, Any]]

    # ---- 需求与数据 ----
    task_requirements: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    windows: List[Dict[str, Any]]
    data_quality: Dict[str, Any]
    image_metadata: Dict[str, Any]
    image_metrics: Dict[str, Any]
    image_path: str

    # ---- 规划与校验 ----
    scored_plans: List[Dict[str, Any]]
    candidate_pipelines: List[Dict[str, Any]]
    rejected_pipelines: List[Dict[str, Any]]
    selected_plan: Optional[Dict[str, Any]]
    selected_pipeline: Optional[Dict[str, Any]]
    constraint_result: Dict[str, Any]
    diagnosis: Dict[str, Any]

    # ---- 审批与执行 ----
    approval_id: Optional[str]
    approval_status: str  # pending / approved / rejected / not_required
    plan_version: int
    idempotency_key: Optional[str]
    execution_result: Dict[str, Any]
    verification_result: Dict[str, Any]

    # ---- 证据与知识 ----
    evidence: List[Dict[str, Any]]
    retrieved_docs: List[Dict[str, Any]]

    # ---- TSLC 订单驱动调度 ----
    orders: List[Dict[str, Any]]
    schedule: List[Dict[str, Any]]
    rejected_orders: List[Dict[str, Any]]
    schedule_metrics: Dict[str, Any]
    schedule_memory: List[Dict[str, Any]]
    schedule_knowledge: List[Dict[str, Any]]
    max_replans: int

    # ---- 流程控制 ----
    current_node: str
    awaiting_approval: bool
    missing_fields: List[str]


def initial_state(task_id: str, trace_id: str, user_request: str) -> PlanningState:
    """构造初始状态。"""
    return PlanningState(
        task_id=task_id,
        trace_id=trace_id,
        user_request=user_request,
        task_type="",
        messages=[],
        errors=[],
        task_requirements={},
        candidates=[],
        windows=[],
        data_quality={},
        image_metadata={},
        image_metrics={},
        image_path="",
        scored_plans=[],
        candidate_pipelines=[],
        rejected_pipelines=[],
        selected_plan=None,
        selected_pipeline=None,
        constraint_result={},
        diagnosis={},
        approval_id=None,
        approval_status="not_required",
        plan_version=1,
        idempotency_key=None,
        execution_result={},
        verification_result={},
        evidence=[],
        retrieved_docs=[],
        orders=[],
        schedule=[],
        rejected_orders=[],
        schedule_metrics={},
        schedule_memory=[],
        schedule_knowledge=[],
        max_replans=3,
        current_node="START",
        awaiting_approval=False,
        missing_fields=[],
    )
