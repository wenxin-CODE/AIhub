"""TSLC 订单驱动调度节点。

参考: Chen J. et al., IEEE TGRS 2026, DOI 10.1109/TGRS.2026.3700605
（"A Tri-Stage LLM-Coordinated Framework for Order-Driven Scheduling of
Earth Observation Satellite Tasks"）。

节点职责：
1. 载入订单队列（satellite_order 表）
2. 调用 :func:`tools.order_scheduling.schedule_orders` 执行三阶段序列决策
   （AS 指派卫星 → VS 选择有效时窗 → ES 确定精确观测时刻），
   由 CV 分阶段校验、PM 迭代验证与重规划
3. 落库：MS（失败记忆）、K（知识库）、订单状态
4. 生成调度报告并送人工审批关口——保留项目主链的安全闭环
"""

from __future__ import annotations

import json
from typing import Any

from database import insert, new_id, query_all, record_audit, update, utcnow
from schemas.models import ApprovalRequest
from services.approval import submit_approval
from tools import order_scheduling


def rows_to_orders(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 satellite_order 表行转成调度内核用的订单结构。"""
    return [order_scheduling.build_order(
        r["order_id"],
        r["area_id"],
        r.get("area_name") or r["area_id"],
        r["time_start"],
        r["time_end"],
        satellite_type=r.get("satellite_type") or "any",
        max_resolution=r.get("max_resolution"),
        max_cloud_rate=r.get("max_cloud_rate"),
        max_incidence_angle=r.get("max_incidence_angle"),
        priority=r.get("priority") or "normal",
        profit=r.get("profit"),
        min_duration_min=r.get("min_duration_min"),
    ) for r in rows]


def _load_orders(state: dict[str, Any]) -> list[dict[str, Any]]:
    """优先用 state 里的订单；否则从数据库读待处理订单。"""
    orders = state.get("orders") or []
    if orders:
        return orders
    rows = query_all("satellite_order", "status = 'pending'", order_by="created_at ASC")
    return rows_to_orders(rows)


def persist_knowledge(knowledge: list[dict[str, Any]]) -> int:
    """把已确认计划写入知识库 K，按 `plan_signature` 去重。

    同一条观测计划（卫星 | 窗口 | 观测起止时刻）无论被确认多少次，
    知识库里只保留一条——重复确认只更新来源订单与确认时间。
    返回实际新增的条数。
    """
    added = 0
    for p in knowledge:
        signature = p.get("plan_signature") or order_scheduling.plan_signature(p)
        existing = query_all(
            "schedule_knowledge", "plan_signature = ?", (signature,), limit=1
        )
        if existing:
            # 同一观测计划已确认过：更新确认时间与来源订单，不重复入库
            update(
                "schedule_knowledge",
                "knowledge_id = ?",
                (existing[0]["knowledge_id"],),
                {
                    "order_id": p["order_id"],
                    "plan": json.dumps(p, ensure_ascii=False, default=str),
                    "created_at": utcnow(),
                },
            )
            continue
        insert(
            "schedule_knowledge",
            {
                "knowledge_id": new_id("kb"),
                "order_id": p["order_id"],
                "satellite_id": p["satellite_id"],
                "window_id": p["window_id"],
                "plan": json.dumps(p, ensure_ascii=False, default=str),
                "plan_signature": signature,
                "created_at": utcnow(),
            },
        )
        added += 1
    return added


def node_order_schedule(state: dict[str, Any], tracer: Any) -> dict[str, Any]:
    """TSLC 订单驱动调度主节点。"""
    state["current_node"] = "order_schedule"
    tracer.emit("node_start", node="order_schedule", label="订单驱动调度（TSLC 三阶段）")

    orders = _load_orders(state)
    state["orders"] = orders
    if not orders:
        tracer.emit(
            "node_end",
            node="order_schedule",
            label="订单驱动调度（TSLC 三阶段）",
            error="订单队列为空",
        )
        return state

    def on_event(evt: dict[str, Any]) -> None:
        payload = {k: v for k, v in evt.items() if k != "event"}
        tracer.emit(evt.get("event", "order_event"), **payload)

    res = tracer.tool(
        "schedule_orders",
        lambda: order_scheduling.schedule_orders(
            orders,
            max_replans=int(state.get("max_replans", 3) or 3),
            on_event=on_event,
        ),
        input_summary={"orders": len(orders), "max_replans": state.get("max_replans", 3)},
        tool_version="tslc-v1",
    )

    data = res.data or {}
    schedule = data.get("schedule", [])
    rejected = data.get("rejected", [])
    metrics = data.get("metrics", {})
    memory = data.get("memory", [])
    knowledge = data.get("knowledge", [])

    state["schedule"] = schedule
    state["rejected_orders"] = rejected
    state["schedule_metrics"] = metrics
    state["schedule_memory"] = memory
    state["schedule_knowledge"] = knowledge

    # ---- 落库：MS（失败记忆）与 K（知识库）----
    for m in memory:
        insert(
            "schedule_memory",
            {
                "memory_id": new_id("mem"),
                "order_id": m["order_id"],
                "stage": m["stage"],
                "reason": m["reason"][:500],
                "attempt": m.get("attempt", 1),
                "created_at": utcnow(),
            },
        )
    persist_knowledge(knowledge)

    # ---- 订单状态回写 ----
    for p in schedule:
        update("satellite_order", "order_id = ?", (p["order_id"],), {"status": "scheduled"})
    for r in rejected:
        update("satellite_order", "order_id = ?", (r["order_id"],), {"status": "rejected"})

    # ---- 报告 ----
    report = order_scheduling.summarize_schedule(data)
    state["execution_result"] = {"report": report, "metrics": metrics}

    # ---- 人工审批关口（论文没有这一环，本项目保留）----
    if schedule:
        appr = tracer.tool(
            "submit_approval",
            lambda: submit_approval(
                ApprovalRequest(
                    target_type="observation_plan",
                    target_id=f"schedule-{state.get('task_id', '')}",
                    plan_version=state.get("plan_version", 1),
                    summary=(
                        f"TSLC 调度计划：{metrics.get('scheduled_orders')}/{metrics.get('total_orders')} "
                        f"订单已排期，完成率 {metrics.get('completion_rate')}"
                    ),
                    snapshot={
                        "schedule": schedule[:20],
                        "rejected": rejected[:20],
                        "metrics": metrics,
                    },
                )
            ),
            input_summary={"scheduled": len(schedule)},
        )
        if appr.data:
            state["approval_id"] = appr.data.get("approval_id")
            state["approval_status"] = "pending"
            state["awaiting_approval"] = True

    record_audit(
        "agent",
        "order_schedule",
        state.get("task_id", ""),
        {
            "metrics": metrics,
            "scheduled": len(schedule),
            "rejected": len(rejected),
        },
    )

    tracer.emit(
        "node_end",
        node="order_schedule",
        label="订单驱动调度（TSLC 三阶段）",
        scheduled=len(schedule),
        rejected=len(rejected),
        metrics=metrics,
        report=report,
        approval_id=state.get("approval_id"),
    )
    return state
