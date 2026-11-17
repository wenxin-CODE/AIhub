"""审批服务与幂等提交。

安全闭环的核心：规划/生成与执行彻底分离，执行必须校验审批状态 + 方案版本 + 幂等键。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from database import (
    insert,
    new_id,
    query_all,
    query_one,
    record_audit,
    update,
    utcnow,
)
from schemas.models import ApprovalRecord, ApprovalRequest, CommitResult, ToolResult


# ---------------------------------------------------------------- 审批

def submit_approval(req: ApprovalRequest, actor: str = "agent") -> ToolResult:
    """创建审批单。"""
    approval_id = new_id("appr")
    insert(
        "approval",
        {
            "approval_id": approval_id,
            "target_type": req.target_type,
            "target_id": req.target_id,
            "plan_version": req.plan_version,
            "status": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "comment": None,
            "snapshot": json.dumps(req.snapshot, ensure_ascii=False, default=str),
            "created_at": utcnow(),
        },
    )
    record_audit(actor, "submit_approval", approval_id, req.model_dump())
    return ToolResult.ok(
        data={"approval_id": approval_id, "status": "pending", "plan_version": req.plan_version}
    )


def get_approval(approval_id: str) -> dict[str, Any] | None:
    row = query_one("approval", "approval_id = ?", (approval_id,))
    if row and row.get("snapshot"):
        try:
            row["snapshot"] = json.loads(row["snapshot"])
        except json.JSONDecodeError:
            pass
    return row


def review_approval(
    approval_id: str,
    decision: str,
    reviewer: str = "operator",
    comment: str = "",
) -> ToolResult:
    """审批决策：approved / rejected。

    拒绝时保持原状态，只记录原因，不自动改写方案。
    """
    if decision not in {"approved", "rejected"}:
        return ToolResult.fail(f"非法审批结论：{decision}")

    row = query_one("approval", "approval_id = ?", (approval_id,))
    if not row:
        return ToolResult.fail(f"审批单不存在：{approval_id}")
    if row["status"] != "pending":
        return ToolResult.fail(f"审批单已处理（当前状态 {row['status']}），不可重复审批")

    update(
        "approval",
        "approval_id = ?",
        (approval_id,),
        {
            "status": decision,
            "reviewer": reviewer,
            "reviewed_at": utcnow(),
            "comment": comment,
        },
    )
    record_audit(
        reviewer,
        f"approval_{decision}",
        approval_id,
        {"target_id": row["target_id"], "comment": comment},
    )
    return ToolResult.ok(
        data={"approval_id": approval_id, "status": decision, "reviewer": reviewer}
    )


def pending_approvals() -> list[dict[str, Any]]:
    return query_all(
        "approval", "status = 'pending'", order_by="created_at DESC"
    )


# ---------------------------------------------------------------- 幂等

def build_idempotency_key(task_id: str, plan_version: int, constraints: dict[str, Any]) -> str:
    """幂等键 = hash(task_id + plan_version + 归一化约束)。

    归一化保证字段顺序不影响结果。
    """
    normalized = json.dumps(constraints, ensure_ascii=False, sort_keys=True, default=str)
    raw = f"{task_id}|{plan_version}|{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def commit_plan(
    task_id: str,
    plan_version: int,
    idempotency_key: str,
    payload: dict[str, Any],
    approval_id: str | None = None,
    actor: str = "system",
) -> ToolResult:
    """幂等提交。

    三重校验：
    1. 审批状态必须为 approved（未审批直接拒绝并写审计）
    2. 方案版本必须匹配
    3. 幂等键命中则返回首次结果，不重复执行副作用
    """
    # ---- 校验 1：审批 ----
    if approval_id:
        appr = query_one("approval", "approval_id = ?", (approval_id,))
        if not appr:
            record_audit(actor, "commit_rejected", task_id, {"reason": "审批单不存在"})
            return ToolResult.fail(f"审批单不存在：{approval_id}")
        if appr["status"] != "approved":
            record_audit(
                actor,
                "commit_rejected",
                task_id,
                {"reason": f"审批状态 {appr['status']}", "approval_id": approval_id},
            )
            return ToolResult.fail(
                f"未获批准，拒绝执行（审批状态：{appr['status']}）"
            )
        # ---- 校验 2：版本 ----
        if int(appr["plan_version"]) != int(plan_version):
            record_audit(
                actor,
                "commit_rejected",
                task_id,
                {"reason": "方案版本不匹配", "expected": appr["plan_version"], "got": plan_version},
            )
            return ToolResult.fail(
                f"方案版本不匹配：审批版本 {appr['plan_version']}，提交版本 {plan_version}"
            )
    else:
        record_audit(actor, "commit_rejected", task_id, {"reason": "缺少审批单"})
        return ToolResult.fail("缺少审批单，拒绝执行")

    # ---- 校验 3：幂等 ----
    existing = query_one("plan_commit", "idempotency_key = ?", (idempotency_key,))
    if existing:
        result = json.loads(existing["result"]) if existing.get("result") else {}
        return ToolResult.ok(
            data=CommitResult(
                commit_id=existing["commit_id"],
                task_id=existing["task_id"],
                plan_version=existing["plan_version"],
                status=existing["status"],
                idempotent_hit=True,
                result=result,
            ).model_dump(),
            reason="幂等键命中，返回首次执行结果，未重复执行副作用",
        )

    # ---- 真正执行 ----
    commit_id = new_id("commit")
    exec_result = {
        "task_id": task_id,
        "plan_version": plan_version,
        "applied_at": utcnow(),
        "payload": payload,
    }
    insert(
        "plan_commit",
        {
            "commit_id": commit_id,
            "task_id": task_id,
            "plan_version": plan_version,
            "idempotency_key": idempotency_key,
            "status": "succeeded",
            "result": json.dumps(exec_result, ensure_ascii=False, default=str),
            "created_at": utcnow(),
        },
    )
    record_audit(
        actor,
        "commit_succeeded",
        task_id,
        {"commit_id": commit_id, "idempotency_key": idempotency_key, "plan_version": plan_version},
    )
    return ToolResult.ok(
        data=CommitResult(
            commit_id=commit_id,
            task_id=task_id,
            plan_version=plan_version,
            status="succeeded",
            idempotent_hit=False,
            result=exec_result,
        ).model_dump()
    )
