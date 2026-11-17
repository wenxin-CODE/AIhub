"""安全闭环测试：审批 / 硬约束 / 幂等。

对应方案 §10.1 的 T02、T04、T08、T09、T10、T11、T12。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# 必须在导入 database 之前指定测试库路径
_TMP_DB = Path(tempfile.mkdtemp(prefix="starops_test_")) / "test.db"
os.environ["STAROPS_DB_PATH"] = str(_TMP_DB)

from database import init_db, insert, query_one, query_all  # noqa: E402
from schemas.models import ApprovalRequest  # noqa: E402
from services.approval import (  # noqa: E402
    build_idempotency_key,
    commit_plan,
    review_approval,
    submit_approval,
)
from tools import planning  # noqa: E402


@pytest.fixture(autouse=True)
def _db():
    init_db()
    yield


@pytest.fixture
def approval_id():
    res = submit_approval(
        ApprovalRequest(
            target_type="observation_plan",
            target_id="plan-test-001",
            plan_version=1,
            summary="测试审批单",
            snapshot={"plan_id": "plan-test-001"},
        )
    )
    return res.data["approval_id"]


# ---------------------------------------------------------------- T09 审批拒绝

def test_t09_rejected_approval_blocks_execution(approval_id):
    """T09：用户拒绝审批 → 执行被拒绝，原状态不变。"""
    review_approval(approval_id, "rejected", "operator", "条件不满足")

    res = commit_plan(
        task_id="task-001",
        plan_version=1,
        idempotency_key=build_idempotency_key("task-001", 1, {}),
        payload={"x": 1},
        approval_id=approval_id,
    )
    assert res.status == "failed"
    assert "未获批准" in (res.reason or "")

    # 审计里应记录拒绝动作
    actions = [r["action"] for r in query_all("audit_log")]
    assert "approval_rejected" in actions
    assert "commit_rejected" in actions


# ---------------------------------------------------------------- T10 审批通过

def test_t10_approved_approval_allows_execution(approval_id):
    """T10：批准后允许执行并写审计。"""
    review_approval(approval_id, "approved", "operator", "同意")
    key = build_idempotency_key("task-001", 1, {"a": 1})

    res = commit_plan(
        task_id="task-001",
        plan_version=1,
        idempotency_key=key,
        payload={"x": 1},
        approval_id=approval_id,
    )
    assert res.status == "success"
    assert res.data["idempotent_hit"] is False
    assert any(r["action"] == "commit_succeeded" for r in query_all("audit_log"))


# ---------------------------------------------------------------- T11 幂等

def test_t11_idempotent_key_prevents_duplicate_execution(approval_id):
    """T11：同一幂等键重复提交 → 返回首次结果，不重复执行。"""
    review_approval(approval_id, "approved")
    key = build_idempotency_key("task-001", 1, {"a": 1})

    first = commit_plan("task-001", 1, key, {"x": 1}, approval_id=approval_id)
    second = commit_plan("task-001", 1, key, {"x": 1}, approval_id=approval_id)

    assert first.status == "success"
    assert second.status == "success"
    assert second.data["idempotent_hit"] is True
    assert second.data["commit_id"] == first.data["commit_id"]

    # 只应有一条提交记录
    assert len(query_all("plan_commit")) == 1


# ---------------------------------------------------------------- 版本不匹配

def test_version_mismatch_is_rejected(approval_id):
    """方案版本与审批版本不一致 → 拒绝执行。"""
    review_approval(approval_id, "approved")
    res = commit_plan(
        "task-001",
        2,  # 审批的是版本 1
        build_idempotency_key("task-001", 2, {}),
        {"x": 1},
        approval_id=approval_id,
    )
    assert res.status == "failed"
    assert "版本不匹配" in (res.reason or "")


# ---------------------------------------------------------------- 无审批单

def test_commit_without_approval_is_rejected():
    """缺少审批单 → 直接拒绝。"""
    res = commit_plan(
        "task-001", 1, build_idempotency_key("task-001", 1, {}), {"x": 1}
    )
    assert res.status == "failed"
    assert "审批单" in (res.reason or "")


# ---------------------------------------------------------------- T08 硬约束

def test_t08_constraint_violation_rejected():
    """T08：违反硬约束的方案被校验器拒绝。"""
    candidates = [
        {
            "satellite_id": "SAT-001",
            "satellite_name": "测试卫星",
            "satellite_type": "optical",
            "resolution": 3.0,
            "imaging_mode": "多光谱",
        }
    ]
    windows = [
        {
            "window_id": "win-001",
            "satellite_id": "SAT-001",
            "area_id": "CN-BJ",
            "start_time": "2099-01-01T00:00:00+00:00",
            "end_time": "2099-01-01T00:10:00+00:00",
            "cloud_rate": 0.8,       # 超出云量上限
            "incidence_angle": 20.0,
            "quality": "FRESH",
        }
    ]
    res = planning.validate_constraints(
        requirements={"max_resolution": 2.0, "satellite_type": "optical"},
        candidates=candidates,
        windows=windows,
    )
    # 该方案同时违反分辨率与云量约束 → 被淘汰
    assert res.data["viable_count"] == 0
    assert res.data["rejected_count"] == 1
    assert res.data["no_solution"] is True
    reasons = res.data["rejected_plans"][0]["reasons"]
    assert any("分辨率" in r for r in reasons)
    assert any("云量" in r for r in reasons)


# ---------------------------------------------------------------- T04 过期窗口

def test_t04_expired_window_is_violation():
    """T04：EXPIRED 窗口被硬性排除。"""
    windows = [
        {
            "window_id": "win-old",
            "satellite_id": "SAT-001",
            "area_id": "CN-BJ",
            "start_time": "2000-01-01T00:00:00+00:00",
            "end_time": "2000-01-01T00:10:00+00:00",
            "cloud_rate": 0.1,
            "incidence_angle": 20.0,
            "quality": "EXPIRED",
        }
    ]
    res = planning.validate_constraints(
        requirements={"satellite_type": "optical"},
        candidates=[],
        windows=windows,
    )
    # 过期窗口必须被报告，不能静默丢弃
    assert res.data["expired_count"] == 1
    assert any("已过期" in v for v in res.data["violations"])
    assert res.data["no_solution"] is True


# ---------------------------------------------------------------- 幂等键稳定性

def test_idempotency_key_is_order_insensitive():
    """归一化保证字段顺序不影响幂等键。"""
    k1 = build_idempotency_key("t", 1, {"a": 1, "b": 2})
    k2 = build_idempotency_key("t", 1, {"b": 2, "a": 1})
    assert k1 == k2

    k3 = build_idempotency_key("t", 2, {"a": 1, "b": 2})
    assert k1 != k3
