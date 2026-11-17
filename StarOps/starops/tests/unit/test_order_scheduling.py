"""TSLC 订单驱动调度测试。

对应论文: Chen J. et al., "A Tri-Stage LLM-Coordinated Framework for Order-Driven
Scheduling of Earth Observation Satellite Tasks", IEEE TGRS 2026,
DOI: 10.1109/TGRS.2026.3700605。

覆盖三阶段决策、分阶段约束校验、资源独占与转移时间、PM 重规划、
MS/K 记忆更新、评估指标与不可行订单的可解释拒绝。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP_DB = Path(tempfile.mkdtemp(prefix="starops_tslc_")) / "test.db"
os.environ["STAROPS_DB_PATH"] = str(_TMP_DB)

from database import init_db  # noqa: E402
from tools import order_scheduling as os_sched  # noqa: E402

# 固定时间基准，避免依赖真实当前时刻
D1 = "2027-03-01T00:00:00+00:00"
D1_END = "2027-03-02T00:00:00+00:00"


def sat(sid: str, res: float = 1.0, trans: float = 10.0, min_dur: float = 5.0,
        stype: str = "optical") -> dict:
    return {
        "satellite_id": sid,
        "satellite_name": f"卫星 {sid}",
        "satellite_type": stype,
        "resolution": res,
        "imaging_mode": "多光谱",
        "status": "simulated",
        "transition_time_min": trans,
        "min_imaging_duration_min": min_dur,
    }


def win(wid: str, sid: str, start: str = D1, end: str = D1_END,
        area: str = "CN-T1", cloud: float = 0.1, inc: float = 30.0,
        quality: str = "FRESH") -> dict:
    return {
        "window_id": wid,
        "satellite_id": sid,
        "area_id": area,
        "area_name": area,
        "start_time": start,
        "end_time": end,
        "cloud_rate": cloud,
        "incidence_angle": inc,
        "quality": quality,
    }


def order(oid: str, **kw) -> dict:
    base = dict(
        area_id="CN-T1",
        area_name="测试区",
        time_start=D1,
        time_end=D1_END,
        satellite_type="optical",
        max_resolution=2.0,
        max_cloud_rate=0.3,
        max_incidence_angle=45.0,
        priority="normal",
        profit=2.0,
    )
    base.update(kw)
    return os_sched.build_order(oid, base.pop("area_id"), base.pop("area_name"),
                                base.pop("time_start"), base.pop("time_end"), **base)


def setup_db():
    init_db()


# ---------------------------------------------------------------- 三阶段决策

def test_stage1_assigns_matching_satellite():
    """Stage 1 — AS：只把满足类型与分辨率的卫星列为候选。"""
    sats = [sat("A", res=0.5), sat("B", res=3.0), sat("C", res=1.0, stype="sar")]
    cands, reasons = os_sched.stage_assign_satellite(order("o1", max_resolution=2.0), sats, {})
    ids = [c["satellite_id"] for c in cands]
    assert "A" in ids
    assert "B" not in ids, "分辨率超限的卫星不应成为候选"
    assert "C" not in ids, "类型不符的卫星不应成为候选"
    assert any("分辨率" in r for r in reasons)


def test_stage2_selects_window_within_order_range():
    """Stage 2 — VS：只选时间范围、区域、云量、入射角都合格的窗口。

    区域不匹配的窗口属于「不适用」而非「违规」，不进候选也不进 reasons，
    避免把真正的约束违反原因淹没。
    """
    wins = [
        win("W1", "A"),
        win("W2", "A", area="CN-OTHER"),
        win("W3", "A", cloud=0.8),
        win("W4", "A", inc=70.0),
        win("W5", "A", start="2027-06-01T00:00:00+00:00", end="2027-06-02T00:00:00+00:00"),
    ]
    cands, reasons = os_sched.stage_select_window(order("o1"), sat("A"), wins)
    assert [c["window_id"] for c in cands] == ["W1"]
    joined = "；".join(reasons)
    assert "云量" in joined and "入射角" in joined and "时间范围" in joined
    assert "区域" not in joined, "区域不匹配不应作为违规原因出现"


def test_stage3_determines_observation_slot_inside_window():
    """Stage 3 — ES：观测时刻必须落在窗口内且时长达标。"""
    slot, reasons = os_sched.stage_exact_schedule(order("o1"), sat("A"), win("W1", "A"), {})
    assert slot is not None
    assert slot["observation_start"] == D1
    assert slot["observation_end"] > slot["observation_start"]
    assert not reasons


def test_stage3_respects_transition_time_from_previous_task():
    """ES：与该星前序任务之间必须留出姿态转移时间。"""
    from datetime import datetime, timedelta, timezone

    prev_end = datetime.fromisoformat(D1) + timedelta(minutes=30)
    occupancy = {"A": [(datetime.fromisoformat(D1), prev_end)]}
    slot, _ = os_sched.stage_exact_schedule(order("o1"), sat("A", trans=15.0), win("W1", "A"), occupancy)
    assert slot is not None
    start = datetime.fromisoformat(slot["observation_start"])
    assert start >= prev_end + timedelta(minutes=15), "观测时刻必须让开转移时间"


# ---------------------------------------------------------------- CV 分阶段校验

def test_cv_stage_checks_are_independent():
    """CV：三阶段各自独立判定，硬约束不可被后续阶段覆盖。"""
    cv = os_sched.ConstraintVerifier()
    o = order("o1", max_resolution=1.0, max_cloud_rate=0.2, max_incidence_angle=40.0)

    ok, v = cv.check_stage1(o, sat("A", res=5.0))
    assert not ok and any("分辨率" in x for x in v)

    ok, v = cv.check_stage2(o, sat("A"), win("W1", "A", cloud=0.5))
    assert not ok and any("云量" in x for x in v)

    ok, v = cv.check_stage3(o, sat("A"), win("W1", "A"), {
        "observation_start": "2027-03-03T00:00:00+00:00",
        "observation_end": "2027-03-03T00:05:00+00:00",
    }, {})
    assert not ok and any("越出窗口边界" in x for x in v)


def test_cv_detects_resource_conflict_between_orders():
    """CV：同一卫星同一时刻不可被两个订单占用。"""
    from datetime import datetime

    cv = os_sched.ConstraintVerifier()
    occupancy = {"A": [(datetime.fromisoformat(D1), datetime.fromisoformat(D1_END))]}
    ok, v = cv.check_stage3(order("o1"), sat("A"), win("W1", "A"), {
        "observation_start": D1,
        "observation_end": "2027-03-01T00:05:00+00:00",
    }, occupancy)
    assert not ok
    assert any("冲突" in x for x in v)


# ---------------------------------------------------------------- PM 重规划

def test_pm_replans_to_another_satellite_after_failure():
    """PM：首选卫星不可行时应换星重规划，而不是直接判死。"""
    sats = [sat("A", res=0.5), sat("B", res=1.5)]
    wins = [win("WA", "A", cloud=0.9), win("WB", "B")]
    res = os_sched.schedule_orders([order("o1")], sats, wins)
    d = res.data
    assert d["metrics"]["scheduled_orders"] == 1
    assert d["schedule"][0]["satellite_id"] == "B"
    assert d["metrics"]["replans"] >= 1


def test_infeasible_order_is_rejected_with_reason():
    """不可行订单必须被拒并给出根因，不得静默丢弃。"""
    sats = [sat("A", res=5.0)]  # 分辨率远不满足
    res = os_sched.schedule_orders([order("o1", max_resolution=1.0)], sats, [win("W1", "A")])
    d = res.data
    assert d["metrics"]["scheduled_orders"] == 0
    assert d["rejected"][0]["reasons"]
    assert d["rejected"][0]["failed_stage"] in {"AS", "VS", "ES"}


def test_resource_exclusive_across_orders():
    """资源独占：窗口只够放一次成像时，第二个订单必须被拒。

    窗口设为 18 分钟：一次成像 5 分钟 + 转移时间 10 分钟，
    放完第一单后剩余时间不足第二单。
    """
    sats = [sat("A", trans=10.0, min_dur=5.0)]
    wins = [win("W1", "A", end="2027-03-01T00:18:00+00:00")]
    res = os_sched.schedule_orders([order("o1"), order("o2")], sats, wins)
    d = res.data
    assert d["metrics"]["scheduled_orders"] == 1, "同一卫星同一窗口不可被两个订单占用"
    assert d["metrics"]["rejected_orders"] == 1
    assert d["rejected"][0]["reasons"]


# ---------------------------------------------------------------- MS / K

def test_memory_and_knowledge_are_updated():
    """MS 记录失败尝试，K 记录已确认计划。"""
    sats = [sat("A", res=0.5), sat("B", res=1.5)]
    wins = [win("WA", "A", cloud=0.9), win("WB", "B")]
    res = os_sched.schedule_orders([order("o1")], sats, wins)
    d = res.data
    assert d["memory"], "重规划过程必须留下失败记忆"
    assert d["knowledge"], "确认计划必须写入知识库"
    assert d["knowledge"][0]["order_id"] == "o1"


def test_knowledge_only_contains_confirmed_plans():
    """K 中不得混入被拒订单。"""
    sats = [sat("A", res=5.0)]
    res = os_sched.schedule_orders([order("o1", max_resolution=1.0)], sats, [win("W1", "A")])
    d = res.data
    assert d["knowledge"] == []


def test_plan_signature_is_stable_and_discriminating():
    """计划指纹：同一计划同指纹，不同观测时刻不同指纹。"""
    p1 = {"satellite_id": "A", "window_id": "W1",
          "observation_start": "2027-03-01T10:00:00+00:00",
          "observation_end": "2027-03-01T10:10:00+00:00"}
    p2 = dict(p1)
    p3 = {**p1, "observation_start": "2027-03-01T11:00:00+00:00"}
    assert os_sched.plan_signature(p1) == os_sched.plan_signature(p2)
    assert os_sched.plan_signature(p1) != os_sched.plan_signature(p3)


def test_confirmed_plan_carries_signature():
    """写入 K 的计划必须带 plan_signature，供落库去重使用。"""
    sats = [sat("A")]
    res = os_sched.schedule_orders([order("o1")], sats, [win("W1", "A")])
    kb = res.data["knowledge"]
    assert kb and kb[0].get("plan_signature"), "K 中计划必须带计划指纹"


def test_knowledge_is_deduplicated_by_signature():
    """同一条观测计划重复确认，知识库里只保留一条。"""
    from agents.nodes.order_nodes import persist_knowledge
    from database import query_all

    init_db()
    sats = [sat("A")]
    res = os_sched.schedule_orders([order("o1")], sats, [win("W1", "A")])
    kb = res.data["knowledge"]
    assert kb

    added = persist_knowledge(kb)
    assert added == 1, "首次入库应新增 1 条"
    rows = query_all("schedule_knowledge", "plan_signature = ?",
                     (kb[0]["plan_signature"],), limit=10)
    assert len(rows) == 1

    # 同一计划再确认一次：只更新，不重复入库
    added_again = persist_knowledge(kb)
    assert added_again == 0, "重复确认不得新增知识条目"
    rows = query_all("schedule_knowledge", "plan_signature = ?",
                     (kb[0]["plan_signature"],), limit=10)
    assert len(rows) == 1, "知识库中同一计划只能有一条记录"


# ---------------------------------------------------------------- 指标

def test_metrics_are_consistent():
    """评估指标与实际结果一致。"""
    sats = [sat("A"), sat("B")]
    wins = [win("WA", "A"), win("WB", "B")]
    res = os_sched.schedule_orders([order("o1"), order("o2")], sats, wins)
    m = res.data["metrics"]
    assert m["total_orders"] == 2
    assert m["scheduled_orders"] + m["rejected_orders"] == m["total_orders"]
    assert 0.0 <= m["completion_rate"] <= 1.0
    assert m["atomic_decisions"] > 0
    assert m["rule_version"] == os_sched.ConstraintVerifier.VERSION
    assert "TSLC" in m["method"]


def test_empty_order_queue_is_handled():
    """空订单队列不报错。"""
    res = os_sched.schedule_orders([], [sat("A")], [win("W1", "A")])
    assert res.status == "success"
    assert res.data["metrics"]["total_orders"] == 0
    assert res.data["metrics"]["completion_rate"] == 0.0


# ---------------------------------------------------------------- 事件流

def test_events_cover_three_stages():
    """SSE 事件必须覆盖三阶段决策与最终结果。"""
    sats = [sat("A"), sat("B")]
    wins = [win("WA", "A"), win("WB", "B")]
    events: list[dict] = []
    os_sched.schedule_orders([order("o1")], sats, wins, on_event=events.append)
    kinds = {e["event"] for e in events}
    assert "order_batch_start" in kinds
    assert "stage_decision" in kinds
    assert "order_scheduled" in kinds
    assert "order_batch_end" in kinds
    stages = {e.get("stage") for e in events if e["event"] == "stage_decision"}
    assert {"AS", "VS", "ES"} <= stages, "三阶段都必须有决策事件"


def test_agent_prompts_define_all_agents():
    """每个 agent 都有 name/ability/responsibilities 三要素的 prompt。"""
    for key in ("AS", "VS", "ES", "CV", "PM"):
        p = os_sched.AGENT_PROMPTS[key]
        assert "能力" in p and "职责" in p
