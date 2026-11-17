"""工作流与工具测试。

对应方案 §10.1 的 T01、T03、T06、T07、T14。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP_DB = Path(tempfile.mkdtemp(prefix="starops_wf_")) / "test.db"
os.environ["STAROPS_DB_PATH"] = str(_TMP_DB)

from agents.graph import run_builtin, _Tracer  # noqa: E402
from agents.state import initial_state  # noqa: E402
from database import init_db  # noqa: E402
from simulator.data_generator import seed_all  # noqa: E402
from tools import planning  # noqa: E402


@pytest.fixture(autouse=True)
def _setup():
    init_db()
    seed_all()
    yield


# ---------------------------------------------------------------- T01 正常流程

def test_t01_end_to_end_produces_plan():
    """T01：正常需求 → 产出候选方案，不报错。"""
    state = initial_state("task-001", "trace-001", "规划北京区域的观测任务")
    events: list[dict] = []
    state = run_builtin(state, _Tracer(state, events.append))

    assert state["task_type"] == "planning"
    assert len(state["candidates"]) > 0
    assert state["selected_plan"] is not None
    assert state["selected_plan"]["score"] >= 0
    assert state["awaiting_approval"] is True
    assert state["approval_status"] == "pending"


def test_t01_execution_blocked_without_approval():
    """未审批 → 状态仍为 pending，执行被阻断。"""
    state = initial_state("task-002", "trace-002", "规划北京区域的观测任务")
    state = run_builtin(state, _Tracer(state, None))
    assert state["approval_status"] != "approved"
    assert state["execution_result"] == {}


# ---------------------------------------------------------------- 事件流

def test_sse_events_are_emitted():
    """SSE 推送应覆盖节点与工具事件。"""
    state = initial_state("task-003", "trace-003", "规划北京的观测任务")
    events: list[dict] = []
    run_builtin(state, _Tracer(state, events.append))

    kinds = {e["event"] for e in events}
    assert "node_start" in kinds
    assert "node_end" in kinds
    assert "tool_start" in kinds
    assert "tool_end" in kinds

    tool_names = {e.get("tool_name") for e in events if e["event"] == "tool_start"}
    assert "query_satellites" in tool_names
    assert "validate_constraints" in tool_names
    assert "calculate_plan_score" in tool_names


# ---------------------------------------------------------------- T03 数据质量

def test_t03_expired_window_excluded_from_scoring():
    """T03：过期窗口不参与评分。"""
    from database import query_all

    windows = query_all("observation_window")
    windows[0]["quality"] = "EXPIRED"
    windows[0]["end_time"] = "2000-01-01T00:00:00+00:00"

    candidates = query_all("satellite")[:1]
    res = planning.calculate_plan_score(candidates=candidates, windows=windows)
    # 该卫星的过期窗口被跳过，可能返回 0 条或不含该卫星
    for p in res.data["plans"]:
        assert p["window_id"] != windows[0]["window_id"]


# ---------------------------------------------------------------- T07 评分排序

def test_t07_plans_are_sorted_by_score_desc():
    """T07：候选方案按评分降序返回。"""
    from database import query_all

    satellites = query_all("satellite")
    windows = query_all("observation_window")
    res = planning.calculate_plan_score(candidates=satellites, windows=windows)
    scores = [p["score"] for p in res.data["plans"]]
    assert scores == sorted(scores, reverse=True)


def test_scoring_is_deterministic():
    """同一输入两次评分结果一致（演示可重复）。"""
    from database import query_all

    satellites = query_all("satellite")
    windows = query_all("observation_window")
    a = planning.calculate_plan_score(candidates=satellites, windows=windows)
    b = planning.calculate_plan_score(candidates=satellites, windows=windows)
    assert a.data["plans"] == b.data["plans"]


def test_viable_plans_are_not_dropped_by_scoring():
    """回归：打分不得丢弃任何已通过硬约束校验的方案。

    历史缺陷：打分按"卫星"聚合（一星只留最后一个窗口），
    导致同卫星的其它可行窗口方案被静默丢弃。
    """
    from database import query_all

    satellites = query_all("satellite")
    windows = query_all("observation_window")
    req = {"max_resolution": 2.0, "satellite_type": "optical", "time_days": 5}

    cr = planning.validate_constraints(req, satellites, windows).data
    viable_ids = {p["plan_id"] for p in cr["viable_plans"]}
    assert viable_ids, "构造前提：至少有一个可行方案"

    scored = planning.calculate_plan_score(
        candidates=satellites, windows=windows, viable_plan_ids=list(viable_ids)
    ).data
    scored_ids = {p["plan_id"] for p in scored["plans"]}

    missing = viable_ids - scored_ids
    assert not missing, f"可行方案未进入排行：{missing}"
    # 反向校验：不可行方案不得混进排行
    assert scored_ids <= viable_ids


def test_multi_window_satellite_yields_multiple_scored_plans():
    """同一颗卫星的多个可行窗口必须各自进入排行（方案粒度而非卫星粒度）。"""
    from database import query_all

    satellites = query_all("satellite")
    windows = query_all("observation_window")
    # 找一颗有多窗口的卫星
    by_sat: dict[str, int] = {}
    for w in windows:
        by_sat[w["satellite_id"]] = by_sat.get(w["satellite_id"], 0) + 1
    multi = [sid for sid, n in by_sat.items() if n >= 2]
    assert multi, "构造前提：存在多窗口卫星"

    res = planning.calculate_plan_score(
        candidates=satellites, windows=windows
    ).data
    per_sat: dict[str, int] = {}
    for p in res["plans"]:
        per_sat[p["satellite_id"]] = per_sat.get(p["satellite_id"], 0) + 1
    assert max(per_sat.values()) >= 2, "多窗口卫星应产生多个方案"


# ---------------------------------------------------------------- T06 缺失参数

def test_t06_missing_params_are_flagged_not_guessed():
    """T06：缺失参数应被标记追问，不得自行假设。"""
    state = initial_state("task-004", "trace-004", "帮我规划一次观测任务")
    state = run_builtin(state, _Tracer(state, None))
    # 至少缺时间或分辨率
    assert len(state["missing_fields"]) > 0
    assert state["task_requirements"].get("max_resolution") is None


# ---------------------------------------------------------------- 模拟数据可重复性

def test_simulated_windows_are_reproducible():
    """模拟窗口必须在同一天内重复生成时完全一致（id 与时刻都一致）。

    这是「演示结果可重复」的硬性前提：窗口 id 用随机 UUID 时，
    每次启动都是全新一批数据，调度结果不可复现，知识库 K 的计划指纹
    也无法跨重启去重。
    """
    from database import query_all
    from simulator.data_generator import seed_windows

    seed_windows(reset=True)
    first = {
        r["window_id"]: (r["satellite_id"], r["area_id"], r["start_time"], r["end_time"])
        for r in query_all("observation_window")
    }
    assert first, "窗口集不应为空"

    seed_windows(reset=True)
    second = {
        r["window_id"]: (r["satellite_id"], r["area_id"], r["start_time"], r["end_time"])
        for r in query_all("observation_window")
    }
    assert first == second, "同一天内重复生成的窗口集必须完全一致"
    assert len(first) == 48, f"窗口规模应为 48，实际 {len(first)}"


def test_simulated_orders_cover_all_window_days():
    """订单时间范围必须覆盖窗口跨度。

    否则被拒原因会混入"窗口不在订单时间范围内"这种数据范围错配，
    淹没真正的约束根因。
    """
    from database import query_all

    orders = query_all("satellite_order")
    windows = query_all("observation_window")
    assert orders and windows
    win_start = min(w["start_time"] for w in windows)
    win_end = max(w["end_time"] for w in windows)
    for o in orders:
        assert o["time_start"] <= win_start, f"订单 {o['order_id']} 起点晚于最早窗口"
        assert o["time_end"] >= win_end, f"订单 {o['order_id']} 终点早于最晚窗口"


def test_route_rejects_unknown_task_type_from_llm(monkeypatch):
    """LLM 返回白名单外的 task_type 时必须被纠正，不能原样写进 state。

    真实接入 DeepSeek 后暴露：模型回了 "observation"（项目中不存在该链路），
    原实现只判 `not task_type`，于是脏值进入 state，路由静默走错主干。
    mock 模式永远返回 "planning"，所以这个缺陷此前被完全掩盖。
    """
    from agents.graph import node_route
    from agents import graph as graph_mod

    class _FakeLLM:
        def chat_json(self, prompt, **kw):
            return {"task_type": "observation", "area_name": "北京", "missing_fields": []}

    monkeypatch.setattr(graph_mod, "llm_client", _FakeLLM())

    state = initial_state("task-llm", "trace-llm", "规划北京区域的观测任务")
    state["task_type"] = None
    out = node_route(state, _Tracer(state, []))

    assert out["task_type"] in {"planning", "sar", "order"}, (
        f"非法 task_type 未被纠正：{out['task_type']!r}"
    )
    assert out["task_type"] == "planning"


def test_route_keeps_preset_task_type(monkeypatch):
    """调用方预设的链路不得被 LLM 覆盖。"""
    from agents.graph import node_route
    from agents import graph as graph_mod

    class _FakeLLM:
        def chat_json(self, prompt, **kw):
            return {"task_type": "planning", "missing_fields": []}

    monkeypatch.setattr(graph_mod, "llm_client", _FakeLLM())

    state = initial_state("task-preset", "trace-preset", "做个 SAR 分析")
    state["task_type"] = "sar"
    out = node_route(state, _Tracer(state, []))

    assert out["task_type"] == "sar"
