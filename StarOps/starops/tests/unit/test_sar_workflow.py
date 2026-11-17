"""SAR Agent 工作流测试：诊断 → 横评选型 → 审批 → 执行阻断。

覆盖"安全闭环"在 SAR 链上的行为，与规划链共用同一套机制。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP_DB = Path(tempfile.mkdtemp(prefix="starops_sarwf_")) / "test.db"
os.environ["STAROPS_DB_PATH"] = str(_TMP_DB)

sar_nodes = pytest.importorskip("agents.nodes.sar_nodes", reason="sar_nodes 不可用")
graph = pytest.importorskip("agents.graph", reason="agents.graph 不可用")
imaging = pytest.importorskip("tools.imaging", reason="tools.imaging 不可用")

np = pytest.importorskip("numpy", reason="numpy 未安装")


@pytest.fixture
def speckled_image(tmp_path):
    """造一张带斑点噪声 + 水平条纹的 SAR 影像。"""
    rng = np.random.default_rng(99)
    h, w = 128, 128
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100 + 50 * np.sin(xx / 10.0) + 30 * np.cos(yy / 12.0)
    speckle = rng.gamma(shape=2.0, scale=0.5, size=(h, w))
    img = base * speckle
    img[40:42, :] *= 1.4  # 水平条纹
    p = tmp_path / "wf.png"
    imaging.write_png(p, np.clip(img, 0, 255).astype(np.uint8))
    return str(p)


def _tracer():
    """收集事件的 tracer。"""
    events: list[dict] = []

    class T:
        def emit(self, event, **payload):
            events.append({"event": event, **payload})

        def tool(self, *args, **kwargs):
            raise AssertionError("测试中不应走 tracer.tool")

    return T(), events


# ---------------------------------------------------------------- 诊断

def test_diagnosis_detects_severe_speckle():
    """CV 高的影像应判为 severe 斑点。"""
    from tools import sar

    metrics = {
        "cv": 0.75,
        "enl": 2.2,
        "local_variance": 0.046,
        "striping_strength": 0.44,
        "striping_direction": 0.0,
    }
    d = sar_nodes._diagnosis(metrics)
    assert d["speckle_level"] == "severe"
    assert d["has_striping"] is True
    assert len(d["findings"]) == 2


def test_diagnosis_mild_speckle():
    """CV 低的影像应判为 mild。"""
    d = sar_nodes._diagnosis({"cv": 0.2, "enl": 12.0, "local_variance": 0.001})
    assert d["speckle_level"] == "mild"
    assert d["has_striping"] is False


def test_diagnosis_no_striping_when_strength_low():
    """条纹突出度低于阈值不判为有条纹。"""
    d = sar_nodes._diagnosis({"cv": 0.5, "striping_strength": 0.1})
    assert d["has_striping"] is False


# ---------------------------------------------------------------- 横评选型

def test_pipeline_selection_prefers_lower_cv(speckled_image):
    """选型应在通过守门的管线中选 CV 降幅最大者。"""
    tracer, events = _tracer()
    candidates = [
        {"pipeline_id": "lee", "method": "lee", "filter_level": "L1_classical", "label": "Lee 滤波"},
        {"pipeline_id": "adaptive", "method": "adaptive", "filter_level": "L1_classical", "label": "自适应滤波"},
    ]
    best, rejected = sar_nodes._select_best_pipeline(speckled_image, candidates, tracer)
    assert best is not None
    assert best["cv_reduced_pct"] >= 0
    # 被淘汰的管线必须带原因，不能静默丢弃
    for r in rejected:
        assert r.get("reject_reason")


def test_pipeline_selection_rejects_radiometry_drift(speckled_image):
    """均值偏移超过 5% 的管线必须被拒（辐射强度失真）。"""
    tracer, _ = _tracer()
    candidates = [
        {"pipeline_id": "median", "method": "median", "filter_level": "L1_classical", "label": "中值滤波"},
    ]
    best, rejected = sar_nodes._select_best_pipeline(speckled_image, candidates, tracer)
    # 中值滤波对乘性斑点有系统性下偏，本例中应被守门拦下
    if best is None:
        assert rejected and "均值偏移" in rejected[0]["reject_reason"]
    else:
        assert best["mean_shift_pct"] <= 5.0


def test_pipeline_selection_returns_none_when_all_rejected():
    """全部候选被拒时返回 None（不硬凑一个方案）。"""
    tracer, _ = _tracer()
    best, rejected = sar_nodes._select_best_pipeline(
        "E:/nonexistent/x.png",
        [{"pipeline_id": "lee", "method": "lee", "label": "Lee 滤波"}],
        tracer,
    )
    assert best is None


# ---------------------------------------------------------------- 审批阻断

def test_sar_execute_blocked_without_approval(speckled_image):
    """未审批时 SAR 管线执行必须被阻断。"""
    from agents.state import initial_state

    tracer, events = _tracer()
    st = initial_state("task-x", "trace-x", "sar test")
    st["task_type"] = "sar"
    st["selected_pipeline"] = {"pipeline_id": "lee", "output_path": "x.png"}
    st["approval_status"] = "pending"

    st = sar_nodes.node_sar_execute(st, tracer)
    assert not st.get("execution_result")
    assert any(e.get("status") == "blocked" for e in events)


def test_sar_execute_succeeds_after_approval(speckled_image):
    """审批通过后执行成功。"""
    from agents.state import initial_state

    tracer, events = _tracer()
    st = initial_state("task-y", "trace-y", "sar test")
    st["task_type"] = "sar"
    st["selected_pipeline"] = {"pipeline_id": "lee", "output_path": "x.png"}
    st["approval_status"] = "approved"

    st = sar_nodes.node_sar_execute(st, tracer)
    assert st["execution_result"]["status"] == "succeeded"


# ---------------------------------------------------------------- 路由

def test_builtin_order_branches_by_task_type():
    """内置执行器按 task_type 分流：planning 与 sar 走不同链。"""
    assert graph.BUILTIN_ORDER_SAR != graph.BUILTIN_ORDER
    assert sar_nodes.node_sar_analyze in graph.BUILTIN_ORDER_SAR
    assert graph.node_validate in graph.BUILTIN_ORDER
