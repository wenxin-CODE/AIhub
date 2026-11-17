"""配置加载测试。

对应方案 §10.1 的 T30：`.env` 配置必须真正生效，且优先级正确。
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from config import Settings, load_dotenv


# ---------------------------------------------------------------------------
# .env 解析
# ---------------------------------------------------------------------------


def _write_env(tmp_path: Path, content: str) -> Path:
    p = tmp_path / ".env"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def test_dotenv_parses_basic_pairs(tmp_path):
    p = _write_env(tmp_path, """
        # 注释行
        DEEPSEEK_API_KEY=sk-test-123456
        DEEPSEEK_MODEL=deepseek-reasoner

        EMPTY_VALUE=
    """)
    parsed = load_dotenv(p)
    assert parsed["DEEPSEEK_API_KEY"] == "sk-test-123456"
    assert parsed["DEEPSEEK_MODEL"] == "deepseek-reasoner"
    assert parsed["EMPTY_VALUE"] == ""


def test_dotenv_supports_export_quotes_and_inline_comment(tmp_path):
    p = _write_env(tmp_path, """
        export DEEPSEEK_API_KEY="sk-quoted-key"
        DEEPSEEK_BASE_URL='https://example.com'
        DEEPSEEK_MODEL=deepseek-chat  # 行尾注释
    """)
    parsed = load_dotenv(p)
    assert parsed["DEEPSEEK_API_KEY"] == "sk-quoted-key"
    assert parsed["DEEPSEEK_BASE_URL"] == "https://example.com"
    assert parsed["DEEPSEEK_MODEL"] == "deepseek-chat"


def test_dotenv_skips_malformed_lines(tmp_path):
    p = _write_env(tmp_path, """
        GOOD_KEY=ok
        this-is-not-a-pair
        =novalue
        ANOTHER=2
    """)
    parsed = load_dotenv(p)
    assert "GOOD_KEY" in parsed
    assert "ANOTHER" in parsed
    assert len(parsed) == 2, f"畸形行不应被解析，实际: {parsed}"


def test_dotenv_missing_file_is_not_an_error(tmp_path):
    parsed = load_dotenv(tmp_path / "does-not-exist.env")
    assert parsed == {}


# ---------------------------------------------------------------------------
# 优先级：真实环境变量 > .env
# ---------------------------------------------------------------------------


def test_process_env_takes_precedence_over_dotenv(tmp_path, monkeypatch):
    """已存在的环境变量不得被 .env 覆盖——CI / 容器注入必须赢。"""
    p = _write_env(tmp_path, "STAROPS_LLM_MODEL=from-dotenv\n")
    monkeypatch.setenv("STAROPS_LLM_MODEL", "from-process-env")

    load_dotenv(p)
    assert os.environ["STAROPS_LLM_MODEL"] == "from-process-env"


def test_dotenv_fills_vars_absent_from_process_env(tmp_path, monkeypatch):
    p = _write_env(tmp_path, "STAROPS_LLM_MODEL=from-dotenv\n")
    monkeypatch.delenv("STAROPS_LLM_MODEL", raising=False)

    load_dotenv(p)
    assert os.environ["STAROPS_LLM_MODEL"] == "from-dotenv"


# ---------------------------------------------------------------------------
# Settings 行为
# ---------------------------------------------------------------------------


def test_api_key_source_reports_dotenv(tmp_path, monkeypatch):
    """Key 来自 .env 时，来源标签应为 dotenv。"""
    monkeypatch.delenv("STAROPS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    p = _write_env(tmp_path, "DEEPSEEK_API_KEY=sk-from-dotenv-file\n")
    load_dotenv(p)

    s = Settings()
    assert s.llm_available is True
    assert s.llm_api_key == "sk-from-dotenv-file"
    assert s.llm_api_key_source == "dotenv"


def test_api_key_source_reports_process_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-process-env")
    monkeypatch.delenv("STAROPS_LLM_API_KEY", raising=False)
    # 空的 .env 不改变来源判定
    load_dotenv(_write_env(tmp_path, "STAROPS_DEBUG=0\n"))

    s = Settings()
    assert s.llm_api_key_source == "process-env"


def test_api_key_source_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("STAROPS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    load_dotenv(_write_env(tmp_path, "STAROPS_DEBUG=0\n"))

    s = Settings()
    assert s.llm_api_key == ""
    assert s.llm_available is False
    assert s.llm_api_key_source == "unset"


def test_generic_key_overrides_provider_specific(monkeypatch):
    """STAROPS_LLM_API_KEY 优先于 DEEPSEEK_API_KEY——provider 解耦。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-specific")
    monkeypatch.setenv("STAROPS_LLM_API_KEY", "sk-generic")

    s = Settings()
    assert s.llm_api_key == "sk-generic"


def test_missing_key_degrades_to_mock_not_crash(tmp_path, monkeypatch):
    """无 Key 时必须降级而不是抛异常——离线演示是硬要求。"""
    monkeypatch.delenv("STAROPS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    load_dotenv(_write_env(tmp_path, "STAROPS_DEBUG=0\n"))

    s = Settings()
    assert s.llm_available is False
    assert s.llm_api_key == ""
    # 降级路径必须可用：模型名、端点仍有默认值，不能因缺 Key 而崩
    assert s.llm_model
    assert s.llm_base_url.startswith("http")


# ---------------------------------------------------------------------------
# 路径解析：相对路径必须锚定项目根，不能随 CWD 漂移
# ---------------------------------------------------------------------------


def test_relative_paths_resolve_against_project_root(monkeypatch):
    """相对路径按 BASE_DIR 解析。

    否则 `cd backend && uvicorn` 与 `cd starops && uvicorn`
    会落到不同的数据库——表现为"数据莫名消失"。
    """
    from config import BASE_DIR

    monkeypatch.setenv("STAROPS_DB_PATH", "./data/starops.db")
    monkeypatch.setenv("STAROPS_CHECKPOINT_DIR", "./data/checkpoints")
    monkeypatch.setenv("STAROPS_IMAGE_DIR", "./data/images")

    s = Settings()
    assert s.db_path == (BASE_DIR / "data" / "starops.db").resolve()
    assert s.checkpoint_dir == (BASE_DIR / "data" / "checkpoints").resolve()
    assert s.image_dir == (BASE_DIR / "data" / "images").resolve()


def test_absolute_paths_are_preserved(monkeypatch, tmp_path):
    monkeypatch.setenv("STAROPS_DB_PATH", str(tmp_path / "custom.db"))

    s = Settings()
    assert s.db_path == tmp_path / "custom.db"


def test_diagnostics_never_leaks_full_key(monkeypatch):
    """诊断输出必须脱敏——它会被写日志、返回前端。

    允许暴露前 6 位与后 4 位（云控制台惯例，用于确认"是不是这把 Key"），
    但中段与完整 Key 一律不得出现。
    """
    key = "sk-SECRETVALUE1234567890"
    monkeypatch.setenv("DEEPSEEK_API_KEY", key)

    s = Settings()
    diag = s.diagnostics()
    blob = repr(diag)

    assert key not in blob, "完整 Key 不得出现在诊断输出中"
    assert diag["llm_api_key_set"] is True
    assert diag["llm_api_key_masked"].startswith(key[:6])
    assert key[-4:] in diag["llm_api_key_masked"]
    assert "SECRETVALUE" not in diag["llm_api_key_masked"], "中段不得泄露"


# ---------------------------------------------------------------------------
# LLM 返回字段归一化
# ---------------------------------------------------------------------------


def test_normalize_maps_synonym_field_names():
    """LLM 可能返回同义结构，归一化后必须落到项目约定的字段名。

    真实接入 DeepSeek 后暴露：模型回了
    {"target_area": {"name": "北京"}, "constraints": {"spatial_resolution_m": {"value": 2}}}
    而下游硬编码读 area_name / max_resolution —— 字段名对不上则需求被解析成空。
    """
    from agents.llm import LLMClient

    raw = {
        "task_type": "satellite_observation",
        "target_area": {"name": "北京"},
        "constraints": {"spatial_resolution_m": {"operator": "lt", "value": 2}},
        "time_window": {"days": 3},
        "missing_fields": [],
    }
    out = LLMClient._normalize_requirements(raw)

    assert out["area_name"] == "北京"
    assert out["max_resolution"] == 2.0
    assert out["time_days"] == 3


def test_normalize_accepts_plain_string_area():
    from agents.llm import LLMClient

    out = LLMClient._normalize_requirements({"target_area": "上海", "area_name": None})
    assert out["area_name"] == "上海"


def test_normalize_does_not_overwrite_existing_values():
    """已有正确字段时不得被同义字段覆盖。"""
    from agents.llm import LLMClient

    out = LLMClient._normalize_requirements(
        {"area_name": "广州", "target_area": {"name": "北京"}}
    )
    assert out["area_name"] == "广州"


def test_normalize_leaves_unknown_fields_untouched():
    """命不中的字段保持原样——不猜、不填。"""
    from agents.llm import LLMClient

    raw = {"objective": "洪水监测", "unknown_field": "keep"}
    out = LLMClient._normalize_requirements(raw)
    assert out["objective"] == "洪水监测"
    assert out["unknown_field"] == "keep"
    assert "area_name" not in out


# ---------------------------------------------------------------------------
# 区域编码反查
# ---------------------------------------------------------------------------


def test_resolve_area_id_from_name():
    """真实 LLM 只能给城市名，需由确定性字典反查区域编码。"""
    from tools.planning import resolve_area_id

    assert resolve_area_id("北京") == "CN-BJ"
    assert resolve_area_id("深圳") == "CN-SZ"
    assert resolve_area_id(" 成都 ") == "CN-CD"


def test_resolve_area_id_keeps_explicit_id():
    from tools.planning import resolve_area_id

    assert resolve_area_id("北京", "CN-XX") == "CN-XX"
    assert resolve_area_id(None, "CN-YY") == "CN-YY"


def test_resolve_area_id_unknown_returns_none():
    from tools.planning import resolve_area_id

    assert resolve_area_id("火星") is None
    assert resolve_area_id(None) is None


def test_simulator_and_planning_share_same_area_dict():
    """区域字典必须只有一份——两处维护迟早漂移。"""
    from tools.planning import AREAS
    from simulator.data_generator import AREAS as SIM_AREAS

    assert SIM_AREAS == AREAS
