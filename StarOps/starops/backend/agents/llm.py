"""LLM 客户端。

未配置 DEEPSEEK_API_KEY 时自动降级为 mock，保证离线可演示、CI 可跑。
LLM 只负责理解、决策和组织输出，不负责数值计算。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是星枢 StarOps 的航天领域任务规划 Agent。
职责：把用户的自然语言需求解析为结构化任务约束，并输出 JSON。
约束：
1. 只输出 JSON，不要任何解释文字或 markdown 代码块。
2. 无法从用户输入中确定的字段，必须置为 null 并列入 missing_fields，绝不猜测。
3. 你不计算分辨率、过境时间或评分，这些由确定性工具完成。

必须严格使用下列字段名，不得改用同义名或自创结构：
{
  "task_type": "planning" | "sar" | "order",
  "area_name": 城市/区域名称字符串或 null,
  "area_id": 区域编号字符串或 null,
  "time_days": 整数（要求多少天内完成）或 null,
  "satellite_type": "optical" | "sar" | "any",
  "max_resolution": 数值（空间分辨率上限，单位米）或 null,
  "imaging_mode": 字符串或 null,
  "priority": "urgent" | "high" | "normal" | "low",
  "objective": 字符串或 null,
  "missing_fields": 缺失字段名数组
}
例如"优于2米"应写入 "max_resolution": 2，不得写成 constraints.spatial_resolution_m。

可用的区域编码（area_name 与 area_id 的对应关系）：
- 北京 -> CN-BJ
- 上海 -> CN-SH
- 广州 -> CN-GZ
- 深圳 -> CN-SZ
- 成都 -> CN-CD
- 西安 -> CN-XA
用户提到上述城市时，请同时给出 area_name 与 area_id。"""


class LLMClient:
    """DeepSeek 客户端。缺 Key 时自动 mock。

    ``available`` / ``model`` / ``base_url`` 走 property 而非构造时快照——
    ``settings`` 是模块级单例，配置若在导入后才变化（测试注入、热更新、
    ``.env`` 加载顺序调整），快照会与实际配置脱节。
    """

    @property
    def available(self) -> bool:
        return settings.llm_available

    @property
    def model(self) -> str:
        return settings.llm_model

    @property
    def base_url(self) -> str:
        return settings.llm_base_url

    def __init__(self) -> None:
        if not self.available:
            logger.warning(
                "未配置 DEEPSEEK_API_KEY，LLM 降级为 mock 模式（演示可用，非真实推理）"
            )

    def chat_json(
        self,
        user_prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """返回结构化 JSON。失败时抛异常，由调用方走降级路径。"""
        if not self.available:
            return self._mock_requirements(user_prompt)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"}

        last_err: Exception | None = None
        for attempt in range(settings.llm_max_retries + 1):
            try:
                resp = httpx.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=settings.llm_timeout,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return self._normalize_requirements(json.loads(content))
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning("LLM 调用失败（第 %s 次）：%s", attempt + 1, exc)
                if attempt < settings.llm_max_retries:
                    time.sleep(0.5 * (attempt + 1))

        raise RuntimeError(f"LLM 调用失败，已重试 {settings.llm_max_retries} 次：{last_err}")

    # ------------------------------------------------------------ 字段归一化

    @staticmethod
    def _normalize_requirements(raw: dict[str, Any]) -> dict[str, Any]:
        """把 LLM 返回的字段名映射到项目约定的 schema。

        模型可能返回 ``target_area`` / ``constraints.spatial_resolution_m``
        这类同义结构——字段名对不上时下游会读到 None，需求被解析成空。
        这里做确定性映射，不改语义、只对齐字段名。

        命不中的字段保持原样（不猜、不填），由 missing_fields 兜底提示。
        """
        out = dict(raw)

        # 区域：target_area / location / region → area_name
        if not out.get("area_name"):
            for alt in ("target_area", "location", "region", "area"):
                v = out.get(alt)
                if isinstance(v, str) and v.strip():
                    out["area_name"] = v.strip()
                    break
                if isinstance(v, dict):
                    name = v.get("name") or v.get("area_name")
                    if isinstance(name, str) and name.strip():
                        out["area_name"] = name.strip()
                        break

        # 分辨率：constraints.spatial_resolution_m 等 → max_resolution
        if out.get("max_resolution") is None:
            c = out.get("constraints")
            if isinstance(c, dict):
                for alt in (
                    "spatial_resolution_m",
                    "max_resolution",
                    "resolution_m",
                    "resolution",
                ):
                    v = c.get(alt)
                    if isinstance(v, (int, float)):
                        out["max_resolution"] = float(v)
                        break
                    if isinstance(v, dict) and isinstance(v.get("value"), (int, float)):
                        out["max_resolution"] = float(v["value"])
                        break

        # 天数：time_window.days / time_days → time_days
        if out.get("time_days") is None:
            tw = out.get("time_window")
            if isinstance(tw, dict) and isinstance(tw.get("days"), (int, float)):
                out["time_days"] = int(tw["days"])
            elif isinstance(tw, (int, float)):
                out["time_days"] = int(tw)

        return out

    def chat_text(self, user_prompt: str, system_prompt: str = "") -> str:
        """自由文本输出。用于报告生成。"""
        if not self.available:
            return self._mock_text(user_prompt)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or "你是航天领域技术助手。"},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
        resp = httpx.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=settings.llm_timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ------------------------------------------------------------ mock 实现

    def _mock_requirements(self, prompt: str) -> dict[str, Any]:
        """离线 mock：用规则从文本抽取关键字段，并显式标注缺失。

        这是降级路径而非伪装——输出中带 "_mock": true，前端与报告会标注。
        """
        text = prompt.lower()
        req: dict[str, Any] = {
            "area_name": None,
            "area_id": None,
            "time_days": None,
            "satellite_type": "optical",
            "max_resolution": None,
            "imaging_mode": None,
            "priority": "normal",
            "objective": None,
            "missing_fields": [],
            "_mock": True,
            "_note": "LLM 未配置，本次由规则降级解析，非真实模型推理",
        }

        for city, cid in (
            ("北京", "CN-BJ"), ("上海", "CN-SH"), ("广州", "CN-GZ"),
            ("深圳", "CN-SZ"), ("成都", "CN-CD"), ("西安", "CN-XA"),
        ):
            if city in prompt:
                req["area_name"] = city
                req["area_id"] = cid
                break

        if "sar" in text or "雷达" in prompt:
            req["satellite_type"] = "sar"

        for token in ("优先光学", "光学优先", "可见光"):
            if token in prompt:
                req["satellite_type"] = "optical"
                break

        # 分辨率
        import re

        m = re.search(r"(\d+(?:\.\d+)?)\s*(米|m)\b", prompt)
        if m:
            req["max_resolution"] = float(m.group(1))
        elif "优于" in prompt or "分辨率" in prompt:
            req["missing_fields"].append("max_resolution")

        # 天数
        m = re.search(r"(\d+)\s*天", prompt)
        if m:
            req["time_days"] = int(m.group(1))
        else:
            req["missing_fields"].append("time_days")

        if req["area_name"] is None:
            req["missing_fields"].append("area_name")

        return req

    def _mock_text(self, prompt: str) -> str:
        return (
            "【离线模式】未配置 DEEPSEEK_API_KEY，本报告由模板生成，非模型推理输出。\n"
            f"原始请求：{prompt[:200]}"
        )


llm_client = LLMClient()
