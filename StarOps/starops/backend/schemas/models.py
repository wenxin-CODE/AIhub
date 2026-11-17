"""Pydantic 结构化 Schema —— 工具入参出参契约。

原则：工具出入参必须结构化、可校验，LLM 只能按 Schema 生成参数。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

TaskType = Literal["planning", "sar"]
DataQuality = Literal["FRESH", "STALE", "EXPIRED", "SUSPECT"]
FilterLevel = Literal["L1_classical", "L2_deep", "L3_patent"]


# ---------------------------------------------------------------- 规划侧

class TaskRequirements(BaseModel):
    """从自然语言抽取的结构化任务约束。缺失项由 Agent 主动追问，不自行假设。"""

    area_id: str | None = Field(default=None, description="观测区域标识")
    area_name: str | None = Field(default=None, description="观测区域名称")
    time_start: str | None = Field(default=None, description="时间范围起 ISO8601")
    time_end: str | None = Field(default=None, description="时间范围止 ISO8601")
    time_days: int | None = Field(default=None, ge=1, le=60, description="相对天数")
    satellite_type: Literal["optical", "sar", "any"] = "optical"
    max_resolution: float | None = Field(default=None, gt=0, description="空间分辨率上限（米）")
    imaging_mode: str | None = None
    priority: Literal["low", "normal", "high", "urgent"] = "normal"
    objective: str | None = Field(default=None, description="优化目标，如尽早完成")

    @field_validator("time_start", "time_end")
    @classmethod
    def _iso(cls, v: str | None) -> str | None:
        if v is None:
            return None
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class Satellite(BaseModel):
    satellite_id: str
    satellite_name: str
    satellite_type: str
    resolution: float
    imaging_mode: str
    status: str = "simulated"


class ObservationWindow(BaseModel):
    window_id: str
    satellite_id: str
    area_id: str
    area_name: str | None = None
    start_time: str
    end_time: str
    cloud_rate: float = Field(ge=0.0, le=1.0)
    incidence_angle: float = Field(ge=0.0, le=90.0)
    quality: DataQuality = "FRESH"


class PlanItem(BaseModel):
    """单个候选观测方案。"""

    plan_id: str
    satellite_id: str
    satellite_name: str
    window_id: str
    start_time: str
    end_time: str
    resolution: float
    cloud_rate: float
    incidence_angle: float
    score: float = Field(ge=0.0, le=1.0)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class ConstraintResult(BaseModel):
    """硬约束校验结果。由独立校验器产出，LLM 不得改写。"""

    passed: bool
    violations: list[str] = Field(default_factory=list)
    checked_rules: list[str] = Field(default_factory=list)
    rule_version: str = "v1"

    @property
    def status(self) -> str:
        return "passed" if self.passed else "failed"


# ---------------------------------------------------------------- SAR 侧

class ImageMetrics(BaseModel):
    """SAR 影像质量指标集合。"""

    gray_mean: float | None = None
    gray_std: float | None = None
    local_variance: float | None = None
    cv: float | None = None
    enl: float | None = None
    noise_estimate: float | None = None
    striping_strength: float | None = None
    striping_direction: float | None = None
    psnr: float | None = None
    ssim: float | None = None


class PipelineResult(BaseModel):
    """单条处理管线的执行结果。"""

    pipeline_id: str
    method: str
    filter_level: FilterLevel
    output_path: str | None = None
    metrics: ImageMetrics = Field(default_factory=ImageMetrics)
    elapsed_ms: int = 0
    model_version: str | None = None
    weight_source: str | None = None
    degraded: bool = False
    degraded_reason: str | None = None
    trace: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- 审批 / 执行

class ApprovalRequest(BaseModel):
    target_type: Literal["observation_plan", "sar_pipeline"]
    target_id: str
    plan_version: int = 1
    summary: str = ""
    snapshot: dict[str, Any] = Field(default_factory=dict)


class ApprovalRecord(BaseModel):
    approval_id: str
    target_type: str
    target_id: str
    plan_version: int
    status: Literal["pending", "approved", "rejected"]
    reviewer: str | None = None
    reviewed_at: str | None = None
    comment: str | None = None
    snapshot: dict[str, Any] = Field(default_factory=dict)


class CommitResult(BaseModel):
    """执行结果。幂等键命中时返回首次结果。"""

    commit_id: str
    task_id: str
    plan_version: int
    status: str
    idempotent_hit: bool = False
    result: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- 工具返回封装

class ToolResult(BaseModel):
    """所有工具的统一返回封装。

    约定：失败时 status != success 且必须带 reason，不允许用自然语言伪装成功。
    """

    status: Literal["success", "failed", "degraded"]
    data: Any = None
    reason: str | None = None
    timestamp: str
    elapsed_ms: int = 0

    @classmethod
    def ok(cls, data: Any, elapsed_ms: int = 0, reason: str | None = None) -> "ToolResult":
        from database import utcnow

        return cls(status="success", data=data, reason=reason, timestamp=utcnow(), elapsed_ms=elapsed_ms)

    @classmethod
    def fail(cls, reason: str, elapsed_ms: int = 0, data: Any = None) -> "ToolResult":
        from database import utcnow

        return cls(status="failed", data=data, reason=reason, timestamp=utcnow(), elapsed_ms=elapsed_ms)

    @classmethod
    def degraded_to(cls, reason: str, data: Any, elapsed_ms: int = 0) -> "ToolResult":
        from database import utcnow

        return cls(status="degraded", data=data, reason=reason, timestamp=utcnow(), elapsed_ms=elapsed_ms)
