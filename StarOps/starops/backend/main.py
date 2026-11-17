"""FastAPI 应用入口。

接口：
- GET  /api/health              健康检查
- GET  /api/meta                运行元信息（LLM 是否可用、模拟数据统计）
- POST /api/seed                初始化模拟数据
- GET  /api/satellites          卫星列表
- GET  /api/windows             观测窗口列表
- POST /api/agent/run           发起任务（SSE 流式返回执行过程）
- GET  /api/approvals           审批列表
- POST /api/approvals/{id}/review  审批决策
- GET  /api/audit               审计查询
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pathlib import Path

from agents.graph import build_graph, run_builtin
from agents.state import initial_state
from agents.llm import llm_client
from config import settings
from database import exec_sql, init_db, insert, new_id, query_all, utcnow
from services.approval import pending_approvals, review_approval
from simulator.data_generator import seed_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("starops")

app = FastAPI(title="StarOps API", version="0.1.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    result = seed_all()
    logger.info("启动完成：%s", result)


# ---------------------------------------------------------------- 请求模型

class AgentRunRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户自然语言需求")
    task_id: str | None = None


class ReviewRequest(BaseModel):
    decision: str = Field(..., description="approved / rejected")
    reviewer: str = "operator"
    comment: str = ""


class SarAnalyzeRequest(BaseModel):
    image_path: str = Field(..., min_length=1, description="影像路径（服务端可访问）")
    message: str = Field("", description="可选的自然语言补充说明")


class OrderCreateRequest(BaseModel):
    """创建一个观测订单（TSLC 订单驱动调度的输入单元）。"""

    area_id: str = Field(..., min_length=1, description="观测区域标识，如 CN-BJ")
    area_name: str = Field("", description="观测区域名称")
    time_start: str = Field(..., description="时间范围起 ISO8601")
    time_end: str = Field(..., description="时间范围止 ISO8601")
    satellite_type: str = Field("any", description="optical / sar / any")
    max_resolution: float | None = Field(None, gt=0, description="空间分辨率上限（米）")
    max_cloud_rate: float = Field(0.3, ge=0.0, le=1.0)
    max_incidence_angle: float = Field(45.0, ge=0.0, le=90.0)
    priority: str = Field("normal", description="low / normal / high / urgent")
    min_duration_min: float | None = Field(None, gt=0)


class OrderRunRequest(BaseModel):
    """运行订单调度。

    - order_ids 为空：处理全部 pending 订单；
    - order_ids 非空：只处理指定订单（无论其当前状态）；
    - reset=True：先把全部订单重置为 pending 再调度（用于重复演示）。
    """

    order_ids: list[str] = Field(default_factory=list)
    max_replans: int = Field(3, ge=0, le=10, description="单个订单的重规划上限")
    message: str = Field("", description="可选说明")
    reset: bool = Field(False, description="调度前把订单状态重置为 pending（可重复演示）")


# ---------------------------------------------------------------- 基础接口

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": settings.app_name, "time": utcnow()}


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    """运行元信息。前端据此提示 LLM 是否处于降级模式。

    附带 config 诊断（Key 脱敏）：用来回答"我配了 Key 怎么还在降级模式"。
    """
    sats = query_all("satellite")
    wins = query_all("observation_window")
    orders = query_all("satellite_order")
    diag = settings.diagnostics()
    return {
        "app": settings.app_name,
        "llm_available": settings.llm_available,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_degraded": not settings.llm_available,
        "degraded_note": (
            "未配置 DEEPSEEK_API_KEY，需求解析由规则降级处理。"
            if not settings.llm_available
            else "DeepSeek 已接入"
        ),
        "config": {
            "env_file": diag["env_file"],
            "env_file_exists": diag["env_file_exists"],
            "llm_api_key_set": diag["llm_api_key_set"],
            "llm_api_key_source": diag["llm_api_key_source"],
            "llm_api_key_masked": diag["llm_api_key_masked"],
        },
        "data": {
            "satellites": len(sats),
            "windows": len(wins),
            "orders": len(orders),
            "source": "simulated",
        },
        "checkpoint_dir": settings.checkpoint_dir.as_posix(),
        "methods": {
            "planning": "需求驱动候选方案排序（Filter + Weighted Sum）",
            "order": "TSLC 订单驱动三阶段调度（DOI 10.1109/TGRS.2026.3700605）",
            "sar": "SAR 质量诊断与多管线横评",
        },
    }


@app.post("/api/seed")
def seed() -> dict[str, Any]:
    result = seed_all()
    return {"seeded": result}


@app.get("/api/satellites")
def satellites() -> dict[str, Any]:
    return {"satellites": query_all("satellite")}


@app.get("/api/windows")
def windows(area_id: str | None = None) -> dict[str, Any]:
    rows = query_all("observation_window")
    if area_id:
        rows = [r for r in rows if r["area_id"] == area_id]
    return {"windows": rows, "count": len(rows)}


# ---------------------------------------------------------------- Agent 执行

@app.post("/api/agent/run")
async def agent_run(req: AgentRunRequest) -> StreamingResponse:
    """发起任务并以 SSE 流式返回执行过程。

    这是"展示执行过程，而不仅是最终答案"的落点：前端能看到
    当前节点、工具名、工具入参、工具返回、耗时。
    """
    task_id = req.task_id or new_id("task")
    trace_id = new_id("trace")
    state = initial_state(task_id, trace_id, req.message)

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def sink(event: dict[str, Any]) -> None:
        """从工作流线程向事件循环投递。"""
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def worker() -> None:
        """工作流在后台线程执行，避免阻塞事件循环。"""
        from agents.graph import _Tracer

        tracer = _Tracer(state, sink)
        try:
            graph = build_graph(tracer)
            if graph is None:
                result_state = dict(run_builtin(state, tracer))
            else:
                compiled = graph.compile()
                result_state = dict(compiled.invoke(state))
            state.clear()
            state.update(result_state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("工作流执行失败")
            sink({"event": "error", "message": str(exc)})
        finally:
            sink({"event": "final", "state": _serialize_state(state)})

    async def event_stream():
        import threading

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while True:
            evt = await queue.get()
            yield f"data: {json.dumps(evt, ensure_ascii=False, default=str)}\n\n"
            if evt.get("event") in {"final", "error"}:
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    """剔除不可序列化的 messages 内容。"""
    out: dict[str, Any] = {}
    for k, v in state.items():
        if k == "messages":
            out[k] = [
                m if isinstance(m, dict) else {"content": str(m)} for m in (v or [])
            ]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- 审批

@app.get("/api/approvals")
def approvals(status: str | None = None) -> dict[str, Any]:
    if status:
        rows = query_all("approval", "status = ?", (status,), order_by="created_at DESC")
    else:
        rows = query_all("approval", order_by="created_at DESC")
    return {"approvals": rows, "count": len(rows)}


@app.post("/api/approvals/{approval_id}/review")
def approvals_review(approval_id: str, req: ReviewRequest) -> dict[str, Any]:
    res = review_approval(approval_id, req.decision, req.reviewer, req.comment)
    if res.status != "success":
        raise HTTPException(status_code=400, detail=res.reason)
    return res.data


@app.get("/api/audit")
def audit(object_id: str | None = None, limit: int = 50) -> dict[str, Any]:
    if object_id:
        rows = query_all(
            "audit_log", "object_id = ?", (object_id,), order_by="created_at DESC", limit=limit
        )
    else:
        rows = query_all("audit_log", order_by="created_at DESC", limit=limit)
    return {"audit": rows, "count": len(rows)}


@app.get("/api/tool-executions")
def tool_executions(trace_id: str | None = None, limit: int = 50) -> dict[str, Any]:
    if trace_id:
        rows = query_all(
            "tool_execution", "trace_id = ?", (trace_id,), order_by="created_at DESC", limit=limit
        )
    else:
        rows = query_all("tool_execution", order_by="created_at DESC", limit=limit)
    return {"executions": rows, "count": len(rows)}


@app.get("/api/tasks")
def tasks(limit: int = 50) -> dict[str, Any]:
    return {
        "tasks": query_all("mission_task", order_by="created_at DESC", limit=limit)
    }


# ---------------------------------------------------------------- SAR 影像分析

@app.post("/api/sar/upload")
async def sar_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """上传 SAR 影像，落盘并登记到 sar_image 表。

    文件名做安全处理（防路径穿越），扩展名保留以选解码器。
    """
    original = file.filename or "image.png"
    safe_name = Path(original).name
    suffix = Path(safe_name).suffix.lower() or ".png"
    allowed = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的格式：{suffix}")

    image_id = new_id("img")
    dest = settings.image_dir / f"{image_id}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    dest.write_bytes(content)

    # 登记元数据
    from tools import sar as sar_tools

    meta_res = sar_tools.read_image_metadata(str(dest))
    meta = meta_res.data if meta_res.status == "success" else {}
    insert(
        "sar_image",
        {
            "image_id": image_id,
            "filename": safe_name,
            "format": suffix.lstrip(".").upper(),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "dtype": meta.get("dtype"),
            "upload_path": str(dest),
            "source": "uploaded",
            "uploaded_at": utcnow(),
        },
    )

    return {
        "image_id": image_id,
        "filename": safe_name,
        "path": str(dest),
        "url": f"/files/images/{dest.name}",
        "metadata": meta,
        "size_bytes": len(content),
    }


@app.get("/api/images")
def images(limit: int = 50) -> dict[str, Any]:
    """已登记的 SAR 影像列表。"""
    return {
        "images": query_all("sar_image", order_by="uploaded_at DESC", limit=limit)
    }


@app.post("/api/sar/analyze")
async def sar_analyze(req: SarAnalyzeRequest) -> StreamingResponse:
    """SAR 影像分析：指标 → 诊断 → 选型 → 审批，同样以 SSE 推送执行过程。

    与规划链共用同一个 SSE 协议，前端复用同一套渲染逻辑。
    """
    task_id = new_id("sar")
    trace_id = new_id("trace")
    state = initial_state(task_id, trace_id, req.message or f"SAR 影像分析 {req.image_path}")
    state["task_type"] = "sar"
    state["image_path"] = req.image_path

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def sink(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def worker() -> None:
        from agents.graph import _Tracer, run_builtin
        from agents.nodes import sar_nodes

        tracer = _Tracer(state, sink)
        try:
            graph = build_graph(tracer)
            if graph is None:
                result_state = dict(run_builtin(state, tracer))
            else:
                compiled = graph.compile()
                result_state = dict(compiled.invoke(state))
            state.clear()
            state.update(result_state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("SAR 分析失败")
            sink({"event": "error", "message": str(exc)})
        finally:
            sink({"event": "final", "state": _serialize_state(state)})

    async def event_stream():
        import threading

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while True:
            evt = await queue.get()
            yield f"data: {json.dumps(evt, ensure_ascii=False, default=str)}\n\n"
            if evt.get("event") in {"final", "error"}:
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# 影像静态服务：前端需要显示原图与处理后对比
app.mount("/files/images", StaticFiles(directory=str(settings.image_dir)), name="images")


# ---------------------------------------------------------------- TSLC 订单驱动调度

@app.get("/api/orders")
def orders(status: str | None = None) -> dict[str, Any]:
    """观测订单队列。"""
    if status:
        rows = query_all("satellite_order", "status = ?", (status,), order_by="created_at ASC")
    else:
        rows = query_all("satellite_order", order_by="created_at ASC")
    return {"orders": rows, "count": len(rows)}


@app.post("/api/orders")
def create_order(req: OrderCreateRequest) -> dict[str, Any]:
    """创建观测订单。"""
    order_id = new_id("ord")
    from tools import order_scheduling

    order = order_scheduling.build_order(
        order_id,
        req.area_id,
        req.area_name or req.area_id,
        req.time_start,
        req.time_end,
        satellite_type=req.satellite_type,
        max_resolution=req.max_resolution,
        max_cloud_rate=req.max_cloud_rate,
        max_incidence_angle=req.max_incidence_angle,
        priority=req.priority,
        min_duration_min=req.min_duration_min,
    )
    insert(
        "satellite_order",
        {
            "order_id": order["order_id"],
            "area_id": order["area_id"],
            "area_name": order["area_name"],
            "time_start": order["time_start"],
            "time_end": order["time_end"],
            "satellite_type": order["satellite_type"],
            "max_resolution": order["max_resolution"],
            "max_cloud_rate": order["max_cloud_rate"],
            "max_incidence_angle": order["max_incidence_angle"],
            "priority": order["priority"],
            "profit": order["profit"],
            "min_duration_min": order["min_duration_min"],
            "status": "pending",
            "source": "user",
            "created_at": utcnow(),
        },
    )
    return {"order": order, "status": "pending"}


@app.post("/api/orders/run")
async def run_orders(req: OrderRunRequest) -> StreamingResponse:
    """运行 TSLC 订单驱动调度，以 SSE 流式推送三阶段决策过程。

    事件：order_batch_start → order_start → stage_decision / stage_failed
          → order_scheduled / order_rejected → order_batch_end → final
    """
    task_id = new_id("ordtask")
    trace_id = new_id("trace")
    state = initial_state(task_id, trace_id, req.message or "TSLC 订单驱动调度")
    state["task_type"] = "order"
    state["max_replans"] = req.max_replans

    if req.reset:
        # 重置订单队列，使调度可重复演示（不删数据，只改状态）
        exec_sql("UPDATE satellite_order SET status = 'pending'")

    if req.order_ids:
        rows = query_all(
            "satellite_order",
            "order_id IN ({})".format(",".join("?" * len(req.order_ids))),
            tuple(req.order_ids),
            order_by="created_at ASC",
        )
    else:
        rows = query_all("satellite_order", "status = 'pending'", order_by="created_at ASC")

    # 订单显式塞进 state，节点直接用这份队列（否则 order_ids 筛选不会生效）
    from agents.nodes.order_nodes import rows_to_orders

    state["orders"] = rows_to_orders(rows)

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def sink(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def worker() -> None:
        from agents.graph import _Tracer, run_builtin

        tracer = _Tracer(state, sink)
        try:
            graph = build_graph(tracer)
            if graph is None:
                result_state = dict(run_builtin(state, tracer))
            else:
                compiled = graph.compile()
                result_state = dict(compiled.invoke(state))
            state.clear()
            state.update(result_state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("订单调度失败")
            sink({"event": "error", "message": str(exc)})
        finally:
            sink({"event": "final", "state": _serialize_state(state)})

    async def event_stream():
        import threading

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while True:
            evt = await queue.get()
            yield f"data: {json.dumps(evt, ensure_ascii=False, default=str)}\n\n"
            if evt.get("event") in {"final", "error"}:
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/schedule/knowledge")
def schedule_knowledge(limit: int = 50) -> dict[str, Any]:
    """TSLC 知识库 K：已确认的调度计划。"""
    rows = query_all("schedule_knowledge", order_by="created_at DESC", limit=limit)
    for r in rows:
        if r.get("plan"):
            try:
                r["plan"] = json.loads(r["plan"])
            except json.JSONDecodeError:
                pass
    return {"knowledge": rows, "count": len(rows)}


@app.get("/api/schedule/memory")
def schedule_memory(limit: int = 50) -> dict[str, Any]:
    """TSLC 记忆 MS：失败尝试记录。"""
    rows = query_all("schedule_memory", order_by="created_at DESC", limit=limit)
    return {"memory": rows, "count": len(rows)}
