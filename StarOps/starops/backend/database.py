"""SQLite 数据访问层。

第一阶段只做建表 + 基础 CRUD。使用标准库 sqlite3，避免为 MVP 引入 ORM 依赖。
表结构对应方案 §7：satellite / observation_window / sar_image / mission_task /
approval / plan_commit / tool_execution / audit_log。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from config import settings

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- 卫星资源
CREATE TABLE IF NOT EXISTS satellite (
    satellite_id   TEXT PRIMARY KEY,
    satellite_name TEXT NOT NULL,
    satellite_type TEXT NOT NULL,          -- optical / sar
    resolution     REAL NOT NULL,           -- 空间分辨率（米）
    imaging_mode   TEXT NOT NULL,           -- 多光谱 / 全色 / 极化 / 条带等
    status         TEXT NOT NULL DEFAULT 'simulated',  -- 模拟资源状态
    -- TSLC 调度所需：姿态转移时间与最短成像时长（分钟）
    transition_time_min      REAL NOT NULL DEFAULT 0,
    min_imaging_duration_min REAL NOT NULL DEFAULT 8,
    source         TEXT NOT NULL DEFAULT 'simulated',
    created_at     TEXT NOT NULL
);

-- 观测窗口（含数据质量标记）
CREATE TABLE IF NOT EXISTS observation_window (
    window_id       TEXT PRIMARY KEY,
    satellite_id    TEXT NOT NULL REFERENCES satellite(satellite_id),
    area_id         TEXT NOT NULL,
    area_name       TEXT,
    start_time      TEXT NOT NULL,          -- ISO8601 UTC
    end_time        TEXT NOT NULL,
    cloud_rate      REAL NOT NULL,          -- 模拟云量 0~1
    incidence_angle REAL NOT NULL,          -- 入射角（度）
    quality         TEXT NOT NULL DEFAULT 'FRESH',  -- FRESH/STALE/EXPIRED/SUSPECT
    source          TEXT NOT NULL DEFAULT 'simulated',
    created_at      TEXT NOT NULL
);

-- SAR 影像
CREATE TABLE IF NOT EXISTS sar_image (
    image_id    TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    format      TEXT NOT NULL,
    width       INTEGER,
    height      INTEGER,
    dtype       TEXT,
    upload_path TEXT,
    source      TEXT NOT NULL DEFAULT 'simulated',
    uploaded_at TEXT NOT NULL
);

-- 任务
CREATE TABLE IF NOT EXISTS mission_task (
    task_id       TEXT PRIMARY KEY,
    task_type     TEXT NOT NULL DEFAULT 'planning',  -- planning / sar
    requirements  TEXT NOT NULL,                    -- JSON
    selected_plan TEXT,                             -- JSON
    status        TEXT NOT NULL DEFAULT 'draft',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

-- 审批单
CREATE TABLE IF NOT EXISTS approval (
    approval_id  TEXT PRIMARY KEY,
    target_type  TEXT NOT NULL,        -- observation_plan / sar_pipeline
    target_id    TEXT NOT NULL,
    plan_version INTEGER NOT NULL DEFAULT 1,
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending/approved/rejected
    reviewer     TEXT,
    reviewed_at  TEXT,
    comment      TEXT,
    snapshot     TEXT,                 -- JSON 方案快照
    created_at   TEXT NOT NULL
);

-- 幂等提交记录
CREATE TABLE IF NOT EXISTS plan_commit (
    commit_id      TEXT PRIMARY KEY,
    task_id        TEXT NOT NULL,
    plan_version   INTEGER NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    status         TEXT NOT NULL DEFAULT 'succeeded',
    result         TEXT,               -- JSON
    created_at     TEXT NOT NULL
);

-- 工具调用记录
CREATE TABLE IF NOT EXISTS tool_execution (
    execution_id   TEXT PRIMARY KEY,
    trace_id       TEXT NOT NULL,
    tool_name      TEXT NOT NULL,
    input_summary  TEXT,
    output_summary TEXT,
    status         TEXT NOT NULL,
    duration_ms    INTEGER,
    tool_version   TEXT,
    created_at     TEXT NOT NULL
);

-- 审计日志
CREATE TABLE IF NOT EXISTS audit_log (
    id         TEXT PRIMARY KEY,
    actor      TEXT NOT NULL,
    action     TEXT NOT NULL,
    object_id  TEXT,
    details    TEXT,                   -- JSON
    created_at TEXT NOT NULL
);

-- 观测订单（TSLC 订单驱动调度的输入单元）
CREATE TABLE IF NOT EXISTS satellite_order (
    order_id            TEXT PRIMARY KEY,
    area_id             TEXT NOT NULL,
    area_name           TEXT,
    time_start          TEXT NOT NULL,       -- 订单要求时间范围起 ISO8601
    time_end            TEXT NOT NULL,
    satellite_type      TEXT NOT NULL DEFAULT 'any',
    max_resolution      REAL,
    max_cloud_rate      REAL NOT NULL DEFAULT 0.3,
    max_incidence_angle REAL NOT NULL DEFAULT 45.0,
    priority            TEXT NOT NULL DEFAULT 'normal',  -- low/normal/high/urgent
    profit              REAL NOT NULL DEFAULT 2.0,        -- 订单收益权重
    min_duration_min    REAL,
    status              TEXT NOT NULL DEFAULT 'pending',  -- pending/scheduled/rejected
    source              TEXT NOT NULL DEFAULT 'simulated',
    created_at          TEXT NOT NULL
);

-- TSLC 知识库 K：已确认的调度计划
CREATE TABLE IF NOT EXISTS schedule_knowledge (
    knowledge_id  TEXT PRIMARY KEY,
    order_id      TEXT NOT NULL,
    satellite_id  TEXT NOT NULL,
    window_id     TEXT NOT NULL,
    plan          TEXT NOT NULL,          -- JSON
    plan_signature TEXT,                 -- 计划指纹（卫星|窗口|起止时刻），去重键
    created_at    TEXT NOT NULL
);

-- TSLC 记忆 MS：失败尝试记录
CREATE TABLE IF NOT EXISTS schedule_memory (
    memory_id  TEXT PRIMARY KEY,
    order_id   TEXT NOT NULL,
    stage      TEXT NOT NULL,             -- AS / VS / ES
    reason     TEXT NOT NULL,
    attempt    INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_win_sat_area  ON observation_window(satellite_id, area_id, start_time);
CREATE INDEX IF NOT EXISTS idx_tool_trace     ON tool_execution(trace_id);
CREATE INDEX IF NOT EXISTS idx_audit_object  ON audit_log(object_id, created_at);
CREATE INDEX IF NOT EXISTS idx_approval_target ON approval(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_order_status   ON satellite_order(status, created_at);
CREATE INDEX IF NOT EXISTS idx_kb_order       ON schedule_knowledge(order_id);
CREATE INDEX IF NOT EXISTS idx_mem_order      ON schedule_memory(order_id);
"""

# plan_signature 是后补的列，其索引必须在补列之后建（见 init_db）
KNOWLEDGE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_kb_signature   ON schedule_knowledge(plan_signature);
CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_signature_uniq ON schedule_knowledge(plan_signature)
    WHERE plan_signature IS NOT NULL;
"""

# 轻量迁移：给已存在的表补列（CREATE TABLE IF NOT EXISTS 不会改已有表结构）
MIGRATIONS: list[tuple[str, str, str]] = [
    ("satellite", "transition_time_min", "REAL NOT NULL DEFAULT 0"),
    ("satellite", "min_imaging_duration_min", "REAL NOT NULL DEFAULT 8"),
    ("schedule_knowledge", "plan_signature", "TEXT"),
]


def utcnow() -> str:
    """统一 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """建表 + 建索引 + 轻量补列。可重复调用。"""
    conn = get_conn()
    try:
        conn.executescript(SCHEMA)
        _apply_migrations(conn)
        # plan_signature 是补列后才有的列，索引必须在补列之后建
        conn.executescript(KNOWLEDGE_INDEX_SQL)
        conn.commit()
    finally:
        conn.close()


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """给缺失列的表补列。幂等。"""
    for table, column, ddl in MIGRATIONS:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


# ---------------------------------------------------------------- 通用 CRUD

def insert(table: str, data: dict[str, Any]) -> str:
    """插入一行，返回主键值。"""
    cols = ", ".join(data.keys())
    marks = ", ".join(["?"] * len(data))
    conn = get_conn()
    try:
        conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(data.values()))
        conn.commit()
    finally:
        conn.close()
    return str(next(iter(data.values())))


def query_all(
    table: str,
    where: str = "",
    params: Iterable[Any] = (),
    order_by: str = "",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """查询多行。

    where 只放条件（不含 ORDER BY / LIMIT），排序与限行通过独立参数传入，
    避免把 ORDER BY 拼进 WHERE 造成语法错误。
    """
    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    if limit is not None:
        sql += " LIMIT ?"
        params = list(params) + [int(limit)]
    conn = get_conn()
    try:
        rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def query_one(table: str, where: str = "", params: Iterable[Any] = ()) -> dict[str, Any] | None:
    rows = query_all(table, where, params)
    return rows[0] if rows else None


def update(table: str, where: str, params: Iterable[Any], data: dict[str, Any]) -> int:
    sets = ", ".join(f"{k} = ?" for k in data)
    conn = get_conn()
    try:
        cur = conn.execute(
            f"UPDATE {table} SET {sets} WHERE {where}",
            list(data.values()) + list(params),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def exec_sql(sql: str, params: Iterable[Any] = ()) -> int:
    """执行一条写语句（UPDATE / DELETE 等），返回受影响行数。

    只用于本项目内部固定 SQL（如订单状态重置），不接受外部拼接的表名。
    """
    conn = get_conn()
    try:
        cur = conn.execute(sql, tuple(params))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def delete_all(table: str) -> int:
    """清空整表。仅用于模拟数据重建。"""
    conn = get_conn()
    try:
        cur = conn.execute(f"DELETE FROM {table}")
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def upsert(table: str, data: dict[str, Any], key: str) -> str:
    """按主键 upsert。"""
    cols = ", ".join(data.keys())
    marks = ", ".join(["?"] * len(data))
    updates = ", ".join(f"{k} = excluded.{k}" for k in data if k != key)
    sql = (
        f"INSERT INTO {table} ({cols}) VALUES ({marks}) "
        f"ON CONFLICT({key}) DO UPDATE SET {updates}"
    )
    conn = get_conn()
    try:
        conn.execute(sql, list(data.values()))
        conn.commit()
    finally:
        conn.close()
    return str(data[key])


# ---------------------------------------------------------------- 审计

def record_tool_execution(
    trace_id: str,
    tool_name: str,
    input_summary: Any,
    output_summary: Any,
    status: str,
    duration_ms: int,
    tool_version: str = "v1",
) -> str:
    """工具调用留痕。input/output 只存摘要，避免把大对象塞进库。"""
    execution_id = new_id("exec")
    insert(
        "tool_execution",
        {
            "execution_id": execution_id,
            "trace_id": trace_id,
            "tool_name": tool_name,
            "input_summary": json.dumps(input_summary, ensure_ascii=False, default=str)[:2000],
            "output_summary": json.dumps(output_summary, ensure_ascii=False, default=str)[:2000],
            "status": status,
            "duration_ms": duration_ms,
            "tool_version": tool_version,
            "created_at": utcnow(),
        },
    )
    return execution_id


def record_audit(actor: str, action: str, object_id: str | None, details: Any) -> str:
    """业务审计。与工作流 checkpoint 分开管理。"""
    if not settings.audit_enabled:
        return ""
    audit_id = new_id("audit")
    insert(
        "audit_log",
        {
            "id": audit_id,
            "actor": actor,
            "action": action,
            "object_id": object_id,
            "details": json.dumps(details, ensure_ascii=False, default=str)[:2000],
            "created_at": utcnow(),
        },
    )
    return audit_id
