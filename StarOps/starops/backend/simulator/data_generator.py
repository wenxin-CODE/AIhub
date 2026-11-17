"""模拟数据生成器。

生成自洽的卫星资源与观测窗口数据。
所有数据带 source='simulated' 标记，不冒充真实卫星数据。
"""

from __future__ import annotations

import hashlib
import math
import random
from datetime import datetime, timedelta, timezone

from database import utcnow
from tools.planning import AREAS  # 区域字典与需求解析共用一份，避免两处维护

SATELLITES = [
    # id, name, type, resolution(m), imaging_mode, 姿态转移时间(min), 最短成像时长(min)
    ("SAT-001", "模拟光学卫星 A", "optical", 0.8, "全色+多光谱", 12.0, 8.0),
    ("SAT-002", "模拟光学卫星 B", "optical", 1.0, "多光谱", 15.0, 10.0),
    ("SAT-003", "模拟光学卫星 C", "optical", 1.2, "多光谱", 18.0, 10.0),
    ("SAT-004", "模拟光学卫星 D", "optical", 2.0, "多光谱", 20.0, 12.0),
    ("SAT-005", "模拟光学卫星 E", "optical", 3.0, "中分辨率多光谱", 22.0, 12.0),
    ("SAT-006", "模拟 SAR 卫星 F", "sar", 1.0, "双极化 C 波段", 25.0, 15.0),
    ("SAT-007", "模拟 SAR 卫星 G", "sar", 0.6, "全极化", 30.0, 20.0),
    ("SAT-008", "模拟光学卫星 H", "optical", 1.5, "条带多光谱", 16.0, 10.0),
]

# AREAS 统一由 tools.planning 提供（需求解析与模拟数据共用同一份字典）


def _window_id(satellite_id: str, area_id: str, start: datetime) -> str:
    """窗口 ID 由「卫星 | 区域 | 起始时刻」派生，不使用随机 UUID。

    随机 ID 会让每次启动生成一批全新的窗口 id：既破坏「演示结果可重复」，
    也让知识库 K 的计划指纹（卫星|窗口|起止时刻）跨重启无法去重。
    """
    raw = f"{satellite_id}|{area_id}|{start.isoformat()}".encode("utf-8")
    return f"win-{hashlib.sha1(raw).hexdigest()[:12]}"


def seed_satellites() -> int:
    """写入模拟卫星资源。幂等：已存在则只补齐调度参数。"""
    from database import insert, query_all, update

    existing = {r["satellite_id"] for r in query_all("satellite")}
    count = 0
    for sid, name, stype, res, mode, trans, min_dur in SATELLITES:
        if sid in existing:
            # 旧库行补齐 TSLC 调度参数（转移时间 / 最短成像时长）
            update(
                "satellite",
                "satellite_id = ?",
                (sid,),
                {"transition_time_min": trans, "min_imaging_duration_min": min_dur},
            )
            continue
        insert(
            "satellite",
            {
                "satellite_id": sid,
                "satellite_name": name,
                "satellite_type": stype,
                "resolution": res,
                "imaging_mode": mode,
                "transition_time_min": trans,
                "min_imaging_duration_min": min_dur,
                "status": "simulated",
                "source": "simulated",
                "created_at": utcnow(),
            },
        )
        count += 1
    return count


def seed_windows(days_ahead: int = 14, per_satellite: int = 3, reset: bool = False) -> int:
    """生成未来 N 天内的模拟过境窗口。

    物理合理性约束：单星单日过境次数有限、云量随机但有区域偏好、
    入射角落在合理区间。

    reset=True 时先清空再生成，避免重复启动导致数据累积膨胀。

    **可重复性**：时间基准锚定到当日 UTC 零点，window_id 由
    「卫星 | 区域 | 起始时刻」派生而非随机 UUID。因此同一天内重复启动
    生成的数据完全一致——演示结果可复现，知识库 K 的计划指纹也能跨重启去重。
    """
    from database import delete_all, insert, query_all

    if reset:
        delete_all("observation_window")

    existing = {r["window_id"] for r in query_all("observation_window")}
    rng = random.Random(20261115)  # 固定种子，保证演示可重复
    # 锚定到当日 UTC 零点：同一天内重启得到完全相同的窗口集
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    count = 0

    for sat_idx, (sid, name, stype, res, mode, _trans, _min_dur) in enumerate(SATELLITES):
        # 区域轮转覆盖：每颗卫星在每个区域都至少有一个过境窗口。
        # 随机分配区域会让某些区域/卫星类型完全没有窗口（例如 SAR 星缺深圳窗口），
        # 演示上会表现成"莫名其妙的订单被拒"，覆盖不均不是调度算法的问题。
        for k in range(per_satellite * 2):
            area_id, area_name = AREAS[(sat_idx + k) % len(AREAS)]
            day_offset = rng.randint(1, max(1, days_ahead))
            base = (now + timedelta(days=day_offset)).replace(
                hour=rng.choice([2, 6, 10, 14, 18]), minute=rng.choice([5, 17, 33, 48]),
                second=0, microsecond=0,
            )
            # 窗口时长必须大于各星的最短成像时长（8~20 分钟），
            # 否则窗口连一次成像都装不下，那颗星永远不可能被选中。
            duration_min = rng.choice([20, 25, 30, 35, 40, 45])
            # 云量：截断正态，中心 0.22。
            # 原 Beta(2,5) 均值 0.286，导致近半窗口超过 30% 阈值，
            # 8 颗星的候选几乎全被否掉，演示变成"随机拒绝"而非调度演示。
            cloud_rate = round(min(0.95, max(0.0, rng.gauss(0.22, 0.16))), 3)
            incidence = round(rng.uniform(15.0, 55.0), 2)
            quality = "FRESH"
            if rng.random() < 0.08:
                quality = "STALE"
            elif rng.random() < 0.04:
                quality = "SUSPECT"

            window_id = _window_id(sid, area_id, base)
            if window_id in existing:
                continue
            insert(
                "observation_window",
                {
                    "window_id": window_id,
                    "satellite_id": sid,
                    "area_id": area_id,
                    "area_name": area_name,
                    "start_time": base.isoformat(),
                    "end_time": (base + timedelta(minutes=duration_min)).isoformat(),
                    "cloud_rate": cloud_rate,
                    "incidence_angle": incidence,
                    "quality": quality,
                    "source": "simulated",
                    "created_at": utcnow(),
                },
            )
            count += 1

    if reset:
        _prune_stale_knowledge()
    return count


def _prune_stale_knowledge() -> int:
    """清除引用了已不存在窗口的知识条目。

    窗口重建后，旧的确认计划所指向的窗口已不存在，计划本身失效。
    留着它们会让知识库页面出现"指向不存在窗口的调度计划"，属于脏数据。
    """
    from database import get_conn, query_all

    live = {r["window_id"] for r in query_all("observation_window")}
    stale = [r["knowledge_id"] for r in query_all("schedule_knowledge")
             if r["window_id"] not in live]
    if not stale:
        return 0
    conn = get_conn()
    try:
        conn.executemany(
            "DELETE FROM schedule_knowledge WHERE knowledge_id = ?",
            [(k,) for k in stale],
        )
        conn.commit()
    finally:
        conn.close()
    return len(stale)


def seed_orders(count: int = 6, reset: bool = False) -> int:
    """生成模拟观测订单队列（TSLC 订单驱动调度的输入）。

    订单覆盖不同区域、优先级与约束松紧，其中刻意包含 1 条极紧约束订单，
    用于验证"无可行方案 → 被拒并留痕"的路径，而不是只造能全部成功的样本。

    时间范围必须覆盖窗口生成周期（seed_windows 的 days_ahead=14），
    否则订单会因"窗口不在时间范围内"被批量拒绝，掩盖真正的调度能力。
    """
    from database import get_conn, insert, query_all

    if reset:
        # 只清理模拟订单，保留用户通过 API 创建的订单
        conn = get_conn()
        try:
            conn.execute("DELETE FROM satellite_order WHERE source = 'simulated'")
            conn.commit()
        finally:
            conn.close()

    existing = {r["order_id"] for r in query_all("satellite_order")}
    rng = random.Random(20261117)
    # 与 seed_windows 同一基准：锚定当日 UTC 零点，保证订单时间范围稳定
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    # (area_id, area_name, satellite_type, max_res, max_cloud, max_inc, priority, days)
    # days 统一取 14，与 seed_windows(days_ahead=14) 的窗口跨度对齐——
    # 否则订单时间范围比窗口短，被拒原因会混入"窗口不在订单时间范围内"这种
    # 数据范围错配，淹没真正的约束根因。
    blueprints = [
        ("CN-BJ", "北京", "optical", 2.0, 0.30, 45.0, "urgent", 14),
        ("CN-SH", "上海", "optical", 1.5, 0.25, 40.0, "high", 14),
        ("CN-GZ", "广州", "any", 3.0, 0.35, 50.0, "normal", 14),
        ("CN-SZ", "深圳", "sar", 1.0, 0.25, 45.0, "high", 14),
        ("CN-CD", "成都", "optical", 2.5, 0.30, 45.0, "normal", 14),
        # 极紧约束：云量上限与入射角上限都很苛刻，大概率无可行方案 → 走被拒路径
        ("CN-XA", "西安", "optical", 0.8, 0.05, 20.0, "low", 14),
    ]

    profit_by_priority = {"urgent": 4.0, "high": 3.0, "normal": 2.0, "low": 1.0}
    written = 0
    for i, (area_id, area_name, stype, res, cloud, inc, priority, days) in enumerate(blueprints):
        if i >= count:
            break
        order_id = f"ord-{i + 1:03d}"
        if order_id in existing:
            continue
        start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=days)
        insert(
            "satellite_order",
            {
                "order_id": order_id,
                "area_id": area_id,
                "area_name": area_name,
                "time_start": start.isoformat(),
                "time_end": end.isoformat(),
                "satellite_type": stype,
                "max_resolution": res,
                "max_cloud_rate": cloud,
                "max_incidence_angle": inc,
                "priority": priority,
                "profit": profit_by_priority[priority],
                "min_duration_min": None,
                "status": "pending",
                "source": "simulated",
                "created_at": utcnow(),
            },
        )
        written += 1
    return written


def seed_all(reset_windows: bool = True, reset_orders: bool = True) -> dict[str, int]:
    """初始化全部模拟数据。

    默认重建窗口与订单队列，保证演示数据规模稳定、结果可重复。
    """
    return {
        "satellites": seed_satellites(),
        "windows": seed_windows(reset=reset_windows),
        "orders": seed_orders(reset=reset_orders),
    }
