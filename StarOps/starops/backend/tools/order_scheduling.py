"""TSLC 订单驱动调度核心。

参考方法
--------
Chen J., Pedrycz W., Wang F., Wu J., Wu G., Xing L., Chen Y., Song Y.,
"A Tri-Stage LLM-Coordinated Framework for Order-Driven Scheduling of
Earth Observation Satellite Tasks", *IEEE Transactions on Geoscience and
Remote Sensing*, Vol. 64, Art. 3002015, 2026.
DOI: 10.1109/TGRS.2026.3700605

论文方法 → 本模块的映射（依据论文 Fig.1–Fig.6 与 Index Terms）
--------------------------------------------------------------
论文把 EOSTSP 从"单体式调度"重构为"实时订单履约"：每个观测任务视为
一个独立 order，顺序处理，每个 order 依次穿过三个专门决策阶段。

======================  ====================================================
论文组件                本模块落点
======================  ====================================================
订单驱动调度策略         :func:`schedule_orders` 逐单处理订单队列
Stage 1  AS              :func:`stage_assign_satellite`   —— 指派卫星
Stage 2  VS              :func:`stage_select_window`       —— 选择有效时窗 VTW
Stage 3  ES              :func:`stage_exact_schedule`      —— 确定精确观测时刻
CV  约束校验器            :class:`ConstraintVerifier`       —— 分阶段约束检查
PM  决策管理器            :class:`PerformanceManager`       —— 迭代验证 + 失败重规划
MS  失败记忆              :func:`_memory_store`             —— 记录失败尝试
K   知识库                :func:`_knowledge_store`          —— 记录已确认计划
======================  ====================================================

与论文的差异（必须如实说明，见 README 与方案文档 §4.5）
------------------------------------------------------
1. 论文面向几十至上百颗敏捷卫星、80–540 个订单的真实规模在线调度；
   本项目是 8 颗模拟卫星、数十条模拟窗口的原型演示，规模不可比。
2. 论文目标是最大化订单利润（profit）的全局最优；本项目是需求驱动的
   可行性优先 + 效用排序，且保留人工审批关口（论文没有这一环）。
3. 论文的 LLM 是真实推理（GPT-4o 等）并统计 token 成本；本项目无 Key 时
   走确定性规则降级，`degraded` 标记会显式输出，不伪装真实推理。
4. 论文的转移时间来自真实轨道与姿态机动模型；本项目用每星固定参数模拟。
5. 论文的 4.58% / 13.38% 提升依赖与 CP/GA/SA/DRL 等 SOTA 的同规模对比实验，
   本项目不复现该数字，也不做该声明。

设计原则（与项目主链一致）
--------------------------
LLM 只做**候选偏好编排**（可用时），所有**可行性判定与数值计算**由本模块
的确定性代码完成——与论文"外部校验保证运行可行性"（external validation）
以及项目主链"LLM 只编排不算"的主张一致。
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from database import query_all, utcnow
from schemas.models import ToolResult

EventSink = Callable[[dict[str, Any]], None]

# ---------------------------------------------------------------- 论文 Fig.4
# 每个 agent 的 prompt 描述其 name / ability / responsibilities。
# 这里只保留"角色定义"，真正的数值裁决仍由确定性代码完成。

AGENT_PROMPTS: dict[str, str] = {
    "AS": """你是卫星指派智能体（Assigning Satellite agent, AS）。
能力：依据订单的成像类型、分辨率上限与优先级，在候选卫星中给出偏好排序。
职责：只输出候选卫星的偏好顺序与简短理由，不得计算分数、不得编造不存在的卫星。
输出 JSON：{"ranking": ["SAT-001", ...], "reasons": {"SAT-001": "..."}}""",
    "VS": """你是有效时窗选择智能体（Valid Time Window agent, VS）。
能力：在已指派卫星的候选过境窗口中，依据云量、入射角、时间紧迫性选择最优窗口。
职责：只输出窗口 ID 的偏好顺序与简短理由，不得改写窗口的物理参数。
输出 JSON：{"ranking": ["win-xxx", ...], "reasons": {"win-xxx": "..."}}""",
    "ES": """是精确观测时刻智能体（Exact Scheduling agent, ES）。
能力：在已选窗口内，结合成像时长与姿态转移时间，给出可执行观测时刻的偏好。
职责：只输出时刻偏好，不得越出窗口边界，不得与其他订单的已排任务冲突。
输出 JSON：{"preferred_start": "ISO8601", "reason": "..."}""",
    "CV": """你是约束校验器（Constraint Verifier, CV）。
能力：按阶段检查分辨率、类型、时间范围、云量、入射角、数据质量、成像时长、
转移时间与资源独占。
职责：只判定通过或违反并给出原因，不做取舍判断，不放宽任何硬约束。""",
    "PM": """你是决策管理器（Performance Manager, PM）。
能力：依据 CV 的校验结论决定接受当前决策还是触发重规划，管理失败记忆与知识库。
职责：不得放宽任何硬约束；记录失败尝试；把最终确认的计划写入知识库。""",
}

# 各阶段的归一化权重（论文未公开具体权重，此处为本项目自设工程参数）
AS_WEIGHTS = {"resolution": 0.6, "idleness": 0.4}
VS_WEIGHTS = {"cloud": 0.35, "incidence": 0.25, "timeliness": 0.25, "slack": 0.15}
ES_WEIGHTS = {"slack": 0.6, "timeliness": 0.4}

PRIORITY_PROFIT = {"urgent": 4.0, "high": 3.0, "normal": 2.0, "low": 1.0}


# ---------------------------------------------------------------- 基础工具

def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _elapsed(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _normalize(values: list[float]) -> list[float]:
    """min-max 归一化到 [0,1]。全等时返回全 1，避免除零与人为放大差异。"""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end


def plan_signature(plan: dict[str, Any]) -> str:
    """计划指纹：卫星 | 窗口 | 观测起止时刻 的规范化串。

    用于知识库 K 去重——同一条观测计划（不论由哪一轮调度、哪个订单确认）
    只保留一条知识记录。
    """
    parts = [
        str(plan.get("satellite_id", "")),
        str(plan.get("window_id", "")),
        str(plan.get("observation_start", "")),
        str(plan.get("observation_end", "")),
    ]
    return "|".join(parts)


# ---------------------------------------------------------------- 订单

DEFAULT_ORDER_FIELDS = {
    "satellite_type": "any",
    "max_resolution": None,
    "max_cloud_rate": 0.3,
    "max_incidence_angle": 45.0,
    "priority": "normal",
    "profit": None,
    "min_duration_min": None,
    # 注意：area_id / area_name / time_start / time_end 由调用方显式传入，
    # 不能出现在这里——`**DEFAULT_ORDER_FIELDS` 会覆盖掉传入值，
    # 曾导致所有订单的 area_id 被悄悄置空、区域约束整体失效。
}


def build_order(
    order_id: str,
    area_id: str,
    area_name: str,
    time_start: str,
    time_end: str,
    **overrides: Any,
) -> dict[str, Any]:
    """构造一个标准订单。字段缺失一律用默认值，不静默猜测。"""
    order = {
        "order_id": order_id,
        "area_id": area_id,
        "area_name": area_name,
        "time_start": time_start,
        "time_end": time_end,
        **DEFAULT_ORDER_FIELDS,
    }
    for k, v in overrides.items():
        if v is not None:
            order[k] = v
    order["profit"] = float(order["profit"] or PRIORITY_PROFIT.get(order["priority"], 2.0))
    order["status"] = "pending"
    return order


# ---------------------------------------------------------------- Stage 1: AS

def stage_assign_satellite(
    order: dict[str, Any],
    satellites: list[dict[str, Any]],
    occupancy: dict[str, list[tuple[datetime, datetime]]],
    excluded: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Stage 1 — AS：为订单指派卫星。

    返回 (候选卫星列表, 拒绝原因列表)。候选按效用降序。
    """
    excluded = excluded or set()
    rejected: list[str] = []
    cands: list[dict[str, Any]] = []

    want_type = order.get("satellite_type", "any")
    max_res = order.get("max_resolution")

    for sat in satellites:
        sid = sat["satellite_id"]
        if sid in excluded:
            rejected.append(f"{sid}：本轮重规划已排除")
            continue
        if want_type and want_type != "any" and sat["satellite_type"] != want_type:
            rejected.append(f"{sid}：类型 {sat['satellite_type']} 不满足 {want_type}")
            continue
        if max_res and sat["resolution"] > max_res:
            rejected.append(f"{sid}：分辨率 {sat['resolution']}m 超出上限 {max_res}m")
            continue
        if sat.get("status") not in (None, "simulated", "active"):
            rejected.append(f"{sid}：卫星状态 {sat.get('status')} 不可用")
            continue
        cands.append(dict(sat))

    if not cands:
        return [], rejected

    # 效用：分辨率能力（越小越好）+ 空闲度（已排任务越少越好）
    res_raw = [1.0 / (1.0 + s["resolution"]) for s in cands]
    idle_raw = [1.0 / (1.0 + len(occupancy.get(s["satellite_id"], []))) for s in cands]
    res_n = _normalize(res_raw)
    idle_n = _normalize(idle_raw)
    for i, s in enumerate(cands):
        s["stage1_score"] = round(
            res_n[i] * AS_WEIGHTS["resolution"] + idle_n[i] * AS_WEIGHTS["idleness"], 4
        )
        s["stage1_breakdown"] = {
            "resolution": round(res_n[i], 4),
            "idleness": round(idle_n[i], 4),
        }
    cands.sort(key=lambda x: x["stage1_score"], reverse=True)
    return cands, rejected


# ---------------------------------------------------------------- Stage 2: VS

def stage_select_window(
    order: dict[str, Any],
    satellite: dict[str, Any],
    windows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Stage 2 — VS：在已指派卫星上选择有效时窗 VTW。

    返回 (候选窗口列表, 拒绝原因列表)。候选按效用降序。
    """
    rejected: list[str] = []
    quality_reasons: list[str] = []
    cands: list[dict[str, Any]] = []

    max_cloud = order.get("max_cloud_rate", 0.3)
    max_inc = order.get("max_incidence_angle", 45.0)
    o_start, o_end = _parse(order["time_start"]), _parse(order["time_end"])
    min_dur = float(
        order.get("min_duration_min") or satellite.get("min_imaging_duration_min") or 8.0
    )

    for w in windows:
        wid = w["window_id"]
        if w["satellite_id"] != satellite["satellite_id"]:
            continue
        if w.get("area_id") and order.get("area_id") and w["area_id"] != order["area_id"]:
            # 区域不匹配是「不适用」而非「违规」：该窗口不是本订单的候选，
            # 记进 reasons 会淹没真正的约束违反原因。
            continue
        if w.get("quality") == "EXPIRED":
            quality_reasons.append(f"{wid}：窗口已过期")
            continue
        if w.get("quality") == "SUSPECT":
            quality_reasons.append(f"{wid}：窗口数据可疑，需人工复核")
            continue
        w_start, w_end = _parse(w["start_time"]), _parse(w["end_time"])
        if w_end < o_start or w_start > o_end:
            rejected.append(f"{wid}：窗口不在订单时间范围内")
            continue
        if w["cloud_rate"] > max_cloud:
            rejected.append(f"{wid}：云量 {w['cloud_rate']} 超出上限 {max_cloud}")
            continue
        if w["incidence_angle"] > max_inc:
            rejected.append(f"{wid}：入射角 {w['incidence_angle']}° 超出上限 {max_inc}°")
            continue
        if (w_end - w_start).total_seconds() / 60.0 < min_dur:
            rejected.append(f"{wid}：窗口时长不足最短成像 {min_dur} 分钟")
            continue
        item = dict(w)
        item["window_duration_min"] = (w_end - w_start).total_seconds() / 60.0
        cands.append(item)

    if not cands:
        # 数值约束违反优先于数据质量标记——前者是硬约束，后者只是数据可信度问题
        return [], rejected + quality_reasons

    now = datetime.now(timezone.utc)
    cloud_raw = [1.0 - w["cloud_rate"] for w in cands]
    inc_raw = [1.0 - (w["incidence_angle"] / 90.0) for w in cands]
    time_raw = [
        1.0 / (1.0 + max(0.0, (_parse(w["start_time"]) - now).total_seconds() / 86400.0))
        for w in cands
    ]
    slack_raw = [w["window_duration_min"] for w in cands]
    cn, inn, tn, sn = (
        _normalize(cloud_raw),
        _normalize(inc_raw),
        _normalize(time_raw),
        _normalize(slack_raw),
    )
    for i, w in enumerate(cands):
        w["stage2_score"] = round(
            cn[i] * VS_WEIGHTS["cloud"]
            + inn[i] * VS_WEIGHTS["incidence"]
            + tn[i] * VS_WEIGHTS["timeliness"]
            + sn[i] * VS_WEIGHTS["slack"],
            4,
        )
        w["stage2_breakdown"] = {
            "cloud": round(cn[i], 4),
            "incidence": round(inn[i], 4),
            "timeliness": round(tn[i], 4),
            "slack": round(sn[i], 4),
        }
    cands.sort(key=lambda x: x["stage2_score"], reverse=True)
    return cands, rejected


# ---------------------------------------------------------------- Stage 3: ES

def stage_exact_schedule(
    order: dict[str, Any],
    satellite: dict[str, Any],
    window: dict[str, Any],
    occupancy: dict[str, list[tuple[datetime, datetime]]],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Stage 3 — ES：在选定 VTW 内确定精确观测时刻。

    约束：成像时长可行、与该星已有任务不冲突且满足姿态转移时间。
    返回 (调度片段, 拒绝原因列表)。
    """
    rejected: list[str] = []
    sid = satellite["satellite_id"]

    min_dur = float(
        order.get("min_duration_min") or satellite.get("min_imaging_duration_min") or 8.0
    )
    transition = float(satellite.get("transition_time_min") or 0.0)

    win_start, win_end = _parse(window["start_time"]), _parse(window["end_time"])
    duration = timedelta(minutes=min_dur)

    occupied = occupancy.get(sid, [])

    # 已被占用的时段两侧各留出转移时间，形成阻塞区间
    blocked = [
        (s - timedelta(minutes=transition), e + timedelta(minutes=transition))
        for (s, e) in occupied
    ]

    t = win_start
    for (bs, be) in sorted(blocked):
        if t < be and bs < t + duration:
            t = be
    if t + duration > win_end:
        rejected.append(
            f"{window['window_id']}：扣除 {transition} 分钟转移时间后无可行观测时刻"
            + (f"（已有 {len(occupied)} 个已排任务）" if occupied else "")
        )
        return None, rejected

    obs_start, obs_end = t, t + duration
    slack = (win_end - obs_end).total_seconds() / 60.0
    urgency = 1.0 / (
        1.0 + max(0.0, (obs_start - datetime.now(timezone.utc)).total_seconds() / 86400.0)
    )
    slack_n = slack / 60.0 if slack < 60 else 1.0
    score = round(
        min(1.0, slack_n) * ES_WEIGHTS["slack"] + urgency * ES_WEIGHTS["timeliness"], 4
    )

    return (
        {
            "observation_start": _iso(obs_start),
            "observation_end": _iso(obs_end),
            "duration_min": min_dur,
            "transition_time_min": transition,
            "slack_min": round(slack, 1),
            "stage3_score": score,
        },
        rejected,
    )


# ---------------------------------------------------------------- CV 约束校验

class ConstraintVerifier:
    """CV — 分阶段约束校验器（论文 external validation 的落点）。

    与论文一致：校验独立于决策者，只判定通过/违反并给出原因，
    不做取舍，不放宽任何硬约束。
    """

    VERSION = "tslc-cv-v1"

    def check_stage1(
        self, order: dict[str, Any], satellite: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        violations: list[str] = []
        want_type = order.get("satellite_type", "any")
        if want_type and want_type != "any" and satellite["satellite_type"] != want_type:
            violations.append(f"类型 {satellite['satellite_type']} 不满足 {want_type}")
        max_res = order.get("max_resolution")
        if max_res and satellite["resolution"] > max_res:
            violations.append(f"分辨率 {satellite['resolution']}m 超出上限 {max_res}m")
        return (not violations), violations

    def check_stage2(
        self, order: dict[str, Any], satellite: dict[str, Any], window: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        violations: list[str] = []
        if window.get("quality") == "EXPIRED":
            violations.append("窗口已过期")
        if window.get("quality") == "SUSPECT":
            violations.append("窗口数据可疑")
        if window["area_id"] and order.get("area_id") and window["area_id"] != order["area_id"]:
            violations.append(f"区域 {window['area_id']} 不匹配订单 {order['area_id']}")
        w_start, w_end = _parse(window["start_time"]), _parse(window["end_time"])
        if w_end < _parse(order["time_start"]) or w_start > _parse(order["time_end"]):
            violations.append("窗口不在订单时间范围内")
        if window["cloud_rate"] > order.get("max_cloud_rate", 0.3):
            violations.append(f"云量 {window['cloud_rate']} 超出上限 {order.get('max_cloud_rate')}")
        if window["incidence_angle"] > order.get("max_incidence_angle", 45.0):
            violations.append(
                f"入射角 {window['incidence_angle']}° 超出上限 {order.get('max_incidence_angle')}°"
            )
        return (not violations), violations

    def check_stage3(
        self,
        order: dict[str, Any],
        satellite: dict[str, Any],
        window: dict[str, Any],
        slot: dict[str, Any],
        occupancy: dict[str, list[tuple[datetime, datetime]]],
    ) -> tuple[bool, list[str]]:
        violations: list[str] = []
        min_dur = float(
            order.get("min_duration_min") or satellite.get("min_imaging_duration_min") or 8.0
        )
        transition = float(satellite.get("transition_time_min") or 0.0)
        o_s, o_e = _parse(slot["observation_start"]), _parse(slot["observation_end"])
        w_s, w_e = _parse(window["start_time"]), _parse(window["end_time"])

        if (o_e - o_s).total_seconds() / 60.0 < min_dur - 1e-6:
            violations.append(f"观测时长不足最短成像 {min_dur} 分钟")
        if o_s < w_s or o_e > w_e:
            violations.append("观测时刻越出窗口边界")
        for (s, e) in occupancy.get(satellite["satellite_id"], []):
            if _overlaps(o_s, o_e, s, e):
                violations.append(f"与已排任务冲突：{s.isoformat()} ~ {e.isoformat()}")
            elif e <= o_s and (o_s - e).total_seconds() / 60.0 < transition - 1e-6:
                violations.append(
                    f"与前序任务间隔 {(o_s - e).total_seconds() / 60.0:.1f} 分钟，"
                    f"小于转移时间 {transition} 分钟"
                )
            elif s >= o_e and (s - o_e).total_seconds() / 60.0 < transition - 1e-6:
                violations.append(
                    f"与后序任务间隔 {(s - o_e).total_seconds() / 60.0:.1f} 分钟，"
                    f"小于转移时间 {transition} 分钟"
                )
        return (not violations), violations


# ---------------------------------------------------------------- PM / MS / K

class PerformanceManager:
    """PM — 决策管理器：迭代验证每阶段决策，失败则触发重规划。

    论文 Fig.6：PM 迭代校验每个 agent 的决策；失败尝试写入 MS，
    最终确认的调度计划写入知识库 K。
    """

    def __init__(self, max_replans: int = 3) -> None:
        self.max_replans = max_replans
        self.memory: list[dict[str, Any]] = []
        self.knowledge: list[dict[str, Any]] = []

    def record_failure(self, order_id: str, stage: str, reason: str, attempt: int) -> None:
        """写入 MS（失败尝试记忆）。"""
        self.memory.append(
            {
                "order_id": order_id,
                "stage": stage,
                "reason": reason,
                "attempt": attempt,
                "created_at": utcnow(),
            }
        )

    def record_plan(self, plan: dict[str, Any]) -> None:
        """写入 K（已确认计划知识库）。

        附带 `plan_signature` 计划指纹：卫星 | 窗口 | 观测起止时刻 的规范化串。
        指纹相同的计划视为同一条知识，落库时去重（见 order_nodes 的 upsert）。
        """
        self.knowledge.append(
            {**plan, "plan_signature": plan_signature(plan), "confirmed_at": utcnow()}
        )

    def replan_budget(self) -> int:
        return self.max_replans


# ---------------------------------------------------------------- 编排

def schedule_orders(
    orders: list[dict[str, Any]],
    satellites: list[dict[str, Any]] | None = None,
    windows: list[dict[str, Any]] | None = None,
    max_replans: int = 3,
    on_event: EventSink | None = None,
) -> ToolResult:
    """订单驱动调度：逐单执行 AS → VS → ES 三阶段序列决策。

    返回统一 ToolResult，data 内含调度计划、被拒订单与评估指标。
    """
    start = time.perf_counter()
    sats = satellites if satellites is not None else query_all("satellite")
    wins = windows if windows is not None else query_all("observation_window")

    cv = ConstraintVerifier()
    pm = PerformanceManager(max_replans=max_replans)

    occupancy: dict[str, list[tuple[datetime, datetime]]] = {}
    scheduled: list[dict[str, Any]] = []
    rejected_orders: list[dict[str, Any]] = []
    atomic_decisions = 0
    replans = 0
    stage_failures = 0

    def emit(event: str, **payload: Any) -> None:
        if on_event:
            on_event({"event": event, **payload})

    emit(
        "order_batch_start",
        total_orders=len(orders),
        satellites=len(sats),
        windows=len(wins),
        rule_version=cv.VERSION,
        source="simulated",
    )

    for order in orders:
        order_id = order["order_id"]
        excluded_sats: set[str] = set()
        excluded_wins: set[str] = set()
        attempt = 0
        scheduled_ok = False
        last_reasons: list[str] = []
        last_stage = ""
        # 跨尝试累积各阶段的拒绝原因：单次尝试只看一颗卫星、一个窗口，
        # 只保留最后一次会让根因被覆盖，报告变成"随机的一句"。
        stage_reasons: dict[str, list[str]] = {}

        while attempt <= pm.replan_budget():
            attempt += 1
            emit("order_start", order_id=order_id, attempt=attempt, area_id=order.get("area_id"))

            # ---- Stage 1: AS
            sat_cands, s_reasons = stage_assign_satellite(
                order, sats, occupancy, excluded_sats
            )
            atomic_decisions += 1
            if not sat_cands:
                # 候选耗尽：区分"一开始就没有可用卫星"和"重规划中已被全部排除"。
                # 若是重规划导致的耗尽，保留此前阶段的实质性失败原因——那才是根因，
                # "候选已耗尽"只是结果，不是原因。
                stage_failures += 1
                if excluded_sats:
                    if not last_reasons:
                        last_reasons = [
                            f"候选卫星在前序重规划中已全部尝试且均不可行"
                            f"（已排除 {len(excluded_sats)} 颗）"
                        ]
                else:
                    last_reasons = s_reasons or ["无可用卫星"]
                    last_stage = "AS"
                stage_reasons.setdefault("AS", []).extend(last_reasons)
                pm.record_failure(order_id, "AS", "；".join(last_reasons), attempt)
                emit("stage_failed", order_id=order_id, stage="AS", reasons=last_reasons)
                break
            sat = sat_cands[0]
            emit(
                "stage_decision",
                order_id=order_id,
                stage="AS",
                agent="AS",
                decision={"satellite_id": sat["satellite_id"], "satellite_name": sat["satellite_name"]},
                score=sat["stage1_score"],
                candidates=len(sat_cands),
                reasons=s_reasons[:3],
            )

            ok, v = cv.check_stage1(order, sat)
            if not ok:
                stage_failures += 1
                pm.record_failure(order_id, "AS", "；".join(v), attempt)
                excluded_sats.add(sat["satellite_id"])
                last_reasons = v
                emit("stage_failed", order_id=order_id, stage="AS", reasons=v)
                replans += 1
                continue

            # ---- Stage 2: VS
            win_cands, w_reasons = stage_select_window(order, sat, wins)
            win_cands = [w for w in win_cands if w["window_id"] not in excluded_wins]
            atomic_decisions += 1
            if not win_cands:
                last_reasons = w_reasons or ["无可用窗口"]
                last_stage = "VS"
                stage_reasons.setdefault("VS", []).extend(last_reasons)
                stage_failures += 1
                pm.record_failure(order_id, "VS", "；".join(last_reasons), attempt)
                excluded_sats.add(sat["satellite_id"])
                emit("stage_failed", order_id=order_id, stage="VS", reasons=last_reasons)
                replans += 1
                continue
            win = win_cands[0]
            emit(
                "stage_decision",
                order_id=order_id,
                stage="VS",
                agent="VS",
                decision={"window_id": win["window_id"], "area_name": win.get("area_name")},
                score=win["stage2_score"],
                candidates=len(win_cands),
                reasons=w_reasons[:3],
            )

            ok, v = cv.check_stage2(order, sat, win)
            if not ok:
                stage_failures += 1
                pm.record_failure(order_id, "VS", "；".join(v), attempt)
                excluded_wins.add(win["window_id"])
                last_reasons = v
                stage_reasons.setdefault("VS", []).extend(v)
                emit("stage_failed", order_id=order_id, stage="VS", reasons=v)
                replans += 1
                continue

            # ---- Stage 3: ES
            slot, e_reasons = stage_exact_schedule(order, sat, win, occupancy)
            atomic_decisions += 1
            if slot is None:
                last_reasons = e_reasons or ["无可行观测时刻"]
                stage_reasons.setdefault("ES", []).extend(last_reasons)
                stage_failures += 1
                pm.record_failure(order_id, "ES", "；".join(last_reasons), attempt)
                excluded_wins.add(win["window_id"])
                # 只排除该窗口，不排除整颗卫星——同一卫星可能还有其它可用窗口，
                # 一刀切排除卫星会把本可排期的订单误杀。
                last_stage = "ES"
                emit("stage_failed", order_id=order_id, stage="ES", reasons=last_reasons)
                replans += 1
                continue

            emit(
                "stage_decision",
                order_id=order_id,
                stage="ES",
                agent="ES",
                decision={
                    "observation_start": slot["observation_start"],
                    "observation_end": slot["observation_end"],
                },
                score=slot["stage3_score"],
                candidates=1,
                reasons=[],
            )

            ok, v = cv.check_stage3(order, sat, win, slot, occupancy)
            if not ok:
                stage_failures += 1
                pm.record_failure(order_id, "ES", "；".join(v), attempt)
                excluded_wins.add(win["window_id"])
                last_reasons = v
                last_stage = "ES"
                stage_reasons.setdefault("ES", []).extend(v)
                emit("stage_failed", order_id=order_id, stage="ES", reasons=v)
                replans += 1
                continue
            # ---- 通过 CV：确认计划
            plan = {
                "order_id": order_id,
                "area_id": order.get("area_id"),
                "area_name": order.get("area_name"),
                "priority": order.get("priority"),
                "profit": order.get("profit"),
                "satellite_id": sat["satellite_id"],
                "satellite_name": sat["satellite_name"],
                "window_id": win["window_id"],
                "observation_start": slot["observation_start"],
                "observation_end": slot["observation_end"],
                "duration_min": slot["duration_min"],
                "transition_time_min": slot["transition_time_min"],
                "resolution": sat["resolution"],
                "cloud_rate": win["cloud_rate"],
                "incidence_angle": win["incidence_angle"],
                "stage_scores": {
                    "AS": sat["stage1_score"],
                    "VS": win["stage2_score"],
                    "ES": slot["stage3_score"],
                },
                "utility": round(
                    (
                        sat["stage1_score"] * 0.34
                        + win["stage2_score"] * 0.33
                        + slot["stage3_score"] * 0.33
                    )
                    * float(order.get("profit", 2.0))
                    / 4.0,
                    4,
                ),
                "attempts": attempt,
                "stage_count": 3,
                "source": "simulated",
            }
            scheduled.append(plan)
            occupancy.setdefault(sat["satellite_id"], []).append(
                (_parse(slot["observation_start"]), _parse(slot["observation_end"]))
            )
            pm.record_plan(plan)
            scheduled_ok = True
            emit(
                "order_scheduled",
                order_id=order_id,
                plan=plan,
                utility=plan["utility"],
                attempts=attempt,
            )
            break

        if not scheduled_ok:
            rejected_orders.append(
                {
                    "order_id": order_id,
                    "area_id": order.get("area_id"),
                    "priority": order.get("priority"),
                    "failed_stage": last_stage or "AS",
                    "reasons": (stage_reasons.get(last_stage or "AS") or last_reasons
                                or ["无可行调度方案"])[:4],
                    "attempts": attempt,
                }
            )
            emit("order_rejected", order_id=order_id, reasons=last_reasons, attempts=attempt)

    total = len(orders)
    done = len(scheduled)
    metrics = {
        "total_orders": total,
        "scheduled_orders": done,
        "rejected_orders": len(rejected_orders),
        "completion_rate": round(done / total, 4) if total else 0.0,
        "constraint_violations": stage_failures,
        "replans": replans,
        "atomic_decisions": atomic_decisions,
        "avg_attempts": round(
            sum(p["attempts"] for p in scheduled) / done, 3
        ) if done else 0.0,
        "total_utility": round(sum(p["utility"] for p in scheduled), 4),
        "computation_ms": _elapsed(start),
        "rule_version": cv.VERSION,
        "method": "TSLC(order-driven) 三阶段序列决策",
    }

    emit(
        "order_batch_end",
        metrics=metrics,
        scheduled=done,
        rejected=len(rejected_orders),
    )

    return ToolResult.ok(
        data={
            "schedule": scheduled,
            "rejected": rejected_orders,
            "metrics": metrics,
            "memory": pm.memory,
            "knowledge": pm.knowledge,
            "agent_prompts": AGENT_PROMPTS,
        },
        elapsed_ms=_elapsed(start),
    )


def summarize_schedule(result: dict[str, Any]) -> str:
    """生成可读的调度摘要（模板化，不依赖 LLM）。"""
    metrics = result.get("metrics", {})
    lines = [
        "# 订单驱动调度报告（TSLC 三阶段）",
        "",
        f"**订单总数**：{metrics.get('total_orders')}",
        f"**成功调度**：{metrics.get('scheduled_orders')}",
        f"**未调度**：{metrics.get('rejected_orders')}",
        f"**完成率**：{metrics.get('completion_rate')}",
        f"**约束违规次数**：{metrics.get('constraint_violations')}",
        f"**重规划次数**：{metrics.get('replans')}",
        f"**原子决策数**：{metrics.get('atomic_decisions')}",
        f"**计算耗时**：{metrics.get('computation_ms')} ms",
        "",
        "## 调度计划",
    ]
    for p in result.get("schedule", []):
        lines.append(
            f"- 订单 {p['order_id']}（{p.get('area_name')}，优先级 {p.get('priority')}）"
            f" → {p['satellite_name']} ｜ {p['window_id']} ｜ "
            f"{p['observation_start']} ~ {p['observation_end']} ｜ 效用 {p['utility']}"
        )
    if result.get("rejected"):
        lines += ["", "## 未调度订单"]
        for r in result["rejected"]:
            lines.append(f"- 订单 {r['order_id']}：{'；'.join(r.get('reasons', []))}")
    lines += ["", "> 说明：本报告基于模拟卫星与模拟观测窗口数据生成，非真实调度结果。"]
    return "\n".join(lines)
