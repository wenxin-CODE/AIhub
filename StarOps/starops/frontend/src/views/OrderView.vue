<template>
  <div class="order-view">
    <div class="head">
      <div class="head-left">
        <div class="title">订单驱动调度 · TSLC 三阶段</div>
        <div class="subtitle">
          参照 IEEE TGRS 2026「Tri-Stage LLM-Coordinated Framework for Order-Driven
          Scheduling of Earth Observation Satellite Tasks」
          <el-tag size="small" type="info" effect="plain" class="doi">DOI 10.1109/TGRS.2026.3700605</el-tag>
        </div>
      </div>
      <div class="head-right">
        <el-checkbox v-model="resetQueue" size="small">
          重置队列后调度
          <el-tooltip content="订单被排期/拒绝后状态即变为 scheduled/rejected，勾选后先把全部订单重置为 pending，保证演示可重复执行" placement="bottom">
            <el-icon class="help"><QuestionFilled /></el-icon>
          </el-tooltip>
        </el-checkbox>
        <el-button type="primary" :loading="running" @click="run">
          {{ running ? '调度中…' : '运行订单调度' }}
        </el-button>
        <el-tag type="warning" effect="plain" size="small">模拟数据 · 非真实卫星调度</el-tag>
      </div>
    </div>

    <div class="metrics">
      <div class="metric">
        <div class="metric-label">订单总数</div>
        <div class="metric-value">{{ metrics.total_orders ?? orders.length }}</div>
      </div>
      <div class="metric ok">
        <div class="metric-label">成功排期</div>
        <div class="metric-value">{{ metrics.scheduled_orders ?? 0 }}</div>
      </div>
      <div class="metric bad">
        <div class="metric-label">未排期</div>
        <div class="metric-value">{{ metrics.rejected_orders ?? 0 }}</div>
      </div>
      <div class="metric">
        <div class="metric-label">完成率</div>
        <div class="metric-value">{{ pct(metrics.completion_rate) }}</div>
      </div>
      <div class="metric">
        <div class="metric-label">约束违规次数</div>
        <div class="metric-value">{{ metrics.constraint_violations ?? 0 }}</div>
      </div>
      <div class="metric">
        <div class="metric-label">重规划次数</div>
        <div class="metric-value">{{ metrics.replans ?? 0 }}</div>
      </div>
      <div class="metric">
        <div class="metric-label">原子决策数</div>
        <div class="metric-value">{{ metrics.atomic_decisions ?? 0 }}</div>
      </div>
      <div class="metric">
        <div class="metric-label">计算耗时</div>
        <div class="metric-value">{{ metrics.computation_ms ?? 0 }}<span class="unit">ms</span></div>
      </div>
    </div>

    <div class="grid">
      <div class="panel">
        <div class="panel-title">
          订单队列
          <el-tag size="small" effect="plain" type="info">
            {{ orders.length }} 单<span v-if="selected.length"> · 已选 {{ selected.length }}</span>
          </el-tag>
        </div>
        <el-table
          :data="orders"
          size="small"
          max-height="240"
          :show-index="false"
          @selection-change="onSelect"
        >
          <el-table-column type="selection" width="38" />
          <el-table-column prop="order_id" label="订单" width="80" />
          <el-table-column prop="area_name" label="区域" width="70" />
          <el-table-column prop="satellite_type" label="类型" width="70" />
          <el-table-column prop="priority" label="优先级" width="80">
            <template #default="{ row }">
              <el-tag size="small" :type="priorityType(row.priority)">{{ row.priority }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="status" label="状态" width="90">
            <template #default="{ row }">
              <el-tag size="small" :type="statusType(row.status)">{{ row.status }}</el-tag>
            </template>
          </el-table-column>
        </el-table>
        <div class="panel-foot dim">
          勾选订单可只调度所选；不勾选则调度全部 pending 订单
        </div>
      </div>

      <div class="panel">
        <div class="panel-title">
          调度计划
          <el-tag size="small" effect="plain" type="success">{{ schedule.length }} 单已排</el-tag>
        </div>
        <el-table :data="schedule" size="small" max-height="240" :show-index="false">
          <el-table-column prop="order_id" label="订单" width="80" />
          <el-table-column prop="satellite_name" label="卫星" min-width="110" />
          <el-table-column prop="observation_start" label="观测时刻" width="160">
            <template #default="{ row }">
              <span class="mono">{{ shortTime(row.observation_start) }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="utility" label="效用" width="70">
            <template #default="{ row }">{{ row.utility }}</template>
          </el-table-column>
          <el-table-column prop="attempts" label="尝试" width="60" />
        </el-table>
      </div>

      <div class="panel">
        <div class="panel-title">
          未排期订单（可解释拒绝）
          <el-tag size="small" effect="plain" type="danger">{{ rejected.length }} 单</el-tag>
        </div>
        <div class="reject-list">
          <div v-for="r in rejected" :key="r.order_id" class="reject-item">
            <div class="reject-head">
              <span class="mono">{{ r.order_id }}</span>
              <el-tag size="small" type="danger" effect="plain">止步 {{ r.failed_stage }}</el-tag>
              <span class="dim">{{ r.attempts }} 次尝试</span>
            </div>
            <ul>
              <li v-for="(rs, i) in r.reasons" :key="i">{{ rs }}</li>
            </ul>
          </div>
          <el-empty v-if="!rejected.length" description="全部订单已排期" :image-size="60" />
        </div>
      </div>

      <div class="panel wide">
        <div class="panel-title">
          三阶段决策流
          <el-tag size="small" effect="plain" type="info">
            AS 指派卫星 → VS 选择时窗 → ES 确定观测时刻
          </el-tag>
        </div>
        <div class="event-log">
          <div v-for="(e, i) in events" :key="i" class="event" :class="e.kind">
            <span class="ev-stage" :style="{ color: stageColor(e.stage) }">{{ stageLabel(e) }}</span>
            <span class="ev-text">{{ e.text }}</span>
          </div>
          <el-empty v-if="!events.length" description="点击「运行订单调度」开始" :image-size="60" />
        </div>
      </div>

      <div class="panel">
        <div class="panel-title">
          记忆 MS / 知识库 K
          <el-tag size="small" effect="plain" type="warning">{{ memory.length }} 失败 · {{ knowledge.length }} 确认</el-tag>
        </div>
        <el-tabs v-model="tab" class="mini-tabs">
          <el-tab-pane label="MS 失败记忆" name="ms">
            <div class="mem-list">
              <div v-for="m in memory" :key="m.memory_id" class="mem-item">
                <span class="mono dim">{{ m.order_id }}</span>
                <el-tag size="small" effect="plain" :type="m.stage === 'AS' ? 'info' : m.stage === 'VS' ? 'warning' : 'danger'">
                  {{ m.stage }}
                </el-tag>
                <span class="mem-text">{{ m.reason }}</span>
              </div>
              <el-empty v-if="!memory.length" description="暂无失败记录" :image-size="50" />
            </div>
          </el-tab-pane>
          <el-tab-pane label="K 知识库" name="k">
            <div class="mem-list">
              <div v-for="k in knowledge" :key="k.knowledge_id" class="mem-item">
                <span class="mono dim">{{ k.order_id }}</span>
                <span class="mem-text">{{ k.plan?.satellite_name }} → {{ shortTime(k.plan?.observation_start) }}</span>
              </div>
              <el-empty v-if="!knowledge.length" description="暂无已确认计划" :image-size="50" />
            </div>
            <div class="panel-foot dim">
              K 按计划指纹（卫星 | 窗口 | 起止时刻）去重：同一条观测计划只保留一条
            </div>
          </el-tab-pane>
        </el-tabs>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getOrders, getScheduleKnowledge, getScheduleMemory, runOrdersStream } from '@/api/client'

const orders = ref<any[]>([])
const schedule = ref<any[]>([])
const rejected = ref<any[]>([])
const metrics = ref<any>({})
const events = ref<any[]>([])
const memory = ref<any[]>([])
const knowledge = ref<any[]>([])
const running = ref(false)
const tab = ref('ms')
const selected = ref<any[]>([])
const resetQueue = ref(true)

const onSelect = (rows: any[]) => {
  selected.value = rows
}

const pct = (v?: number) => (v === undefined || v === null ? '—' : `${(v * 100).toFixed(1)}%`)
const shortTime = (s?: string) => (s ? s.replace('T', ' ').slice(0, 19) : '—')
const priorityType = (p: string) =>
  p === 'urgent' ? 'danger' : p === 'high' ? 'warning' : p === 'low' ? 'info' : 'success'
const statusType = (s: string) =>
  s === 'scheduled' ? 'success' : s === 'rejected' ? 'danger' : 'info'
const stageColor = (s?: string) =>
  s === 'AS' ? '#00f0ff' : s === 'VS' ? '#ff2e93' : s === 'ES' ? '#ffd166' : '#8b98a9'
const stageLabel = (e: any) => {
  if (e.event === 'stage_decision') return e.stage
  if (e.event === 'stage_failed') return `${e.stage}✕`
  if (e.event === 'order_scheduled') return '✓'
  if (e.event === 'order_rejected') return '⊘'
  return '·'
}

function pushEvent(kind: string, stage: string | undefined, text: string) {
  events.value.push({ kind, stage, text })
}

async function run() {
  running.value = true
  events.value = []
  const orderIds = selected.value.map((o: any) => o.order_id)
  await runOrdersStream(
    (e) => {
      switch (e.event) {
        case 'order_batch_start':
          pushEvent('info', undefined, `开始处理 ${e.total_orders} 个订单（卫星 ${e.satellites} / 窗口 ${e.windows}）`)
          break
        case 'order_start':
          pushEvent('dim', undefined, `订单 ${e.order_id} 第 ${e.attempt} 轮`)
          break
        case 'stage_decision': {
          const d = e.decision || {}
          const detail = d.satellite_name || d.area_name || d.observation_start || ''
          pushEvent('ok', e.stage, `${e.stage} → ${detail}（评分 ${e.score}，候选 ${e.candidates}）`)
          break
        }
        case 'stage_failed':
          pushEvent('bad', e.stage, `${e.stage} 失败：${(e.reasons || []).join('；')}`)
          break
        case 'order_scheduled':
          pushEvent('ok', undefined, `订单 ${e.order_id} 已排期，效用 ${e.utility}`)
          break
        case 'order_rejected':
          pushEvent('bad', undefined, `订单 ${e.order_id} 未排期（${e.attempts} 次尝试）`)
          break
        case 'order_batch_end':
          pushEvent('info', undefined, `完成：排期 ${e.scheduled} / 拒绝 ${e.rejected}`)
          break
      }
    },
    async (state) => {
      schedule.value = state.schedule || []
      rejected.value = state.rejected_orders || []
      metrics.value = state.schedule_metrics || {}
      await refresh()
      running.value = false
    },
    (err) => {
      pushEvent('bad', undefined, `执行失败：${err.message}`)
      running.value = false
    },
    3,
    orderIds,
    resetQueue.value
  )
}

async function refresh() {
  orders.value = await getOrders()
  memory.value = await getScheduleMemory()
  knowledge.value = await getScheduleKnowledge()
}

onMounted(refresh)
</script>

<style scoped>
.order-view {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  background: #131a24;
  border: 0.5px solid #22303f;
  border-radius: 8px;
  padding: 14px 16px;
}
.title {
  font-size: 15px;
  font-weight: 600;
  color: #c8d4e0;
}
.subtitle {
  margin-top: 6px;
  font-size: 12px;
  color: #6b7a8d;
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.doi {
  margin-left: 4px;
}
.head-right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  gap: 10px;
}
.metric {
  background: #131a24;
  border: 0.5px solid #22303f;
  border-radius: 8px;
  padding: 10px 12px;
}
.metric-label {
  font-size: 11px;
  color: #6b7a8d;
}
.metric-value {
  margin-top: 4px;
  font-size: 18px;
  font-weight: 600;
  color: #c8d4e0;
  font-variant-numeric: tabular-nums;
}
.metric.ok .metric-value {
  color: #00f0ff;
}
.metric.bad .metric-value {
  color: #ff2e93;
}
.unit {
  font-size: 11px;
  color: #6b7a8d;
  margin-left: 2px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}
.panel {
  background: #131a24;
  border: 0.5px solid #22303f;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.panel.wide {
  grid-column: span 2;
}
.panel-title {
  padding: 10px 14px;
  font-size: 13px;
  font-weight: 500;
  color: #c8d4e0;
  border-bottom: 0.5px solid #22303f;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.reject-list,
.event-log,
.mem-list {
  padding: 10px 12px;
  overflow-y: auto;
  max-height: 300px;
}
.reject-item {
  border-left: 2px solid #ff2e93;
  padding: 6px 0 8px 10px;
  margin-bottom: 8px;
}
.reject-head {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #c8d4e0;
}
.reject-item ul {
  margin: 6px 0 0;
  padding-left: 16px;
  font-size: 11.5px;
  color: #8b98a9;
  line-height: 1.7;
}
.event {
  display: flex;
  gap: 10px;
  font-size: 12px;
  padding: 3px 0;
  border-bottom: 0.5px dashed #1b2531;
}
.ev-stage {
  width: 34px;
  flex: none;
  font-weight: 600;
  font-size: 11px;
}
.ev-text {
  color: #8b98a9;
}
.event.ok .ev-text {
  color: #b6c4d4;
}
.event.bad .ev-text {
  color: #ff8ab8;
}
.event.info .ev-text {
  color: #00f0ff;
}
.event.dim .ev-text {
  color: #5c6b7d;
}
.mem-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11.5px;
  padding: 4px 0;
  border-bottom: 0.5px dashed #1b2531;
}
.mem-text {
  color: #8b98a9;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.mono {
  font-family: 'JetBrains Mono', Consolas, monospace;
}
.dim {
  color: #5c6b7d;
}
.mini-tabs :deep(.el-tabs__content) {
  padding: 8px 12px;
}
.panel-foot {
  padding: 7px 12px;
  font-size: 11px;
  border-top: 0.5px solid #1b2531;
}
.help {
  color: #6b7a8d;
  vertical-align: -2px;
  margin-left: 2px;
}
</style>
