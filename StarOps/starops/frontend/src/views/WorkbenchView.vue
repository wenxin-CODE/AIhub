<template>
  <div class="workbench">
    <!-- 左：会话 -->
    <div class="panel panel-chat">
      <div class="panel-title">自然语言任务</div>

      <div ref="chatBox" class="chat-box">
        <div v-for="(m, i) in messages" :key="i" class="chat-msg" :class="m.role">
          <div class="chat-avatar">{{ m.role === 'user' ? '我' : 'AI' }}</div>
          <div class="chat-body">
            <div v-if="m.kind === 'text'" class="chat-text">{{ m.content }}</div>
            <div v-else-if="m.kind === 'report'" class="chat-report">
              <pre>{{ m.content }}</pre>
            </div>
            <div v-else-if="m.kind === 'requirement'" class="chat-req">
              <el-descriptions :column="1" size="small" border>
                <el-descriptions-item label="观测区域">
                  {{ m.content.area_name || '（未识别）' }}
                </el-descriptions-item>
                <el-descriptions-item label="时间范围">
                  {{ m.content.time_days ? m.content.time_days + ' 天内' : '（未识别）' }}
                </el-descriptions-item>
                <el-descriptions-item label="分辨率上限">
                  {{ m.content.max_resolution ? m.content.max_resolution + ' m' : '（未识别）' }}
                </el-descriptions-item>
                <el-descriptions-item label="卫星类型">{{ m.content.satellite_type }}</el-descriptions-item>
              </el-descriptions>
            </div>
          </div>
        </div>
      </div>

      <div class="chat-input">
        <el-input
          v-model="input"
          type="textarea"
          :rows="3"
          placeholder="例：帮我规划一次北京区域的卫星观测任务，要求空间分辨率优于2米，尽量在未来3天内完成，优先选择光学卫星。"
          @keyup.ctrl.enter="send"
        />
        <div class="chat-actions">
          <el-button type="primary" :loading="running" @click="send">执行任务 (Ctrl+Enter)</el-button>
          <el-button :disabled="running" @click="reset">清空</el-button>
        </div>
      </div>
    </div>

    <!-- 中：执行过程 -->
    <div class="panel panel-trace">
      <div class="panel-title">
        Agent 执行过程
        <el-tag size="small" :type="currentNode ? 'success' : 'info'">{{ currentNode || '等待' }}</el-tag>
      </div>

      <div class="trace-list">
        <div v-for="(s, i) in steps" :key="i" class="trace-step">
          <div class="trace-dot" :class="s.status"></div>
          <div class="trace-content">
            <div class="trace-head">
              <span class="trace-label">{{ s.label }}</span>
              <span class="trace-time">{{ s.elapsed_ms }}ms</span>
            </div>
            <div v-if="s.tool_name" class="trace-tool">
              <el-tag size="small" type="info" effect="plain">{{ s.tool_name }}</el-tag>
              <span class="trace-reason" :class="s.status">{{ s.reason || '' }}</span>
            </div>
            <details v-if="s.output" class="trace-detail">
              <summary>查看工具返回</summary>
              <pre>{{ pretty(s.output) }}</pre>
            </details>
          </div>
        </div>

        <el-empty v-if="!steps.length" description="等待任务执行" :image-size="70" />
      </div>

      <!-- 审批面板：未审批时执行按钮置灰（后端仍会独立拒绝） -->
      <div v-if="pendingApproval" class="approval-box">
        <div class="approval-title">
          <el-icon><Stamp /></el-icon>
          待人工审批
        </div>
        <div class="approval-summary">{{ pendingApproval.summary }}</div>
        <el-input v-model="approvalComment" placeholder="审批意见（可选）" size="small" />
        <div class="approval-actions">
          <el-button size="small" type="success" :loading="reviewing" @click="review('approved')">
            批准
          </el-button>
          <el-button size="small" type="danger" plain :loading="reviewing" @click="review('rejected')">
            拒绝
          </el-button>
        </div>
      </div>
    </div>

    <!-- 右：结构化结果 -->
    <div class="panel panel-result">
      <div class="panel-title">结构化结果</div>

      <div v-if="constraintResult && Object.keys(constraintResult).length" class="result-block">
        <div class="result-label">硬约束校验</div>
        <el-tag :type="constraintResult.passed ? 'success' : 'danger'" effect="dark">
          {{ constraintResult.passed ? '通过' : '未通过' }}
        </el-tag>
        <ul v-if="constraintResult.violations?.length" class="violations">
          <li v-for="(v, i) in constraintResult.violations" :key="i">{{ v }}</li>
        </ul>
      </div>

      <div v-if="plans.length" class="result-block">
        <div class="result-label">候选方案评分</div>
        <el-table :data="plans" size="small" :show-index="false" max-height="260">
          <el-table-column prop="satellite_name" label="卫星" width="130" />
          <el-table-column prop="resolution" label="分辨率" width="80">
            <template #default="{ row }">{{ row.resolution }}m</template>
          </el-table-column>
          <el-table-column prop="cloud_rate" label="云量" width="70" />
          <el-table-column prop="incidence_angle" label="入射角" width="80" />
          <el-table-column prop="score" label="评分" width="80">
            <template #default="{ row }">
              <el-tag size="small" :type="row.score > 0.5 ? 'success' : 'info'">{{ row.score }}</el-tag>
            </template>
          </el-table-column>
        </el-table>
      </div>

      <div v-if="windows.length" class="result-block">
        <div class="result-label">观测窗口</div>
        <el-table :data="windows" size="small" :show-index="false" max-height="220">
          <el-table-column prop="area_name" label="区域" width="80" />
          <el-table-column prop="start_time" label="开始" width="180" />
          <el-table-column prop="cloud_rate" label="云量" width="70" />
          <el-table-column prop="quality" label="数据质量" width="90">
            <template #default="{ row }">
              <el-tag size="small" :type="qualityType(row.quality)">{{ row.quality }}</el-tag>
            </template>
          </el-table-column>
        </el-table>
      </div>

      <el-empty v-if="!plans.length && !windows.length" description="等待结果" :image-size="70" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { runAgentStream, reviewApproval } from '@/api/client'

interface Step {
  status: string
  label: string
  tool_name?: string
  elapsed_ms: number
  reason?: string
  output?: any
}

const input = ref('')
const running = ref(false)
const reviewing = ref(false)
const messages = ref<any[]>([])
const steps = ref<Step[]>([])
const currentNode = ref('')
const plans = ref<any[]>([])
const windows = ref<any[]>([])
const constraintResult = ref<any>({})
const pendingApproval = ref<any>(null)
const approvalComment = ref('')
const chatBox = ref<HTMLElement>()

const pretty = (o: any) => JSON.stringify(o, null, 2)

const qualityType = (q: string) =>
  q === 'FRESH' ? 'success' : q === 'STALE' ? 'warning' : q === 'EXPIRED' ? 'danger' : 'info'

async function scrollBottom() {
  await nextTick()
  if (chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight
}

function reset() {
  messages.value = []
  steps.value = []
  plans.value = []
  windows.value = []
  constraintResult.value = {}
  pendingApproval.value = null
  currentNode.value = ''
}

async function send() {
  const msg = input.value.trim()
  if (!msg || running.value) return

  messages.value.push({ role: 'user', kind: 'text', content: msg })
  input.value = ''
  steps.value = []
  plans.value = []
  windows.value = []
  constraintResult.value = {}
  pendingApproval.value = null
  running.value = true
  await scrollBottom()

  await runAgentStream(
    msg,
    (evt) => {
      // 节点开始
      if (evt.event === 'node_start') {
        currentNode.value = evt.label || evt.node
        steps.value.push({
          status: 'running',
          label: evt.label || evt.node,
          elapsed_ms: 0
        })
      }
      // 工具开始
      if (evt.event === 'tool_start') {
        const last = steps.value[steps.value.length - 1]
        if (last && last.status === 'running') {
          last.tool_name = evt.tool_name
        } else {
          steps.value.push({
            status: 'running',
            label: evt.tool_name,
            tool_name: evt.tool_name,
            elapsed_ms: 0
          })
        }
      }
      // 工具结束
      if (evt.event === 'tool_end') {
        const last = steps.value[steps.value.length - 1]
        if (last) {
          last.status = evt.status
          last.elapsed_ms = evt.elapsed_ms
          last.reason = evt.reason
          last.output = evt.output
        }
      }
      // 节点结束
      if (evt.event === 'node_end') {
        const last = steps.value[steps.value.length - 1]
        if (last && last.label === (evt.label || evt.node)) {
          last.status = 'success'
        }
      }
      // 需要补充参数
      if (evt.event === 'need_input') {
        ElMessage.warning(evt.message)
      }
    },
    (state) => {
      running.value = false
      if (!state) return
      plans.value = state.scored_plans || []
      windows.value = state.windows || []
      constraintResult.value = state.constraint_result || {}
      if (state.task_requirements && Object.keys(state.task_requirements).length) {
        messages.value.push({
          role: 'assistant',
          kind: 'requirement',
          content: state.task_requirements
        })
      }
      if (state.approval_status === 'pending' && state.approval_id) {
        pendingApproval.value = {
          approval_id: state.approval_id,
          summary: state.selected_plan
            ? `${state.selected_plan.satellite_name}｜评分 ${state.selected_plan.score}｜分辨率 ${state.selected_plan.resolution}m`
            : '待审批方案'
        }
        messages.value.push({
          role: 'assistant',
          kind: 'text',
          content:
            '方案已生成并通过硬约束校验，等待人工审批。**未获批准前不会执行任何落库操作。**'
        })
      }
      scrollBottom()
    },
    (err) => {
      running.value = false
      ElMessage.error(`执行失败：${err.message}`)
    }
  )
}

async function review(decision: string) {
  if (!pendingApproval.value) return
  reviewing.value = true
  try {
    await reviewApproval(pendingApproval.value.approval_id, decision, approvalComment.value)
    ElMessage.success(decision === 'approved' ? '已批准' : '已拒绝')
    messages.value.push({
      role: 'assistant',
      kind: 'text',
      content:
        decision === 'approved'
          ? '审批通过，方案可进入执行环节。'
          : '审批拒绝，保留原状态并记录拒绝原因。'
    })
    pendingApproval.value = null
    approvalComment.value = ''
    scrollBottom()
  } catch (e: any) {
    ElMessage.error(e.response?.data?.detail || '审批失败')
  } finally {
    reviewing.value = false
  }
}
</script>

<style scoped>
.workbench {
  display: grid;
  grid-template-columns: 1fr 1.1fr 1fr;
  gap: 12px;
  height: calc(100vh - 100px);
}
.panel {
  background: #131a24;
  border: 0.5px solid #22303f;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
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
.chat-box {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}
.chat-msg {
  display: flex;
  gap: 10px;
  margin-bottom: 14px;
}
.chat-avatar {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  background: #1f2b3a;
  color: #7d8ea3;
}
.chat-msg.user .chat-avatar {
  background: #00f0ff;
  color: #06212a;
}
.chat-body {
  flex: 1;
  min-width: 0;
}
.chat-text {
  font-size: 13px;
  line-height: 1.7;
  color: #d5dee8;
  white-space: pre-wrap;
  word-break: break-word;
}
.chat-report pre {
  margin: 0;
  font-size: 12px;
  line-height: 1.6;
  color: #a8b8c8;
  white-space: pre-wrap;
  background: #0d131b;
  padding: 10px;
  border-radius: 6px;
  max-height: 320px;
  overflow-y: auto;
}
.chat-input {
  padding: 12px;
  border-top: 0.5px solid #22303f;
}
.chat-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
}
.trace-list {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}
.trace-step {
  display: flex;
  gap: 10px;
  margin-bottom: 12px;
  position: relative;
}
.trace-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-top: 5px;
  flex-shrink: 0;
  background: #3d4c5e;
}
.trace-dot.running {
  background: #ffb020;
  animation: pulse 1.2s infinite;
}
.trace-dot.success {
  background: #00d087;
}
.trace-dot.failed {
  background: #ff4d4f;
}
.trace-dot.degraded {
  background: #ff9f40;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}
.trace-content {
  flex: 1;
  min-width: 0;
}
.trace-head {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
}
.trace-label {
  color: #d5dee8;
}
.trace-time {
  color: #5c6b7d;
  font-size: 12px;
}
.trace-tool {
  margin-top: 4px;
  display: flex;
  gap: 8px;
  align-items: center;
}
.trace-reason {
  font-size: 12px;
  color: #7d8ea3;
}
.trace-reason.failed { color: #ff7875; }
.trace-detail {
  margin-top: 4px;
  font-size: 12px;
}
.trace-detail summary {
  cursor: pointer;
  color: #5c6b7d;
}
.trace-detail pre {
  margin: 6px 0 0;
  padding: 8px;
  background: #0d131b;
  border-radius: 4px;
  max-height: 200px;
  overflow: auto;
  color: #8b98a9;
}
.approval-box {
  border-top: 0.5px solid #22303f;
  padding: 12px;
  background: #171f2b;
}
.approval-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: #ffb020;
  margin-bottom: 8px;
}
.approval-summary {
  font-size: 12px;
  color: #8b98a9;
  margin-bottom: 8px;
}
.approval-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
}
.panel-result {
  overflow-y: auto;
}
.result-block {
  padding: 12px;
  border-bottom: 0.5px solid #22303f;
}
.result-label {
  font-size: 12px;
  color: #7d8ea3;
  margin-bottom: 8px;
}
.violations {
  margin: 8px 0 0;
  padding-left: 18px;
  font-size: 12px;
  color: #ff7875;
  line-height: 1.8;
}
</style>
