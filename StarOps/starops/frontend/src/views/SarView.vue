<template>
  <div class="sar-page">
    <div class="upload-row">
      <el-upload
        :auto-upload="false"
        :show-file-list="false"
        accept="image/*,.tif,.tiff"
        :on-change="onFileChange"
      >
        <el-button type="primary" :loading="uploading">
          <el-icon><Upload /></el-icon>上传 SAR 影像
        </el-button>
      </el-upload>

      <el-select
        v-model="selectedImagePath"
        placeholder="或选择已登记影像"
        size="small"
        style="width: 260px"
        :disabled="uploading || analyzing"
        @change="onSelectImage"
      >
        <el-option
          v-for="img in imageList"
          :key="img.image_id"
          :label="`${img.filename}（${img.width}×${img.height}）`"
          :value="img.upload_path"
        />
      </el-select>

      <span v-if="fileName" class="file-name">{{ fileName }}</span>
      <el-tag size="small" type="info" effect="plain">
        L1 经典基线实测横评；L2 深度模型待接入权重
      </el-tag>
    </div>

    <!-- 执行过程 -->
    <div v-if="trace.length" class="panel trace-panel">
      <div class="panel-title">执行过程</div>
      <div class="trace-body">
        <div v-for="(t, i) in trace" :key="i" class="trace-item" :class="t.kind">
          <span class="trace-badge">{{ t.kind === 'node' ? '节点' : t.kind === 'tool' ? '工具' : '结果' }}</span>
          <span class="trace-text">{{ t.text }}</span>
          <span v-if="t.ms" class="trace-ms">{{ t.ms }}ms</span>
        </div>
      </div>
    </div>

    <!-- 诊断 -->
    <div v-if="diagnosis" class="panel">
      <div class="panel-title">诊断结论</div>
      <div class="diag-body">
        <el-tag
          v-for="f in diagnosis.findings || []"
          :key="f"
          size="small"
          :type="diagnosis.speckle_level === 'severe' ? 'danger' : 'warning'"
          effect="plain"
          class="diag-tag"
        >
          {{ f }}
        </el-tag>
        <el-tag
          v-if="diagnosis.has_striping"
          size="small"
          type="danger"
          effect="plain"
          class="diag-tag"
        >
          检出条纹
        </el-tag>
      </div>
    </div>

    <div class="metric-row">
      <div v-for="m in metricCards" :key="m.label" class="metric-card">
        <div class="metric-label">{{ m.label }}</div>
        <div class="metric-value">{{ m.value }}</div>
      </div>
    </div>

    <div class="sar-grid">
      <div class="panel">
        <div class="panel-title">原始影像</div>
        <div class="image-box">
          <img v-if="imageUrl" :src="imageUrl" alt="原始影像" />
          <el-empty v-else description="等待上传" :image-size="60" />
        </div>
      </div>

      <div class="panel">
        <div class="panel-title">
          处理结果
          <el-tag v-if="selectedPipeline" size="small" effect="plain">
            {{ selectedPipeline.label }}
          </el-tag>
        </div>
        <div class="image-box">
          <img v-if="processedUrl" :src="processedUrl" alt="处理后影像" />
          <el-empty v-else description="等待处理" :image-size="60" />
        </div>
      </div>

      <div class="panel">
        <div class="panel-title">指标对比</div>
        <div class="compare-body">
          <el-table v-if="comparison" :data="comparisonRows" size="small" :show-index="false">
            <el-table-column prop="metric" label="指标" width="110" />
            <el-table-column prop="before" label="处理前" />
            <el-table-column prop="after" label="处理后" />
            <el-table-column prop="delta" label="变化" />
          </el-table>
          <el-empty v-else description="等待处理" :image-size="60" />
        </div>
      </div>
    </div>

    <!-- 管线横评与被拒原因 -->
    <div v-if="candidatePipelines.length" class="panel">
      <div class="panel-title">候选管线横评（统一指标口径）</div>
      <div class="compare-body">
        <el-table :data="pipelineRows" size="small" :show-index="false">
          <el-table-column prop="label" label="管线" width="120" />
          <el-table-column prop="level" label="层级" width="100" />
          <el-table-column prop="result" label="结果" />
          <el-table-column prop="detail" label="说明" />
        </el-table>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { getImages, runSarStream, uploadSarImage } from '../api/client'

const fileName = ref('')
const uploading = ref(false)
const analyzing = ref(false)
const imageUrl = ref('')
const processedUrl = ref('')
const selectedImagePath = ref('')
const imageList = ref<any[]>([])

const metrics = ref<any>(null)
const comparison = ref<any>(null)
const diagnosis = ref<any>(null)
const candidatePipelines = ref<any[]>([])
const rejectedPipelines = ref<any[]>([])
const selectedPipeline = ref<any>(null)
const trace = ref<any[]>([])

const metricCards = computed(() => {
  const m = metrics.value
  if (!m) return []
  return [
    { label: '灰度均值', value: fmt(m.gray_mean) },
    { label: '变异系数 CV', value: fmt(m.cv) },
    { label: '局部方差', value: fmt(m.local_variance) },
    { label: '等效视数 ENL', value: fmt(m.enl) },
    { label: '条纹突出度', value: fmt(m.striping_strength) }
  ]
})

const comparisonRows = computed(() => {
  const c = comparison.value
  if (!c) return []
  const keys = ['gray_mean', 'gray_std', 'cv', 'local_variance', 'enl']
  const labels: Record<string, string> = {
    gray_mean: '灰度均值',
    gray_std: '灰度标准差',
    cv: '变异系数',
    local_variance: '局部方差',
    enl: '等效视数'
  }
  return keys.map((k) => ({
    metric: labels[k],
    before: c.before[k],
    after: c.after[k],
    delta: c.delta[k]
  }))
})

const pipelineRows = computed(() => {
  const rows: any[] = []
  const chosen = selectedPipeline.value?.pipeline_id
  for (const p of candidatePipelines.value) {
    const isChosen = p.pipeline_id === chosen
    const rej = rejectedPipelines.value.find((r) => r.pipeline_id === p.pipeline_id)
    rows.push({
      label: p.label,
      level: p.filter_level === 'L1_classical' ? 'L1 经典' : p.filter_level,
      result: isChosen ? '已选中' : '已淘汰',
      detail: isChosen
        ? `CV ${p.before_cv?.toFixed(4)} → ${p.after_cv?.toFixed(4)}（降 ${p.cv_reduced_pct?.toFixed(1)}%），均值偏移 ${p.mean_shift_pct}%`
        : rej?.reject_reason || '—'
    })
  }
  return rows
})

const fmt = (v: any) => (v === null || v === undefined ? '—' : Number(v).toFixed(4))

function pushTrace(kind: string, text: string, ms?: number) {
  trace.value.push({ kind, text, ms })
}

async function loadImageList() {
  try {
    imageList.value = await getImages()
  } catch (e) {
    console.warn('影像列表加载失败', e)
  }
}

async function onFileChange(file: any) {
  fileName.value = file.name
  uploading.value = true
  try {
    const res = await uploadSarImage(file.raw)
    imageUrl.value = res.url
    selectedImagePath.value = res.path
    await loadImageList()
    await analyze(res.path)
  } catch (err: any) {
    ElMessage.error(`上传失败：${err?.message || err}`)
  } finally {
    uploading.value = false
  }
}

async function onSelectImage(path: string) {
  const img = imageList.value.find((i) => i.upload_path === path)
  if (img) {
    imageUrl.value = `/files/images/${img.upload_path.split(/[\\/]/).pop()}`
  }
  await analyze(path)
}

async function analyze(imagePath: string) {
  analyzing.value = true
  trace.value = []
  metrics.value = null
  comparison.value = null
  diagnosis.value = null
  candidatePipelines.value = []
  rejectedPipelines.value = []
  selectedPipeline.value = null
  processedUrl.value = ''

  await runSarStream(
    imagePath,
    (e: any) => {
      if (e.event === 'node_start') pushTrace('node', `▶ ${e.label}`)
      else if (e.event === 'tool_start') pushTrace('tool', `→ ${e.tool_name}`)
      else if (e.event === 'tool_end') {
        const reason = e.reason ? ` — ${e.reason}` : ''
        pushTrace('tool', `← ${e.tool_name} [${e.status}]${reason}`, e.elapsed_ms)
      } else if (e.event === 'node_end') {
        pushTrace('result', `■ ${e.label}${e.status ? ` (${e.status})` : ''}`)
      } else if (e.event === 'error') {
        pushTrace('result', `! ${e.message}`)
      }
    },
    (state: any) => {
      metrics.value = state.image_metrics
      diagnosis.value = state.diagnosis
      candidatePipelines.value = state.candidate_pipelines || []
      rejectedPipelines.value = state.rejected_pipelines || []
      selectedPipeline.value = state.selected_pipeline
      const sp = state.selected_pipeline
      if (sp?.comparison) {
        comparison.value = sp.comparison
      }
      if (sp?.output_path) {
        processedUrl.value = `/files/images/${sp.output_path.split(/[\\/]/).pop()}`
      }
      if (state.approval_status === 'pending') {
        ElMessage.warning('处理管线需人工审批后方可正式执行，请在「审批」页操作')
      }
    },
    (err: any) => {
      ElMessage.error(`分析失败：${err?.message || err}`)
    }
  )
  analyzing.value = false
}

onMounted(loadImageList)
</script>

<style scoped>
.sar-page {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.upload-row {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.file-name {
  font-size: 13px;
  color: #8b98a9;
}
.panel {
  background: #131a24;
  border: 0.5px solid #22303f;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
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
  gap: 8px;
}
.trace-panel .trace-body {
  max-height: 200px;
  overflow-y: auto;
  padding: 8px 14px;
}
.trace-item {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  padding: 3px 0;
  color: #9fb0c3;
}
.trace-badge {
  flex: 0 0 auto;
  min-width: 34px;
  text-align: center;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 4px;
  background: #1d2836;
  color: #6f8195;
}
.trace-item.node .trace-badge {
  background: #12303a;
  color: #00f0ff;
}
.trace-item.tool .trace-badge {
  background: #2a2036;
  color: #b388ff;
}
.trace-item.result .trace-badge {
  background: #33261a;
  color: #ffb26b;
}
.trace-text {
  flex: 1;
}
.trace-ms {
  color: #5d6d80;
  font-size: 11px;
}
.diag-body {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  padding: 12px 14px;
}
.diag-tag {
  margin: 0;
}
.metric-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
}
.metric-card {
  background: #131a24;
  border: 0.5px solid #22303f;
  border-radius: 8px;
  padding: 12px 14px;
}
.metric-label {
  font-size: 12px;
  color: #7d8ea3;
  margin-bottom: 6px;
}
.metric-value {
  font-size: 20px;
  font-weight: 500;
  color: #00f0ff;
}
.sar-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 12px;
}
.image-box {
  flex: 1;
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 10px;
}
.image-box img {
  max-width: 100%;
  max-height: 340px;
  object-fit: contain;
}
.compare-body {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
}
</style>
