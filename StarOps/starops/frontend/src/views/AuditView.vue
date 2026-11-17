<template>
  <div class="audit-page">
    <div class="head-row">
      <div>
        <div class="page-title">审计追溯</div>
        <div class="page-desc">
          双表留痕：tool_execution 记录工具调用，audit_log 记录业务动作。可从一次审批回溯到工具输入输出。
        </div>
      </div>
      <el-tabs v-model="tab" @tab-change="load">
        <el-tab-pane label="工具调用" name="tool" />
        <el-tab-pane label="业务审计" name="audit" />
      </el-tabs>
    </div>

    <el-table v-if="tab === 'tool'" :data="toolRows" size="small" border max-height="calc(100vh - 260px)">
      <el-table-column prop="created_at" label="时间" width="180" />
      <el-table-column prop="tool_name" label="工具" width="160" />
      <el-table-column prop="status" label="状态" width="90">
        <template #default="{ row }">
          <el-tag size="small" :type="statusType(row.status)">{{ row.status }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="duration_ms" label="耗时" width="90">
        <template #default="{ row }">{{ row.duration_ms }}ms</template>
      </el-table-column>
      <el-table-column prop="input_summary" label="输入摘要" show-overflow-tooltip />
      <el-table-column prop="output_summary" label="输出摘要" show-overflow-tooltip />
    </el-table>

    <el-table v-else :data="auditRows" size="small" border max-height="calc(100vh - 260px)">
      <el-table-column prop="created_at" label="时间" width="180" />
      <el-table-column prop="actor" label="操作者" width="110" />
      <el-table-column prop="action" label="动作" width="150" />
      <el-table-column prop="object_id" label="对象" width="200" show-overflow-tooltip />
      <el-table-column prop="details" label="详情" show-overflow-tooltip />
    </el-table>

    <el-empty v-if="!rows().length" description="暂无记录" />
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getAudit, getToolExecutions } from '@/api/client'

const tab = ref('tool')
const toolRows = ref<any[]>([])
const auditRows = ref<any[]>([])

const rows = () => (tab.value === 'tool' ? toolRows.value : auditRows.value)

const statusType = (s: string) =>
  s === 'success' ? 'success' : s === 'failed' ? 'danger' : s === 'degraded' ? 'warning' : 'info'

async function load() {
  if (tab.value === 'tool') {
    toolRows.value = await getToolExecutions()
  } else {
    auditRows.value = await getAudit()
  }
}

onMounted(load)
</script>

<style scoped>
.audit-page {
  background: #131a24;
  border: 0.5px solid #22303f;
  border-radius: 8px;
  padding: 16px;
}
.head-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 16px;
}
.page-title {
  font-size: 15px;
  font-weight: 500;
  color: #d5dee8;
  margin-bottom: 4px;
}
.page-desc {
  font-size: 12px;
  color: #7d8ea3;
  line-height: 1.6;
  max-width: 560px;
}
</style>
