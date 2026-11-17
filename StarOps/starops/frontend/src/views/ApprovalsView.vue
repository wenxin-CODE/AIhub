<template>
  <div class="approvals-page">
    <div class="head-row">
      <div>
        <div class="page-title">审批中心</div>
        <div class="page-desc">
          规划与执行严格分离：只有审批通过的方案才允许落库。未审批的执行请求会被后端直接拒绝并写入审计。
        </div>
      </div>
      <el-radio-group v-model="filter" @change="load">
        <el-radio-button value="pending">待审批</el-radio-button>
        <el-radio-button value="approved">已批准</el-radio-button>
        <el-radio-button value="rejected">已拒绝</el-radio-button>
        <el-radio-button value="">全部</el-radio-button>
      </el-radio-group>
    </div>

    <el-table :data="rows" size="small" border>
      <el-table-column prop="approval_id" label="审批单号" width="200" />
      <el-table-column prop="target_type" label="类型" width="130" />
      <el-table-column prop="target_id" label="对象" width="150" />
      <el-table-column prop="status" label="状态" width="90">
        <template #default="{ row }">
          <el-tag size="small" :type="statusType(row.status)">{{ row.status }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="created_at" label="创建时间" width="180" />
      <el-table-column prop="reviewer" label="审批人" width="100" />
      <el-table-column prop="comment" label="意见" />
      <el-table-column label="操作" width="150" fixed="right">
        <template #default="{ row }">
          <template v-if="row.status === 'pending'">
            <el-button size="small" type="success" @click="review(row, 'approved')">批准</el-button>
            <el-button size="small" type="danger" plain @click="review(row, 'rejected')">拒绝</el-button>
          </template>
          <span v-else class="done-mark">已处理</span>
        </template>
      </el-table-column>
    </el-table>

    <el-empty v-if="!rows.length" description="暂无审批单" />
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { getApprovals, reviewApproval } from '@/api/client'

const rows = ref<any[]>([])
const filter = ref('pending')

const statusType = (s: string) =>
  s === 'approved' ? 'success' : s === 'rejected' ? 'danger' : 'warning'

async function load() {
  rows.value = await getApprovals(filter.value || undefined)
}

async function review(row: any, decision: string) {
  try {
    await reviewApproval(row.approval_id, decision, '')
    ElMessage.success(decision === 'approved' ? '已批准' : '已拒绝')
    await load()
  } catch (e: any) {
    ElMessage.error(e.response?.data?.detail || '审批失败')
  }
}

onMounted(load)
</script>

<style scoped>
.approvals-page {
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
  max-width: 620px;
}
.done-mark {
  font-size: 12px;
  color: #5c6b7d;
}
</style>
