<template>
  <el-container class="app-shell">
    <el-header class="app-header">
      <div class="brand">
        <span class="brand-mark">STAROPS</span>
        <span class="brand-sub">Satellite Mission Intelligence Platform</span>
      </div>
      <div class="header-right">
        <el-tag v-if="meta?.llm_degraded" type="warning" effect="dark" size="small">
          LLM 离线降级模式
        </el-tag>
        <el-tag v-else type="success" effect="dark" size="small">
          {{ meta?.llm_model || 'DeepSeek' }} 已接入
        </el-tag>
        <el-tag type="info" effect="plain" size="small">数据源：模拟数据</el-tag>
      </div>
    </el-header>

    <el-container>
      <el-aside width="200px" class="app-aside">
        <el-menu :default-active="$route.path" router background-color="#0f1620" text-color="#8b98a9" active-text-color="#00f0ff">
          <el-menu-item index="/workbench">
            <el-icon><DataAnalysis /></el-icon>
            <span>工作台</span>
          </el-menu-item>
          <el-menu-item index="/planning">
            <el-icon><MapLocation /></el-icon>
            <span>观测任务规划</span>
          </el-menu-item>
          <el-menu-item index="/sar">
            <el-icon><Picture /></el-icon>
            <span>SAR 影像分析</span>
          </el-menu-item>
          <el-menu-item index="/orders">
            <el-icon><List /></el-icon>
            <span>订单驱动调度</span>
          </el-menu-item>
          <el-menu-item index="/approvals">
            <el-icon><Stamp /></el-icon>
            <span>审批中心</span>
          </el-menu-item>
          <el-menu-item index="/audit">
            <el-icon><Document /></el-icon>
            <span>审计追溯</span>
          </el-menu-item>
        </el-menu>
      </el-aside>

      <el-main class="app-main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getMeta } from '@/api/client'

const meta = ref<any>(null)

onMounted(async () => {
  try {
    meta.value = await getMeta()
  } catch (e) {
    console.warn('元信息加载失败', e)
  }
})
</script>

<style scoped>
.app-shell {
  height: 100vh;
}
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #0f1620;
  border-bottom: 1px solid #1e2937;
}
.brand {
  display: flex;
  align-items: baseline;
  gap: 12px;
}
.brand-mark {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 3px;
  color: #00f0ff;
}
.brand-sub {
  font-size: 12px;
  color: #6b7a8d;
  letter-spacing: 1px;
}
.header-right {
  display: flex;
  gap: 8px;
  align-items: center;
}
.app-aside {
  background: #0f1620;
  border-right: 1px solid #1e2937;
}
.app-aside :deep(.el-menu) {
  border-right: none;
}
.app-main {
  background: #0d131b;
  padding: 16px;
  overflow-y: auto;
}
</style>
