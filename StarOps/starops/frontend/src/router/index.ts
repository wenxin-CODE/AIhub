import { createRouter, createWebHashHistory } from 'vue-router'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', redirect: '/workbench' },
    {
      path: '/workbench',
      name: 'workbench',
      component: () => import('@/views/WorkbenchView.vue'),
      meta: { title: 'Agent 工作台' }
    },
    {
      path: '/planning',
      name: 'planning',
      component: () => import('@/views/PlanningView.vue'),
      meta: { title: '观测任务规划' }
    },
    {
      path: '/sar',
      name: 'sar',
      component: () => import('@/views/SarView.vue'),
      meta: { title: 'SAR 影像分析' }
    },
    {
      path: '/orders',
      name: 'orders',
      component: () => import('@/views/OrderView.vue'),
      meta: { title: '订单驱动调度' }
    },
    {
      path: '/approvals',
      name: 'approvals',
      component: () => import('@/views/ApprovalsView.vue'),
      meta: { title: '审批中心' }
    },
    {
      path: '/audit',
      name: 'audit',
      component: () => import('@/views/AuditView.vue'),
      meta: { title: '审计追溯' }
    }
  ]
})

export default router
