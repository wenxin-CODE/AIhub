import axios from 'axios'

const http = axios.create({
  baseURL: '/api',
  timeout: 120000
})

export async function getMeta() {
  const { data } = await http.get('/meta')
  return data
}

export async function getSatellites() {
  const { data } = await http.get('/satellites')
  return data.satellites
}

export async function getWindows(areaId?: string) {
  const { data } = await http.get('/windows', { params: areaId ? { area_id: areaId } : {} })
  return data.windows
}

export async function getApprovals(status?: string) {
  const { data } = await http.get('/approvals', { params: status ? { status } : {} })
  return data.approvals
}

export async function reviewApproval(id: string, decision: string, comment = '') {
  const { data } = await http.post(`/approvals/${id}/review`, { decision, comment })
  return data
}

export async function getAudit(objectId?: string) {
  const { data } = await http.get('/audit', { params: objectId ? { object_id: objectId } : {} })
  return data.audit
}

export async function getToolExecutions(traceId?: string) {
  const { data } = await http.get('/tool-executions', { params: traceId ? { trace_id: traceId } : {} })
  return data.executions
}

export async function getImages() {
  const { data } = await http.get('/images')
  return data.images
}

/** 观测订单队列（TSLC 订单驱动调度的输入）。 */
export async function getOrders(status?: string) {
  const { data } = await http.get('/orders', { params: status ? { status } : {} })
  return data.orders
}

/** TSLC 知识库 K：已确认的调度计划。 */
export async function getScheduleKnowledge() {
  const { data } = await http.get('/schedule/knowledge')
  return data.knowledge
}

/** TSLC 记忆 MS：失败尝试记录。 */
export async function getScheduleMemory() {
  const { data } = await http.get('/schedule/memory')
  return data.memory
}

/**
 * 订阅 TSLC 订单驱动调度执行过程。
 *
 * 事件协议：order_batch_start → order_start → stage_decision / stage_failed
 *          → order_scheduled / order_rejected → order_batch_end → final
 *
 * @param orderIds 只调度指定订单；为空表示调度全部 pending 订单
 * @param reset    调度前把订单状态重置为 pending（可重复演示）
 */
export async function runOrdersStream(
  onEvent: (e: any) => void,
  onDone?: (state: any) => void,
  onError?: (err: any) => void,
  maxReplans = 3,
  orderIds: string[] = [],
  reset = false
) {
  try {
    const resp = await fetch('/api/orders/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        max_replans: maxReplans,
        order_ids: orderIds,
        reset
      })
    })

    if (!resp.ok || !resp.body) {
      throw new Error(`请求失败：HTTP ${resp.status}`)
    }

    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const chunks = buffer.split('\n\n')
      buffer = chunks.pop() ?? ''

      for (const chunk of chunks) {
        const line = chunk.trim()
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        try {
          const evt = JSON.parse(payload)
          onEvent(evt)
          if (evt.event === 'final') onDone?.(evt.state)
          if (evt.event === 'error') onError?.(new Error(evt.message))
        } catch (e) {
          console.warn('SSE 解析失败', payload)
        }
      }
    }
  } catch (err) {
    onError?.(err)
  }
}

/**
 * 上传 SAR 影像。
 *
 * 用 FormData 直传，文件名保留扩展名（后端据此选解码器）。
 */
export async function uploadSarImage(file: File) {
  const fd = new FormData()
  fd.append('file', file)
  const { data } = await http.post('/sar/upload', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000
  })
  return data
}

/**
 * 订阅 SAR 分析执行过程。与 runAgentStream 同协议，只是端点不同。
 */
export async function runSarStream(
  imagePath: string,
  onEvent: (e: any) => void,
  onDone?: (state: any) => void,
  onError?: (err: any) => void
) {
  try {
    const resp = await fetch('/api/sar/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_path: imagePath })
    })

    if (!resp.ok || !resp.body) {
      throw new Error(`请求失败：HTTP ${resp.status}`)
    }

    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const chunks = buffer.split('\n\n')
      buffer = chunks.pop() ?? ''

      for (const chunk of chunks) {
        const line = chunk.trim()
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        try {
          const evt = JSON.parse(payload)
          onEvent(evt)
          if (evt.event === 'final') onDone?.(evt.state)
          if (evt.event === 'error') onError?.(new Error(evt.message))
        } catch (e) {
          console.warn('SSE 解析失败', payload)
        }
      }
    }
  } catch (err) {
    onError?.(err)
  }
}

/**
 * 发起 Agent 任务并订阅 SSE 执行过程。
 *
 * 用 fetch + ReadableStream 而非 EventSource，因为需要 POST body。
 * 回调：onEvent(event) 逐条接收，onDone(finalState) 接收最终状态。
 */
export async function runAgentStream(
  message: string,
  onEvent: (e: any) => void,
  onDone?: (state: any) => void,
  onError?: (err: any) => void
) {
  try {
    const resp = await fetch('/api/agent/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    })

    if (!resp.ok || !resp.body) {
      throw new Error(`请求失败：HTTP ${resp.status}`)
    }

    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const chunks = buffer.split('\n\n')
      buffer = chunks.pop() ?? ''

      for (const chunk of chunks) {
        const line = chunk.trim()
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        try {
          const evt = JSON.parse(payload)
          onEvent(evt)
          if (evt.event === 'final') onDone?.(evt.state)
          if (evt.event === 'error') onError?.(new Error(evt.message))
        } catch (e) {
          console.warn('SSE 解析失败', payload)
        }
      }
    }
  } catch (err) {
    onError?.(err)
  }
}

export default http
