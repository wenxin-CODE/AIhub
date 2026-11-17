<template>
  <div class="planning">
    <div class="map-panel">
      <div class="panel-title">
        观测区域地图
        <el-tag size="small" type="info" effect="plain">Leaflet · 模拟数据</el-tag>
      </div>
      <div id="map" class="map"></div>
    </div>

    <div class="side-panel">
      <el-tabs v-model="tab">
        <el-tab-pane label="卫星资源" name="satellites">
          <el-table :data="satellites" size="small" max-height="300" :show-index="false">
            <el-table-column prop="satellite_name" label="名称" />
            <el-table-column prop="satellite_type" label="类型" width="70" />
            <el-table-column prop="resolution" label="分辨率" width="80">
              <template #default="{ row }">{{ row.resolution }}m</template>
            </el-table-column>
          </el-table>
        </el-tab-pane>

        <el-tab-pane label="观测窗口" name="windows">
          <el-table :data="windows" size="small" max-height="300" :show-index="false">
            <el-table-column prop="area_name" label="区域" width="70" />
            <el-table-column prop="start_time" label="开始" width="170" />
            <el-table-column prop="quality" label="质量" width="90">
              <template #default="{ row }">
                <el-tag size="small" :type="qualityType(row.quality)">{{ row.quality }}</el-tag>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>

        <el-tab-pane label="评分分布" name="chart">
          <div ref="chartRef" class="chart"></div>
        </el-tab-pane>
      </el-tabs>
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { getSatellites, getWindows } from '@/api/client'

const satellites = ref<any[]>([])
const windows = ref<any[]>([])
const chartRef = ref<HTMLElement>()
const tab = ref('satellites')
let map: L.Map | null = null

const AREA_CENTER: Record<string, [number, number]> = {
  'CN-BJ': [39.9042, 116.4074],
  'CN-SH': [31.2304, 121.4737],
  'CN-GZ': [23.1291, 113.2644],
  'CN-SZ': [22.5431, 114.0579],
  'CN-CD': [30.5728, 104.0668],
  'CN-XA': [34.3416, 108.9398]
}

const qualityType = (q: string) =>
  q === 'FRESH' ? 'success' : q === 'STALE' ? 'warning' : q === 'EXPIRED' ? 'danger' : 'info'

function renderMap() {
  if (!map) {
    map = L.map('map').setView([35.0, 108.0], 4)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap &copy; CARTO',
      maxZoom: 18
    }).addTo(map)
  }
  map.eachLayer((l) => {
    if ((l as any)._staropsMarker) map!.removeLayer(l)
  })

  const satColor: Record<string, string> = { optical: '#00f0ff', sar: '#ff2e93' }
  const seen = new Set<string>()
  for (const w of windows.value) {
    if (seen.has(w.area_id)) continue
    seen.add(w.area_id)
    const center = AREA_CENTER[w.area_id]
    if (!center) continue
    L.circle(center, {
      radius: 60000,
      color: '#00f0ff',
      fillColor: '#00f0ff',
      fillOpacity: 0.12,
      weight: 1
    })
      .bindPopup(`<b>${w.area_name}</b><br/>观测窗口 ${windows.value.filter((x: any) => x.area_id === w.area_id).length} 条`)
      .addTo(map)
  }

  for (const s of satellites.value) {
    L.marker([35 + Math.random() * 3, 100 + Math.random() * 15], {
      icon: L.divIcon({
        className: 'sat-marker',
        html: `<div style="width:8px;height:8px;border-radius:50%;background:${satColor[s.satellite_type] || '#888'};box-shadow:0 0 8px ${satColor[s.satellite_type] || '#888'}"></div>`,
        iconSize: [8, 8]
      })
    })
      .bindPopup(`<b>${s.satellite_name}</b><br/>${s.satellite_type}｜${s.resolution}m<br/>${s.imaging_mode}`)
      .addTo(map)
  }
}

function renderChart() {
  if (!chartRef.value) return
  const chart = echarts.init(chartRef.value)
  const bySat: Record<string, number> = {}
  for (const w of windows.value) {
    bySat[w.satellite_id] = (bySat[w.satellite_id] || 0) + 1
  }
  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    grid: { left: 50, right: 20, top: 20, bottom: 30 },
    xAxis: { type: 'category', data: Object.keys(bySat), axisLabel: { color: '#8b98a9' } },
    yAxis: { type: 'value', axisLabel: { color: '#8b98a9' }, splitLine: { lineStyle: { color: '#1e2937' } } },
    series: [
      {
        type: 'bar',
        data: Object.values(bySat),
        itemStyle: { color: '#00f0ff', borderRadius: [3, 3, 0, 0] },
        barMaxWidth: 22
      }
    ]
  })
}

onMounted(async () => {
  satellites.value = await getSatellites()
  windows.value = await getWindows()
  await nextTick()
  renderMap()
  renderChart()
})

watch(windows, () => {
  renderMap()
  renderChart()
})
</script>

<style scoped>
.planning {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  height: calc(100vh - 100px);
}
.map-panel,
.side-panel {
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
}
.map {
  flex: 1;
  min-height: 420px;
}
.side-panel :deep(.el-tabs) {
  flex: 1;
  display: flex;
  flex-direction: column;
}
.side-panel :deep(.el-tabs__content) {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}
.chart {
  height: 340px;
  width: 100%;
}
</style>
