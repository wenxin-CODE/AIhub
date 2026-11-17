# StarOps 项目长期记忆

## 环境约定（重要）
- **后端一律用项目内虚拟环境**：`E:/AIhub/StarOps/starops/.venv/Scripts/python.exe`（CPython 3.13.14，uv 创建）。
- **禁止用裸 `python`**：本机 `python` 在不同 shell 里会解析到 Anaconda base（3.14.6，缺 fastapi）或 workbuddy 内置 Python（3.13.14，无第三方包），都会导致 `ModuleNotFoundError`。
- 离线 wheel 缓存：`starops/.wheels/`（numpy 2.5.4 cp313、opencv-python-headless 5.0.0.93、fastapi 0.163.0、uvicorn 0.54.0、pydantic 2.14.3、httpx、httpcore、pydantic_core），PyPI 慢时用 `--no-index --find-links .wheels`。
- 前端：`starops/frontend`，Node 18+（开发于 22），Vite 代理 `/api`、`/files` → 127.0.0.1:8000。

## 常用命令
```bash
# 后端启动
cd E:/AIhub/StarOps/starops/backend && ../.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
# 单元测试
cd E:/AIhub/StarOps/starops && ./.venv/Scripts/python.exe -m pytest tests/unit -q
```

## 技术栈
- 后端：FastAPI + LangGraph（缺失降级到内置顺序执行器）+ SQLite **11 表** + SSE；DeepSeek API（无 Key 走离线规则模式）。
- 三条编排链：`planning` / `sar` / `order`（TSLC 订单驱动三阶段调度）。`node_route` 以**调用方预设的 task_type 优先**，不再被 LLM 结果覆盖。
- 前端：Vue3 + TS + Vite + Element Plus + Leaflet + ECharts。Element Plus 图标在 `main.ts` 全局注册。
- SAR：纯 numpy 实现（局部方差 / ENL / Lee / 中值 / 自适应 / FFT 条纹检测 / PSNR / SSIM），OpenCV 仅作 TIFF 可选增强。
- TSLC（IEEE TGRS 2026, DOI 10.1109/TGRS.2026.3700605）：AS→VS→ES 三阶段 + CV 校验器 + PM 重规划 + MS/K 记忆；内核在 `backend/tools/order_scheduling.py`。

## 已知坑位（别再踩）
- **SQLite 迁移顺序**：新增列的索引不能写进建表脚本，必须在 `_apply_migrations` 之后建（如 `KNOWLEDGE_INDEX_SQL`），否则旧库启动报 "no such column"。
- **模拟数据必须确定性**：窗口 ID 由「卫星|区域|起始时刻」派生（`_window_id`），时间基准锚定当日 UTC 零点。随机 UUID + `datetime.now()` 会让每次重启数据全变，"演示结果可重复"名不副实。
- **演示数据要与约束自洽**：云量分布、窗口时长、订单时间范围三项必须同时对齐阈值，否则测的是数据噪声。
- **订单调度要支持重复执行**：订单排期后状态即变，`/api/orders/run` 需带 `reset=true` 才能再次跑出结果。

## 项目边界（对外表达纪律）
- 观测窗口、卫星状态、订单均为**模拟数据**，不宣称真实卫星实时调度。
- 基础条纹/斑点检测 ≠ ScanSAR 扇贝效应算法；普通滤波 ≠ CycleGAN。
- 5% 辐射强度守门是**项目自设工程阈值**，非行业标准。
- 数字必须有可溯源来源；二手引用（CSDN 转引白皮书）不得进文档与口播。
- **TSLC 论文只说"吸收三阶段方法骨架"，不说"复现论文"**；论文的 4.58% / 13.38% 提升数字**不引用**（无同口径基线实验）。
- 订单调度是**逐单贪心**，优先级→利润映射与每星标称姿态转移时间都是**项目自设简化模型**。
