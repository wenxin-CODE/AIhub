# 星枢 StarOps 软件测试评估报告

- **评估对象**：`e:\AIhub\StarOps\starops`（FastAPI + LangGraph 后端、Vue3 前端、SQLite 数据层）
- **评估日期**：2026-11-17
- **评估角色**：软件测试工程师
- **评估方式**：源码审阅 + 现有单测执行 + 隔离数据库运行时探针 + 前端构建验证
- **结论摘要**：项目作为演示原型完成度较高，单元测试 71/71 通过、前端严格模式构建通过；但存在 **3 项 P0 级问题**，其中「审批闭环并未真正约束执行」和「SAR 接口可读写任意本地路径」直接推翻了 README 中最核心的安全与可信承诺。建议先修复 P0，再进入对外演示或容器化部署。

> 说明：本报告只记录能从代码或实测复现的问题；对依赖并发、压测、真实 LLM、真实浏览器矩阵才能确认的项目，明确标注为「风险/待验证」，不臆测为确定缺陷。运行时探针均使用临时目录与临时 SQLite 库，未修改项目数据。

---

## 1. 评估范围与方法

| 项目 | 内容 |
|---|---|
| 代码范围 | 后端 17 个 Python 文件（约 3.5k 行）、前端 12 个 TS/Vue 文件、5 个测试文件、Docker/Compose/配置 |
| 已执行 | `pytest tests/unit -q` → **71 passed（14.27s）**；`npm run build` → **构建成功（21.15s）**，有 chunk 体积告警 |
| 运行时探针 | 用临时数据库启动 FastAPI TestClient，验证区域/时间约束、审批绑定、审批前落库、无时区时间戳、CORS、上传与任意路径 SAR 分析 |
| 未执行 | 真实 DeepSeek 调用、Docker 实际起容器、压测/并发注入、渗透测试、真实浏览器/移动端矩阵、L2 深度模型 |
| 基线资料 | 仅 README 与代码；`docs/` 为空，无需求规格、验收用例、性能指标、安全基线、历史缺陷库 |

---

## 2. 严重度与优先级定义

| 级别 | 定义 | 处理时限建议 |
|---|---|---|
| **P0 阻断** | 安全边界失效、核心安全承诺不成立、可造成数据/权限事故 | 对外演示/部署前必须修复 |
| **P1 高** | 主要功能错误、部署不可用、稳定性和安全性明显缺口 | 本迭代修复 |
| **P2 中** | 局部功能、可用性、可观测性、可维护性与测试质量问题 | 下一迭代修复 |
| **P3 低** | 优化项、文档与体验改进 | 纳入计划 |

---

## 3. 问题总览（按严重度排序）

| ID | 严重度/优先级 | 维度 | 问题 | 证据类型 |
|---|---|---|---|---|
| P0-01 | P0 / 立即 | 安全 | SAR 分析接口可读取任意本地路径，并把处理结果写到该路径所在目录 | 实测 |
| P0-02 | P0 / 立即 | 功能正确性、安全 | 审批关口未真正约束执行：执行节点不可达，订单/管线在审批前已落库 | 实测 |
| P0-03 | P0 / 立即 | 安全、正确性 | 审批单未绑定任务/方案/载荷，已批准的审批单可授权任意提交 | 实测 |
| P1-01 | P1 / 本迭代 | 功能正确性 | 规划链忽略显式时间范围；缺少区域时跨全部区域自动选方案 | 实测 |
| P1-02 | P1 / 本迭代 | 安全 | 全站无认证授权；CORS 通配 + 凭证；接口泄露服务器绝对路径 | 实测 + 代码审阅 |
| P1-03 | P1 / 本迭代 | 稳定性、正确性 | 订单时间字符串未校验；无时区输入触发 TypeError，SSE 只发 error 不发 final | 实测 |
| P1-04 | P1 / 本迭代 | 稳定性 | LLM 异常不降级、每次请求都调用 LLM、LLM JSON 未按 Schema 校验、`.env` 未加载 | 代码审阅 + 实测降级路径 |
| P1-05 | P1 / 本迭代 | 安全、稳定性 | 上传无大小限制，PNG 解码信任宽高与压缩流，存在内存/CPU 耗尽风险；坏图仍登记 200 | 代码审阅 |
| P1-06 | P1 / 本迭代 | 兼容性、部署 | Docker 前端 nginx 未代理 `/files/`，SAR 图片 404；构建上下文无 `.dockerignore`，`npm install` 不可复现 | 代码审阅 |
| P1-07 | P1 / 本迭代 | 功能正确性 | 订单调度资源独占只在单次运行内生效；MS 只写不读、K 不参与决策 | 代码审阅 |
| P2-01 | P2 / 下迭代 | 正确性、并发 | 审批/幂等为「先查后写」，存在并发重复审批与唯一键冲突 500 风险；无去重与过期 | 代码审阅（待并发验证） |
| P2-02 | P2 / 下迭代 | 数据完整性 | 每次启动删除并重建窗口与模拟订单，知识库/审计产生孤儿；多副本启动有竞态 | 代码审阅 + 实测启动日志 |
| P2-03 | P2 / 下迭代 | 稳定性 | SSE 无断连检测、无心跳、队列无上限；error 时丢弃 final；三处 SSE 代码重复 | 代码审阅 + 实测 |
| P2-04 | P2 / 下迭代 | 可追溯性 | `mission_task` 从不写入；审计/工具输出截断 2000 字符；审计人固定 `operator`；无分页 | 代码审阅 |
| P2-05 | P2 / 下迭代 | 稳定性 | SQLite 每次查询新建连接、无 busy_timeout/连接池，多 SSE 并发写入可能锁库 | 代码审阅（待压测验证） |
| P2-06 | P2 / 下迭代 | 易用性 | 多个页面加载无异常处理，后端不可用时白屏/未处理 Promise；SSE 无超时/取消 | 代码审阅 |
| P2-07 | P2 / 下迭代 | 易用性、兼容性 | 无响应式布局；时间显示原始 UTC；地图卫星坐标为随机；TIF 预览不可用 | 代码审阅 + 构建 |
| P2-08 | P2 / 下迭代 | 数据安全 | 订单页默认勾选「重置队列」，且重置会改全部订单状态（含用户创建/已排期） | 代码审阅 |
| P2-09 | P2 / 下迭代 | 输入校验、性能 | `limit` 可传 -1 变为无限制；审批列表返回完整快照；接口无分页 | 实测 + 代码审阅 |
| P2-10 | P2 / 下迭代 | 可维护性 | `insert()` 用「字典第一个值」当主键返回，键顺序变化会静默返回错误 ID | 代码审阅 |
| P2-11 | P2 / 下迭代 | 回归风险 | 生产路径（LangGraph）单测覆盖不足，现有测试主要走 `run_builtin`；跨文件共享 DB 路径 | 代码审阅 + 执行结果 |
| P2-12 | P2 / 下迭代 | 性能 | 前端首屏包体大：主包 1.28MB、规划页 1.19MB（gzip 合计约 800KB），未做分包/按需引入 | 构建实测 |
| P2-13 | P2 / 下迭代 | 兼容性、可维护性 | 依赖仅 `>=` 无锁；README 要求 Python 3.13、Docker 用 3.12；`on_event` 已弃用；多处配置为死字段 | 代码审阅 |
| P2-14 | P2 / 下迭代 | 可解释性 | 区域不匹配窗口被静默跳过；过期/可疑原因会被后续尝试覆盖；不可行根因不完整 | 代码审阅 |
| P2-15 | P2 / 下迭代 | 回归/交付 | 项目整体未纳入版本控制；`.gitignore` 忽略 `data/images/`，与「仓库自带演示影像」声明冲突 | 实测 |
| P2-16 | P2 / 下迭代 | 质量门禁 | 无 CI、无后端 lint/类型检查、无前端 ESLint/单测/E2E、无覆盖率门禁 | 代码审阅 |
| P3-01 | P3 / 计划 | 可观测性 | `/api/health` 不检查依赖；无请求 ID/结构化日志；审计与日志未关联 trace | 代码审阅 |
| P3-02 | P3 / 计划 | 兼容性 | 地图瓦片依赖外部 CDN，离线环境地图空白；无 CSP/安全响应头 | 代码审阅 |
| P3-03 | P3 / 计划 | 易用性、无障碍 | 大量 11–12px 字号、无键盘/读屏适配；审批无二次确认与意见输入 | 代码审阅 |
| P3-04 | P3 / 计划 | 文档 | 缺少需求/验收/测试计划文档，README 多处能力声明与实现不一致 | 代码审阅 |

---

## 4. 详细问题说明

### P0-01 SAR 分析接口可读写任意本地路径

- **维度**：安全性（路径穿越 / 本地文件访问 / 任意文件写入）
- **证据**：
  - 接口直接接收服务端绝对路径，无白名单、无目录约束：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L345-L355)、[sar_nodes.py](file:///e:/AIhub/StarOps/starops/backend/agents/nodes/sar_nodes.py#L70-L90)
  - 处理结果路径由输入路径派生并直接写盘：[sar_nodes.py](file:///e:/AIhub/StarOps/starops/backend/agents/nodes/sar_nodes.py#L229-L237)、[sar.py](file:///e:/AIhub/StarOps/starops/backend/tools/sar.py#L211-L247)
  - 实测：把 `demo_sar.png` 复制到项目外的临时目录后调用 `/api/sar/analyze`，该目录新增 `any__adaptive.png`、`any__lee.png`、`any__median.png`
- **可能触发场景**：任意客户端（含浏览器中的恶意网页，见 P1-02）提交 `image_path` 指向用户目录、共享盘、其他服务数据目录中的 PNG/TIF。
- **潜在风险**：本地影像/截图等信息泄露；在任意可写目录制造文件、耗尽磁盘；配合通配 CORS 可被远程网页驱动，形成本地服务的「读写代理」。
- **优化建议**：
  1. 只接受 `image_id` 或 `image_dir` 内的相对文件名，服务端解析为绝对路径后校验 `resolve().is_relative_to(settings.image_dir)`；
  2. 处理结果统一写入独立 `output_dir`，禁止写在输入目录；
  3. 加认证与源限制（见 P1-02）。
- **验证标准**：
  - 传入 `image_dir` 之外路径返回 4xx，磁盘无新增文件；
  - 处理输出仅出现在 `data/images/processed/`；
  - 自动化用例覆盖 `..`、绝对路径、符号链接、Windows 盘符路径。

---

### P0-02 审批关口未真正约束执行（核心安全闭环不成立）

- **维度**：功能正确性、安全性、合规可追溯
- **证据**：
  - `commit_plan` 仅被测试引用，API 层没有任何提交入口：[approval.py](file:///e:/AIhub/StarOps/starops/backend/services/approval.py#L119-L216)、[test_safety_loop.py](file:///e:/AIhub/StarOps/starops/tests/unit/test_safety_loop.py#L20-L133)
  - `execute` / `sar_execute` 节点虽注册，但没有从审批节点或外部恢复流程连入的边，实际不可达：[graph.py](file:///e:/AIhub/StarOps/starops/backend/agents/graph.py#L405-L442)
  - 订单链在创建审批单**之前**已把订单改为 `scheduled/rejected` 并写入知识库：[order_nodes.py](file:///e:/AIhub/StarOps/starops/backend/agents/nodes/order_nodes.py#L140-L190)
  - SAR 链在生成审批单前已把所有候选滤波结果写入磁盘：[sar_nodes.py](file:///e:/AIhub/StarOps/starops/backend/agents/nodes/sar_nodes.py#L160-L197)
  - 实测：调度后订单状态 `pending -> scheduled` 且审批仍为 `pending`；随后批准，`mission_task` 仍为 0 条、`plan_commit` 无新增
- **可能触发场景**：任何一次规划/SAR/订单运行；用户以为「批准后才执行」，实际审批只是改状态。
- **潜在风险**：README 的「审批不可绕过」「审批前不落地」不成立；演示中被追问时无法自证；若接入真实执行器，将直接造成未审批任务下发。
- **优化建议**：
  1. 明确定义「执行」：审批通过后由独立 `/api/commits`（或审批回调）触发 `commit_plan` 与执行节点；
  2. 订单/SAR 的副作用全部后移到审批通过之后；运行期只产出候选与审批单，不写业务状态、不写知识库、不落处理文件；
  3. 审批拒绝必须保持原状态并可回滚已产生的临时产物；
  4. 前端在批准后轮询/订阅执行结果，而不是只提示「可进入执行环节」。
- **验证标准**：
  - 端到端用例：运行后订单仍 `pending`、无 `plan_commit`、无输出影像；批准后恰好 1 条 `plan_commit`、状态变 `scheduled`、输出影像存在；拒绝后无任何状态变化；
  - 绕过前端直接调用提交接口，未审批必须 4xx 且写审计；
  - 审计中可串联「审批单 → 提交 → 执行产物」。

---

### P0-03 审批单未绑定任务、方案与载荷

- **维度**：安全性、正确性、审计一致性
- **证据**：`commit_plan` 只校验审批状态与 `plan_version`，不校验审批单的 `target_type/target_id` 与本次 `task_id`、`payload` 是否一致：[approval.py](file:///e:/AIhub/StarOps/starops/backend/services/approval.py#L134-L180)
- **实测**：为 `target_id=plan-A` 创建并批准审批单后，用同一 `approval_id` 提交 `task-B` 且 payload 为完全不同的 `{"plan":"COMPLETELY-DIFFERENT"}`，返回 `success`
- **可能触发场景**：审批中心同时存在多张审批单；脚本/前端传错 `approval_id`；或恶意客户端复用历史已批准单。
- **潜在风险**：越权执行未审批方案；审计记录与实际审批对象不一致，无法追责。
- **优化建议**：
  1. 审批单保存 `task_id`、`plan_hash`（规范化载荷哈希）与目标类型；
  2. 提交时校验 `appr.target_id == payload.plan_id`、`appr.task_id == task_id`、`appr.plan_hash == hash(payload)`；
  3. 审批单一次性使用（`used_at`），或按 `(target_id, plan_version)` 唯一约束。
- **验证标准**：
  - 用 A 单提交 B 方案必须失败并写 `commit_rejected`；
  - 篡改 payload 后提交失败；
  - 同一审批单二次提交按幂等返回首次结果，不得执行新载荷。

---

### P1-01 规划链忽略显式时间范围，缺区域时跨区域自动选方案

- **维度**：功能正确性（需求理解与硬约束）
- **证据**：
  - 需求解析虽输出 `missing_fields`，但流程继续执行，未阻断：[graph.py](file:///e:/AIhub/StarOps/starops/backend/agents/graph.py#L139-L152)
  - 窗口查询节点没有把 `time_start/time_end` 传给工具：[graph.py](file:///e:/AIhub/StarOps/starops/backend/agents/graph.py#L177-L196)；工具本身支持这两个参数但未被使用：[planning.py](file:///e:/AIhub/StarOps/starops/backend/tools/planning.py#L47-L80)
  - 硬约束校验只检查分辨率/类型/质量/云量/入射角/`time_days`，**没有区域约束**，也没有 `time_start/time_end` 检查：[planning.py](file:///e:/AIhub/StarOps/starops/backend/tools/planning.py#L121-L211)
  - `PlanItem` 不含 `area_id`，最终方案无法体现区域：[models.py](file:///e:/AIhub/StarOps/starops/backend/schemas/models.py#L64-L79)
  - 实测 A：请求「规划任务，分辨率优于2米，未来3天内」未给区域，窗口覆盖 6 个区域，最终选中 **CN-SH**，方案中无区域字段；
  - 实测 B：显式要求 `2027-06-01 ~ 2027-06-30`，最终选中 `2026-11-30` 的窗口（早于要求 7 个月），而 `checked_rules` 仍声称包含 `time_range`
- **可能触发场景**：用户只说「规划未来3天任务」而不说区域；或给出明确起止日期；LLM 只返回 `area_name` 未返回 `area_id`。
- **潜在风险**：把错误区域/错误时段的方案当作合规结果展示；违反 README「缺失项主动追问、不自行假设」的核心主张。
- **优化建议**：
  1. `missing_fields` 非空时终止并返回追问，不进入查询/评分；
  2. 将显式时间范围传入窗口查询与校验器，统一以 UTC 比较；
  3. 校验器增加 `area_id` 规则，`PlanItem` 增加 `area_id/area_name`；
  4. 把 `checked_rules` 与实际执行规则绑定，避免「声明检查但未实现」。
- **验证标准**：
  - 缺区域请求返回 `need_input` 且不产出方案；
  - 指定 2027-06 时，所有候选窗口均落在该区间，越界方案进入 rejected；
  - 选中方案包含且仅包含请求区域；
  - 单测覆盖上述三条。

---

### P1-02 全站无认证授权、CORS 通配、暴露绝对路径

- **维度**：安全性
- **证据**：
  - CORS `allow_origins=["*"]` + `allow_credentials=True`：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L46-L52)
  - 所有接口（审批、审计、订单重置、上传、SAR 分析、Swagger）无认证/角色校验：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L240-L265)、[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L111-L165)
  - `/api/images` 与上传响应返回服务器绝对路径 `upload_path`：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L327-L342)
  - 实测：带 `Origin: https://evil.example` 的预检返回 `Access-Control-Allow-Origin: https://evil.example` 且 `Allow-Credentials: true`
- **可能触发场景**：用户在本机/内网运行服务，浏览器打开任意恶意页面即可跨源调用；Docker Compose 将 8000 端口暴露到宿主机。
- **潜在风险**：未授权审批、清空订单状态、触发高开销任务、读取审计与影像元数据、获取本机用户名/目录结构；形成 CSRF/本地代理类攻击面。
- **优化建议**：
  1. 默认仅监听 `127.0.0.1`；生产加登录与 RBAC（审批者/操作员/只读）；
  2. CORS 改为显式来源白名单；`allow_credentials=False` 或不通配；
  3. 响应中去掉绝对路径，仅返回 `image_id` 与相对 URL；
  4. 审批接口强制校验审批人身份并写入真实 actor；错误信息不回显完整路径。
- **验证标准**：
  - 未携带有效凭证访问审批/审计/订单变更返回 401/403；
  - 非白名单 Origin 预检被拒绝；
  - `/api/images` 不出现盘符/绝对路径；
  - 审批审计 actor 等于登录用户。

---

### P1-03 订单时间入参未校验，无时区输入导致 TypeError 与 SSE 断流

- **维度**：稳定性、输入校验、易用性
- **证据**：
  - 创建订单的请求模型只声明 `time_start/time_end: str`，未做 ISO/时区校验：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L80-L93)
  - 解析用 `datetime.fromisoformat`，未强制 UTC：[order_scheduling.py](file:///e:/AIhub/StarOps/starops/backend/tools/order_scheduling.py#L95-L96)
  - 实测：`time_start="not-a-date"` 创建订单返回 **200**；随后调度抛 `TypeError: can't compare offset-naive and offset-aware datetimes`；SSE 只收到 `tool_end(failed)` 与 `error`，没有 `order_rejected`，也没有 `final`
  - `main.py` 的 SSE 循环在 `error` 时直接 `break`，丢弃 `final`：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L207-L222)
- **可能触发场景**：手工输入命令、第三方集成、前端未使用日期控件、LLM 生成的订单时间不带时区。
- **潜在风险**：接口 500/SSE 中断、页面卡在「调度中」；部分订单已调度而其余未处理，状态半成品；错误信息直接暴露内部异常文本。
- **优化建议**：
  1. 用 Pydantic `datetime` + `AwareDatetime`（或自定义 validator）强制 ISO8601 且带时区，并校验 `start < end`；
  2. 统一规范化到 UTC 后入库；
  3. 业务异常在节点内捕获并转成结构化 `order_rejected`；
  4. SSE 无论 error 与否都应发送 `final`（可含 `ok=false`）。
- **验证标准**：
  - 非法/无时区时间返回 422，且不写库；
  - 合法无时区字符串被明确拒绝或统一补 UTC，调度不抛异常；
  - 人为触发异常时客户端仍收到 `final`，运行态可恢复；
  - 单测覆盖 `naive`、`Z`、`+08:00`、非法字符串。

---

### P1-04 LLM 异常不降级、调用位置浪费、输出未按 Schema 校验、`.env` 未加载

- **维度**：稳定性、功能正确性、可配置性
- **证据**：
  - `chat_json` 重试后直接抛异常，没有回退 mock：[llm.py](file:///e:/AIhub/StarOps/starops/backend/agents/llm.py#L41-L80)
  - 路由节点先调用 LLM，再判断 `task_type` 是否已由 API 预设，SAR/订单链也承担 LLM 延迟与失败风险：[graph.py](file:///e:/AIhub/StarOps/starops/backend/agents/graph.py#L92-L119)
  - LLM 返回的 `task_requirements` 直接进入下游，未经 `TaskRequirements` 校验：[graph.py](file:///e:/AIhub/StarOps/starops/backend/agents/graph.py#L120-L123)、[models.py](file:///e:/AIhub/StarOps/starops/backend/schemas/models.py#L20-L40)
  - 代码中没有 `load_dotenv`/`python-dotenv`，README 的「复制 `.env.example` 为 `.env`」在本地启动时不会生效：[config.py](file:///e:/AIhub/StarOps/starops/backend/config.py#L16-L53)
- **可能触发场景**：配置了 Key 但网络不通、配额耗尽、模型返回非 JSON/markdown 包裹；LLM 返回 `"max_resolution": "两米"`、`time_days: "abc"`；用户按 README 配置 `.env`。
- **潜在风险**：单次运行最长阻塞约 3×60s 后整链失败；下游 `int()/比较` 抛异常；「配置了 Key 却一直离线」造成误判。
- **优化建议**：
  1. 先按 `preset task_type` 路由，SAR/订单链不调用需求解析 LLM；
  2. `chat_json` 失败时返回带 `_degraded/_mock` 的规则结果，并在 UI 显式标注；
  3. 对 LLM 输出强制走 `TaskRequirements.model_validate`，失败则降级或追问；
  4. 增加 `.env` 加载（`python-dotenv` 或 uvicorn `--env-file`），并补充启动日志说明配置来源；
  5. 增加 LLM 熔断/超时预算（整链超时而非单次超时）。
- **验证标准**：
  - 注入 500/超时/非 JSON 响应，接口仍完成并标注 degraded，不 500；
  - SAR/订单请求在 LLM 关闭时不产生 LLM 调用（可用 mock 计数验证）；
  - 非法类型字段被 Schema 拒绝并进入追问/降级；
  - 按 README 配置 `.env` 后 `/api/meta` 显示 `llm_available=true`。

---

### P1-05 上传与 PNG 解码缺少资源上限，坏图仍登记成功

- **维度**：安全性、稳定性
- **证据**：
  - 上传一次性 `await file.read()` 全量入内存，无大小限制；仅按扩展名放行：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L288-L325)
  - 解码直接信任 IHDR 的宽高并解压全部 IDAT，无像素数上限：[imaging.py](file:///e:/AIhub/StarOps/starops/backend/tools/imaging.py#L136-L191)
  - 反滤波逐行 Python 循环，尺寸越大 CPU 成本越高：[imaging.py](file:///e:/AIhub/StarOps/starops/backend/tools/imaging.py#L97-L133)
  - 上传后即使元数据解析失败仍写入 `sar_image` 并返回 200：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L307-L334)
- **可能触发场景**：上传超大文件；构造「小文件、巨大声明宽高 + 高压缩比 IDAT」的 PNG；扩展名正确但内容损坏。
- **潜在风险**：内存耗尽/进程被杀；单请求长时间占用 CPU；无效影像污染影像列表并让后续分析反复失败。
- **优化建议**：
  1. 限制上传大小（如 50MB）与像素数（如 4096×4096）、位深/通道；
  2. 解码前从 IHDR 读取宽高并先做上限校验；限制解压后字节数；
  3. 先校验再登记，坏图返回 4xx 且不落库、不留文件；
  4. 内容类型/魔数校验，不仅看扩展名。
- **验证标准**：
  - 超大文件/超大声明尺寸被拒绝且内存不显著增长；
  - 伪造 PNG 返回 4xx，`sar_image` 无记录，磁盘无残留；
  - 正常 512×512 与 16 位 PNG 仍通过；
  - 增加对应单测与资源上限配置项。

---

### P1-06 Docker 部署链路不完整、构建不可复现

- **维度**：兼容性、部署、可复现性
- **证据**：
  - 前端 nginx 仅代理 `/api/`，未代理 `/files/`；而 SAR 页面图片地址为 `/files/images/...`：[frontend/Dockerfile](file:///e:/AIhub/StarOps/starops/frontend/Dockerfile#L16-L32)、[SarView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/SarView.vue#L233-L239)
  - 后端构建上下文为仓库根，无 `.dockerignore`，会把 `.venv/.wheels/data/*.db` 一并送入构建上下文：[docker-compose.yml](file:///e:/AIhub/StarOps/starops/docker-compose.yml#L1-L16)、[backend/Dockerfile](file:///e:/AIhub/StarOps/starops/backend/Dockerfile#L1-L16)
  - 前端 `COPY package.json ./` 后 `npm install`，随后 `COPY . .` 会用宿主机 `node_modules` 覆盖容器内依赖；无 `.dockerignore`：[frontend/Dockerfile](file:///e:/AIhub/StarOps/starops/frontend/Dockerfile#L1-L14)
- **可能触发场景**：`docker compose up -d` 后打开 SAR 页面；在 Windows 构建前端镜像；CI 从干净环境构建。
- **潜在风险**：SAR 原图/处理图 404，功能在容器中不可用；构建极慢或直接失败；依赖版本漂移导致「本机能跑、容器不能跑」。
- **优化建议**：
  1. nginx 增加 `location /files/ { proxy_pass http://backend:8000/files/; }`；
  2. 增加根目录与前端 `.dockerignore`（`.venv`、`node_modules`、`dist`、`data/*.db` 等）；
  3. 使用 `npm ci` 与锁文件；后端固定版本或提供 constraints 锁；
  4. CI 中执行一次 compose 冒烟测试。
- **验证标准**：
  - 容器内 `curl http://localhost:5173/files/images/demo_sar.png` 返回 200；
  - 干净机器构建成功且镜像体积合理；
  - SAR 页面在容器环境中原图/处理图均可显示。

---

### P1-07 资源独占只在单次调度内生效；MS 只写不读、K 不参与决策

- **维度**：功能正确性（调度约束）、一致性
- **证据**：
  - `occupancy` 每次调度从空字典开始，只记录本次运行内已排计划：[order_scheduling.py](file:///e:/AIhub/StarOps/starops/backend/tools/order_scheduling.py#L527-L530)
  - 订单链只加载 `pending` 订单，已 `scheduled` 的占用不进入占用表：[order_nodes.py](file:///e:/AIhub/StarOps/starops/backend/agents/nodes/order_nodes.py#L45-L51)
  - `schedule_memory` 落库后没有任何读取/避坑逻辑；知识库 K 只做去重写入：[order_nodes.py](file:///e:/AIhub/StarOps/starops/backend/agents/nodes/order_nodes.py#L54-L93)、[order_nodes.py](file:///e:/AIhub/StarOps/starops/backend/agents/nodes/order_nodes.py#L140-L153)
- **可能触发场景**：勾选部分订单重复调度；不重置队列追加新订单；服务重启后继续调度。
- **潜在风险**：同一卫星同一时段生成两条互相冲突的计划；README「资源独占」「失败经验被记住」等承诺只在单次运行内成立。
- **优化建议**：
  1. 从持久化的已排计划（建议新增 `schedule_plan` 表或使用 `plan_commit`）初始化 `occupancy`；
  2. 决策前读取 MS/K，用于排除已知失败卫星/窗口或复用可行计划；
  3. 写入计划时在 DB 层做时间区间唯一/冲突约束。
- **验证标准**：
  - 两次运行（第二次不重置）不产生重叠观测；
  - 勾选子集调度时仍避开已排任务；
  - 同一订单第二次调度命中 K 或明确跳过；
  - 单测覆盖跨运行冲突。

---

### P2-01 审批与幂等的并发安全、去重与过期

- **维度**：正确性、并发、安全
- **证据**：审批「查询 pending → 更新」非原子：[approval.py](file:///e:/AIhub/StarOps/starops/backend/services/approval.py#L60-L98)；提交「查幂等键 → 插入」非原子，且 `idempotency_key` 有 UNIQUE 约束，竞态下可能抛 `IntegrityError` 变 500：[approval.py](file:///e:/AIhub/StarOps/starops/backend/services/approval.py#L165-L200)、[database.py](file:///e:/AIhub/StarOps/starops/backend/database.py#L90-L99)；无审批单去重/过期。
- **可能触发场景**：两个审批人同时点批准；前端重复提交批准；并发重试提交；同一方案多次运行生成多张 pending 审批单。
- **潜在风险**：后一次审批覆盖前一次；历史审批单无限期可用；审批中心堆积；并发提交报 500。
- **优化建议**：使用条件更新 `UPDATE ... WHERE status='pending'` 并判断 `rowcount`；对 `(target_type,target_id,plan_version)` 建唯一约束或复用 pending 单；增加 `expires_at`；幂等插入捕获唯一冲突后回读首次结果。
- **验证标准**：并发 20 次审批仅 1 次成功；并发同幂等键提交返回同一 `commit_id` 且无 500；过期审批不可用。

---

### P2-02 启动即破坏性重建模拟数据，产生孤儿知识/审计

- **维度**：数据完整性、可运维性
- **证据**：启动固定执行 `seed_all()`，默认 `reset_windows=True/reset_orders=True`：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L55-L59)、[data_generator.py](file:///e:/AIhub/StarOps/starops/backend/simulator/data_generator.py#L255-L265)；窗口全删重建，模拟订单全删重建，只清理引用不存在窗口的知识：[data_generator.py](file:///e:/AIhub/StarOps/starops/backend/simulator/data_generator.py#L83-L154)、[data_generator.py](file:///e:/AIhub/StarOps/starops/backend/simulator/data_generator.py#L157-L179)
- **可能触发场景**：每天重启服务；容器滚动重启；多副本同时启动；演示中途重启。
- **潜在风险**：订单状态与知识库/审计不一致；K 指向已删除订单；多副本同时 DELETE/INSERT 竞态；用户误以为数据已持久化。
- **优化建议**：启动仅做建表/迁移与「空表才播种」；提供显式 `/api/seed?reset=true` 或管理命令；为 K/MS 增加外键或启动一致性清理；多副本加启动锁。
- **验证标准**：重启后已排订单/知识库/审计不丢失；空库可自动播种；多进程同时启动无异常且数据一致。

---

### P2-03 SSE 健壮性：断连、心跳、队列与 final

- **维度**：稳定性、性能
- **证据**：三套 SSE 实现重复：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L169-L222)、[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L383-L398)、[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L516-L531)；循环遇 `error` 即 break，`final` 被丢弃；无 `request.is_disconnected()`；`asyncio.Queue` 无上限；后台线程为 daemon；前端无 AbortController/超时：[client.ts](file:///e:/AIhub/StarOps/starops/frontend/src/api/client.ts#L75-L127)
- **可能触发场景**：用户切页/刷新；代理 60s 超时；客户端断开；长耗时 LLM 调用；重复点击运行。
- **潜在风险**：任务继续占用 CPU/LLM 配额；队列堆积内存；前端卡在运行态；无法获得最终状态。
- **优化建议**：抽取统一 SSE 生成器；监听断连并设置取消事件；error 后仍发 final；加心跳注释行；限制并发运行数；前端 AbortController 与超时提示。
- **验证标准**：断开后后台任务在 N 秒内停止或标记取消；error 场景客户端收到 final；60s 空闲无代理超时；并发上限生效。

---

### P2-04 审计与任务追溯不完整

- **维度**：可追溯性、合规
- **证据**：`mission_task` 表只被查询、从未写入（全仓 grep 仅建表与 GET）：[database.py](file:///e:/AIhub/StarOps/starops/backend/database.py#L65-L74)、[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L279-L283)；工具输入/输出与审计详情截断 2000 字符：[database.py](file:///e:/AIhub/StarOps/starops/backend/database.py#L332-L376)；前端审批不传 reviewer，默认 `operator`：[ApprovalsView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/ApprovalsView.vue#L56-L68)；`/api/audit` 默认 50 条且无分页：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L257-L265)
- **可能触发场景**：需要复现一次决策、追责审批、导出审计；大快照（订单 20 条）被截断。
- **潜在风险**：无法证明「谁在何时批准了什么」；工具输出缺失导致无法复盘；审计查询只能看到最近记录。
- **优化建议**：写入任务生命周期（created/running/awaiting_approval/approved/rejected/executed）；审计存完整快照引用（对象存储/分表），摘要只用于列表；审计人来自身份；分页/时间范围过滤；保留期策略。
- **验证标准**：每次运行有 task 记录并可关联 trace/approval/commit；审计可完整还原载荷；审批人非固定默认值；分页可用。

---

### P2-05 SQLite 连接与并发写入

- **维度**：稳定性、性能
- **证据**：每次 CRUD 新建连接并关闭，未设置 `busy_timeout`，无连接池：[database.py](file:///e:/AIhub/StarOps/starops/backend/database.py#L197-L201)、[database.py](file:///e:/AIhub/StarOps/starops/backend/database.py#L227-L298)；SSE 在工作线程中写库。
- **可能触发场景**：多人同时运行规划/订单调度；长事务与上传/审批并发。
- **潜在风险**：`database is locked` 间歇性失败；连接开销随请求增长。当前为演示规模，尚未实测复现，列为风险。
- **优化建议**：设置 `timeout`/`PRAGMA busy_timeout`；写操作串行化或使用单写线程；必要时迁移 PostgreSQL；批量写入减少连接次数。
- **验证标准**：20 并发调度/审批压测无锁错误；P95 延迟满足目标；连接数受控。

---

### P2-06 前端加载与错误处理

- **维度**：易用性、稳定性
- **证据**：`onMounted(load)` / `refresh()` 无 try/catch：[ApprovalsView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/ApprovalsView.vue#L56-L70)、[AuditView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/AuditView.vue#L56-L64)、[OrderView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/OrderView.vue#L278-L284)、[PlanningView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/PlanningView.vue#L135-L141)；Workbench 审批后仅提示「可进入执行」，没有执行结果：[WorkbenchView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/WorkbenchView.vue#L291-L313)
- **可能触发场景**：后端未启动/重启、接口 500、网络抖动。
- **潜在风险**：白屏、控制台未处理 Promise、用户不知道发生了什么；审批后无后续反馈加剧 P0-02 的误解。
- **优化建议**：统一请求封装与错误提示；加载骨架/空态/重试；SSE 超时与取消；审批后轮询执行结果。
- **验证标准**：后端全下线时各页面显示可理解错误而非白屏；断网恢复可重试；审批后能看到执行状态。

---

### P2-07 前端可用性与兼容性

- **维度**：易用性、兼容性
- **证据**：各页面固定 2–3 列网格，无任何 `@media`：[WorkbenchView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/WorkbenchView.vue#L316-L322)、[OrderView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/OrderView.vue#L324-L357)、[SarView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/SarView.vue#L400-L404)；卫星地图坐标为随机数：[PlanningView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/PlanningView.vue#L98-L108)；时间直接截取 ISO 字符串显示 UTC：[OrderView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/OrderView.vue#L208-L209)；TIFF 被允许上传但用 `<img>` 预览：[SarView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/SarView.vue#L1-L9)、[SarView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/SarView.vue#L82-L100)
- **可能触发场景**：1366×768 笔记本、移动端、UTC+8 用户、上传 TIFF。
- **潜在风险**：内容挤压/横向滚动；用户误读时间；地图位置误导（随机）；TIFF 预览破图。
- **优化建议**：加断点与可折叠面板；时间按本地时区格式化并标注 UTC；卫星轨迹使用确定性/真实坐标或明确「示意」；TIFF 显示转换预览或提示不支持预览。
- **验证标准**：1280/1920/390 宽度无横向溢出；时间显示与本地一致；地图位置稳定；TIFF 上传有明确提示。

---

### P2-08 订单重置默认开启且范围过大

- **维度**：数据安全、易用性
- **证据**：`resetQueue` 默认 `true`：[OrderView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/OrderView.vue#L200-L203)；后端 `reset=true` 执行 `UPDATE satellite_order SET status='pending'`，不区分来源/是否已审批：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L472-L484)
- **可能触发场景**：用户直接点「运行订单调度」；只想重跑一个订单却勾选重置。
- **潜在风险**：已审批/已排期订单被批量回退；用户创建订单状态被覆盖；与审计记录不一致。
- **优化建议**：默认不重置；重置需二次确认并限定 `source='simulated'` 或仅选中订单；记录重置审计。
- **验证标准**：默认运行不改变已排订单状态；重置仅影响明确范围并留审计；UI 有确认提示。

---

### P2-09 接口 limit 与分页

- **维度**：输入校验、性能
- **证据**：`limit` 直接传入 SQL `LIMIT ?`，SQLite 中 `-1` 表示无限制：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L257-L283)、[database.py](file:///e:/AIhub/StarOps/starops/backend/database.py#L240-L265)；审批列表返回完整 `snapshot` 字符串：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L240-L246)
- **可能触发场景**：`/api/audit?limit=-1`；审批/知识库数据量增长。
- **潜在风险**：一次拉取全表造成内存/响应膨胀；敏感大快照暴露。实测 `limit=-1` 可用（当前仅 7 条，未造成压力）。
- **优化建议**：`limit` 限制 1–200；增加 offset/cursor 分页；列表不返回 snapshot，详情接口单独取。
- **验证标准**：非法 limit 返回 422 或被钳制；大量数据下响应大小稳定；分页可用。

---

### P2-10 `insert()` 返回主键的隐式假设

- **维度**：可维护性、回归风险
- **证据**：`insert` 返回 `str(next(iter(data.values())))`，依赖调用方把主键放在字典第一项：[database.py](file:///e:/AIhub/StarOps/starops/backend/database.py#L227-L237)
- **可能触发场景**：新增调用方调整字段顺序；重构数据字典；批量插入。
- **潜在风险**：静默返回错误 ID，审计/关联错乱，且测试不易发现。
- **优化建议**：显式传入 `pk` 字段名，或用 `RETURNING`/`lastrowid`；或在表定义中声明主键并由数据层校验。
- **验证标准**：任意字段顺序均返回真实主键；单测覆盖；调用方无需关心顺序。

---

### P2-11 生产路径测试覆盖不足与测试隔离

- **维度**：回归风险、测试质量
- **证据**：工作流单测主要调用 `run_builtin`（降级执行器），未覆盖实际 LangGraph 路由/状态合并：[test_workflow.py](file:///e:/AIhub/StarOps/starops/tests/unit/test_workflow.py#L17-L70)；各测试文件在导入前设置 `STAROPS_DB_PATH`，但 `config.settings` 是模块级单例，首个导入文件的路径会决定整个进程的 DB，后续设置不生效：[test_safety_loop.py](file:///e:/AIhub/StarOps/starops/tests/unit/test_safety_loop.py#L14-L26)、[conftest.py](file:///e:/AIhub/StarOps/starops/tests/conftest.py#L1-L8)
- **可能触发场景**：修改图定义/节点签名；调整测试顺序或并行执行（pytest-xdist）。
- **潜在风险**：测试全绿但生产路径回归；测试间数据互相污染、顺序依赖。
- **优化建议**：增加 `build_graph().invoke` 的集成测试；用 `monkeypatch` + fixture 在用例级重建配置/DB；启用 `pytest-xdist` 验证隔离；补 API/SSE/上传测试。
- **验证标准**：LangGraph 与 run_builtin 行为一致用例通过；随机顺序/并行运行稳定；覆盖率报告可生成。

---

### P2-12 前端包体与按需加载

- **维度**：性能
- **证据**：构建输出主包 `index-*.js` 1,278.71 kB、规划页 `PlanningView-*.js` 1,189.68 kB（gzip 约 800KB 合计），Vite 告警 chunk >500kB；全量注册 Element Plus 与全部图标：[main.ts](file:///e:/AIhub/StarOps/starops/frontend/src/main.ts#L1-L20)；规划页同时引入 echarts 与 leaflet：[PlanningView.vue](file:///e:/AIhub/StarOps/starops/frontend/src/views/PlanningView.vue#L43-L48)
- **可能触发场景**：首次访问、弱网、低配设备。
- **潜在风险**：首屏慢、演示现场卡顿。
- **优化建议**：Element Plus/图标按需引入；echarts 按需模块；规划页图表/地图懒加载；`manualChunks` 拆分 vendor。
- **验证标准**：首屏 JS gzip < 300KB；规划页异步加载；Lighthouse 性能分达标。

---

### P2-13 依赖、运行时与配置漂移

- **维度**：兼容性、可维护性
- **证据**：后端依赖仅 `>=` 无锁：[requirements.txt](file:///e:/AIhub/StarOps/starops/backend/requirements.txt#L1-L25)；README 要求 Python 3.13，Docker 基础镜像为 3.12：[README.md](file:///e:/AIhub/StarOps/starops/README.md#L32-L40)、[backend/Dockerfile](file:///e:/AIhub/StarOps/starops/backend/Dockerfile#L1)；使用已弃用的 `@app.on_event("startup")`：[main.py](file:///e:/AIhub/StarOps/starops/backend/main.py#L55-L59)；`debug/db_echo/max_iterations/llm_provider/checkpoint_dir` 未实际使用：[config.py](file:///e:/AIhub/StarOps/starops/backend/config.py#L16-L39)
- **可能触发场景**：半年后重新构建；升级 FastAPI/LangGraph；按 README 切换 Python 版本。
- **潜在风险**：构建随机失败；行为不兼容；配置项给人「可调」错觉。
- **优化建议**：生成并提交锁文件；统一 Python 3.12/3.13；改用 lifespan；删除死配置或在代码中落实。
- **验证标准**：干净环境可重复构建；CI 固定版本；启动无弃用告警；配置项均有行为或移除。

---

### P2-14 不可行根因与拒绝理由的完整性

- **维度**：可解释性
- **证据**：VS 对区域不匹配窗口直接 `continue`，不进入原因：[order_scheduling.py](file:///e:/AIhub/StarOps/starops/backend/tools/order_scheduling.py#L254-L267)；过期/可疑原因只在「无候选」时合并返回：[order_scheduling.py](file:///e:/AIhub/StarOps/starops/backend/tools/order_scheduling.py#L285-L287)；最终拒绝只取 `last_stage` 对应原因的最多 4 条：[order_scheduling.py](file:///e:/AIhub/StarOps/starops/backend/tools/order_scheduling.py#L737-L749)
- **可能触发场景**：订单因区域无窗口而失败；多阶段/多尝试失败。
- **潜在风险**：报告显示「无可用窗口」但真实根因是区域不匹配；用户误判为算法能力问题。
- **优化建议**：区分「不适用」与「违反」但都留可查询明细；按阶段聚合完整根因并给出首选根因；提供拒绝详情接口。
- **验证标准**：区域无窗口时原因明确包含区域；多因失败按阶段完整可见；单测覆盖。

---

### P2-15 版本基线与演示资产

- **维度**：回归风险、交付
- **证据**：`git rev-parse --show-toplevel` 为 `E:/AIhub`，当前 `starops` 整体为未跟踪（`git status` 显示 `?? ../`）；`.gitignore` 忽略 `data/images/` 与 `data/*.db`：[.gitignore](file:///e:/AIhub/StarOps/starops/.gitignore#L8-L11)，但 README 声明仓库自带 `data/images/demo_sar.png`
- **可能触发场景**：换机/协作/回滚；重新 clone；交付评审。
- **潜在风险**：无回归基线可 diff；演示影像在干净检出不存在的风险；测试/演示不可复现。
- **优化建议**：为 starops 建独立仓库并提交基线；将演示影像纳入版本控制（或用生成脚本重建）；数据目录仅忽略运行时输出。
- **验证标准**：干净 clone 后一条命令可启动并完成三条演示动线；`git diff` 可追踪变更。

---

### P3 级优化项

| ID | 问题 | 建议 | 验证 |
|---|---|---|---|
| P3-01 | `/api/health` 固定 ok，不检查 DB/磁盘/LLM；无请求 ID 与结构化日志 | 增加依赖健康检查、request_id、耗时日志 | 模拟 DB 不可用返回 unhealthy |
| P3-02 | 地图瓦片依赖外部 CDN；无 CSP/X-Frame-Options 等安全头 | 提供离线瓦片或降级图；nginx 加安全响应头 | 离线环境地图不空白；安全头存在 |
| P3-03 | 11–12px 字号较多、无键盘/读屏适配；审批无二次确认与意见输入 | 提升可读性、加 aria/键盘操作、审批确认框 | 无障碍扫描无严重项；审批可填意见 |
| P3-04 | 缺需求/验收/测试计划文档；README 部分能力声明与实现不符 | 补齐测试计划与验收矩阵，修正 README 口径 | 文档评审通过；声明可逐条验证 |

---

## 5. 维度评估小结

| 维度 | 结论 | 主要依据 |
|---|---|---|
| **功能正确性** | **不通过（需整改）** | 区域/时间约束被忽略；审批后无执行；订单/管线审批前落库；资源独占仅单次运行 |
| **稳定性** | 中等偏弱 | 无时区输入直接异常；LLM 失败拖垮整链；SSE 断连/无心跳；SQLite 并发风险 |
| **易用性** | 中等 | 三条链路可视化较好；但错误无反馈、非响应式、UTC 时间、随机地图、重置默认开启 |
| **性能** | 演示规模可用，前端偏重 | 后端 48 窗口/6 订单毫秒级；前端首屏包体约 800KB gzip；每次请求编译图 |
| **安全性** | **不通过（严重）** | 任意路径读写、无认证、通配 CORS、绝对路径泄露、上传/解码无上限、审批绑定缺失 |
| **兼容性** | 中等偏弱 | 本地 Windows 可跑；Docker `/files` 缺失；Python 3.12/3.13 不一致；TIFF 预览问题 |
| **回归风险** | 较高 | 无版本控制基线、生产 LangGraph 路径覆盖不足、测试共享 DB、无 CI/覆盖率门禁 |

---

## 6. 现有测试资产评价

**优点**
- 71 个单测全部通过，覆盖三阶段调度、CV、PM、MS/K、硬约束、幂等、PNG/滤波一致性等关键内核；
- 有跨路径一致性护栏（自研 PNG vs OpenCV、PNG vs TIFF、多窗口方案不丢失、模拟数据可重复）；
- 前端在 `strict` + `noUnusedLocals` 下可构建。

**缺口**
1. 无 API 层测试（路由、状态码、SSE 协议、上传、CORS、鉴权）；
2. 无并发/故障注入测试（LLM 超时、DB 锁、客户端断连）；
3. 无前端单测/E2E/可视化回归；
4. 无覆盖率门禁与 CI；
5. 生产执行路径（LangGraph）与降级路径覆盖不均；
6. 测试 DB 路径跨文件共享，隔离性不足。

**建议质量门禁**
- 后端：`pytest` + 覆盖率 ≥70%（核心 `services/tools` ≥85%）+ `ruff` + `mypy`；
- 前端：`vue-tsc` + ESLint + Vitest（组件）+ Playwright（三条演示动线）；
- 部署：compose 冒烟（health、SSE、`/files`、上传分析）；
- 安全：依赖扫描 + 路径/上传/鉴权专项用例。

---

## 7. 建议修复路线

1. **第 1 批（阻断）**：P0-01 路径白名单与输出目录；P0-02 审批后置执行与副作用后移；P0-03 审批绑定 task/plan hash。
2. **第 2 批（高）**：P1-01 区域/时间硬约束与缺参阻断；P1-02 鉴权/CORS/路径脱敏；P1-03 时间校验与 SSE final；P1-04 LLM 降级与 Schema 校验；P1-05 上传/解码上限；P1-06 Docker `/files` 与构建上下文。
3. **第 3 批（中）**：P1-07 跨运行资源占用、MS/K 读取；P2-02 启动不破坏数据；P2-03 SSE 取消/心跳；P2-04 任务与审计闭环；并发与限流；前端错误态/响应式/分包。
4. **第 4 批（工程化）**：CI、锁文件、覆盖率、测试隔离、版本基线、文档口径统一。

---

## 8. 需要补充的关键资料

以下资料缺失会限制结论从「代码/实测可证」扩展到「按业务验收」：

1. **需求规格与验收标准**：README 提到的 T01–T29 全量定义、期望结果、优先级；尤其「区域/时间约束是否为硬约束」「缺参是否必须阻断」。
2. **安全与合规基线**：是否需要登录/RBAC、部署边界（本机/内网/公网）、审计留存期限、数据分级、是否允许上传外部影像。
3. **非功能指标**：并发用户数/并发任务数、SSE 首事件与总时长目标、调度规模上限（8 星 48 窗口 6 单是否为最终规模）、单图尺寸/大小上限。
4. **部署目标**：Docker Compose 是否正式交付、是否需要 HTTPS/域名/反向代理、目标浏览器与分辨率矩阵。
5. **LLM 运行策略**：供应商配额、超时/重试/熔断策略、提示词与模型版本基线、离线模式的验收口径。
6. **「执行/落库」的业务定义**：审批通过后应执行什么、是否需要真实下发、失败如何回滚、幂等副作用范围。
7. **历史资料**：缺陷库、回归用例、CI 配置、代码版本基线（当前项目未纳入 git，无法建立可对比基线）。
8. **L2 深度模型接入计划**：权重来源、许可证、与 L1 的统一指标口径与验收阈值。

在上述资料补齐前，本报告中标注「风险/待验证」的并发、压测、浏览器兼容与 LLM 相关结论应视为待确认项，而非既定缺陷。

---

## 9. 附录：实测证据摘要

| 探针 | 输入/操作 | 结果 |
|---|---|---|
| 单测 | `pytest tests/unit -q` | 71 passed，14.27s |
| 前端构建 | `npm run build` | 成功 21.15s；主包 1.28MB、规划页 1.19MB，chunk 告警 |
| 缺区域规划 | 未给区域，分辨率优于 2m，未来 3 天 | 窗口覆盖 6 区，最终选中 **CN-SH**，方案无区域字段 |
| 显式时间 | 要求 2027-06-01~06-30 | 选中 2026-11-30 窗口，早于要求，`checked_rules` 仍含 `time_range` |
| 无时区订单 | `time_start="2027-03-01T00:00:00"` | `TypeError: can't compare offset-naive and offset-aware datetimes`；SSE 只有 error |
| 非法时间建单 | `time_start="not-a-date"` | HTTP 200，订单创建成功（未校验） |
| 审批绑定 | 批准 `plan-A` 后提交 `task-B` + 不同 payload | 提交成功，payload 为 `COMPLETELY-DIFFERENT` |
| 审批前落库 | 运行订单调度后查状态 | `pending -> scheduled`，审批仍 `pending`，知识库已写入 |
| 审批后执行 | 批准后查库 | `mission_task=0`，`plan_commit` 无新增 |
| CORS | `Origin: https://evil.example` 预检 | 回显该 Origin 且 `Allow-Credentials: true` |
| 任意路径 SAR | 分析项目外临时目录 PNG | 成功，并在该目录生成 `__lee/__median/__adaptive.png` |
| 路径泄露 | 上传后查 `/api/images` | 返回 `C:\Users\...\images\img-*.png` 绝对路径 |
| limit | `/api/audit?limit=-1` | 200，SQLite 语义为不限制（当前 7 条） |

> 以上运行时探针均使用独立临时 DB/影像目录，未对 `starops/data` 下现有数据做修改。
