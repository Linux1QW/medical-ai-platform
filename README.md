# 基于多智能体的医生临床问诊评估平台

面向医学教育、标准化问诊训练和教学复核的全栈 AI 平台。医生用户与虚拟患者完成多轮问诊、提交诊断和治疗方案；平台将评估请求可靠地投递到 Celery，由 LangGraph 多智能体工作流生成五维评分、证据引用、风险提示、改进建议和人工复核状态。V1.2 进一步加入默认关闭的 Interview Coach、三层记忆、Prompt/A-B 管理、语音 Beta 和数据反馈闭环。

> **医疗安全边界**：本项目只用于教学、训练、研究和质量改进，不提供真实医疗服务，也不能替代执业医师的诊断、处方、急救决策或人工审核。出现现实医疗问题或急危重症时，应立即转交合格医疗机构和专业人员。平台输出属于**辅助质量评估，不是自主诊疗结论**。

> **隐私边界**：不得向模型、日志、测试集、知识库或 Git 仓库写入可识别患者身份的信息、生产病历、密钥或密码。数据导入前必须取得合法授权，并完成最小化、去标识化和访问控制。对话、导出、备份、链路追踪和评测报告同样属于敏感数据。

## 1. 项目状态

| 项目 | 当前状态 |
|---|---|
| 功能迭代 | V1.2 Agent Intelligence 代码已实现 |
| 默认主流程 | 虚拟患者问诊 → 异步多智能体评估 → 报告 → 人工复核 |
| Interview Coach | opt-in，`COACH_ENABLED=false`，默认关闭 |
| Voice | optional beta，`VOICE_ENABLED=false`，无真实 LiveKit 验收证据，不属于已接受生产范围 |
| 本地质量验证 | 后端 pytest、Ruff、前端 test/lint/build 已在当前修复分支通过；详见“测试与验收” |
| 生产验收 | 必须以 `docs/release-evidence/v1.2/final-acceptance.json` 为唯一事实来源；当前证据仍为 `passed=false`，不能宣称生产发布已验收 |
| Git 状态 | 当前修复分支尚未合并到 `master` |

后端 API/OpenAPI 和健康端点的版本元数据已统一为 `1.2.0`。生产是否可发布仍由候选 SHA 对应的验收证据决定，不能只根据版本号判断。

## 2. 谁使用这个项目

| 角色 | 主要能力 | 数据边界 |
|---|---|---|
| 医生/受训者 | 注册登录、选择病例、模拟问诊、提交诊断、查看本人评估、使用已启用的 Coach/Voice、管理本人记忆授权 | 只能访问本人问诊、评估和导出数据 |
| 教师/管理员 | 管理病例、查看全局统计、处理人工复核、查看运行 Trace、管理知识库、Prompt、实验与模型版本 | 依赖 RBAC 权限；高敏操作写审计日志 |
| 运维/发布人员 | 数据库迁移、异步任务、索引发布、监控、告警、备份恢复、发布证据与回滚演练 | 不应通过运维权限绕过业务数据最小化原则 |
| 开发/测试人员 | 本地开发、自动化测试、RAG/Agent 评测、故障注入、E2E 和负载测试 | Fixture 必须是合成或去标识化数据 |

## 3. 核心能力

### 3.1 问诊训练

- React 19 + TypeScript + Ant Design 前端，支持注册登录、病例选择、问诊、诊断提交、结果查看、统计和管理员工作台。
- 虚拟患者由患者 Agent 驱动，包含对话记忆、信息披露策略、覆盖度、情绪/生理状态和合理症状工具。
- 问诊资源执行 owner/admin 访问控制；医生看不到病例标准答案和隐藏 Prompt。
- 普通响应与 SSE 流式响应并存；前端统一通过 `VITE_API_BASE_URL` 或同源 `/api/v1` 访问 API。

### 3.2 多智能体评估

- LangGraph 评估图包含 Safety、计划、并行评估、确定性评分、Reflection、复核门和建议生成。
- 五个专业维度覆盖问诊逻辑与效率、人文沟通、诊断、治疗和医学知识。
- 评分区分 `pass`、`partial`、`fail`、`unassessed`、`not_applicable`；未评估不等于零分。
- 高危信号、证据不足、维度缺失或一致性异常触发 fail-closed，并进入人工复核，而不是生成看似确定的结果。
- RunBudget、工具白名单、超时、重试、取消与 checkpoint 恢复限制失控调用和资源消耗。

### 3.3 RAG 知识检索

- 医学词法 tokenizer + `bm25s` 实现 BM25 词法召回。
- Chroma 提供 Dense 向量召回；BGE-M3 Sparse 是可选能力，默认依赖不安装、功能不启用。
- BM25、Dense、可选 Sparse 通过加权 RRF 融合，再执行分级检索（Base → MQE → HyDE）、两阶段 rerank、来源多样性和可选上下文压缩。
- 引用 ID 基于知识库版本、文档、分块和内容哈希确定性生成；诊断/治疗 claim 可关联到具体 evidence。
- 索引使用不可变 generation：构建候选 → 校验 manifest/Chroma/BM25/可选 Sparse → Redis CAS 切换 active generation → Pub/Sub 通知 Worker 热加载。

### 3.4 V1.2 Agent Intelligence

- **Interview Coach**：7 节点图 `intent → planner → evidence → draft → critic → finalize → persist`，通过 SSE 输出建议并支持幂等重放。
- **上下文隔离**：`CoachContextView` 禁止注入预期诊断等隐藏字段；Working Memory 校验来源并限制上下文预算。
- **三层记忆**：当前对话 Working Memory、历史问诊 Episodic Memory、需用户同意且管理员审批的 Trainee Profile Memory。
- **Skill/MCP 安全边界**：最小工具白名单、预算限制、非可信证据包装和控制指令清洗；MCP Demo 只使用 stdio 和静态去标识化 fixture。
- **Prompt 与实验**：PromptBundle 生命周期和确定性 ExperimentAssignment 支持版本化、激活与 A/B 分配。
- **数据反馈闭环**：Coach trace → eval → attribution candidate → 管理员复核/去标识化 → eligible for training。
- **LiveKit Voice Beta**：短期令牌、部分转录仅内存、最终转录去重、不保存原始录音；默认关闭且尚未完成真实环境验收。

## 4. 系统架构

```text
Browser
  │
  ▼
React / Nginx
  │ REST + SSE + WebSocket
  ▼
FastAPI ──────────────── MySQL 8
  │                         ├─ 用户 / 病例 / 问诊 / 消息
  │                         ├─ 评估 / Run / 节点结果 / Checkpoint
  │                         ├─ Outbox / 审计 / 人工复核
  │                         └─ Coach / Memory / Prompt / Experiment / Trace
  │
  ├─ redis-state（AOF + noeviction）
  │    ├─ LangGraph checkpoint
  │    ├─ Celery broker/result
  │    ├─ JWT 吊销、评估控制与进度
  │    └─ active RAG generation / Pub/Sub
  │
  ├─ redis-cache（allkeys-lru）
  │    ├─ LLM response cache
  │    └─ retrieval cache
  │
  ├─ Evaluation Dispatcher：Transactional Outbox → Celery
  ├─ Celery Worker：评估、RAG 索引、清理与对账任务
  ├─ Celery Beat：周期调度（必须保持单实例）
  └─ RAG：Chroma Dense + BM25 + optional Sparse → RRF / rerank
```

### 4.1 为什么使用 Outbox 和 Dispatcher

提交评估时，API 在同一个 MySQL 事务中写入 `evaluation_runs`、锁、审计和 `evaluation_dispatch_outbox`，但不直接调用 Celery。独立 Dispatcher 使用租约批量领取 Outbox 记录并投递任务。这样可以避免“数据库已提交但 Celery 消息丢失”或“消息已发出但业务事务回滚”的双写不一致。

投递语义是**至少一次**，不是恰好一次。Worker 依靠 run 状态机、幂等键、租约和 fencing 防止重复执行覆盖有效结果；Reconciliation 周期任务负责扫描过期租约、卡住的 dispatch 和重试状态。

### 4.2 一次完整问诊与评估的数据流

1. 用户通过 `/api/v1/auth/login` 获取 JWT，前端保存在 `sessionStorage`。
2. 用户选择脱敏虚拟患者并创建 Consultation。
3. 每轮提问写入 ConsultationMessage，患者 Agent 根据可见上下文生成响应。
4. 用户提交诊断/治疗方案并请求评估。
5. FastAPI 校验问诊归属，在单事务内创建 EvaluationRun、锁、审计和 Outbox。
6. Dispatcher 将 Outbox 投递至 Celery；Worker 领取 run 并进入 `running`。
7. LangGraph 执行安全门、并行专业 Agent、RAG、评分、Reflection 和建议节点。
8. 运行进度经 Redis Progress Bus 推送到 WebSocket；Redis 进度不可用时，客户端仍可查询数据库状态。
9. 结果写入 Evaluation、节点结果、Trace 和引用；run 进入 `completed`、`needs_review` 或失败/取消终态。
10. 管理员在复核工作台处理需要人工判断的结果，形成后续评测和数据治理反馈。

### 4.3 EvaluationRun 状态机

```text
queued → running → completed
             ├──→ needs_review → reviewed
             ├──→ retrying ────→ running
             ├──→ failed
             └──→ cancelled
```

数据库终态是权威状态，Redis 中的进度事件只能补充百分比和提示，不能把终态重新改成运行中。取消采用“DB/Redis 取消标志 + Celery revoke + Worker 协作轮询”，因此不是强制杀进程。

## 5. 技术栈

| 层 | 主要技术 |
|---|---|
| 前端 | React 19、TypeScript 5.9、Vite 7、Ant Design 6、Axios、Recharts、Vitest、Playwright |
| API | Python 3.10、FastAPI、Pydantic 2、SQLAlchemy 2 Async、Alembic |
| Agent | LangGraph、OpenAI-compatible Provider、DashScope/Qwen、外置 Prompt |
| 异步任务 | Celery 5.6、Redis broker/result、Transactional Outbox、Dispatcher |
| RAG | ChromaDB、`bm25s`、jieba、可选 FlagEmbedding/BGE-M3、PyMuPDF |
| 数据 | MySQL 8、Redis Stack 7 state（含 RedisJSON/RediSearch）与 Redis 7 cache 双实例 |
| 可观测性 | JSON 日志、Prometheus、Grafana、可选 Langfuse、HMAC 隐私遥测 |
| 工程质量 | pytest、mypy、Ruff、ESLint、Vitest、Playwright、Trivy、GitHub Actions |

依赖版本以 `backend/requirements.txt` 和 `frontend/package-lock.json` 为准，不要仅依据本表安装。

## 6. 目录结构

```text
medical-ai-platform/
├─ backend/
│  ├─ app/
│  │  ├─ api/v1/              # REST、SSE、WebSocket 路由
│  │  ├─ agent_runtime/       # V1.2 Coach 图、策略、Skill 与遥测
│  │  ├─ orchestration/       # 主评估 LangGraph、状态、节点和 checkpoint
│  │  ├─ services/            # 业务服务、Agent、RAG、LLM、工具和可观测性
│  │  ├─ tasks/               # Celery 任务、Dispatcher、对账和资源初始化
│  │  ├─ models/              # SQLAlchemy 模型
│  │  ├─ schemas/             # Pydantic API 契约
│  │  ├─ repositories/        # Coach/Prompt/Memory/Trace 持久化访问
│  │  ├─ prompts/             # 版本化系统 Prompt 与 manifest
│  │  └─ core/                # 配置、认证、权限、审计、日志和安全
│  ├─ alembic/versions/       # 数据库迁移链
│  ├─ scripts/                # 管理、迁移、评测、CI 与故障演练脚本
│  └─ tests/                  # 单元、API、集成、RAG、安全和 E2E 测试
├─ frontend/src/
│  ├─ pages/                  # 登录、问诊、评估、统计、复核等页面
│  ├─ components/             # 报告、Coach、Voice、Trace 等组件
│  ├─ api/                    # 按领域封装的 API 客户端
│  ├─ hooks/                  # SSE/WebSocket/评估任务 hooks
│  ├─ store/                  # 认证状态
│  └─ utils/                  # 请求、URL 与评分工具
├─ data/                      # 本地知识库源文件（禁止提交敏感材料）
├─ docs/                      # 总手册、专题、runbook、ADR 与验收证据
├─ monitoring/                # Prometheus/Grafana 配置
├─ nginx/                     # 前端反向代理配置
├─ .github/workflows/         # CI、RC、部署工作流
├─ docker-compose.yml         # 本地/基线 Compose
└─ docker-compose.prod.yml    # 生产覆盖配置
```

## 7. 主要页面与接口

### 7.1 前端路由

| 路由 | 页面 | 权限 |
|---|---|---|
| `/login`、`/register` | 登录与注册 | 公开 |
| `/dashboard` | 工作台 | 登录用户 |
| `/patients` | 虚拟患者列表 | 登录用户 |
| `/consultations` | 本人问诊列表 | 登录用户 |
| `/consultation/:id` | 问诊、Coach、Voice | owner/admin；功能还受 Feature Flag 控制 |
| `/evaluation/:id` | 评估状态、报告、证据和风险 | owner/admin |
| `/profile` | 个人资料与授权 | 登录用户 |
| `/stats` | 统计 | 页面内部按角色展示 |
| `/admin/consultations` | 全部问诊 | admin |
| `/admin/reviews` | 人工复核工作台 | admin |
| `/admin/patients` | 病例管理 | admin |

### 7.2 API 领域

所有业务 API 默认位于 `/api/v1`；OpenAPI JSON 位于 `/api/v1/openapi.json`，开发环境可在 `/docs` 查看交互文档。

| 前缀 | 用途 |
|---|---|
| `/auth` | 注册、登录、刷新、登出、当前用户和资料 |
| `/patients` | 脱敏病例查询及管理员 CRUD/导出 |
| `/consultations` | 创建、查询、消息、流式消息、延长、提交诊断、结束和删除 |
| `/evaluations` | 202 异步提交、run 状态、取消、锁状态、结果及 WebSocket |
| `/reviews` | owner/admin 状态查询、管理员待复核队列和决策提交 |
| `/knowledge-base` | 索引统计、全量重建、增删文档、任务状态和缓存清理 |
| `/admin` | 缓存、清理、Tool runtime、Trace、失败和用量汇总 |
| `/cases` | 病例推荐与难度 |
| `/model-versions` | 模型版本登记、查询、废弃和状态回滚 |
| `/coach` | 建议 SSE、状态、反馈和管理员 Trace |
| `/voice` | Voice Beta session 创建、状态和结束 |
| `/trainee-memory` | 记忆、审批和用户 consent |
| `/prompt-registry` | Prompt bundle、激活、实验和分配 |

健康与监控端点：`/health/live` 只表示进程存活；`/health/ready` 检查关键依赖并可返回 503；`/health` 返回较完整状态；`/metrics` 在生产环境必须使用 Bearer token 保护。

## 8. 本地启动

### 8.1 前置要求

- Python 3.10
- Node.js 18 或更高版本
- MySQL 8
- Redis Stack 7（state，必须包含 RedisJSON/RediSearch）和 Redis 7（cache）；完整拓扑应物理分离
- 可用的 OpenAI-compatible 或 DashScope/Qwen API 凭据（运行真实 Agent 时）

Windows 上的 Celery Worker 建议使用 `-P solo`。生产环境不要使用 solo pool。

### 8.2 配置环境变量

```powershell
cd backend
Copy-Item .env.example .env
```

至少检查以下配置：

- `MYSQL_*`：数据库连接；生产使用专用最小权限账户。
- `SECRET_KEY`：生产必须是随机长密钥，禁止使用示例值。
- `DASHSCOPE_API_KEY` 或通用 `LLM_*`：真实模型调用凭据。
- `REDIS_CHECKPOINT_URL`、`CELERY_BROKER_URL`、`CELERY_RESULT_BACKEND`、`PROGRESS_REDIS_URL`：必须指向 state Redis。
- `LLM_CACHE_REDIS_URL`、`RETRIEVAL_CACHE_REDIS_URL`：必须指向 cache Redis。
- `COACH_ENABLED`、`VOICE_ENABLED`：默认保持 `false`，满足依赖与验收条件后再启用。

权威字段见 `backend/app/core/config.py`，可复制模板见 `backend/.env.example`。

### 8.3 初始化数据库与后端

创建空数据库 `medical_ai`。**不要先执行 `database/init.sql` 再叠加 baseline 迁移**；Alembic 是当前 schema 的权威迁移路径。

```powershell
cd backend
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe scripts/init_admin.py
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 8.4 启动异步任务组件

至少另外打开三个终端：

```powershell
# 终端 1：Outbox Dispatcher
cd backend
.\.venv\Scripts\python.exe -m app.tasks.evaluation_dispatcher

# 终端 2：Celery Worker（Windows 本地）
cd backend
.\.venv\Scripts\python.exe -m celery -A app.celery_app worker --loglevel=info -P solo

# 终端 3：Celery Beat；全环境只能有一个 Beat 调度实例
cd backend
.\.venv\Scripts\python.exe -m celery -A app.celery_app beat --loglevel=info
```

只启动 FastAPI 可以浏览接口，但无法完成真实异步评估闭环。

### 8.5 启动前端

```powershell
cd frontend
npm ci
npm run dev
```

默认地址：

- 前端：<http://localhost:5173>
- Swagger UI：<http://localhost:8000/docs>
- 存活检查：<http://localhost:8000/health/live>
- 就绪检查：<http://localhost:8000/health/ready>

若前后端不同源，构建或启动前设置 `VITE_API_BASE_URL=http://localhost:8000/api/v1`，并同步后端 `CORS_ORIGINS`。

## 9. Docker Compose

Compose 包含 MySQL、双 Redis、迁移、FastAPI、Dispatcher、Celery Worker、Celery Beat、Nginx 前端和可选监控：

```powershell
# 构建并启动本地完整栈
docker compose up -d --build

# 查看状态和日志
docker compose ps
docker compose logs -f backend evaluation-dispatcher celery-worker celery-beat

# 启用 Prometheus/Grafana
docker compose --profile monitoring up -d

# 生产覆盖配置
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 停止服务；默认保留 volume 数据
docker compose down
```

`migrate` 服务成功执行 `alembic upgrade head` 后，backend/dispatcher/worker 才应启动。执行 `docker compose down -v` 会删除数据库和 Redis volume，属于破坏性操作，不应作为普通停止命令。

## 10. 关键配置与 Feature Flag

| 变量 | 默认值 | 作用与生产要求 |
|---|---|---|
| `ENVIRONMENT` | `development` | `staging/production` 会启用更严格的启动校验 |
| `SECRET_KEY` | 不安全示例值 | 生产必须替换 |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | 生产不得超过 60 |
| `JWT_BLACKLIST_FAIL_CLOSED` | `false` | staging/production 必须为 `true` |
| `LANGGRAPH_ENABLED` | `true` | 主评估图开关；启用时 checkpoint Redis 必须可用 |
| `ENABLE_TOOL_USE` | `true` | 知识 Agent 工具调用；同时受轮次、次数、时长和结果长度预算限制 |
| `COACH_ENABLED` | `false` | 启用前必须配置 `COACH_HMAC_KEY` 及生产依赖 |
| `VOICE_ENABLED` | `false` | 启用前配置 LiveKit URL/key/secret；当前仍是 NOT_ACCEPTED beta |
| `BGE_M3_ENABLED` | `false` | 可选 Sparse；需额外安装 FlagEmbedding 并做资源/质量验证 |
| `LANGFUSE_ENABLED` | `false` | staging/production 启用时要求 HMAC 密钥且禁止原始内容采集 |
| `METRICS_TOKEN` | 空 | 生产必须配置，否则 `/metrics` 拒绝访问 |

Redis 默认逻辑分工：checkpoint DB 0（RedisVL/RediSearch 要求）、Celery broker DB 4、result DB 5、progress DB 6、evaluation control DB 8；缓存 Redis 使用独立实例的 DB 0/1。数据库编号不能替代 state/cache 的物理隔离。

## 11. 数据、迁移与持久化

- MySQL 表由 SQLAlchemy 模型和 Alembic 迁移共同维护；当前迁移 head 为 `6f7a8b9c0d1e`。
- 核心实体包括 User、VirtualPatient、Consultation/Message、Evaluation、EvaluationRun/Lock/Checkpoint/NodeResult、Outbox、Review、Audit 和 ModelVersion。
- V1.2 新增 CoachSession/Decision/StreamEvent、TraineeMemory/Consent、PromptBundle、ExperimentAssignment 和 AgentTraceEvent。
- `data/` 是知识库源文件目录；RAG generation artifact 和 Chroma 数据必须使用共享持久卷，所有相关进程看到相同路径。
- 备份至少包含 MySQL、RAG artifact/Chroma、Prompt/配置版本和恢复所需密钥元数据；Redis cache 无需当作业务主数据，state Redis 的恢复策略必须按实际用途评估。
- 日志、导出和评测产物必须进入保留与销毁策略；不要把真实数据产物提交到仓库。

## 12. RAG 索引发布与验证

管理员可通过 API 获取统计、触发全量重建、添加/替换或删除文档，并查询 Celery 任务状态。索引任务不会直接原地覆盖 active collection，而是发布新的不可变 generation。

发布前至少验证：

1. manifest 的 generation、文档数、哈希、tokenizer/parser/chunker 版本一致；
2. Chroma、BM25 和启用的 Sparse artifact 均可加载；
3. 质量指标未低于基线，安全/一致性计数为零；
4. CAS 切换没有覆盖另一个并发发布者的新 generation；
5. Worker 可通过持久指针自愈，不能只依赖 Pub/Sub 通知；
6. 检索缓存 key 包含 generation，切换后不存在旧 generation 污染。

当前没有面向运维人员的正式 RAG generation 回滚 REST/CLI。紧急恢复必须遵循 runbook 并保留审计证据，不能把模型版本登记接口的 rollback 误认为 RAG 索引回滚。

## 13. 安全模型

- JWT 包含 `jti`，登出写入 Redis 黑名单；生产强制 fail-closed，吊销存储不可用时拒绝请求。
- 路由使用 role/permission 与资源归属双重校验，避免只“已登录”却可读取他人对象的 IDOR。
- WebSocket 首帧传递 JWT，不把 token 放在 URL 和访问日志中。
- API 统一加入 request ID、结构化错误码和安全响应头；生产启用 HSTS。
- Coach 不接收隐藏病例字段，工具结果按非可信证据处理；记忆 consent 默认关闭并执行 PHI 拒绝规则。
- `/metrics`、Trace、Prompt、实验、知识库和复核均属于受保护管理面。
- 示例管理员、数据库密码、API Key、LiveKit secret、HMAC key 不能进入生产镜像、日志和 Git 历史。

这仍是教学/研究软件，不代表通过医疗器械、等保、HIPAA、GDPR 或任何特定地区合规认证。

## 14. 测试与验收

### 14.1 本地质量检查

```powershell
# 后端
cd backend
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy app
.\.venv\Scripts\python.exe -m compileall -q app
.\.venv\Scripts\python.exe -m pytest tests -q

# 前端
cd ..\frontend
npm run lint
npm test
npm run build
```

按风险还应执行迁移、RAG、安全、API、E2E、Playwright、故障矩阵和负载测试。CI 工作流位于 `.github/workflows/ci.yml`；V1.2 RC 实测工作流位于 `.github/workflows/v12-rc.yml`。

### 14.2 如何解读“通过”

- **单元/集成测试通过**：说明当前自动化合约未发现回归，不代表真实 LLM、LiveKit、MySQL/Redis 故障或生产负载已经验证。
- **代码完整**：表示实现和测试存在，不等于 Feature Flag 应立即打开。
- **生产验收通过**：必须由受保护 RC 工作流在候选 SHA 上实测，并写入测试数、mypy/Ruff、迁移、E2E、负载、回滚和人工审批证据。
- `docs/release-evidence/v1.2/final-acceptance.json` 是机器可读的唯一验收事实源；当前文件为 `passed=false`，因此 README 不宣称 V1.2 已获生产验收。

## 15. 部署与运维要点

生产部署前必须完成：

- 使用空库执行 Alembic upgrade，并对已有库做结构 diff、备份和升级/恢复演练；
- 配置强随机密钥、专用数据库账户、双 Redis、HTTPS、最小 CORS 和 metrics token；
- 确认 Dispatcher 与 Beat 各单实例，Worker/API 可按共享依赖容量扩展；
- 校验 Outbox backlog、stale lease、failed/retrying run、Celery queue、Redis memory、LLM 错误率和 RAG generation；
- 执行真实 E2E、故障矩阵、负载测试、备份恢复和回滚演练；
- 由代码、QA、安全负责人完成候选 SHA 审批并封存验收 bundle。

故障处理和回滚命令不要从 README 临时拼接，请使用 [V1.2 Runbook](docs/runbooks/production-v1.2.md)。

## 16. 当前已知限制

1. `database/init.sql` 与当前 ORM/Alembic schema 不完整等价，新环境必须以 Alembic 为准。
2. 生产验收 bundle 当前为 `passed=false`；真实 LLM 性能、负载、审批和真实部署 E2E 仍需受保护 RC 工作流生成。
3. Voice 是默认关闭的 Beta，缺少真实 LiveKit 房间、音频链路和转录持久化验收证据。
4. BGE-M3、OCR 和部分 RAG 增强默认关闭，额外依赖不随基础 requirements 安装。
5. 没有正式的 RAG generation 回滚 REST/CLI；模型版本 rollback 只操作模型版本登记状态。
6. 知识库、模型版本、监控等管理能力以 API 为主，前端管理 UI 尚未覆盖全部运维操作。
7. ChromaDB 使用较大的 `hnsw:sync_threshold` 规避跨进程段加载问题；旧 collection 需重建才能继承新 metadata。
8. Dispatcher 和 Beat 必须各自保持单实例；跨主机部署需要共享 MySQL、Redis 和 RAG artifact，并重新做故障与一致性验证。

## 17. 文档导航

- [文档中心](docs/README.md)：按新成员、开发、测试、运维、管理员等角色导航，并说明文档优先级。
- [PROJECT_GUIDE](docs/PROJECT_GUIDE.md)：配置、API、数据模型、RAG/Celery、测试、运维与限制的源码级总手册。
- [技术架构说明](docs/technical-document.md)：状态机、Outbox/Dispatcher、双 Redis、LangGraph、RAG generation 和数据保留。
- [平台操作说明](docs/platform-documentation.md)：医生、教师/管理员和一线运维操作流程。
- [V1.2 Production Guide](docs/release-evidence/v1.2/production-v1.2.md)：V1.2 API、安全和生产行为。
- [V1.2 Runbook](docs/runbooks/production-v1.2.md)：部署、故障处理和回滚流程。
- [V1.2 Acceptance Summary](docs/release-evidence/v1.2/acceptance-summary.md)：由验收 bundle 生成的摘要，不应手工美化状态。
- [评测基线](docs/evaluation-baseline.md)：五维评分语义、图节点和人工反馈闭环。
- [Prompt 与 Provider](docs/prompt-and-provider-adapter.md)：Prompt 文件、版本覆盖和 LLM Provider 适配。
- [贡献指南](CONTRIBUTING.md) 与 [变更记录](CHANGELOG.md)。

## 18. 贡献与文档维护

任何涉及路由、schema、默认值、环境变量、Compose、迁移、Agent 图、RAG generation、验收门禁或前端操作的修改，都必须同步文档。文档中的数字必须来自代码常量、配置或可追溯的当次实测；Mock、测试合约和离线模拟不能包装成真实生产指标。

提交前建议运行 pre-commit 和与改动范围匹配的测试。不要提交 `.env`、API Key、数据库 dump、真实病历、原始录音、未脱敏 Trace 或本地大模型文件。
