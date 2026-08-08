# 项目总手册：基于多智能体的医生临床问诊评估平台

本文是项目的权威说明，事实基线为分支 `codex/rag-bm25-optimization` 的当前代码。内容由代码、配置、FastAPI 路由、Docker Compose、SQLAlchemy/Alembic、前端调用、测试和 CI 交叉核对。其他文档如与本文冲突，应回到对应源码确认并同步本文。

## 1. 项目定位、安全与边界

平台服务于医学教育、虚拟病例问诊训练、结构化评估和教学复核。医生用户选择虚拟患者并完成对话、诊断和治疗方案；系统保存全过程，通过多智能体与知识库生成五维分析、分数、引用、建议和复核状态；管理员维护病例并查看全局统计、运行追踪、知识库和人工复核数据。

平台不是医疗器械或临床决策系统：

- 输出不能替代真实诊断、处方、急救决策、会诊或专业审核。
- Safety Agent 只是教学工作流的风险门控，不保证发现全部急危重症。
- LLM、Embedding、Rerank 和 OCR 均可能失败、幻觉、遗漏或受第三方服务变化影响。
- 真实医疗问题必须转交合格医务人员；急危重症必须立即联系当地急救和医疗机构。

隐私与数据治理要求：

- 不采集或提交不必要的真实身份、联系方式、证件号、生产病历和密钥。
- 导入数据前取得合法授权，完成最小化、去标识化、用途限制和访问控制。
- 对话、Prompt、知识库、日志、审计、缓存、导出、备份、trace 和评测报告都可能包含敏感信息，必须按同一数据等级管理。
- 普通医生接口会隐藏虚拟患者标准答案和系统 Prompt，并对姓名脱敏；管理员仍可访问完整病例，因此管理员账户必须最小化分配。
- `database/seed.sql` 的 `admin/admin123` 只用于隔离开发，生产必须删除或立即改密。

## 2. 产品角色与端到端流程

### 2.1 角色

数据库只定义 `doctor` 和 `admin` 两种角色。用户还可在 `users.permissions` 中设置自定义权限列表；一旦非空，它会覆盖角色默认权限。

- `doctor`：查看脱敏病例、创建和访问自己的问诊、触发/取消自己的评估、查看自己的统计和导出自己的数据。
- `admin`：继承默认的评估、问诊、病例、用户、系统和模型管理权限；可查看全平台问诊、维护完整病例、操作知识库、查看监控并提交人工复核。

默认权限来自 `backend/app/core/permissions.py`。部分路由只检查角色，部分检查细粒度权限，不能仅凭前端菜单判断后端授权。

### 2.2 医生流程

1. 注册或登录，前端把 access token 和用户信息存入 `sessionStorage`。
2. 在“虚拟患者”中筛选病例；普通医生看到姓名脱敏后的病例，不会收到 `expected_diagnosis` 和 `system_prompt`。
3. 创建问诊。默认 `consultation_type=initial`、`max_rounds=20`；每次医生消息产生医生与患者两条消息，可使用普通 JSON 或 SSE 流式回复。
4. 达到轮次限制时可调用延长接口，每次增加 10 轮。
5. 提交诊断和治疗方案会结束问诊；也可直接结束问诊。
6. 触发评估。非测试模式下任务应交给 Celery；前端同时使用 WebSocket 和锁状态轮询显示进度。
7. 查看五个维度、总分、分析、建议、引用和复核提示。缺少维度或风险/一致性门控失败时，总分可为 `null`，状态可进入 `needs_review`。

### 2.3 管理员流程

- 创建、编辑、删除和导出虚拟患者；查看全平台问诊与统计。
- 通过 API 查看待复核项、提交复核意见和可选评分调整。
- 将仓库根目录 `data/` 中的 PDF/DOCX 异步添加、替换、删除或全量重建为新 RAG generation。
- 查看缓存、Tool runtime、run trace、失败归因和 Token 用量。
- 管理模型版本注册记录。注意模型版本注册表与实际 `settings`/Provider 路由不是自动绑定的部署系统。

## 3. 系统架构

```text
React SPA (5173 / Nginx 80,443)
        │ REST / SSE / WebSocket
        ▼
FastAPI (8000)
  ├─ Auth/RBAC/Audit/Rate limit/Security headers
  ├─ SQLAlchemy async ───────────── MySQL 8
  ├─ LangGraph orchestration ────── Redis checkpoint db=1
  ├─ cache/JWT/token/RAG pointers ─ Redis db=2/3 + shared instance
  ├─ task submission ────────────── Celery broker db=4/result db=5
  └─ RAG retrieval
       ├─ Chroma Dense: backend/data/medical_kb
       ├─ BM25 artifacts: backend/data/rag_indexes/<generation>/bm25
       ├─ optional Sparse: .../<generation>/sparse
       └─ manifest + Redis active pointer + Pub/Sub switch

Celery Worker
  ├─ run_evaluation
  ├─ rebuild/add/replace/delete_rag_index
  └─ per-process generation switch listener

Celery Beat ── daily cleanup_expired_records
Prometheus/Grafana ── optional monitoring profile
```

FastAPI lifespan 会执行 `Base.metadata.create_all`、安全检查、Agent adapter 注册、Redis checkpointer 初始化和可选 Tool 健康探测。`LANGGRAPH_ENABLED=true` 且 Redis checkpointer 初始化失败时，服务会拒绝启动；若要显式使用旧评估路径，应设置 `LANGGRAPH_ENABLED=false`。

## 4. 目录与模块职责

| 路径 | 责任 |
|---|---|
| `backend/app/main.py` | FastAPI 生命周期、中间件、异常处理、`/health`、`/metrics` |
| `backend/app/api/v1/` | REST/WS 路由；总前缀 `/api/v1` |
| `backend/app/core/` | 配置、JWT、依赖、权限、访问控制、审计、脱敏、日志、限流 |
| `backend/app/models/`、`schemas/` | SQLAlchemy 模型和 Pydantic API 契约 |
| `backend/app/services/agents/` | 患者 Agent、五维评估 Agent、Safety、Reflection、Suggestion |
| `backend/app/orchestration/` | LangGraph state、route plan、graph、adapter、checkpointer |
| `backend/app/services/rag/` | 检索、索引、artifact、generation、cache、rerank、OCR |
| `backend/app/tasks/`、`celery_app.py` | 评估、索引、清理任务和 Worker 生命周期 |
| `backend/evaluation/`、`backend/scripts/eval/` | Gold cases、指标、报告、A/B、BM25 评测和调参 |
| `backend/alembic/` | 权威迁移链 |
| `database/` | Compose 初始化 SQL 和演示数据；不等价于完整迁移链 |
| `frontend/src/api/`、`pages/`、`components/` | API 封装、页面和展示组件 |
| `docker-compose*.yml`、`Dockerfile.*` | 容器编排、开发/预发/生产覆盖 |
| `monitoring/` | Prometheus 配置和 Grafana provisioning/dashboard |
| `.github/workflows/` | CI 与手动部署 |

## 5. 技术栈

- 后端镜像/CI：Python 3.10；FastAPI 0.115.6、Uvicorn 0.34.0、Pydantic 2.10.4。
- 数据：SQLAlchemy 2.0.36、Alembic 1.18.5、aiomysql/pymysql、MySQL 8.0。
- 编排与队列：LangGraph 1.2.6、Redis 6.4 Python client、Redis 7 server、Celery 5.6.3。
- RAG：ChromaDB 1.5.7、PyMuPDF、jieba、bm25s 0.3.9；可选 FlagEmbedding/BGE-M3 未写入默认 requirements。
- LLM：OpenAI-compatible client、DashScope；默认 Provider adapter 为 `openai_compatible`。
- 前端：Node 18 构建；React 19.2、TypeScript 5.9、Vite 7.3、Ant Design 6.3、Axios、Recharts。
- 可观测性：Prometheus client、可选 Langfuse 2.60.10、Prometheus 2.53、Grafana 11.1。

精确锁定版本见 `backend/requirements.txt` 和 `frontend/package.json`。

## 6. 配置与默认值

后端在**当前工作目录**读取 `.env`；因此本地命令应从 `backend/` 执行并使用 `backend/.env`。权威字段是 `backend/app/core/config.py`，模板是 `backend/.env.example`。环境变量优先于模板和代码默认值。

### 6.1 基础、认证与数据

| 变量 | 代码默认值 | 说明 |
|---|---:|---|
| `ENVIRONMENT` | `development` | `production` 时默认 SECRET_KEY 会阻止启动，`staging` 没有独立安全分支 |
| `API_V1_PREFIX` | `/api/v1` | OpenAPI 为 `/api/v1/openapi.json` |
| `SECRET_KEY` | `change-this-to-a-secure-random-string` | 生产必须替换 |
| `ALGORITHM` | `HS256` | JWT 算法 |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` | access token 24 小时 |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `7` | refresh token |
| `JWT_TOKEN_BLACKLIST_ENABLED` | `true` | 黑名单 Redis 不可用时当前实现会 fail open |
| `MYSQL_HOST/PORT/USER/PASSWORD/DATABASE` | `localhost/3306/root/空/medical_ai` | 应用账户应最小授权 |
| `DB_POOL_SIZE/MAX_OVERFLOW/RECYCLE/TIMEOUT` | `20/10/3600/30` | SQLAlchemy 连接池 |
| `CORS_ORIGINS` | `localhost:5173`、`localhost:3000` | JSON 数组格式 |

注册密码允许 6–128 字符；密码先做 SHA-256 归一化再由 passlib bcrypt_sha256/bcrypt 存储。登录和注册各限流 5 次/分钟。

### 6.2 LLM、LangGraph 与任务

| 变量 | 代码默认值 | 说明 |
|---|---:|---|
| `QWEN_API_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI 兼容地址 |
| `QWEN_MODEL` | `qwen3.7-plus` | `backend/.env.example` 和 Compose 显式给出 `qwen3.7-max`，以实际环境值为准 |
| `LLM_PROVIDER_TYPE` | `openai_compatible` | `LLM_*` 为空时回退 `QWEN_*` |
| `LLM_MAX_CONCURRENT` / `LLM_SEMAPHORE_TIMEOUT` | `10` / `60` 秒 | 全局 LLM 并发 |
| `LANGGRAPH_ENABLED` / `LANGGRAPH_SHADOW_MODE` | `true` / `false` | Shadow 配置存在，返回仍以实现路径为准 |
| `LANGGRAPH_GRAPH_VERSION` | `evaluation-graph-v1` | 写入运行/评估记录 |
| `REDIS_CHECKPOINT_URL` / `REDIS_CHECKPOINT_TTL` | `redis://localhost:6379/1` / `86400` | Checkpoint |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | `redis://localhost:6379/4` / `redis://localhost:6379/5` | API 与 Worker 必须一致 |
| `EVALUATION_RUN_TIMEOUT_SECONDS` | `240` | 应小于 Celery soft limit 300 秒 |
| `AGENT_TIMEOUT_SECONDS` | `180` | 旧编排单 Agent 超时 |
| `EVAL_RESUME_FROM_CHECKPOINT` | `true` | Celery 重试尝试复用失败 run checkpoint |
| `EVAL_CANCEL_POLL_SECONDS` | `2` | 协作式取消轮询 |

Celery 全局 hard/soft time limit 为 600/300 秒，`worker_prefetch_multiplier=1`。评估任务最多重试 2 次，只重试网络/连接/超时类异常，退避 30 秒、60 秒。Beat 每 86400 秒调度一次留存清理。

### 6.3 RAG 与缓存

| 变量 | 默认值 |
|---|---:|
| `ACTIVE_INDEX_VERSION` | `rag-v1`（仅启动/旧兼容回退；集群 active 以 Redis 指针为准） |
| `RAG_LEGACY_COLLECTION_FALLBACK` | `false` |
| `BM25_K1/BM25_B/BM25_METHOD` | `1.2/0.8/lucene` |
| `BM25_TOKENIZER_VERSION` | `medical-lexical-v3` |
| `BM25_ENABLE_CJK_BIGRAM` | `false` |
| `BM25_HEADING_BOOST/BM25_ENTITY_BOOST` | `2/3`，均限制 1–3 |
| `RRF_K` | `35` |
| `RRF_WEIGHT_BM25/DENSE/SPARSE` | `0.30/0.45/0.25` |
| `BGE_M3_ENABLED` | `false` |
| `RERANK_MODEL` | `gte-rerank` |
| `ENABLE_METADATA_FILTER`、`ENABLE_DIVERSITY_RERANK` | `false`、`false` |
| `ENABLE_CONTEXT_EXPANSION`、`ENABLE_CONTEXT_COMPRESSION` | `false`、`false` |
| `ENABLE_OCR` | `false` |
| `RETRIEVAL_CACHE_ENABLED/TTL/MAX_SIZE` | `true/86400/5000` |
| `LLM_CACHE_ENABLED/TTL/MAX_SIZE` | `true/86400/10000` |

固定实现值：Embedding 模型 `qwen3.7-text-embedding`、维度 1024；分块大小 500、重叠 100；Embedding LRU 是每进程内存缓存，最大 1000 条。检索缓存 key 含 generation、query shape、top-k 和检索设置，缓存体不持久化正文，命中时从指定 generation 的 Chroma 回填并丢弃 stale candidate。

### 6.4 Tool、监控与留存

代码默认 `ENABLE_TOOL_USE=true`、`ENABLE_PATIENT_TOOL_USE=true`、`TOOL_EXECUTOR_HARDENED=true`、`TOOL_BUDGET_MANAGER_ENABLED=true`、`TOOL_HEALTH_CHECK_ENABLED=false`。基础 Compose 对 FastAPI 显式设置 `ENABLE_TOOL_USE=false`，所以容器默认与代码默认不同。

`AUDIT_LOG_RETENTION_DAYS=90`、`EVALUATION_RUN_RETENTION_DAYS=180`、`TOKEN_DAILY_LIMIT=1000000`、`COST_PER_1K_TOKENS=0.02`、`LANGFUSE_ENABLED=false`、`METRICS_TOKEN` 为空。生产环境若 `METRICS_TOKEN` 为空，`/metrics` 返回 403；非空时必须携带精确的 Bearer token。

## 7. 启动与部署

### 7.1 本地进程：推荐的可控路径

前置：Python 3.10、Node 18、MySQL 8、Redis 7。先创建空数据库，不要先运行 `database/init.sql`。

```powershell
cd backend
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

另开终端：

```powershell
cd backend
.\.venv\Scripts\python.exe -m celery -A app.celery_app worker --loglevel=info -P solo
.\.venv\Scripts\python.exe -m celery -A app.celery_app beat --loglevel=info
```

Linux Worker 可使用默认 prefork；基础 Compose 使用 `--concurrency=2`。Beat 必须保持单实例。每个 fork Worker 在 `worker_process_init` 启动 `rag:index-switched` 监听器。

前端：

```powershell
cd frontend
npm ci
npm run dev
```

Vite 固定代理 `/api` 到 `http://localhost:8000`；Axios 固定 `baseURL=/api/v1`。当前 `VITE_API_BASE_URL` 虽在 Docker 构建/override 中出现，但 `frontend/src/utils/request.ts` 未读取它。

### 7.2 Compose 服务与端口

`docker-compose.yml` 定义：MySQL 3306、Redis 6379、FastAPI 8000、Nginx 80/443、Celery Worker、Celery Beat；monitoring profile 增加 Prometheus 9090 和 Grafana 3000。端口均可由同名 Compose 变量覆盖。

```powershell
docker compose up -d
docker compose ps
docker compose logs -f backend celery-worker
docker compose --profile monitoring up -d
```

当前 Compose 必须在部署前修正或外部覆盖：

1. `backend` 没有注入 `CELERY_BROKER_URL=redis://redis:6379/4` 和 `CELERY_RESULT_BACKEND=redis://redis:6379/5`，API `.delay()` 会使用代码默认的容器内 `localhost`。
2. `builder.PDF_DIR` 在容器中解析为 `/app/data`；Compose 当前把 `./data` 挂到 `/app/backend/data/medical_pdfs`。RAG Worker 需要可读的 `/app/data`，且 FastAPI 与所有 Worker 必须共享同一 source、Chroma 和 artifact 存储。
3. `database/init.sql` 不是当前 ORM/Alembic 的完整等价物，见下一节。
4. `container_name` 会限制 `docker compose --scale celery-worker=N`；多容器 Worker 部署前应调整编排，但 Beat 仍只能一个。

`docker-compose.prod.yml` 和 `staging.yml` 使用 GHCR 镜像并将 Uvicorn workers 改为 4/2；`REPO` 必须配置。部署工作流仅执行 SSH pull/up，不自动执行 Alembic，因此迁移必须成为发布前明确步骤。

### 7.3 TLS、备份和持久化

前端容器检测 `certs/server.crt` 与 `certs/server.key`；存在时启用 443 并把 80 重定向到 HTTPS，否则仅 HTTP。生产必须使用可信证书、外部密钥管理和定期恢复演练。

Compose 命名卷：`mysql_data`、`redis_data`、`chroma_data`、`prometheus_data`、`grafana_data`。RAG generation 的 Chroma、BM25/Sparse artifact 必须一起备份；只备份一个组件无法恢复可验证 generation。

## 8. 数据库与迁移

### 8.1 权威迁移链

当前 Alembic head 为 `1a2b3c4d5e6f`：

- `0c1dfb4fea5f`：当前模型 baseline，创建 audit、checkpoint、model version、review、user、patient、consultation、lock、run、evaluation、node result 等表。
- `1a2b3c4d5e6f`：增加 `virtual_patients.case_id` 唯一索引，以及 `(consultation_id, sequence)` 消息唯一约束。

空数据库从 `backend/` 执行：

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic current
```

升级已有数据库前先备份、在副本执行 `alembic current/history/upgrade head` 并运行应用测试。不要在不了解结构时 `stamp head`，因为 stamp 只写版本号，不补表或列。

### 8.2 SQL 初始化的当前差异

`database/init.sql` 创建 7 张主表，但当前模型还需要 `audit_logs`、`evaluation_checkpoints`、`model_versions`、`review_records`、`evaluation_locks`，且 SQL 中缺少至少 `users.permissions`、`consultations.max_rounds`、`consultations.memory_state`、`evaluations.review_completed_by` 和 `evaluations.review_completed_at`。应用启动的 `create_all` 只会创建缺失表，不会给既有表补列。

因此：

- 新环境优先使用空库 + Alembic；如需演示数据，在确认 schema 后单独审阅并执行 `database/seed.sql`。
- Compose 的 MySQL entrypoint 会在空卷首次启动时自动执行 `init.sql` 和 `seed.sql`，当前不能视为已完成 Alembic 初始化。
- 不能把 `init.sql` 后直接 `alembic upgrade head` 当作安全流程：baseline 会尝试创建已有表，后续迁移也会重复添加 `case_id`/约束。

### 8.3 数据模型

| 表/模型 | 用途 |
|---|---|
| `users` | 账户、角色、自定义权限、资料和密码哈希 |
| `virtual_patients` | 稳定 `case_id`、人格、主诉、病史、标准答案和患者 Prompt |
| `consultations` | 医生/患者关联、状态、诊断、治疗、类型、轮次、患者 memory state |
| `consultation_messages` | 医生/患者消息和会话内唯一 sequence |
| `evaluations` | 五维结果、总分、引用、RAG trace、Safety、复核和版本信息 |
| `evaluation_locks` | 防重复评估、状态、run_id、心跳和过期时间 |
| `evaluation_runs` | graph run、计划、选择的 Agent、执行结果、错误和 attempt |
| `evaluation_node_results` | 节点级状态、耗时、脱敏摘要和错误 |
| `evaluation_checkpoints` | 旧/兼容数据库 checkpoint 模型；当前主 checkpointer 为 Redis |
| `review_records` | 人工复核意见与评分调整 |
| `model_versions` | 模型版本登记状态，不自动改变 Provider 配置 |
| `audit_logs` | 用户、动作、资源、来源请求信息和脱敏详情 |

问诊状态主要为 `in_progress/completed/evaluated`；评估涉及 `pending/running/completed/needs_review/reviewed/failed` 等上下文状态，调用方应按具体响应字段处理，不要把所有状态混成一个枚举。

## 9. 前端实现与操作界面

React Router 路由：

| 页面 | 路径 | 访问 |
|---|---|---|
| 登录、注册 | `/login`、`/register` | 公共 |
| 工作台、病例、我的问诊、个人资料 | `/dashboard`、`/patients`、`/consultations`、`/profile` | 登录 |
| 问诊、评估 | `/consultation/:id`、`/evaluation/:id` | 登录；后端再校验归属 |
| 数据统计 | `/stats` | 登录；后端对医生返回本人、管理员返回全局 |
| 全部问诊、患者管理 | `/admin/consultations`、`/admin/patients` | 前端 AdminRoute + 后端管理员检查 |

`AdminReviews` 页面源码存在，但没有挂到 `App.tsx` 路由或菜单。知识库、模型版本和运行监控也没有完整管理 UI，需使用 OpenAPI/curl。前端 token 存于 sessionStorage，不是 HttpOnly cookie；生产必须防范 XSS，避免在前端记录 token。

普通 Axios 请求超时 60 秒，包含 `/evaluation`、`/evaluate` 或 `/reports` 的请求为 300 秒。问诊消息支持 SSE；评估进度使用 WebSocket。

## 10. 认证、权限与通用协议

- REST 使用 `Authorization: Bearer <access_token>`。
- WebSocket 连接 `/api/v1/evaluations/ws/{consultation_id}` 后，必须在 5 秒内发送 `{"type":"auth","token":"<JWT>"}`；成功收到 `{"type":"auth_ok"}`。token 不放 URL。
- 医生只能访问自己的问诊；管理员可访问所有问诊。对应检查在 `require_consultation_access`。
- 登出把 access token JTI 写入 Redis db=2；Redis 不可用时黑名单检查和写入会降级，已登出 token 可能在过期前继续有效。
- 请求中间件生成/透传 `X-Request-ID`，统一错误含 `error_code/message/detail/request_id`；部分旧路由的 detail 仍可能是字符串。
- 安全响应头包含 nosniff、DENY、no-referrer 和 API CSP；production 增加 HSTS。
- CORS 允许 credentials；生产必须缩小 origins/methods/headers。

## 11. 实际 REST、SSE 与 WebSocket API

下面路径均含 `/api/v1`；“登录”表示 JWT，“管理员”表示 `role=admin`，“权限”表示细粒度权限依赖。

### 11.1 认证、病例与问诊

| 方法 | 路径 | 访问与行为 |
|---|---|---|
| POST | `/auth/register` | 公共；5/min |
| POST | `/auth/login` | 公共；5/min；返回 access/refresh token |
| POST | `/auth/refresh` | 公共；refresh token 换 access token |
| POST | `/auth/logout` | 登录；access token 入黑名单 |
| GET | `/auth/me` | 登录 |
| PUT | `/auth/profile` | 登录 |
| GET | `/patients/` | 登录；筛选 `personality_type/difficulty_level`，分页默认 200、上限 1000 |
| GET | `/patients/{patient_id}` | 登录；医生脱敏、管理员完整 |
| GET | `/patients/export` | `patient:export` |
| POST/PUT/DELETE | `/patients/`、`/patients/{patient_id}` | 管理员；写操作 30/min |
| GET | `/cases/recommend` | 登录；`count` 默认 5、范围 1–20 |
| GET | `/cases/{case_id}/difficulty` | 登录；少于 2 个历史样本时可返回 null |
| POST | `/consultations/` | 登录；body `patient_id` |
| GET | `/consultations/` | 登录；本人列表，分页默认 200、上限 1000 |
| GET | `/consultations/all` | 管理员；筛选用户、人格、分数、时间 |
| GET | `/consultations/{consultation_id}` | 本人或管理员；含消息 |
| POST | `/consultations/{id}/messages` | 本人或管理员；10/min；返回医生/患者消息 |
| POST | `/consultations/{id}/messages/stream` | 本人或管理员；10/min；SSE |
| POST | `/consultations/{id}/extend` | 本人或管理员；增加 10 轮 |
| POST | `/consultations/{id}/submit-diagnosis` | 本人或管理员；结束问诊 |
| POST | `/consultations/{id}/end` | 本人或管理员 |
| DELETE | `/consultations/{id}` | 所有者/服务层授权 |

### 11.2 评估、复核与统计

| 方法 | 路径 | 访问与行为 |
|---|---|---|
| POST | `/evaluations/` | `evaluation:create` + 问诊访问；5/hour |
| POST | `/evaluations/{consultation_id}/cancel` | 同上；Redis flag + best-effort Celery revoke |
| GET | `/evaluations/{consultation_id}/lock-status` | 本人或管理员 |
| GET | `/evaluations/{consultation_id}` | 本人或管理员 |
| GET | `/evaluations/task/{task_id}/status` | 登录；当前未校验 task 归属 |
| WS | `/evaluations/ws/{consultation_id}` | 首消息 JWT + 问诊访问校验 |
| POST | `/reviews/{evaluation_id}/submit` | 管理员 |
| GET | `/reviews/{evaluation_id}/status` | 登录；当前未额外校验评估归属 |
| GET | `/reviews/pending` | 管理员 |
| GET | `/stats/` | 登录；医生本人/管理员全局 |
| GET | `/users/me/data-export` | `consultation:view`；导出本人账户、问诊、消息和评估 |

重要契约限制：`POST /evaluations/` 声明 `response_model=EvaluationOut`，但非 `TESTING` 分支实际返回 `{"task_id":...,"status":"submitted"}`。FastAPI 响应校验可能因此失败；修复前不能把该生产异步响应当成稳定契约。测试模式会同步返回 `EvaluationOut`。

### 11.3 知识库、管理和模型版本

| 方法 | 路径 | 访问与行为 |
|---|---|---|
| GET | `/knowledge-base/stats` | 管理员 |
| POST | `/knowledge-base/add-pdf` | 管理员；202；PDF/DOCX；add 或 replace |
| DELETE | `/knowledge-base/sources/{source_name:path}` | 管理员；202 |
| POST | `/knowledge-base/rebuild` | 管理员；202 |
| GET | `/knowledge-base/rebuild/status?task_id=` | 管理员；任务状态；无 task_id 返回 active generation |
| POST | `/knowledge-base/cache/clear` | 管理员；仅清进程内 Embedding LRU |
| POST/GET | `/admin/cache/retrieval/clear`、`/admin/cache/retrieval/stats` | 管理员 |
| GET | `/admin/cache-stats` | 管理员；LLM + retrieval cache |
| POST | `/admin/cleanup` | 管理员；异步留存清理 |
| GET | `/admin/monitoring/tool-runtime` | 管理员 |
| GET | `/admin/monitoring/runs/{run_id}/trace` | 管理员 |
| GET | `/admin/monitoring/failures/summary?days=7` | 管理员；days 收敛到 1–90 |
| GET | `/admin/monitoring/usage/summary?days=7` | 管理员；days 收敛到 1–30 |
| GET | `/model-versions/`、`/model-versions/{name}/active` | 当前实现未要求登录 |
| POST | `/model-versions/` | `model:manage` |
| PUT | `/model-versions/{id}/deprecate` | `model:manage` |
| POST | `/model-versions/{id}/rollback` | `model:manage`；仅回滚登记状态，不切 RAG generation |

系统端点：`GET /health` 公共，检查 MySQL/Redis并返回缓存、LLM、Token 和 checkpointer 状态；依赖不可用时 503。`GET /metrics` 不出现在 OpenAPI，按 `METRICS_TOKEN` 和环境保护。

运行时 OpenAPI 是最终 HTTP 契约：<http://localhost:8000/docs>。

## 12. LangGraph 多智能体实现

### 12.1 图流程

```text
START → load_context → classify_consultation → safety_check
  ├─ high/undetermined/immediate review → finalize_needs_review → END
  └─ continue → plan_evaluation → validate_plan
       ├─ invalid → finalize_needs_review
       └─ Wave 1: knowledge + inquiry + humanistic (按计划存在项并行)
            → extract_knowledge_citations
            → Wave 2: diagnosis + treatment (按提交情况并行，注入知识引用)
            → aggregate_results → deterministic_scoring
            → reflection_check → review_gate_node
                 ├─ needs_review → finalize_needs_review
                 └─ generate_suggestion → finalize_completed → END
```

`initial/follow_up/emergency` 必选 inquiry、humanistic，条件选择 diagnosis、treatment、knowledge；未提交诊断/治疗时对应维度跳过。`communication` 只选择 inquiry、humanistic。

Safety 先执行确定性红旗规则；硬高风险不可被 LLM 降级。无规则命中且 LLM 失败时返回 `undetermined` 并 fail closed 到人工复核。

### 12.2 五维评估与评分

- inquiry：病史采集与问诊技巧。
- knowledge：医学知识一致性、RAG/Tool evidence 和引用。
- humanistic：同理、沟通和人文关怀。
- diagnosis：诊断合理性；可消费 knowledge citations。
- treatment：治疗方案合理性；可消费 knowledge citations。

默认固定权重 inquiry 0.25、knowledge 0.25、humanistic 0.20、diagnosis 0.15、treatment 0.15。评分器不会因缺失维度临时重分配权重；五项未全部为 scored 时 `total_score=null`。Reflection 是一致性检查和复核信号，不直接替代确定性评分器。

### 12.3 运行、checkpoint、取消和降级

每次运行写入 `evaluation_runs`，节点结果写入 `evaluation_node_results`；Redis thread id 为 `evaluation:<run_id>`。Celery 重试在配置允许时查找可恢复 run，并在 checkpoint 有待执行节点时以 `None` 输入继续；读取失败会降级为全新执行。

取消同时使用 Redis 取消标志和 Celery revoke：排队任务 best-effort revoke，运行中任务由轮询看守协作中断。单 Agent 异常会生成 error envelope 并推动人工复核，而不是让其余维度全部丢失。

## 13. RAG 检索与生成链

### 13.1 Tokenizer、BM25、Dense 与 Sparse

`medical-lexical-v3` tokenizer 保护医学缩写、基因变异、剂量/单位、ICD 风格代码等精确项，再对残余中文使用 jieba 和可选 CJK bigram；查询扩展保留稳定去重顺序。BM25 文档 token 可对 heading 和 entity 做 1–3 次有界 boost。

Dense 使用 `qwen3.7-text-embedding` 1024 维向量和 Chroma cosine collection。BGE-M3 learned sparse 默认关闭；启用时还需手工安装兼容的 FlagEmbedding，并为同一 generation 构建/校验 Sparse artifact。

`hybrid_recall` 从 Redis 读取 active generation，对 BM25、Dense 和启用时的 Sparse 并行召回；单个 channel 失败会记录 warning 并以空结果降级。融合只按稳定字符串 `doc_id`，拒绝跨 generation 结果。无 Sparse 时 BM25/Dense 权重归一化。

### 13.2 RRF、tiered 与 rerank

- Base：每个查询执行 hybrid recall + weighted RRF。
- MQE：Base 未达 high 时，每次总计最多 2 个扩展，先做 embedding 相似度漂移过滤。
- HyDE：仍为 low 时选择优先查询执行一次 hypothetical-document retrieval。
- 候选上限 20。High 需要至少 5 条、3 个来源、最大 vector ≥0.7、覆盖至少 2 个 query type；Medium 需要至少 3 条、2 个来源，且 vector ≥0.5 或 RRF ≥0.015。
- 两阶段 rerank：最多 20 条进入专用 reranker，最多 5 条进入 LLM 精排，最终分数组合 relevance/completeness/authority/freshness。任一阶段失败可降级到前序排序。

Metadata filter、来源多样性、邻居上下文扩展和句级压缩均存在但默认关闭。OCR 只在启用且页面文本低于阈值时调用 Qwen-VL。

### 13.3 Generation、manifest 与 artifact

生产发布链由 `backend/app/tasks/rag_index_task.py` 驱动：

```text
snapshot → parse → chunk → embed → chroma → bm25 → sparse
         → validate → switch → publish
```

generation 名称固定为 `rag-YYYYMMDDHHMMSS-<corpus_sha256前8位>`。一个 generation 包含：

- Chroma collection `medical_guidelines_<generation>`，每条 metadata 写入 `index_generation`；
- `backend/data/rag_indexes/<generation>/manifest.json`；
- `<generation>/bm25/`：bm25s 原生文件、组件 manifest、SHA-256 inventory、`READY`；加载默认 mmap；
- 启用 BGE-M3 时 `<generation>/sparse/`：documents、sparse payload、manifest、hash 和 `READY`。

顶层 manifest 记录 corpus hash、source/chunk 数、parser/chunker/tokenizer、embedding 模型/维度和各组件 generation。顶层 manifest 最后原子写入；generation 目录与 component identity 必须完全一致，不能用其他 generation 的 artifact 补位。

### 13.4 激活、缓存与多 Worker

RAG 构建使用 Redis key `rag:index-build-lock`，TTL 30 分钟并由 heartbeat 续期。候选校验通过后，`rag:active_generation` 通过 compare-and-set 从任务快照中的旧 generation 原子切到新 generation；并发发布者改变指针时任务失败。之后发布 `rag:index-switched`，消息含 new、previous 和 manifest SHA-256。

Worker 收到事件后先加载并验证 manifest、Chroma count、BM25 和可选 Sparse，再在进程锁内一次性替换本地引用。任何预加载或安装失败都保留旧引用。检索 cache key 含 generation；回填时还会过滤 generation 不匹配文档。

注意：FastAPI 自身没有在 lifespan 启动 Pub/Sub listener，但检索入口会读取 Redis active generation 并按 generation 获取组件；Celery fork Worker 会启动 listener。

### 13.5 旧兼容路径

`python -m app.services.rag.build_medical_index` 调用旧 `build_medical_index(target_version="rag-v2")`：构建一个临时指定 collection，结束后恢复 `ACTIVE_INDEX_VERSION`，不写 Task 7 顶层 manifest、不 CAS active pointer、也不发布切换事件。`index_single_pdf`、`switch_index_version` 和 `rebuild_kb_from_cache.py` 同样是兼容/维护接口，不等价于 immutable generation 发布。

`backend/data/embed_cache/*.npz` 属于旧 rebuild-from-cache 脚本的输入；当前 generation builder 使用进程内 1000 条 Embedding LRU，不会自动生成该磁盘 cache。

## 14. 知识库操作与回滚

### 14.1 API 操作

源文件放在仓库根 `data/`。API 只接收相对文件名，拒绝绝对路径、盘符、`..` 和 PDF_DIR 外文件；扩展名只允许 `.pdf`/`.docx`。服务端将相对路径哈希为稳定的 `source-<24 hex>`，避免信任客户端 source id。

```bash
# 全量重建
curl -H "Authorization: Bearer $TOKEN" -X POST \
  http://localhost:8000/api/v1/knowledge-base/rebuild

# 添加；已存在 source 时失败
curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"filename":"guide.pdf","force_replace":false}' \
  http://localhost:8000/api/v1/knowledge-base/add-pdf

# 替换
curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"filename":"guide.pdf","force_replace":true}' \
  http://localhost:8000/api/v1/knowledge-base/add-pdf

# 状态或 active generation
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/knowledge-base/rebuild/status?task_id=<id>"
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/knowledge-base/rebuild/status
```

状态可见 `PENDING/PROGRESS/SUCCESS/FAILURE`，PROGRESS 含真实到达的 phase；成功可含 generation、manifest、switch 和 validation。

### 14.2 回滚原则

当前没有 RAG generation 回滚 REST API。`/model-versions/{id}/rollback` 只改模型登记表，不能回滚 RAG。安全回滚必须由运维使用 `versioning` 的 manifest 校验、CAS active pointer 和同一 Pub/Sub 事件完成，且旧 generation 的 Chroma/BM25/可选 Sparse 必须仍完整存在。

回滚步骤：

1. 停止新的索引写任务并确认 `rag:index-build-lock` 所有者。
2. 记录当前 active generation 和目标旧 generation；加载并执行 `validate_candidate_manifest`，检查 Chroma count、BM25 `READY`/hash 和可选 Sparse。
3. 以当前 generation 为 expected 执行 CAS，避免覆盖并发发布。
4. 计算目标 manifest SHA-256，向 `rag:index-switched` 发布与正常发布相同的消息。
5. 在每个 Worker 验证 `index_generation` trace、候选数、cache hit 和真实查询；失败 Worker 保留旧本地引用，必须排查或重启。

仓库没有封装上述操作的受支持 CLI/API，因此生产回滚应先在演练环境编写并审阅运维脚本，不能直接手改 `settings.ACTIVE_INDEX_VERSION`。至少保留两个已验证 generation；清理前确认无进程引用并保留对应 manifest、评测和审计记录。

## 15. 评测、调参与质量门禁

### 15.1 数据与指标

- `backend/scripts/eval/bm25_golden_set.json`：BM25 词法集，要求至少 40 条，覆盖 `disease_alias/drug_dose/gene_variant/lab_unit/negation/icd_code`。
- `backend/scripts/eval/golden_set.json`：一般检索来源命中集。
- `backend/evaluation/rag_cases/*.jsonl`：dev/test/regression Gold cases；自动 bootstrap 标签必须人工复核，不能当最终 gold。
- 指标：Recall@1/3/5/10、MRR、nDCG、source hit/citation validity、hallucination、refusal/false acceptance、Tool 成功/预算/延迟、claim coverage/contradiction。

BM25 evaluator 是只读的，不重建或切换索引；没有 initialized active generation 时，普通运行诚实跳过，带 `--fail-on-regression` 则失败。

### 15.2 调参

`tune_weights.py` 的联合网格确实存在：BM25 `k1=[0.9,1.2,1.5]`、`b=[0.5,0.7,0.8]`、`heading_boost=[1,2]`、`entity_boost=[1,2,3]`、`RRF_K=[30,35,60]`。另有 6 组固定 BM25/Dense/Sparse 权重候选。

```powershell
cd backend
.\.venv\Scripts\python.exe scripts/eval/tune_weights.py `
  --dev-golden <reviewed-dev.json> `
  --test-golden <independent-reviewed-test.json> `
  --retriever mqe --top-k 10 --primary ndcg@10 `
  --output <tuning-report.json>
```

只允许 dev 选参；test 只对最终胜出参数运行一次。示例 `golden_set.json` 不能同时伪装成 dev 和 test。

### 15.3 Task 8 真实 generation 门禁

候选必须同时满足：

- overall Recall@10 不低于实测 baseline；
- overall nDCG@10 不低于实测 baseline；
- exact-term Recall@10 至少比 baseline 高 0.05；
- cold load ≤10 秒；search p95 ≤5 ms；
- generation mismatch count = 0；stale cache hit count = 0；
- 缺失测量按失败处理，绝不能填 0 或使用 mock 数字。

当前不能宣称门禁已通过：工作区中的 `backend/evaluation_reports/bm25-v1.json` 是未跟踪的本地报告，schema 缺少 Task 8 所需 overall/exact-term/consistency；而 `evaluate_bm25.py` CLI 当前没有注入真实一致性计数的参数，默认报告把一致性标记为未测量。因此 `--fail-on-regression` 的完整绿灯需要先提供同 schema 的真实 baseline/candidate 和实际一致性遥测接线。

CI 的 `python -m evaluation.rag_eval --mode mock --limit 5 --fail-on-threshold` 只阻断评测框架回归；它不证明真实检索、真实 LLM 或候选 generation 的质量。CI 只有在 checkout 同时存在 measured report 和 generation/Chroma artifacts 时才尝试真实门禁，否则明确 SKIP。

## 16. 监控、日志与安全运维

- `/health`：MySQL、Redis、LLM、缓存、Token、LangGraph/checkpointer；degraded 返回 503。
- `/metrics`：HTTP 数量/耗时、LLM/RAG/缓存等 Prometheus 指标。生产必须配置 `METRICS_TOKEN`。
- RAG trace/metrics 应关注 `index_generation`、`bm25_load_seconds`、`bm25_query_seconds`、`bm25_candidates`、`bm25_top_score`、`lexical_expansion_count`、`filter_fallback`、`cache_hit`、`retrieval_level` 和 channel candidate counts。
- 管理 API 提供 run/node tree、failure reason 和 Redis Token 用量；Token run 数据保留 7 天。
- 日志支持 JSON/text 和 `X-Request-ID`；审计 detail 应保持脱敏。
- Tool health check 默认关闭，因为探测会产生真实调用和成本。
- pip-audit、npm audit、Trivy 在 CI 当前为告警模式，不是阻断门禁。

生产必须使用强 SECRET_KEY、独立数据库账号、最小 CORS、TLS、受保护 metrics、外部密钥管理、网络分段、备份恢复演练和数据删除流程。模型 Provider、Prompt 和知识库变更都应保留版本、审批、评测和回滚证据。

## 17. 测试与 CI

本地全量：

```powershell
cd backend
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy app
.\.venv\Scripts\python.exe -m compileall -q app
.\.venv\Scripts\python.exe -m pytest tests -q

cd ..\frontend
npm run lint
npm test
npm run build
```

CI 在 push/PR 到 `main/master` 时执行：

- MySQL/Redis service 下的 mypy、pytest + app coverage（最低 40%）、RAG mock gate、Task 8 合约测试和全部 RAG 测试；
- 数据库初始化/迁移脚本检查；
- 前端 lint/build；
- Ruff、compileall、`git diff --check`；
- 告警模式依赖/文件系统安全扫描；
- master 上构建并推送 GHCR 镜像。

`npm test` 存在但 CI 的 frontend job 当前只执行 lint/build。文档变更至少执行链接/路径审计、Markdown fence 配对和 `git diff --check`。

## 18. 故障排查

### 后端启动失败

- production 报 SECRET_KEY：替换默认值。
- LangGraph/checkpointer 初始化失败：确认 `REDIS_CHECKPOINT_URL` 可达；若明确接受旧路径再设 `LANGGRAPH_ENABLED=false`。
- MySQL unknown column/table：检查是否误用了不完整 `database/init.sql`；在备份副本对照 Alembic schema，不要盲目 stamp。
- ModuleNotFound：从 `backend/` 使用 `app.main:app`；容器命令和本地模块路径不同。

### 评估提交失败或一直 pending

- 检查 `POST /evaluations/` 的当前 response-model 不一致限制。
- 确认 FastAPI 与 Worker 的 broker/result backend 完全一致，Redis db=4/5 可达。
- 查看 `celery -A app.celery_app inspect registered/active/reserved`、锁状态、task status、Worker 日志和取消标志。
- 任务超时按 run/node trace 判断是 LLM、Tool、RAG 还是 checkpoint；不要只增加超时。

### WebSocket 立即关闭

- 连接后 5 秒内发送 auth 首消息；确认 access token 类型、用户存在且拥有问诊访问权。
- 反向代理必须传 Upgrade/Connection；Nginx Dockerfile 已配置 `/api/` WebSocket。

### RAG 无结果或 generation mismatch

- 无 active pointer：检查 Redis `rag:active_generation` 和 status API。
- Compose 中先修正 `/app/data` 源文件挂载。
- 检查 `<generation>/manifest.json`、BM25/Sparse `READY`、hash、Chroma count 和 BGE 开关一致性。
- 事件加载失败时 Worker 会保留旧引用；查看 `rag:index-switched` listener 日志。
- 切换后 stale 结果：检查 cache key generation、回填过滤和每个 Worker trace，不要全局手工伪造 cache 命中计数。

### BM25 门禁失败

- “unavailable measurement” 不是性能回归等同于 0，而是缺少真实测量。
- 先准备同 schema baseline/candidate 和真实一致性遥测；确认 active generation 真正加载。
- cold load 过慢时检查 mmap、Chroma WAL 冷重建、磁盘和 Worker 冷启动；p95 过慢时检查候选规模和并发。

### 前端 API 异常

- 本地 Vite 只代理 `/api` 到 localhost:8000；`VITE_API_BASE_URL` 当前无效。
- 401 会清 sessionStorage 并跳登录；检查 token 过期/黑名单。
- `AdminReviews` 无路由；请用 OpenAPI，而不是寻找菜单。

## 19. 生产发布检查清单

- [ ] 医疗用途边界、人工责任、应急转交流程已批准。
- [ ] 数据已合法授权、去标识化；日志/导出/备份/评测目录纳入敏感数据管理。
- [ ] 默认管理员和所有示例凭据已移除；SECRET_KEY/API Key/数据库密码来自密钥系统。
- [ ] 新库使用 Alembic 完成，或已有库经结构 diff 和迁移演练；没有把 init.sql 与 baseline 盲目串联。
- [ ] FastAPI 与所有 Celery Worker 的 broker/result/checkpoint Redis 一致且网络可达。
- [ ] Compose 的 `/app/data` 和共享 RAG artifact/Chroma 存储已修正。
- [ ] Beat 单实例；Worker 并发、超时、内存和 LLM 配额已压测。
- [ ] HTTPS、最小 CORS、metrics token、最小 DB/RBAC、网络访问控制已启用。
- [ ] 真实 RAG baseline/candidate、dev/test 独立性、一致性遥测和 Task 8 所有门禁有可审计报告；没有用 mock 结果冒充。
- [ ] active/previous generation 均完整可加载，回滚脚本和恢复演练已通过。
- [ ] 后端 lint/type/test、前端 lint/test/build、迁移和安全扫描已审阅。
- [ ] `/health`、`/metrics`、run trace、failure summary、告警与备份恢复已验证。

## 20. 已知限制

1. `POST /api/v1/evaluations/` 的生产异步返回体与 `EvaluationOut` 响应模型不一致。
2. 基础 Compose 未给 FastAPI backend 注入容器内 Celery broker/result URL。
3. Compose RAG source 挂载路径与代码 `PDF_DIR=/app/data` 不一致。
4. `database/init.sql` 与当前 ORM/Alembic schema 不完整等价；Compose 首次初始化不可视为迁移完成。
5. Task 8 候选真实性能尚未以完整一致性遥测实测通过；当前不得宣称门禁已通过。
6. `evaluate_bm25.py` CLI 当前不能传入真实一致性计数，完整 gate 会把它们判为 unavailable。
7. RAG generation 没有受支持的回滚 REST/CLI；旧 `switch_index_version` 不是集群 immutable generation 回滚。
8. `VITE_API_BASE_URL` 未被 Axios 使用；部署依赖同源 `/api/v1` 和反向代理。
9. `AdminReviews` 页面未接路由；知识库、模型版本、监控也缺少完整 UI。
10. 模型版本 GET 路由当前公开；review status 和 evaluation task status 只要求登录，未做对象归属校验。
11. JWT 黑名单 Redis 不可用时 fail open；登出不保证立刻吊销。
12. Compose 固定 `container_name`，不适合直接水平 scale Worker。
13. ChromaDB 1.5.7 使用极大 `hnsw:sync_threshold` 规避已知跨进程段加载问题，代价是冷查询可能从 WAL 重建；旧 collection 需重建才继承 metadata。
14. BGE-M3 依赖默认未安装，Sparse/OCR/多项增强默认关闭；启用前必须做资源和质量验证。
15. CI 安全扫描为告警模式，前端单元测试未在 CI frontend job 中执行。

## 21. 文档维护规则

改变路由、schema、默认值、Compose、迁移、Agent 图、RAG generation、评测门禁或前端操作时，必须同步本手册。数字只允许来自代码常量、配置或当次可追溯实测；测试合约通过不等于真实性能通过。专题文档：[技术架构说明](technical-document.md)、[平台操作说明](platform-documentation.md)、[Prompt 与 Provider](prompt-and-provider-adapter.md)。
