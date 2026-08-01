# 基于多智能体的医生临床问诊评估平台

> 本文是当前仓库的项目总手册。文中的目录、命令、端口、接口和配置以当前源代码、docker-compose.yml、backend/app/core/config.py 与 backend/.env.example 为准。API 的完整参数约束以运行中的 OpenAPI 为最终准绳：<http://localhost:8000/docs>。

## 1. 项目定位与边界

本项目是面向医学教育和临床问诊训练的 AI 评估平台。医生或学员与虚拟患者进行模拟问诊，系统保存会话，再由多智能体评估流程从五个维度分析问诊质量，并返回结构化评分、建议和知识库证据。

五个评估维度为：

| 维度 | 关注点 |
| --- | --- |
| 问诊分析 | 信息采集的完整性、顺序和效率 |
| 医学知识核对 | 关键信息与医学知识库证据的一致性 |
| 人文关怀 | 沟通方式、解释、共情和患者体验 |
| 诊断评估 | 鉴别诊断、风险识别和诊断依据 |
| 治疗方案 | 方案完整性、合理性、风险和随访建议 |

平台的评估结果用于训练、教学和质量改进，不替代真实临床诊断、处方或医疗机构的专业审核。仓库中的医学资料、虚拟患者数据和模型输出都应按实际部署的数据治理要求管理。

### 1.1 当前实现的主要能力

- 用户注册、登录、刷新令牌、登出和个人资料管理。
- 虚拟患者列表、病例推荐、难度查询和管理员维护。
- 问诊会话创建、消息发送、流式患者回复、延长会话、提交诊断、结束会话和删除会话。
- 异步评估任务、评估锁、取消、状态查询、WebSocket 进度推送和结果查看。
- LangGraph 编排的多智能体评估流程；Redis checkpoint 支持长任务恢复。
- 混合知识检索、PDF 增量入库、索引重建、来源删除、Embedding 缓存清理和检索统计。
- 人工复核队列、复核提交和复核状态查询。
- 审计日志、Prometheus 指标、管理员监控查询和定期数据清理。
- Docker Compose 部署，包含 MySQL、Redis、FastAPI、Celery Worker、Celery Beat、Nginx 前端，以及可选的 Prometheus/Grafana。

### 1.2 明确不应从文档推断的内容

- 没有配置真实 LLM API Key 时，系统不能完成真实模型推理。
- data/ 中是否存在医学 PDF 取决于当前部署目录；仓库不会自动生成或下载医学资料。
- RAG 索引、模型版本和提示词版本必须以当前实例中的数据和配置为准，不能把示例版本当成已经完成的生产基线。
- Docker Compose 的首次初始化脚本只在 MySQL 空数据卷第一次创建时执行；已有数据卷不会自动重新执行 database/init.sql。

## 2. 系统架构

~~~mermaid
flowchart TD
    U[医生或管理员] --> FE[React + Vite 前端]
    FE -->|REST /api/v1| API[FastAPI API]
    FE -->|WebSocket 评估进度| API
    API --> DB[(MySQL 8)]
    API --> R[(Redis 7)]
    API --> V[(ChromaDB 持久化目录)]
    API --> LLM[兼容 OpenAI 协议的 Qwen API]
    API --> Q[Celery Broker]
    Q --> W[Celery Worker]
    W --> G[LangGraph 评估编排]
    G --> R
    G --> V
    W --> DB
    B[Celery Beat] --> Q
    M[Prometheus / Grafana 可选] --> API
~~~

### 2.1 运行组件

| 组件 | 开发入口 | Docker 服务 | 作用 |
| --- | --- | --- | --- |
| 前端 | frontend/，Vite 5173 | frontend，Nginx 80/443 | 登录、问诊、评估、统计和管理界面 |
| API | backend/app/main.py | backend，8000 | 认证、业务 API、健康检查和指标 |
| 数据库 | MySQL 8 | mysql，3306 | 用户、患者、问诊、评估、复核和审计数据 |
| Redis | Redis 7 | redis，6379 | Celery 队列、结果、缓存、取消标志和 LangGraph checkpoint |
| 向量库 | ChromaDB | chroma_data 卷 | 医学文档向量索引；BM25 等索引也依赖持久化目录 |
| 异步执行 | Celery 5.6 | celery-worker | 执行长耗时评估任务 |
| 定时调度 | Celery Beat | celery-beat | 每日触发过期数据清理 |
| 监控 | Prometheus/Grafana | monitoring profile | 指标抓取和可视化 |

### 2.2 请求与评估时序

~~~mermaid
sequenceDiagram
    participant D as 医生
    participant F as 前端
    participant A as FastAPI
    participant DB as MySQL
    participant C as Celery
    participant G as LangGraph
    participant R as Redis/Chroma

    D->>F: 创建问诊并发送问题
    F->>A: REST 请求，携带 Bearer access_token
    A->>DB: 保存会话和消息
    A-->>F: 患者回复或流式响应
    D->>F: 提交诊断并触发评估
    F->>A: POST /evaluations/
    A->>DB: 获取评估锁
    A->>C: 提交 run_evaluation
    C->>G: 执行评估图
    G->>R: 读取 checkpoint 和医学证据
    G->>DB: 保存节点结果、运行记录和最终评估
    G-->>A: WebSocket 推送进度
    A-->>F: 结果或 needs_review
~~~

评估锁用于防止同一问诊被并发重复评估。Celery 任务默认软超时 300 秒、硬超时 600 秒；网络、超时、临时不可用和限流类错误最多重试 2 次，退避时间为 30 秒、60 秒。重试时会尝试从 LangGraph checkpoint 继续执行。

## 3. 代码目录与职责

~~~text
medical-ai-platform/
├─ backend/
│  ├─ app/
│  │  ├─ api/v1/          REST/WebSocket 路由
│  │  ├─ core/            配置、认证、权限、审计、限流、日志、中间件
│  │  ├─ db/              SQLAlchemy 异步会话
│  │  ├─ models/          SQLAlchemy ORM 模型
│  │  ├─ schemas/         Pydantic 请求/响应模型
│  │  ├─ services/        问诊、评估、RAG、缓存、用户等业务服务
│  │  ├─ orchestration/   LangGraph 状态、图、节点和 checkpoint
│  │  ├─ tasks/           Celery 评估与清理任务
│  │  ├─ prompts/         提示词和 manifest
│  │  └─ main.py          FastAPI 应用入口
│  ├─ evaluation/         基准集、Rubric、安全、复核、RAG 评估工具
│  ├─ alembic/            数据库迁移
│  ├─ scripts/            初始化、知识库、数据迁移和评估脚本
│  ├─ tests/              后端单元、集成、RAG、编排和 E2E 测试
│  ├─ requirements.txt
│  └─ .env.example
├─ frontend/
│  ├─ src/pages/          登录、仪表盘、患者、问诊、评估、统计和管理页面
│  ├─ src/components/     通用 UI 组件
│  ├─ src/api/            API 调用封装
│  ├─ src/store/          登录和应用状态
│  ├─ src/utils/          请求客户端及工具
│  └─ package.json
├─ database/               init.sql、seed.sql 和历史迁移 SQL
├─ monitoring/             Prometheus 配置和 Grafana dashboard
├─ data/                   医学 PDF 挂载目录，不保证包含实际资料
├─ docker-compose*.yml
├─ Dockerfile.backend / Dockerfile.frontend
└─ docs/                   主题文档和本项目总手册
~~~

后端源码从 backend 目录启动时使用 app.main:app；生产 Docker 镜像使用 backend.app.main:app 并设置 PYTHONPATH=/app/backend。这两个入口是由运行位置不同造成的正常差异。

## 4. 环境与配置

### 4.1 前置条件

本地开发需要 Python 3.10 或更高版本、Node.js/npm、MySQL 8 和 Redis 7。真实评估还需要兼容 OpenAI API 的 Qwen/DashScope 凭据。Docker 用户需要 Docker Engine 与 Docker Compose v2。

### 4.2 后端环境文件

~~~powershell
cd backend
Copy-Item .env.example .env
~~~

backend/.env 由 Pydantic Settings 读取；Docker Compose 则主要读取仓库根目录的 .env，并把关键值传入容器。两种方式不要混为一谈。任何环境都不得提交真实密钥。

| 变量 | 默认/示例 | 说明 |
| --- | --- | --- |
| ENVIRONMENT | development | production 时必须显式设置安全的 SECRET_KEY |
| MYSQL_HOST | localhost | Docker 内应为 mysql |
| MYSQL_PORT | 3306 | MySQL 端口 |
| MYSQL_USER / MYSQL_PASSWORD | root / 空 | 生产应使用最小权限业务账户；Compose 默认将密码用于 root 初始化 |
| MYSQL_DATABASE | medical_ai | 业务数据库名 |
| SECRET_KEY | 占位字符串 | JWT 签名密钥，生产必须替换为高强度随机值 |
| ACCESS_TOKEN_EXPIRE_MINUTES | 1440 | access token 有效期 |
| REFRESH_TOKEN_EXPIRE_DAYS | 7 | refresh token 有效期 |
| DASHSCOPE_API_KEY | 空 | Qwen API Key；QWEN_API_KEY 默认从它读取 |
| QWEN_API_BASE_URL | DashScope 兼容地址 | OpenAI 兼容接口地址 |
| QWEN_MODEL | qwen3.7-plus（代码默认） | Compose 默认覆盖为 qwen3.7-max，以实际部署变量为准 |
| REDIS_CHECKPOINT_URL | redis://localhost:6379/1 | LangGraph checkpoint Redis 数据库 |
| CELERY_BROKER_URL | redis://localhost:6379/4 | Celery 消息代理 |
| CELERY_RESULT_BACKEND | redis://localhost:6379/5 | Celery 结果存储 |
| LANGGRAPH_ENABLED | true | 是否启用 LangGraph 编排 |
| ACTIVE_INDEX_VERSION | rag-v1 | 当前 RAG 索引版本标识 |
| ENABLE_TOOL_USE | true（代码默认） | Compose 默认传入 false，必须以运行环境为准 |
| LLM_MAX_CONCURRENT | 10 | LLM 全局并发上限 |
| LLM_CACHE_ENABLED | true | LLM 响应缓存开关 |
| AUDIT_LOG_ENABLED | true | 审计日志开关 |
| METRICS_TOKEN | 空 | 生产建议配置，保护 /metrics |

完整配置项及注释见 [backend/.env.example](../backend/.env.example)；配置实现见 [backend/app/core/config.py](../backend/app/core/config.py)。模型、RAG、Tool Use、缓存、数据留存和可观测性参数不要只改文档，应直接改环境变量或配置实现。

## 5. 启动方式

### 5.1 本地启动后端

推荐使用仓库已有的后端虚拟环境；若不存在则创建：

~~~powershell
cd backend
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
~~~

当前依赖包含 celery[redis]==5.6.3。在 Windows 本地启动 Celery Worker 时使用 -P solo：

~~~powershell
cd backend
venv\Scripts\python.exe -m celery -A app.celery_app worker --loglevel=info -P solo
venv\Scripts\python.exe -m celery -A app.celery_app beat --loglevel=info
~~~

Worker 和 Beat 应在独立终端运行。若只做 API、认证或同步测试，可暂不启动它们；真实生产评估依赖 Worker，定时清理依赖 Beat。项目虚拟环境中的 Celery 可用性可用下面命令确认：

~~~powershell
backend\venv\Scripts\python.exe -c "import celery; print(celery.__version__)"
~~~

### 5.2 本地启动前端

~~~powershell
cd frontend
npm install
npm run dev
~~~

Vite 默认监听 5173，/api 和 WebSocket 请求代理到 http://localhost:8000。前端请求客户端的 API 前缀是 /api/v1，普通请求超时 60 秒，评估相关请求超时 300 秒。

默认访问地址：

- 前端：<http://localhost:5173>
- API 文档：<http://localhost:8000/docs>
- ReDoc：<http://localhost:8000/redoc>
- OpenAPI JSON：<http://localhost:8000/api/v1/openapi.json>
- 健康检查：<http://localhost:8000/health>
- Prometheus 指标：<http://localhost:8000/metrics>

### 5.3 Docker Compose

在仓库根目录创建 .env，至少设置 MYSQL_PASSWORD、SECRET_KEY 和 DASHSCOPE_API_KEY；启用 monitoring profile 还要设置 GRAFANA_ADMIN_PASSWORD。然后执行：

~~~powershell
docker compose up -d
docker compose ps
docker compose logs -f backend
~~~

基础栈端口为：前端 80、后端 8000、MySQL 3306、Redis 6379。可用 BACKEND_PORT、FRONTEND_PORT、MYSQL_PORT、REDIS_PORT 修改宿主机映射。前端容器通过 Nginx 提供 SPA；提供 certs/server.crt 与 certs/server.key 后才适合启用 HTTPS 监听。

监控栈：

~~~powershell
docker compose --profile monitoring up -d
~~~

Prometheus 默认 9090，Grafana 默认 3000。Grafana 默认用户名为 admin，密码由 GRAFANA_ADMIN_PASSWORD 提供。docker-compose.override.yml 是本地开发覆盖配置；docker-compose.staging.yml 和 docker-compose.prod.yml 是镜像/资源/日志覆盖配置，不是独立的完整 Compose 文件。

## 6. 数据库初始化与迁移

项目提供两种互斥的全新数据库初始化方式，请不要把它们串联执行：

1. **Alembic 路径**：创建空数据库后，在 backend 目录执行 alembic upgrade head。baseline migration 会创建表，新迁移会继续增加后续结构。
2. **SQL/Compose 路径**：执行 database/init.sql 和 database/seed.sql。Docker Compose 在 MySQL 空数据卷第一次创建时自动执行这两个文件；当前 init.sql 已包含当前仓库需要的结构。

如果已经用 init.sql 建好当前结构，不要再对一个没有 Alembic 版本记录的数据库直接执行 alembic upgrade head，因为 baseline migration 会尝试重复创建表。需要让后续迁移接管时，先确认 init.sql 对应的结构完整，再执行一次 alembic stamp head；这一步只登记版本，不执行结构变更。生产数据库应由备份、变更窗口和人工核对保护。

### 6.1 Alembic 路径（空数据库）

~~~powershell
cd backend
venv\Scripts\python.exe -m alembic upgrade head
~~~

### 6.2 已有数据库

不要用 Base.metadata.create_all() 代替迁移，也不要在有业务数据的环境中直接删除数据卷。当前修复新增的迁移为：

1a2b3c4d5e6f_add_case_id_and_message_sequence_constraint

它会：

1. 给 virtual_patients 增加可空的稳定病例标识 case_id，并建立唯一索引。
2. 给 consultation_messages 增加 (consultation_id, sequence) 唯一约束，避免同一会话出现重复序号。

已有库升级前先备份，再执行 alembic upgrade head。如果历史数据已经存在重复消息序号或重复病例标识，迁移会因唯一约束冲突失败，应先清理并核对数据。Compose 的 init.sql 已同步包含新结构，但它只对空 MySQL 卷生效。

管理账号初始化脚本：

~~~powershell
cd backend
venv\Scripts\python.exe scripts\init_admin.py
~~~

脚本支持的参数和默认行为以 backend/scripts/init_admin.py --help 及源码为准；不要把初始密码写入文档或提交到仓库。

## 7. 用户功能与前端页面

前端实际路由定义在 frontend/src/App.tsx：

| 路由 | 页面 | 权限 |
| --- | --- | --- |
| /login | 登录 | 公开 |
| /register | 注册 | 公开 |
| /dashboard | 仪表盘 | 登录用户 |
| /patients | 虚拟患者 | 登录用户，数据操作受后端权限控制 |
| /consultations | 问诊列表 | 登录用户 |
| /consultation/:id | 问诊详情与交互 | 有该问诊访问权 |
| /evaluation/:id | 评估详情 | 有该问诊访问权 |
| /stats | 统计 | 登录用户；具体数据仍由 API 权限控制 |
| /admin/consultations | 管理员问诊管理 | 管理员 |
| /admin/patients | 管理员患者管理 | 管理员 |
| /profile | 个人资料 | 登录用户 |

典型使用流程：

1. 注册或由管理员创建账号并登录。
2. 选择虚拟患者或使用病例推荐创建问诊。
3. 逐轮发送问诊问题，等待患者回复；必要时延长或结束问诊。
4. 提交诊断，确认会话内容后触发评估。
5. 通过页面轮询、WebSocket 或评估状态 API 查看进度。
6. 查看五维评分、证据、风险提示和建议；标记为 needs_review 的结果进入人工复核流程。

## 8. 认证、权限与通用 API 约定

### 8.1 JWT

登录成功后返回 access_token、refresh_token 和用户信息。REST 请求使用：

~~~http
Authorization: Bearer <access_token>
~~~

前端将令牌放在 sessionStorage。401 响应会触发前端清理登录状态并跳转登录页；登出会将 access token 放入 Redis 黑名单（黑名单启用时）。

WebSocket 评估进度连接建立后，客户端必须在 5 秒内发送以下首条消息，服务端不会从 URL 读取 token：

~~~json
{"type":"auth","token":"<access_token>"}
~~~

鉴权成功后服务端返回 {"type":"auth_ok"}，然后才开始接收该问诊的进度事件。

### 8.2 角色

代码内置 admin 和 doctor 两种角色，并允许用户使用自定义 permissions 覆盖角色默认权限：

| 角色 | 默认权限摘要 |
| --- | --- |
| admin | 评估创建/查看/复核、问诊创建/查看、患者创建/查看/导出、用户/系统/模型管理 |
| doctor | 评估创建/查看、问诊创建/查看、患者查看 |

问诊详情、消息、评估等资源还会校验资源访问权：问诊所属医生或管理员才能访问。不要只依赖前端路由隐藏管理页面，后端权限才是安全边界。

### 8.3 响应与错误

后端统一错误响应通常包含 error_code、message、detail、request_id，并在响应头返回 X-Request-ID。客户端可以把 request_id 提供给运维人员定位日志。常见状态码：

| 状态码 | 含义 |
| --- | --- |
| 401 | 未登录、令牌无效或过期 |
| 403 | 权限不足或资源不属于当前用户 |
| 404 | 资源不存在 |
| 409 | 状态冲突，例如同一问诊已有评估在运行 |
| 422 | 请求体或参数校验失败 |
| 429 | 触发接口限流 |
| 503 | 数据库、Redis 或其他依赖不可用 |

## 9. REST 与 WebSocket API 目录

所有路径都以 /api/v1 开头。列表中的“登录”表示需要有效 JWT；实际请求字段和响应模型请直接查看 /docs。

### 9.1 认证 /auth

| 方法 | 路径 | 作用 | 权限 |
| --- | --- | --- | --- |
| POST | /auth/register | 注册用户 | 公开 |
| POST | /auth/login | 用户名密码登录 | 公开 |
| POST | /auth/refresh | 用 refresh token 换新 access token | 公开 |
| POST | /auth/logout | 登出并处理 token 黑名单 | 登录 |
| GET | /auth/me | 获取当前用户 | 登录 |
| PUT | /auth/profile | 更新当前用户资料 | 登录 |

### 9.2 虚拟患者 /patients 与病例 /cases

| 方法 | 路径 | 作用 | 权限 |
| --- | --- | --- | --- |
| GET | /patients/ | 查询患者列表 | patient:view |
| GET | /patients/export | 导出患者数据 | patient:export |
| GET | /patients/{patient_id} | 查询患者详情 | 登录/资源权限 |
| POST | /patients/ | 创建虚拟患者 | patient:create |
| PUT | /patients/{patient_id} | 更新患者 | 后端权限控制 |
| DELETE | /patients/{patient_id} | 删除患者 | 后端权限控制 |
| GET | /cases/recommend | 按条件推荐病例 | 登录 |
| GET | /cases/{case_id}/difficulty | 查询病例难度 | 登录 |

case_id 是数据集病例的稳定映射标识，名称不是可靠的唯一键。新增病例数据时应保持 case_id 稳定且唯一。

### 9.3 问诊 /consultations

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| POST | /consultations/ | 创建问诊 |
| GET | /consultations/ | 查询当前用户可见问诊 |
| GET | /consultations/all | 查询全量/管理视图问诊 |
| GET | /consultations/{consultation_id} | 获取问诊详情 |
| POST | /consultations/{consultation_id}/messages | 发送问题并获取患者回复 |
| POST | /consultations/{consultation_id}/messages/stream | 流式发送问题和患者回复 |
| POST | /consultations/{consultation_id}/extend | 延长问诊 |
| POST | /consultations/{consultation_id}/submit-diagnosis | 提交诊断 |
| POST | /consultations/{consultation_id}/end | 结束问诊 |
| DELETE | /consultations/{consultation_id} | 删除问诊 |

同一会话消息序号由数据库唯一约束保护；客户端不要自行复用已经使用过的 sequence。

### 9.4 评估 /evaluations

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| POST | /evaluations/ | 创建评估；生产模式返回 Celery task 信息 |
| GET | /evaluations/{consultation_id} | 查询问诊对应评估结果 |
| POST | /evaluations/{consultation_id}/cancel | 协作式取消评估 |
| GET | /evaluations/{consultation_id}/lock-status | 查询评估锁状态 |
| GET | /evaluations/task/{task_id}/status | 查询 Celery 任务状态 |
| WS | /evaluations/ws/{consultation_id} | 订阅评估进度 |

评估可能最终为 completed、needs_review 或其他失败/中间状态。不要只依据 Celery 的 SUCCESS 判断业务评估成功，应继续读取评估记录和锁状态。

### 9.5 人工复核 /reviews

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | /reviews/pending | 查询待复核评估 |
| GET | /reviews/{evaluation_id}/status | 查询复核状态 |
| POST | /reviews/{evaluation_id}/submit | 提交人工复核结果 |

### 9.6 知识库 /knowledge-base

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | /knowledge-base/stats | 查询索引和构建统计 |
| POST | /knowledge-base/add-pdf | 增量添加单个 PDF |
| DELETE | /knowledge-base/sources/{source_name} | 删除指定来源的全部索引块 |
| POST | /knowledge-base/rebuild | 后台触发全量重建 |
| GET | /knowledge-base/rebuild/status | 查询重建状态 |
| POST | /knowledge-base/cache/clear | 清空 Embedding 缓存 |

知识库写操作要求管理员权限。删除来源后系统会尝试重建 BM25 索引；失败会记录警告，需通过统计接口和日志确认最终状态。

### 9.7 统计、管理、模型和导出

| 模块 | 方法与路径 |
| --- | --- |
| 统计 | GET /stats/ |
| 管理缓存 | POST /admin/cache/retrieval/clear；GET /admin/cache/retrieval/stats；GET /admin/cache-stats |
| 管理清理 | POST /admin/cleanup |
| 运行监控 | GET /admin/monitoring/tool-runtime；GET /admin/monitoring/runs/{run_id}/trace；GET /admin/monitoring/failures/summary；GET /admin/monitoring/usage/summary |
| 模型版本 | GET/POST /model-versions/；GET /model-versions/{name}/active；PUT /model-versions/{version_id}/deprecate；POST /model-versions/{version_id}/rollback |
| 数据导出 | GET /users/me/data-export |

## 10. 评估与多智能体实现

评估服务保存一次运行的 run_id，通过 LangGraph 图执行多个评估节点，读取问诊消息、患者资料和检索证据，最后写入结构化评估结果。节点结果、运行状态、token 使用、耗时、风险信息和引用信息分别落库或写入 trace，具体字段见 backend/app/models/evaluation*.py 与 backend/app/evaluation/。

当前运行时支持以下保护机制：

- 评估锁防重复提交，锁有状态迁移和心跳续期。
- 任务取消同时使用 Redis 取消标志和 Celery revoke；执行中的任务由评估侧协作退出。
- 网络/超时类失败才重试，业务校验错误不会盲目重试。
- 评估级 deadline 默认 240 秒；单个 Agent 默认超时 180 秒。
- 对话超过 6000 字符时可压缩上下文，保留最近 20 条完整消息。
- Tool Use 有最大轮数、调用次数、超时、结果字符数、重试和熔断限制。
- 安全红旗、证据不足或结果不确定时可以进入 needs_review，需人工复核，不应被当成自动通过。

提示词版本在 backend/app/prompts/manifest.json 和对应目录中维护。修改模型、提示词、Rubric、RAG 索引或安全规则后，应重新运行回归集并记录版本，不要仅凭单次手工对话判断质量。

## 11. RAG 知识库

知识库以医学 PDF 为输入，经过文本抽取、分块、Embedding/向量索引，并结合检索与重排为评估节点提供证据。代码中提供 BM25、Dense 及可选 Sparse/重排相关配置；默认情况下部分增强开关关闭，是否启用必须以环境变量为准。

常用参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| ACTIVE_INDEX_VERSION | rag-v1 | 当前索引版本标识 |
| RRF_WEIGHT_BM25 / DENSE / SPARSE | 0.30/0.45/0.25 | 多路检索融合权重 |
| ENABLE_METADATA_FILTER | false | 是否先做元数据/文档过滤 |
| ENABLE_DIVERSITY_RERANK | false | 是否限制单一来源占比 |
| ENABLE_CONTEXT_EXPANSION | false | 是否拼接相邻文本块 |
| ENABLE_CONTEXT_COMPRESSION | false | 是否在送入 LLM 前抽取关键句 |
| ENABLE_OCR | false | 扫描版 PDF 的 Qwen-VL OCR 兜底 |

建议操作顺序：

1. 将授权且脱敏的 PDF 放入 data/ 或通过管理员接口上传。
2. 使用 POST /knowledge-base/add-pdf 做单文件增量入库，或使用 POST /knowledge-base/rebuild 全量构建。
3. 通过 /knowledge-base/stats 和 /rebuild/status 检查块数、来源和构建状态。
4. 改变索引或 Embedding 参数后清理缓存，并用 backend/scripts/eval/evaluate_retrieval.py 或 RAG 测试验证召回质量。

知识库文件和 Chroma 数据卷都属于持久化数据。升级前备份，不能把大型索引直接当成可随意删除的临时文件。

## 12. 数据模型

核心表由 backend/app/models/__init__.py 汇总：

| 表 | 用途 |
| --- | --- |
| users | 用户、角色、权限和认证相关信息 |
| virtual_patients | 虚拟患者资料、病例配置、难度和 case_id |
| consultations | 医生与虚拟患者的一次问诊会话 |
| consultation_messages | 会话消息和会话内唯一序号 |
| evaluations | 五维评分、总分、建议、证据、状态和风险数据 |
| evaluation_runs | 一次评估运行的状态、耗时和 token 等运行信息 |
| evaluation_node_results | 各评估节点结果和审计信息 |
| evaluation_checkpoints | LangGraph checkpoint 关联数据 |
| evaluation_locks | 防止重复评估及锁状态 |
| review_records | 人工复核结果和状态迁移 |
| model_versions | 模型版本注册、激活、弃用和回滚 |
| audit_logs | 登录、评估、导出和管理操作审计 |

涉及结构变化时同时更新 ORM 模型、Pydantic schema、Alembic migration 和 database/init.sql， 并为并发、唯一性和权限补充回归测试。

## 13. 监控、日志与运维

- /health 检查 MySQL、Redis 和 LangGraph checkpointer 状态；容器健康检查依赖该接口。
- /metrics 暴露 HTTP 请求量和耗时等 Prometheus 指标。生产环境应设置 METRICS_TOKEN，避免公开暴露指标。
- 每个 HTTP 请求生成或复用 X-Request-ID，日志中记录方法、路径、状态和耗时。
- 生产环境追加安全响应头，/docs、/redoc 和 /openapi.json 为文档资源例外路径。
- LANGFUSE_ENABLED 可选；启用前必须配置对应公钥、私钥和地址。
- Celery Worker、Beat、backend、MySQL 和 Redis 分别查看日志，不要只看前端浏览器日志。
- Beat 每 24 小时触发 cleanup_expired_records：审计日志默认保留 90 天，评估运行和节点结果默认保留 180 天。

监控 profile 的默认地址为：Prometheus http://localhost:9090，Grafana http://localhost:3000。Grafana dashboard 和数据源配置位于 monitoring/grafana/。

## 14. 测试与质量检查

后端在 backend 目录执行：

~~~powershell
cd backend
venv\Scripts\python.exe -m ruff check .
venv\Scripts\python.exe -m compileall -q .
venv\Scripts\python.exe -m pytest tests/ -q
~~~

前端在 frontend 目录执行：

~~~powershell
npm run lint
npm test
npm run build
~~~

测试范围包括认证与权限、API、WebSocket、评估服务、LangGraph 编排、Tool Use、RAG 检索、缓存、数据治理和 E2E。涉及评估流程的改动至少应运行对应 tests/services、tests/orchestration、tests/evaluation 或 tests/tasks 子集，并记录真实结果。测试模式由配置中的 TESTING 控制；测试模式可同步执行评估，生产模式默认通过 Celery 异步执行。

提交前建议额外执行：

~~~powershell
git diff --check
docker compose config
~~~

docker compose config 需要根目录 .env 中的必填变量已经提供；它只校验 Compose 展开结果，不会替代容器健康检查。

## 15. 故障排查

### API 启动失败

先检查 Python 解释器是否来自 backend\venv，再检查 MySQL/Redis 是否可连通、backend/.env 是否存在、SECRET_KEY 和 API Key 是否填写。使用 python -m uvicorn app.main:app 时必须位于 backend 目录；在仓库根目录使用该命令会因模块路径不同而失败。

### 评估一直 pending 或没有结果

确认 Celery Worker 已启动且使用同一 CELERY_BROKER_URL/CELERY_RESULT_BACKEND，Redis DB 4/5 可写；再查看 /evaluations/{consultation_id}/lock-status、Celery 日志和 /health。不要重复点击创建评估，系统会用评估锁返回 409。

### WebSocket 立即断开

客户端必须在连接后的 5 秒内发送 JSON 鉴权消息；检查 access token 是否有效、当前用户是否有该问诊访问权，以及反向代理是否转发 WebSocket Upgrade 头。

### RAG 没有结果

先用知识库统计接口确认 PDF 是否成功解析和入索引，再确认 ACTIVE_INDEX_VERSION、Chroma 数据卷、Embedding 缓存和 Qwen 配置。扫描版 PDF 默认不会自动 OCR，必要时显式启用 OCR 并评估成本和延迟。

### Alembic 迁移失败

先备份数据库，查看 alembic current 和 alembic history。当前新增唯一约束要求历史数据无重复 case_id 和消息序号；清理重复数据后再重试。不要通过删除生产表或重建数据卷绕过迁移。

### Docker 前端空白或 API 访问失败

检查 backend 健康状态和前端 Nginx 日志；开发模式使用 docker-compose.override.yml 时前端地址是 5173，基础生产栈前端地址是 80。修改宿主机端口后，应同步检查浏览器访问地址和 CORS 配置。

## 16. 相关文档与维护规则

- [平台说明](./platform-documentation.md)：平台背景和功能说明。
- [技术文档](./technical-document.md)：更细的技术设计记录。
- [评估基线](./evaluation-baseline.md)：评估、回归和基线相关说明。
- [患者评估与知识库重建](./patient-eval-and-kb-rebuild.md)：患者评估及知识库操作专题。
- [Prompt 与 Provider 适配](./prompt-and-provider-adapter.md)：提示词和模型提供方适配专题。
- [贡献指南](../CONTRIBUTING.md)：分支、提交和质量要求。
- [变更记录](../CHANGELOG.md)：版本变更历史。

当源码、Compose、配置、迁移或 API 发生变化时，应在同一变更中更新本手册或相关专题文档，并通过运行中的 OpenAPI 和实际命令校验文档，不记录未验证的测试数字、服务能力或生产结论。

## 17. 版本与许可

后端配置中的项目版本默认是 1.0.0；实际发布版本以 CHANGELOG.md 和 Git 标签为准。本项目的版权和使用限制以仓库中的项目声明为准，未经授权不得复制、修改、分发或商用。
