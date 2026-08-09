# 基于多智能体的医生临床问诊评估平台

这是一个面向医学教育、标准化问诊训练和教学复核的全栈平台。医生用户与虚拟患者对话、提交诊断和治疗方案；后台通过 Celery 执行 LangGraph 多智能体评估，输出五维评分、分析、改进建议、知识库证据和人工复核状态。当前版本 **v1.1.0**，覆盖 Task 0–16 全部迭代实现。

> **医疗安全声明**：本项目仅用于教学、训练、研究和质量改进，不提供真实诊疗服务，也不能替代执业医师的诊断、处方、急救决策或人工审核。出现急危重症或现实医疗问题时，应立即转交合格医疗机构和专业人员。所有评估报告均为**辅助质量评估，非自主诊断/治疗决策**。

> **隐私声明**：不要向模型、日志、测试集、知识库或版本库写入可识别患者身份的信息、生产病历、密钥或密码。导入数据前必须取得合法授权并完成最小化、去标识化和访问控制；日志、导出、备份与评测报告也属于敏感数据。

## 核心能力与架构

- React 19 + TypeScript + Ant Design 前端：注册登录、病例选择、模拟问诊、诊断提交、评估进度与结果、统计和管理员页面。
- FastAPI + SQLAlchemy：JWT/Redis 黑名单、医生/管理员权限、问诊资源归属校验、审计日志、REST、SSE 与 WebSocket。
- LangGraph 评估图：Safety 门控、Plan-Execute、两波 Send fan-out/fan-in、五个评估 Agent、确定性评分、Reflection、人工复核门和建议生成。
- RAG：医学词法 tokenizer、BM25、Chroma Dense、可选 BGE-M3 Sparse、加权 RRF、分级检索（Base → MQE → HyDE）、两阶段 rerank、引用与 generation 追踪。
- 不可变索引发布：Celery 构建候选 generation，校验 Chroma/BM25/可选 Sparse 与 manifest，Redis CAS 切换 active generation，再通过 Pub/Sub 通知 Worker 原子热加载。
- **Transactional Outbox + Dispatcher**：评估任务通过 Outbox 表与业务事务原子写入，独立 Dispatcher 进程轮询派发至 Celery，保证至少一次投递。
- **双 Redis 物理拓扑**：`redis-state`（AOF + noeviction）承载 checkpoint/broker/result/JWT/控制/进度；`redis-cache`（allkeys-LRU）承载 LLM 缓存和检索缓存。
- **EvaluationRun 状态机**：queued → running → completed/needs_review/failed/cancelled，含 retrying 重试路径和协作式取消。
- MySQL 8、双 Redis 7、Celery Worker/Beat、Evaluation Dispatcher、Prometheus/Grafana 和 Docker Compose。

```text
Browser → React/Nginx → FastAPI → MySQL
                         ├─ redis-state：checkpoint/broker/result/JWT/control/progress
                         ├─ redis-cache：LLM cache/retrieval cache
                         ├─ Celery Worker：评估、RAG 索引、数据清理
                         ├─ Dispatcher：Outbox → Celery 派发循环
                         └─ RAG：Chroma + BM25 + optional Sparse → RRF/rerank
```

## 10 分钟 Quickstart

前置要求：Python 3.10、Node.js 18、MySQL 8、Redis 7。

### 1. 复制环境变量并编辑

```powershell
cd backend
Copy-Item .env.example .env
# 编辑 .env：至少设置 MYSQL_PASSWORD、SECRET_KEY、DASHSCOPE_API_KEY
```

### 2. 创建空数据库并启动服务

先创建一个**空的** MySQL 数据库 `medical_ai`（不要先运行 `database/init.sql`），确保 Redis 在 `localhost:6379` 可达。

### 3. 安装依赖、迁移、启动后端

```powershell
cd backend
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. 启动 Celery Worker 和 Beat（另开终端）

```powershell
cd backend
.\.venv\Scripts\python.exe -m celery -A app.celery_app worker --loglevel=info -P solo
.\.venv\Scripts\python.exe -m celery -A app.celery_app beat --loglevel=info
```

### 5. 初始化管理员

```powershell
cd backend
.\.venv\Scripts\python.exe scripts/init_admin.py
```

### 6. 启动前端（另开终端）

```powershell
cd frontend
npm ci
npm run dev
```

### 7. 验证

- 前端：<http://localhost:5173>
- API 文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>
- 用初始化管理的账户登录

### 8. 停止

在前端和各个后端终端按 `Ctrl+C`。Compose 用户使用 `docker compose down`。

## 开发 vs 生产差异

| 项目 | 开发环境 | 生产环境 |
|---|---|---|
| 启动方式 | 本地进程（上述步骤） | `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d` |
| `ENVIRONMENT` | `development` | `production` |
| `SECRET_KEY` | 可临时使用默认值（有 warning） | **必须**配置安全随机密钥 |
| Celery Worker 池 | `-P solo`（Windows） | prefork + `--concurrency=N` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 可放宽（有 warning） | **不得超过 60 分钟** |
| `JWT_BLACKLIST_FAIL_CLOSED` | `false` 可接受 | **必须 `true`** |
| `/metrics` | 无 token 可访问 | **必须**配置 `METRICS_TOKEN` |
| TLS | 无 | 必须使用可信证书 |
| 数据库 | 空库 + Alembic | 空库 + Alembic + 备份验证 |
| Compose 迁移 | 手动 `alembic upgrade head` | `migrate` 服务自动执行 |

## Docker Compose

Compose 定义 MySQL、双 Redis（state + cache）、FastAPI、Evaluation Dispatcher、Celery Worker、Celery Beat、Nginx 前端，以及可选 Prometheus/Grafana：

```powershell
# 开发/本地
docker compose up -d

# 生产（使用 GHCR 镜像 + 4 Uvicorn workers）
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 启用监控
docker compose --profile monitoring up -d
```

Compose 的 `migrate` 服务会在启动时自动执行 Alembic 迁移。`backend` 依赖 `migrate` 成功后才启动。

## 环境变量

权威字段是 `backend/app/core/config.py`，模板是 `backend/.env.example`。关键变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` 时强制安全配置 |
| `SECRET_KEY` | 默认值（不安全） | 生产必须替换为随机字符串 |
| `MYSQL_HOST/PORT/USER/PASSWORD/DATABASE` | `localhost/3306/root/空/medical_ai` | 生产使用专用账户 |
| `REDIS_CHECKPOINT_URL` | `redis://localhost:6379/1` | LangGraph checkpoint |
| `CELERY_BROKER_URL` | `redis://localhost:6379/4` | Celery broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/5` | Celery result |
| `LLM_CACHE_REDIS_URL` | `redis://localhost:6380/0` | LLM 缓存（独立 redis-cache） |
| `RETRIEVAL_CACHE_REDIS_URL` | `redis://localhost:6380/1` | 检索缓存（独立 redis-cache） |
| `EVALUATION_CONTROL_REDIS_URL` | `redis://localhost:6379/8` | 取消标志和 task_id 映射 |
| `PROGRESS_REDIS_URL` | `redis://localhost:6379/6` | 跨进程进度广播 |
| `DASHSCOPE_API_KEY` | 空 | 阿里云百炼 API Key |
| `LANGGRAPH_ENABLED` | `true` | Feature Flag |
| `ENABLE_TOOL_USE` | `true` | Function Call / Tool Use |

完整变量列表见 `backend/.env.example` 和 [PROJECT_GUIDE](docs/PROJECT_GUIDE.md#6-配置与默认值)。

## 文档导航

- [PROJECT_GUIDE](docs/PROJECT_GUIDE.md)：唯一权威总手册；配置、API、数据、RAG/Celery、评测、运维与限制均以此为准。
- [技术架构说明](docs/technical-document.md)：面向开发者，聚焦状态机、Outbox/Dispatcher、双 Redis、LangGraph、RAG generation 和数据保留。
- [平台操作说明](docs/platform-documentation.md)：面向医生、教师/管理员和运维人员的操作流程。
- [评测基线](docs/evaluation-baseline.md)：五维分数语义、图节点清单、复核反馈闭环。
- [Prompt 与 Provider 适配](docs/prompt-and-provider-adapter.md)：Prompt 文件与 LLM Provider 专题。
- [贡献指南](CONTRIBUTING.md)、[变更记录](CHANGELOG.md)。

## 验证

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

CI 还执行后端覆盖率门槛、RAG mock/Task 8 合约测试、迁移检查、前端构建和告警模式安全扫描。文档中的接口和默认值应继续以 `backend/app/core/config.py`、`backend/.env.example`、路由源码、Compose、迁移和测试为准。
