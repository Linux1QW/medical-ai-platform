# 基于多智能体的医生临床问诊评估平台

这是一个面向医学教育、标准化问诊训练和教学复核的全栈平台。医生用户与虚拟患者对话、提交诊断和治疗方案；后台通过 Celery 执行 LangGraph 多智能体评估，输出五维评分、分析、改进建议、知识库证据和人工复核状态。当前文档以分支 `codex/rag-bm25-optimization` 的当前代码为事实基线，覆盖已完成的 RAG/BM25 Tasks 1–8 实现。

> **医疗安全声明**：本项目仅用于教学、训练、研究和质量改进，不提供真实诊疗服务，也不能替代执业医师的诊断、处方、急救决策或人工审核。出现急危重症或现实医疗问题时，应立即转交合格医疗机构和专业人员。

> **隐私声明**：不要向模型、日志、测试集、知识库或版本库写入可识别患者身份的信息、生产病历、密钥或密码。导入数据前必须取得合法授权并完成最小化、去标识化和访问控制；日志、导出、备份与评测报告也属于敏感数据。

## 核心能力与架构

- React 19 + TypeScript + Ant Design 前端：注册登录、病例选择、模拟问诊、诊断提交、评估进度与结果、统计和管理员页面。
- FastAPI + SQLAlchemy：JWT/Redis 黑名单、医生/管理员权限、问诊资源归属校验、审计日志、REST、SSE 与 WebSocket。
- LangGraph 评估图：Safety 门控、Plan-Execute、两波 Send fan-out/fan-in、五个评估 Agent、确定性评分、Reflection、人工复核门和建议生成。
- RAG：医学词法 tokenizer、BM25、Chroma Dense、可选 BGE-M3 Sparse、加权 RRF、分级检索（Base → MQE → HyDE）、两阶段 rerank、引用与 generation 追踪。
- 不可变索引发布：Celery 构建候选 generation，校验 Chroma/BM25/可选 Sparse 与 manifest，Redis CAS 切换 active generation，再通过 Pub/Sub 通知 Worker 原子热加载。
- MySQL 8、Redis 7、Celery/Beat、Prometheus/Grafana 和 Docker Compose。

```text
Browser → React/Nginx → FastAPI → MySQL
                         ├─ Redis：checkpoint/cache/JWT/Celery/generation
                         ├─ Celery：评估、RAG 索引、数据清理
                         └─ RAG：Chroma + BM25 + optional Sparse → RRF/rerank
```

## 最短可验证启动

建议先走本地进程路径；它可以避开当前 Compose 已知的 Celery 与 RAG 挂载问题。要求 Python 3.10、Node.js 18、MySQL 8、Redis 7，以及真实 LLM/Embedding 调用所需的 API Key。

1. 创建一个**空的** MySQL 数据库 `medical_ai`，启动 Redis。
2. 在仓库根目录执行：

```powershell
cd backend
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 backend/.env：至少核对 MYSQL_*、SECRET_KEY、DASHSCOPE_API_KEY
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

3. 新终端启动前端：

```powershell
cd frontend
npm ci
npm run dev
```

默认入口：前端 <http://localhost:5173>、API 文档 <http://localhost:8000/docs>、OpenAPI <http://localhost:8000/api/v1/openapi.json>、健康检查 <http://localhost:8000/health>。

首次管理员建议使用交互脚本创建：

```powershell
cd backend
.\.venv\Scripts\python.exe scripts/init_admin.py
```

`database/seed.sql` 含演示管理员 `admin/admin123` 和演示病例，只能用于隔离开发环境；若使用，必须立即更换凭据，禁止用于生产。

## Celery 与 RAG 关键命令

本地 Windows 开发可使用 `solo` 池；Worker 负责评估和索引任务，Beat 每日投递留存清理任务：

```powershell
cd backend
.\.venv\Scripts\python.exe -m celery -A app.celery_app worker --loglevel=info -P solo
.\.venv\Scripts\python.exe -m celery -A app.celery_app beat --loglevel=info
```

生产知识库构建应由管理员 JWT 调用异步 API；源文件必须位于代码实际的 `PDF_DIR`，即仓库根目录 `data/`：

```bash
curl -H "Authorization: Bearer $TOKEN" \
  -X POST http://localhost:8000/api/v1/knowledge-base/rebuild

curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/knowledge-base/rebuild/status?task_id=<task-id>"
```

单文件添加/替换和删除分别使用：

```bash
curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"filename":"guide.pdf","force_replace":false}' \
  http://localhost:8000/api/v1/knowledge-base/add-pdf

curl -H "Authorization: Bearer $TOKEN" \
  -X DELETE http://localhost:8000/api/v1/knowledge-base/sources/<source-name>
```

`python -m app.services.rag.build_medical_index` 仍存在，但它调用旧兼容构建器，默认临时构建 `rag-v2` collection，**不会**完成 immutable generation 的 manifest、Redis CAS 与 Pub/Sub 发布；不要把它当作生产发布命令。

Task 8 提供真实索引评测与调参脚本，但仓库当前不能据此宣称候选 generation 已通过门禁：

```powershell
cd backend
.\.venv\Scripts\python.exe scripts/eval/evaluate_bm25.py `
  --output evaluation_reports/candidate.json `
  --compare evaluation_reports/baseline.json `
  --fail-on-regression

.\.venv\Scripts\python.exe scripts/eval/tune_weights.py `
  --dev-golden evaluation_reports/dev.json `
  --test-golden evaluation_reports/test.json `
  --retriever mqe --top-k 10 --primary ndcg@10 `
  --output evaluation_reports/tuning.json
```

调参只能用独立 dev split 选参，test split 只对胜出组合评估一次。CI 中的 mock RAG 阈值步骤验证的是评测管道，不代表真实候选性能；普通 CI 缺少真实 artifact 时会明确 SKIP measured gate，因此生产发布必须另设缺失输入即失败的 required gate。真实门禁还要求真实 active generation、真实基线和一致性遥测。详见 [PROJECT_GUIDE](docs/PROJECT_GUIDE.md#15-评测调参与质量门禁)。

## Docker Compose

Compose 定义 MySQL、Redis、FastAPI、Celery Worker、Celery Beat、Nginx 前端，以及可选 Prometheus/Grafana：

```powershell
docker compose up -d
docker compose --profile monitoring up -d
```

它目前不是无条件的一键生产路径。启动前必须处理三项已知差异：

- `backend` 服务未注入容器内 `CELERY_BROKER_URL`/`CELERY_RESULT_BACKEND`；
- RAG 代码读取 `/app/data`，Compose 却把宿主 `./data` 挂载到 `/app/backend/data/medical_pdfs`；
- `database/init.sql` 与当前 ORM/Alembic 基线并非完整等价，不能把 SQL 初始化和 Alembic baseline 直接串联。

完整修正建议、数据库路径和生产清单见 [PROJECT_GUIDE](docs/PROJECT_GUIDE.md#7-启动与部署)。

## 文档导航

- [PROJECT_GUIDE](docs/PROJECT_GUIDE.md)：唯一权威总手册；配置、API、数据、RAG/Celery、评测、运维与限制均以此为准。
- [技术架构说明](docs/technical-document.md)：面向开发者，聚焦模块边界、LangGraph、RAG generation 和 Celery 实现。
- [平台操作说明](docs/platform-documentation.md)：面向医生、教师/管理员和运维人员的操作流程。
- [Prompt 与 Provider 适配](docs/prompt-and-provider-adapter.md)：Prompt 文件与 LLM Provider 专题。
- [患者评测与知识库重建](docs/patient-eval-and-kb-rebuild.md)、[评测基线](docs/evaluation-baseline.md)：历史专题；如与总手册冲突，以代码和总手册为准。
- [贡献指南](CONTRIBUTING.md)、[变更记录](CHANGELOG.md)。

## 验证

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

CI 还执行后端覆盖率门槛、RAG mock/Task 8 合约测试、迁移检查、前端构建和告警模式安全扫描。文档中的接口和默认值应继续以 `backend/app/core/config.py`、`backend/.env.example`、路由源码、Compose、迁移和测试为准。
