# 基于多智能体的医生临床问诊评估平台

面向医学教育和临床问诊训练的 AI 评估平台。用户与虚拟患者进行模拟问诊，系统保存问诊过程，并通过 LangGraph 多智能体流程从问诊分析、医学知识核对、人文关怀、诊断评估、治疗方案五个维度生成结构化评估结果、建议和知识库证据。

> 评估结果用于训练和教学质量改进，不替代真实临床诊断、处方或专业医疗审核。

## 文档入口

完整说明请阅读 [项目总手册](docs/PROJECT_GUIDE.md)。其中包含：

- 系统架构、代码目录、前后端页面和端到端业务流程；
- 本地开发、Docker Compose、MySQL/Alembic、Celery 和 RAG 知识库操作；
- 环境变量、认证权限、REST/WebSocket API 全目录；
- 数据模型、评估编排、监控、测试、故障排查和当前限制。

专题文档：

- [平台说明](docs/platform-documentation.md)
- [技术文档](docs/technical-document.md)
- [评估基线](docs/evaluation-baseline.md)
- [患者评估与知识库重建](docs/patient-eval-and-kb-rebuild.md)
- [Prompt 与 Provider 适配](docs/prompt-and-provider-adapter.md)
- [贡献指南](CONTRIBUTING.md)
- [变更记录](CHANGELOG.md)

## 技术栈

- 后端：Python、FastAPI、SQLAlchemy、Alembic、Pydantic
- AI 编排：LangGraph、多智能体评估、Qwen/DashScope 兼容 API
- 检索：ChromaDB、BM25、Embedding、可选重排和 OCR
- 异步任务：Celery + Redis
- 数据：MySQL 8、Redis 7
- 前端：React、TypeScript、Vite、Ant Design、WebSocket
- 部署：Docker Compose、Nginx；可选 Prometheus/Grafana

## 最短本地启动路径

前置条件：Python 3.10+、Node.js/npm、MySQL 8、Redis 7，以及真实评估所需的 Qwen/DashScope API Key。

### 1. 后端

~~~powershell
cd backend
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env，至少填写数据库、SECRET_KEY 和 Qwen API 配置
venv\Scripts\python.exe -m alembic upgrade head
venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
~~~

当前依赖包含 celery[redis]==5.6.3。Windows 本地执行异步评估时，另开终端运行：

~~~powershell
cd backend
venv\Scripts\python.exe -m celery -A app.celery_app worker --loglevel=info -P solo
venv\Scripts\python.exe -m celery -A app.celery_app beat --loglevel=info
~~~

### 2. 前端

~~~powershell
cd frontend
npm install
npm run dev
~~~

默认地址：

- 前端：http://localhost:5173
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health
- OpenAPI：http://localhost:8000/api/v1/openapi.json

### 3. Docker Compose

在仓库根目录创建 .env，至少设置 MYSQL_PASSWORD、SECRET_KEY、DASHSCOPE_API_KEY；启用监控时再设置 GRAFANA_ADMIN_PASSWORD：

~~~powershell
docker compose up -d
docker compose ps
docker compose logs -f backend
~~~

启用 Prometheus/Grafana：

~~~powershell
docker compose --profile monitoring up -d
~~~

基础端口：前端 80、后端 8000、MySQL 3306、Redis 6379；监控端口：Prometheus 9090、Grafana 3000。完整配置、端口覆盖和证书说明见 [项目总手册](docs/PROJECT_GUIDE.md)。

## 数据库注意事项

- 全新数据库可选择 Alembic 路径（空库直接执行 `venv\Scripts\python.exe -m alembic upgrade head`），或选择 SQL/Compose 路径（执行 `database/init.sql` + `database/seed.sql`）；两条路径不要串联。
- Docker 的 `database/init.sql` 和 `database/seed.sql` 只在 MySQL 空数据卷第一次初始化时执行。已经由当前 `init.sql` 建好结构的数据库，不要无版本记录地直接执行 Alembic baseline；后续迁移接管前应确认结构后执行 `alembic stamp head`。
- 已有 Alembic 数据库升级前先备份，再在 backend 目录执行 `venv\Scripts\python.exe -m alembic upgrade head`。
- 当前迁移增加虚拟患者稳定 case_id，并为同一问诊的消息序号增加唯一约束；历史重复数据需先清理。
- 不要提交 .env、真实 API Key、密码、医疗隐私数据或生产数据卷。

## 验证命令

~~~powershell
cd backend
venv\Scripts\python.exe -m ruff check .
venv\Scripts\python.exe -m compileall -q .
venv\Scripts\python.exe -m pytest tests/ -q

cd ..\frontend
npm run lint
npm test
npm run build
~~~

变更源码、配置、数据库迁移或 API 时，请同步维护 [项目总手册](docs/PROJECT_GUIDE.md)，并以运行中的 /docs OpenAPI 和实际测试结果核对说明内容。

## RAG 索引与 Task 8 质量门禁

### 索引目录与构建

运行时目录按 generation 隔离，目录不存在时由构建流程创建：

- `backend/data/`：待索引的 PDF/DOCX 源文件（`PDF_DIR`）。
- `backend/data/medical_kb/`：按 `medical_guidelines_<generation>` 命名的 Chroma 持久化集合。
- `backend/data/rag_indexes/<generation>/`：该 generation 的 `manifest.json`、`bm25/` 和可选 `sparse/` artifact；`READY`、manifest 校验和组件 generation 必须一致。
- `backend/data/embed_cache/`：可复用的 embedding 中间产物，不是 active generation 的替代品。

生产环境通过 API 投递 Celery 任务，由 Worker 完成 snapshot → parse/chunk/embed → Chroma/BM25/Sparse → validate → switch → publish：

~~~bash
cd backend
python -m app.services.rag.build_medical_index
~~~

管理员更推荐使用异步接口，以便获得任务状态和 generation：

~~~bash
curl -X POST http://localhost:8000/api/v1/knowledge-base/rebuild
curl "http://localhost:8000/api/v1/knowledge-base/rebuild/status?task_id=<task-id>"
~~~

对应的 Celery task 为 `rebuild_rag_index`、`add_rag_index`、`replace_rag_index` 和 `delete_rag_index`。`POST /api/v1/knowledge-base/add-pdf`、`POST /api/v1/knowledge-base/rebuild` 与 `DELETE /api/v1/knowledge-base/sources/<source>` 返回 `task_id`；状态接口返回 `PENDING`、`PROGRESS`、`SUCCESS` 或 `FAILURE`，并在可用时附带 phase、generation、manifest 和错误信息。

### generation 切换、回滚与清理

每个 generation 是不可变候选。只有 manifest、Chroma、BM25（以及启用时的 Sparse）全部通过校验后，Worker 才会原子更新 Redis `rag:active_generation`，并发布 `rag:index-switched` 事件；各 Worker 收到事件后先加载新对象，加载失败则继续使用旧对象，不清空本地引用。缓存 key 必须包含 generation，避免切换后读取旧结果。

切换使用已构建且已验证的 candidate generation；回滚只把 Redis active 指针切回最近通过验证的 generation，并广播同一个切换事件，不重新构建索引。回滚后检查 `/api/v1/knowledge-base/rebuild/status`、manifest、`index_generation` trace 和检索结果，确认所有 Worker 已加载同一 generation。生产至少保留最近两个通过验证的 generation；旧 generation 仅在没有 Worker 引用、超过保留期且管理员确认后清理，同时保留对应审计/评估报告。

### BM25/RRF 调参与评估

调参只使用 dev split 选参，test split 对最终组合只评估一次；禁止用 test split 反复搜索。当前联合网格是 BM25 `k1=[0.9,1.2,1.5]`、`b=[0.5,0.7,0.8]`、`heading_boost=[1,2]`、`entity_boost=[1,2,3]`，以及 `RRF_K=[30,35,60]`。

~~~bash
cd backend
python scripts/eval/tune_weights.py \
  --dev-golden evaluation_reports/task8-dev.json \
  --test-golden evaluation_reports/task8-test.json \
  --retriever mqe --top-k 10 --primary ndcg@10 \
  --output evaluation_reports/task8-tuning.json
~~~

上述 `task8-dev.json` 与 `task8-test.json` 必须是相互独立、已审核的真实 split 产物；仓库内的示例 golden set 不能同时冒充两个 split。命令要求 active generation 已加载；没有真实索引时只应运行测试和脚本导入检查，不应填写或宣称候选性能。对已有实测 baseline 和 active generation，才运行：

~~~bash
python scripts/eval/evaluate_bm25.py \
  --compare evaluation_reports/bm25-v1.json \
  --fail-on-regression
python -m evaluation.rag_eval --mode both --split regression --fail-on-threshold
~~~

### 发布门禁、指标与故障处理

候选 generation 必须同时满足：overall Recall@10 和 nDCG@10 不低于 baseline；exact-term Recall@10 至少高于 baseline `0.05`；cold load ≤ `10s`；search p95 ≤ `5ms`；generation mismatch 和 stale cache hit 均为 `0`。分层报告必须覆盖 `disease_alias`、`drug_dose`、`gene_variant`、`lab_unit`、`negation`、`icd_code` 六类，并输出 Recall@1/3/5/10、MRR、nDCG@10。

Prometheus/trace 至少记录 `index_generation`、`bm25_load_seconds`、`bm25_query_seconds`、`bm25_candidates`、`bm25_top_score`、`lexical_expansion_count`、`filter_fallback`、`cache_hit`、`retrieval_level` 和各检索 channel 候选数。门禁失败时先保存报告和 generation/manifest，检查 Redis active 指针、Worker 切换事件、artifact `READY`/校验和、Chroma 文档数及缓存 generation；发生加载失败或 mismatch 时保持旧 generation 服务并回滚 active 指针。若 search p95 或 cold load 超阈值，先检查 Worker 冷启动、BM25 mmap、Chroma 持久化目录和候选规模，再重新评估，不能用 mock 或手工数字替代真实门禁。
