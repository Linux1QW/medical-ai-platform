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
