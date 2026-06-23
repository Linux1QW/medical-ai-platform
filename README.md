# 基于多智能体的医生临床问诊评估平台

## 项目简介

本平台是一个面向临床问诊实训的多智能体自动评估系统。医生可与虚拟患者进行模拟问诊，系统自动调用多个 AI 智能体，从问诊技巧、医学知识、人文关怀、诊断能力和治疗方案五个维度进行加权评估，并生成综合改进建议。核心技术亮点包括：RAG V2 分级检索与两阶段重排管线、统一 Pydantic 数据契约、五维加权评分模型（含拒答权重重分配）、MQE 全局预算控制、版本化索引管理与向后兼容，以及幂等数据库迁移。

## 系统架构

### 整体架构

```
问诊对话记录 → 问诊分析智能体 ─┐
              → 诊断评估智能体 ─┤（并行）
              → 治疗评估智能体 ─┤
              → 人文关怀智能体 ─┤
              → 知识核对智能体 ─┘（RAG 检索 + 一致性评估）
                                ↓
                        综合评分智能体（五维加权 + 权重重分配）
                                ↓
                        建议指导智能体
                                ↓
                          评估报告
```

前五个评估智能体通过 `asyncio.gather` 并行执行；综合评分与建议指导依次串行完成。所有智能体均由阿里云百炼平台 Qwen API 驱动，全局通过 `asyncio.Semaphore` 控制 LLM 并发调用数，防止 API 限流。

#### LangGraph 编排层（v5）

自 v5 版本起，评估流程由 **LangGraph StateGraph** 统一编排，替代原有的 `asyncio.gather` 硬编码方式。

**核心特性**：
- **状态图编排**：`EvaluationState` TypedDict 定义所有节点共享状态
- **Safety 门控**：确定性红旗规则优先 → LLM 语义补充 → fail closed 策略
- **动态路由**：基于咨询类型（initial/follow_up/emergency/communication）和提交状态决定执行哪些 Agent
- **并行分支**：通过 `Annotated[list, add]` reducer 支持多 Agent 并行结果合并
- **SQLite/Redis Checkpoint**：支持中断恢复和断点续传

**关键文件**：
- `backend/app/orchestration/state.py` — 状态定义
- `backend/app/orchestration/graph.py` — StateGraph 主图实现
- `backend/app/orchestration/checkpointer.py` — Redis Checkpoint 持久化
- `backend/app/orchestration/adapters/` — Agent 适配器模式（统一输出契约）
- `backend/app/orchestration/routes.py` — 路由矩阵与场景分类

**Feature Flag**：
```bash
LANGGRAPH_ENABLED=true   # 启用 LangGraph 编排（默认 false）
LANGGRAPH_SHADOW_MODE=true  # 影子模式：新旧路径并行对比
```

#### Function Call / Tool Use（v5.1）

知识 Agent 引入 **Function Calling** 能力，LLM 可自主调用检索工具完成医学证据检索和引用校验。

**核心设计**：
- **Agent 内部 Tool Use**：不改变 LangGraph 主编排，仅在知识 Agent 内部启用
- **工具白名单**：5 个医学检索工具 + 1 个引用校验工具，全部经过 Pydantic 参数校验
- **预算控制**：限制 RAG 调用次数（最多 3 次）、MQE 扩展（最多 2 次）、HyDE（最多 1 次）
- **确定性边界**：总分计算、Safety 门控、拒答规则保持代码控制，LLM 不可干预
- **Trace 审计**：所有工具调用记录写入 `tool_trace`，前端可追溯

**关键文件**：
- `backend/app/services/tools/base.py` — BaseTool 基类 + ToolContext
- `backend/app/services/tools/registry.py` — 工具注册表
- `backend/app/services/tools/executor.py` — 统一执行器（校验/预算/超时/截断）
- `backend/app/services/tools/medical_retrieval.py` — 4 个检索工具
- `backend/app/services/tools/citation.py` — 引用校验工具
- `backend/app/services/agents/knowledge_agent.py` — `run_knowledge_check_with_tools()`

**Feature Flag**：
```bash
ENABLE_TOOL_USE=true   # 启用知识 Agent Tool Use（默认 false）
TOOL_USE_FALLBACK_TO_LEGACY=true  # 失败时回退旧路径
```

#### 评分引擎重构

评分逻辑从单体拆分为三个独立组件，支持版本化策略和确定性计算。

**核心组件**：
- **ScoringPolicy**：版本化权重配置（`v1`, `v2`...），支持 A/B 测试
- **ScoreCalculator**：纯代码加权计算，禁止 None 临时权重重分配
- **SummaryGenerator**：LLM 摘要生成 + 五维确定性降级模板

**关键文件**：
- `backend/app/services/scoring/policies.py` — 评分策略
- `backend/app/services/scoring/calculator.py` — 确定性计算器
- `backend/app/services/scoring/summary.py` — 摘要生成器

### RAG 检索系统（V2）

知识核对智能体依托 RAG V2 检索管线，从 80+ 部医学教材与 CSCO/NCCN 指南中检索循证证据。核心特性：

- **统一数据契约（Pydantic Schema）** — `RetrievalQuery`、`EvidenceItem`、`RetrievalBundle`、`Citation`、`KnowledgeAssessment` 等结构化模型，作为检索子系统各模块间的强类型接口，替代原有 dict + raw_response 模式。
- **三类独立查询构建** — 从问诊对话中提取结构化病例事实（`ClinicalFacts`），分别构建病例查询（case）、诊断查询（diagnosis）和治疗查询（treatment），消除仅依赖"医生诊断+治疗方案"导致的确认偏误。
- **三级分级检索** — Level 1: BM25 + 向量混合检索 + RRF 融合；Level 2: LLM 多查询扩展（MQE，全局预算 ≤2 次扩展）+ 语义漂移过滤；Level 3: HyDE（假设文档 embedding，全局预算 ≤1 次调用）。每级判断召回充分性，足够则提前返回，避免不必要的 LLM 调用开销。
- **两阶段重排序** — Stage 1: DashScope gte-rerank 专用模型粗排（20→10），截断上限 800 字；Stage 2: LLM Cross-Encoder 精排（10→5），评估 relevance 与 completeness，未评分证据排后；最终排序融合权威性评分（authority_score）和时效性评分（freshness_score，动态年份基准）。
- **增强元数据** — 从 PDF 文件名和配置文件中解析机构、年份、版本、文档类型、科室、疾病标签、推荐等级、证据等级等 16 个元数据字段，支撑权威性与时效性计算。
- **拒答与引用追溯** — `retrieval_status`（检索充分性）与 `evidence_stance`（证据立场）分离判断；当检索不充分或证据立场不确定时，`score=None` 直接拒答，不填充默认分数。`Citation` 模型将 LLM 结论与知识库证据块绑定（`citation_id` 格式：`rag-v2:doc-hash:p{page}:c{seq}`），支持完整审计链。
- **版本化索引管理** — 支持 `rag-v1`、`rag-v2` 等多版本索引共存，通过 `ACTIVE_INDEX_VERSION` 配置热切换；构建脚本支持 `--version` 参数化构建；旧版本 collection 自动回退兼容。
- **召回充分性阈值** — 候选数 ≥3、查询类型覆盖 ≥2、不同来源 ≥2、RRF 分数 ≥0.015、向量相似度 ≥0.5。

### 评估维度

系统采用五维加权评分模型，各维度权重如下：

| 维度 | 权重 | 智能体 | 说明 |
|------|------|--------|------|
| 问诊技巧（inquiry） | 25% | 问诊分析智能体 | 评估问诊的系统性、完整性与临床规范 |
| 医学知识（knowledge） | 25% | 知识核对智能体 | 基于 RAG 检索对比临床指南，评估知识一致性 |
| 人文关怀（humanistic） | 20% | 人文关怀智能体 | 评估沟通态度、共情能力与患者教育 |
| 诊断能力（diagnosis） | 15% | 诊断评估智能体 | 评估诊断方向与鉴别诊断思路 |
| 治疗方案（treatment） | 15% | 治疗评估智能体 | 评估治疗方案的合理性与指南符合度 |

**权重重分配机制**：当某维度评分为 `None`（拒答/未评估）时，该维度权重自动重分配至其余有效维度，确保总分始终基于实际参评维度的加权比例计算。彻底移除了原有的默认 50 分兜底逻辑。

综合评分智能体汇总五个维度生成加权总分与综合摘要，建议指导智能体输出针对性改进建议。

## 技术栈

| 层级 | 技术 | 版本 |
|------|------|------|
| 前端 | React + TypeScript + Vite + Ant Design | React 19 / Vite 7 / Antd 6 |
| 后端 | FastAPI + Python | Python 3.10 / FastAPI 0.115 |
| 数据库 | MySQL | 8.0 |
| AI/LLM | 阿里云百炼平台 Qwen API | qwen3.7-max |
| 向量检索 | ChromaDB + text-embedding-v3 | — |
| 关键词检索 | BM25（rank_bm25） | — |
| 重排序 | DashScope gte-rerank + LLM Cross-Encoder | — |
| PDF 解析 | PyMuPDF | — |
| 并发控制 | asyncio.Semaphore（全局 LLM 限流） | — |
| 编排框架 | LangGraph（StateGraph 状态图编排） | — |
| 缓存/Checkpoint | Redis（生产环境 Checkpoint 持久化） | — |
| Function Calling | OpenAI SDK（Qwen Function Calling 兼容接口） | — |

## 项目结构

```
medical-ai-platform/
├── backend/                              # 后端服务（FastAPI）
│   ├── app/
│   │   ├── api/v1/                       # REST API 路由
│   │   │   ├── auth.py                   #   认证（注册/登录）
│   │   │   ├── patients.py               #   虚拟患者管理
│   │   │   ├── consultations.py          #   问诊交互
│   │   │   ├── evaluations.py            #   评估触发与报告
│   │   │   ├── knowledge_base.py         #   知识库管理
│   │   │   └── stats.py                  #   管理员统计
│   │   ├── core/                         # 核心基础设施
│   │   │   ├── config.py                 #   配置管理（含五维权重、索引版本）
│   │   │   ├── security.py               #   密码加密（bcrypt_sha256）+ JWT
│   │   │   └── deps.py                   #   认证依赖注入
│   │   ├── models/                       # 数据库模型（SQLAlchemy）
│   │   ├── schemas/                      # 请求/响应模型（Pydantic）
│   │   ├── services/
│   │   │   ├── agents/                   # AI 智能体
│   │   │   │   ├── inquiry_agent.py      #   问诊分析
│   │   │   │   ├── diagnosis_agent.py    #   诊断评估
│   │   │   │   ├── treatment_agent.py    #   治疗评估
│   │   │   │   ├── knowledge_agent.py    #   知识核对（RAG）
│   │   │   │   ├── humanistic_agent.py   #   人文关怀
│   │   │   │   ├── scoring_agent.py      #   综合评分（五维加权 + 权重重分配）
│   │   │   │   └── suggestion_agent.py   #   建议指导
│   │   │   ├── rag/                      # RAG 检索系统（V2）
│   │   │   │   ├── types.py              #   统一数据契约 + 阈值常量
│   │   │   │   ├── retriever.py          #   分级检索 + MQE 预算控制
│   │   │   │   ├── reranker.py           #   两阶段重排（配额优化/截断/动态年份）
│   │   │   │   ├── bm25_search.py        #   BM25 索引
│   │   │   │   ├── embeddings.py         #   Embedding 接口
│   │   │   │   ├── medical_store.py      #   ChromaDB 存储 + 版本兼容
│   │   │   │   ├── metadata_config.py    #   元数据配置
│   │   │   │   └── build_medical_index.py#   索引构建脚本（参数化版本）
│   │   │   ├── evaluation_service.py     #   评估编排器
│   │   │   ├── consultation_service.py   #   问诊逻辑
│   │   │   ├── qwen_client.py            #   Qwen API 客户端
│   │   │   ├── tools/                    # Function Call 工具系统
│   │   │   │   ├── base.py               #   BaseTool 基类 + ToolContext
│   │   │   │   ├── registry.py           #   工具注册表
│   │   │   │   ├── executor.py           #   统一执行器（校验/预算/超时/截断）
│   │   │   │   ├── medical_retrieval.py  #   医学检索工具
│   │   │   │   ├── citation.py           #   引用校验工具
│   │   │   │   └── scoring.py            #   评分工具
│   │   │   └── scoring/                  # 评分引擎
│   │   │       ├── policies.py           #   版本化策略
│   │   │       ├── calculator.py         #   确定性计算器
│   │   │       └── summary.py            #   摘要生成器
│   │   ├── orchestration/                # LangGraph 编排层
│   │   │   ├── state.py                  #   EvaluationState 状态定义
│   │   │   ├── graph.py                  #   StateGraph 主图
│   │   │   ├── checkpointer.py           #   Redis Checkpoint
│   │   │   ├── adapters/                 #   Agent 适配器
│   │   │   └── routes.py                 #   路由矩阵
│   │   └── db/session.py                 # 数据库连接
│   ├── tests/                            # 测试
│   │   ├── rag/                          #   RAG 单元测试与离线评测
│   │   ├── tools/                        #   Tool Use 单元测试
│   │   │   ├── test_registry.py
│   │   │   └── test_executor.py
│   │   ├── services/
│   │   │   └── test_qwen_client_tools.py #   qwen_client Tool Calling
│   │   ├── agents/
│   │   │   └── test_knowledge_agent_tool_use.py  # 知识 Agent Tool Use
│   │   ├── orchestration/
│   │   │   └── test_knowledge_adapter_tool_flag.py  # Feature Flag 切换
│   │   ├── test_auth_error_handling.py
│   │   └── test_auth_password_length.py
│   ├── requirements.txt
│   └── .env                              # 环境变量
├── frontend/                             # 前端应用（React + Vite）
│   ├── src/
│   │   ├── api/                          # 后端接口封装
│   │   ├── pages/                        # 页面组件
│   │   ├── store/useAuth.ts              # 状态管理
│   │   └── utils/request.ts              # Axios 封装
│   ├── vite.config.ts
│   └── package.json
├── database/
│   ├── init.sql                          # 建表 SQL
│   ├── migrate_v2.sql                    # 诊断/治疗字段 + 五维度评估
│   ├── migrate_v3.sql                    # 密码字段扩容
│   ├── migrate_v4.sql                    # RAG 审计字段（幂等迁移）
│   └── seed.sql                          # 种子数据
├── dataset/                              # 评测数据集（150+ 病例）
└── data/                                 # 医学教材与指南 PDF（80+ 部）
```

## 快速开始

### Docker Compose 一键部署（推荐）

使用 Docker Compose 可一键启动整个平台，包括 MySQL 数据库、Redis、后端和前端服务。

**环境要求**：Docker 20.10+ 和 Docker Compose 2.0+

#### 1. 配置环境变量

```powershell
# 复制环境变量模板
cp .env.docker .env

# 编辑 .env 文件，至少修改以下配置：
# - MYSQL_PASSWORD：数据库密码
# - SECRET_KEY：JWT 签名密钥（使用随机字符串）
# - DASHSCOPE_API_KEY：阿里云百炼平台 API Key（必填）
```

#### 2. 启动服务

```powershell
# 生产环境启动
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f

# 停止服务
docker compose down
```

#### 3. 本地开发环境

```powershell
# 使用开发配置启动（支持热重载）
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

#### 4. 访问服务

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端界面 | http://localhost | Nginx 托管的 React SPA |
| 后端 API | http://localhost:8000 | FastAPI 服务 |
| API 文档 | http://localhost:8000/docs | Swagger UI |
| MySQL | localhost:3306 | 数据库（仅开发环境暴露） |
| Redis | localhost:6379 | 缓存（仅开发环境暴露） |

#### 5. 数据持久化

以下数据通过 Docker Volume 持久化存储：

| Volume | 用途 |
|--------|------|
| `medical-ai-mysql-data` | MySQL 数据库文件 |
| `medical-ai-redis-data` | Redis AOF 持久化 |
| `medical-ai-chroma-data` | ChromaDB 向量数据库 |

```powershell
# 查看所有 volume
docker volume ls | findstr medical-ai

# 清除所有数据（谨慎！）
docker compose down -v
```

#### 6. 知识库构建

将医学教材 PDF 放入 `data/` 目录后，在容器内构建索引：

```powershell
# 进入 backend 容器
docker compose exec backend bash

# 构建 RAG 索引
python -m backend.app.services.rag.build_medical_index

# 指定版本构建
python -m backend.app.services.rag.build_medical_index --version rag-v2
```

### 手动安装部署

如果不使用 Docker，可按以下步骤手动安装。

### 环境要求

| 软件 | 最低版本 | 用途 |
|------|----------|------|
| Python | 3.10+ | 后端运行环境 |
| Node.js | 18+ | 前端运行环境 |
| MySQL | 8.0 | 数据存储 |
| Redis | 6.0+ | Checkpoint 持久化（LangGraph 生产环境，可选） |

此外，需在系统环境变量中配置阿里云百炼平台 API Key：

```
DASHSCOPE_API_KEY=sk-xxx
```

### 安装依赖

```powershell
# 后端
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 前端
cd frontend
npm install
```

### 数据库初始化与迁移

**新环境部署**（推荐）：只需执行 `init.sql`，已包含全部表结构和字段：

```powershell
# 1. 创建数据库
mysql -u root -p -e "CREATE DATABASE medical_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 2. 建表（init.sql 已包含所有表结构，含 LangGraph 编排相关表）
mysql -u root -p medical_ai < database/init.sql

# 3. 导入种子数据（含管理员账号 + 虚拟患者）
mysql -u root -p medical_ai < database/seed.sql
```

**从旧版本升级**：按顺序执行增量迁移脚本（均可重复执行，已存在的列自动跳过）：

```powershell
mysql -u root -p medical_ai < database/migrate_v2.sql
mysql -u root -p medical_ai < database/migrate_v3.sql
mysql -u root -p medical_ai < database/migrate_v4.sql
```

> **注意**：`migrate_v5.sql` 的变更（`consultation_type` 字段、`evaluation_runs` / `evaluation_node_results` 表、`evaluations` 审计字段）已合并到 `init.sql`，不再提供独立迁移脚本。如需在已有数据库上添加这些结构，请手动执行对应 `ALTER TABLE` / `CREATE TABLE` 语句。

或使用初始化脚本：

```powershell
cd backend
python init_admin.py   # 交互式创建管理员账号
```

### 知识库构建与索引切换

将医学教材与指南 PDF 放入 `data/` 目录后，运行索引构建脚本：

```powershell
cd backend

# 构建默认版本索引
python -m app.services.rag.build_medical_index

# 指定版本构建
python -m app.services.rag.build_medical_index --version rag-v2
```

索引将存储在项目根目录的 ChromaDB 持久化目录中。通过 `.env` 中的 `ACTIVE_INDEX_VERSION` 切换活跃版本，旧版本 collection 自动回退兼容。

### 启动服务

```powershell
# 终端 1 — 后端（默认 8000 端口）
cd backend
.\venv\Scripts\Activate.ps1

# 启用 LangGraph 编排（可选）
$env:LANGGRAPH_ENABLED="true"

# 启用知识 Agent Tool Use（可选）
$env:ENABLE_TOOL_USE="true"

uvicorn app.main:app --reload --port 8000

# 终端 2 — 前端（默认 5173 端口）
cd frontend
npm run dev
```

访问 **http://localhost:5173** 即可使用。

## 数据库迁移

所有迁移脚本均采用幂等设计（存储过程 + `IF NOT EXISTS` 判断），可安全重复执行。

| 文件 | 说明 |
|------|------|
| `init.sql` | 初始建表：users、patients、consultations、evaluations、evaluation_runs、evaluation_node_results 等（含全部字段） |
| `migrate_v2.sql` | 新增诊断/治疗方案字段（`diagnosis`、`treatment_plan`）及五维度评估字段 |
| `migrate_v3.sql` | `users.hashed_password` 字段扩容为 TEXT |
| `migrate_v4.sql` | RAG 审计字段（幂等版本）：`citation_data`、`retrieval_status`、`evidence_stance`、`human_review_needed`、`review_reason`、`rag_trace_data`、`evaluation_status`；`knowledge_score` 和 `total_score` 允许 NULL |

> **说明**：原 `migrate_v5.sql`（LangGraph 编排支持：`consultation_type`、`evaluation_runs`、`evaluation_node_results`、`evaluations` 审计字段）已合并至 `init.sql`，新环境直接执行 `init.sql` 即可，无需额外迁移。

## 配置说明

所有配置项通过 `backend/.env` 或系统环境变量管理：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MYSQL_HOST` | localhost | MySQL 地址 |
| `MYSQL_PORT` | 3306 | MySQL 端口 |
| `MYSQL_USER` | root | MySQL 用户 |
| `MYSQL_PASSWORD` | （空） | MySQL 密码 |
| `MYSQL_DATABASE` | medical_ai | 数据库名 |
| `SECRET_KEY` | （内置默认值） | JWT 签名密钥 |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 1440 | Token 有效期（24 小时） |
| `DASHSCOPE_API_KEY` | （系统环境变量） | 阿里云百炼 API Key |
| `QWEN_API_BASE_URL` | dashscope 兼容模式 | Qwen API 地址 |
| `QWEN_MODEL` | qwen3.7-max | 默认 LLM 模型 |
| `RERANK_MODEL` | gte-rerank | 专用重排模型 |
| `LLM_MAX_CONCURRENT` | 10 | LLM 最大并发调用数 |
| `LLM_SEMAPHORE_TIMEOUT` | 60 | 信号量等待超时（秒） |
| `ACTIVE_INDEX_VERSION` | rag-v1 | 当前活跃 RAG 索引版本 |

**五维评分权重**（`scoring_agent.py`）：

| 维度 | 权重 |
|------|------|
| inquiry（问诊技巧） | 0.25 |
| knowledge（医学知识） | 0.25 |
| humanistic（人文关怀） | 0.20 |
| diagnosis（诊断能力） | 0.15 |
| treatment（治疗方案） | 0.15 |

**检索阈值常量**（`rag/types.py`）：

| 常量 | 值 | 说明 |
|------|----|------|
| `MIN_CANDIDATE_COUNT` | 3 | 最少候选证据数 |
| `MIN_QUERY_TYPE_COVERAGE` | 2 | 最少覆盖查询类型数 |
| `MIN_SOURCE_COUNT` | 2 | 最少不同来源数 |
| `MIN_RRF_SCORE` | 0.015 | RRF 融合最低分 |
| `MIN_VECTOR_SCORE` | 0.5 | 向量相似度最低阈值 |
| `MAX_MQE_EXPANSIONS` | 2 | MQE 全局预算上限 |
| `MAX_HYDE_CALLS` | 1 | HyDE 全局预算上限 |
| `MAX_RERANK_INPUT` | 20 | 专用 reranker 最大输入条数 |
| `LLM_RERANK_INPUT` | 5 | LLM 精排最大输入条数 |

**LangGraph 编排配置**：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LANGGRAPH_ENABLED` | false | 启用 LangGraph 编排 |
| `LANGGRAPH_SHADOW_MODE` | false | 影子模式：新旧路径并行对比 |
| `LANGGRAPH_GRAPH_VERSION` | evaluation-graph-v1 | 图版本标识 |
| `REDIS_CHECKPOINT_URL` | redis://localhost:6379/1 | Redis Checkpoint 连接地址 |
| `REDIS_CHECKPOINT_TTL` | 86400 | Checkpoint 过期时间（秒） |

**Function Call / Tool Use 配置**：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ENABLE_TOOL_USE` | false | 启用知识 Agent Tool Use |
| `TOOL_USE_MODEL` | qwen-max | Tool Use 专用模型 |
| `TOOL_USE_MAX_ROUNDS` | 4 | 最大工具调用轮次 |
| `TOOL_USE_MAX_CALLS` | 8 | 最大工具调用总次数 |
| `KNOWLEDGE_TOOL_MAX_RAG_CALLS` | 3 | RAG 检索最大调用次数 |
| `KNOWLEDGE_TOOL_MAX_MQE_CALLS` | 2 | MQE 扩展最大调用次数 |
| `KNOWLEDGE_TOOL_MAX_HYDE_CALLS` | 1 | HyDE 最大调用次数 |
| `TOOL_USE_FALLBACK_TO_LEGACY` | true | 失败时回退旧路径 |

## 测试

```powershell
cd backend

# 运行全部单元测试（186+ 个用例）
pytest tests/ -v --tb=short

# 仅运行 RAG 相关测试
pytest tests/rag/ -v

# 仅运行 Tool Use 相关测试
pytest tests/tools/ -v
pytest tests/services/test_qwen_client_tools.py -v
pytest tests/agents/test_knowledge_agent_tool_use.py -v

# 离线评测（评估 RAG 检索质量）
python tests/rag/eval_offline.py
```

**测试覆盖**：
- LangGraph 编排层：28 个基线测试 + 17 个图节点测试
- Tool Use：10 个 Registry 测试 + 10 个 Executor 测试 + 8 个 qwen_client 测试
- 知识 Agent：17 个分数映射测试 + 6 个 Feature Flag 测试
- 适配器：16 个适配器测试 + 13 个路由测试

## API 文档

后端启动后，访问自动生成的交互式 API 文档：

| 文档 | 地址 |
|------|------|
| Swagger UI | http://localhost:8000/api/v1/openapi.json |
| 健康检查 | http://localhost:8000/health |

### 主要接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/auth/register | 注册 |
| POST | /api/v1/auth/login | 登录 |
| GET | /api/v1/auth/me | 获取当前用户 |
| GET | /api/v1/patients/ | 虚拟患者列表 |
| POST | /api/v1/consultations/ | 创建问诊 |
| POST | /api/v1/consultations/{id}/messages | 发送消息 |
| POST | /api/v1/consultations/{id}/end | 结束问诊 |
| POST | /api/v1/evaluations/ | 触发评估 |
| GET | /api/v1/evaluations/{id} | 查看评估报告 |
| GET | /api/v1/knowledge-base/status | 知识库状态 |
| GET | /api/v1/stats/ | 管理员统计 |
