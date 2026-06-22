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
│   │   │   └── qwen_client.py            #   Qwen API 客户端
│   │   └── db/session.py                 # 数据库连接
│   ├── tests/                            # 测试
│   │   ├── rag/                          #   RAG 单元测试与离线评测
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

### 环境要求

| 软件 | 最低版本 | 用途 |
|------|----------|------|
| Python | 3.10+ | 后端运行环境 |
| Node.js | 18+ | 前端运行环境 |
| MySQL | 8.0 | 数据存储 |

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

```powershell
# 1. 创建数据库
mysql -u root -p -e "CREATE DATABASE medical_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 2. 建表
mysql -u root -p medical_ai < database/init.sql

# 3. 导入种子数据（含管理员账号 + 虚拟患者）
mysql -u root -p medical_ai < database/seed.sql

# 4. 执行增量迁移（可重复执行，已存在的列自动跳过）
mysql -u root -p medical_ai < database/migrate_v2.sql
mysql -u root -p medical_ai < database/migrate_v3.sql
mysql -u root -p medical_ai < database/migrate_v4.sql
```

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
| `init.sql` | 初始建表：users、patients、consultations、evaluations 等 |
| `migrate_v2.sql` | 新增诊断/治疗方案字段（`diagnosis`、`treatment_plan`）及五维度评估字段 |
| `migrate_v3.sql` | `users.hashed_password` 字段扩容为 TEXT |
| `migrate_v4.sql` | RAG 审计字段（幂等版本）：`citation_data`、`retrieval_status`、`evidence_stance`、`human_review_needed`、`review_reason`、`rag_trace_data`、`evaluation_status`；`knowledge_score` 和 `total_score` 允许 NULL |

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

## 测试

```powershell
cd backend

# 运行全部单元测试
pytest tests/ -v

# 仅运行 RAG 相关测试
pytest tests/rag/ -v

# 离线评测（评估 RAG 检索质量）
python tests/rag/eval_offline.py
```

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
