# 基于多智能体的医生临床问诊评估平台

一个面向医生临床问诊训练与考核场景的 AI 评估平台：医生与「虚拟患者」进行模拟问诊，平台通过 **LangGraph 编排的多智能体系统** 对问诊全过程进行 **五维度自动评估**（问诊分析 / 医学知识核对 / 人文关怀 / 诊断评估 / 治疗方案），并结合 **混合 RAG 检索 200+ 本医学指南与教材** 为每条评分提供可追溯的循证依据。

**核心技术栈**

| 层次 | 技术选型 |
|---|---|
| 后端 | Python 3.10 + FastAPI 0.115（异步非阻塞） |
| 前端 | React 19 + TypeScript + Vite 7 + Ant Design 6 |
| 智能体编排 | LangGraph（Wave DAG 三波次并行） |
| LLM | 阿里云百炼 Qwen API（默认 `qwen3.7-max`，支持模型路由与跨 Provider 熔断） |
| 向量存储 | ChromaDB |
| 关键词检索 | bm25s（自实现医学分词 + RRF 融合） |
| 数据库 | MySQL 8.0（Alembic 管理迁移） |
| 缓存 / 队列 | Redis（Checkpoint / 检索缓存 / Celery broker） |
| 异步任务 | Celery worker + Celery beat（定时任务） |
| 可观测性 | Prometheus + Grafana + Langfuse 链路追踪 |

---

## 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [多智能体系统](#多智能体系统)
- [RAG 检索系统](#rag-检索系统)
- [后端架构](#后端架构)
- [前端架构](#前端架构)
- [数据库设计](#数据库设计)
- [快速开始](#快速开始)
- [Docker 部署](#docker-部署)
- [测试与代码质量](#测试与代码质量)
- [可观测性与监控](#可观测性与监控)
- [安全设计](#安全设计)
- [项目目录结构](#项目目录结构)
- [版本与贡献](#版本与贡献)

---

## 功能特性

- **虚拟患者模拟问诊**：基于 Prompt 角色控制（无微调）构建具有稳定人格、病情设定的虚拟患者，支持 150+ 病例数据集；滑动窗口记忆压缩保证长对话上下文一致性
- **五维度自动评估**：问诊结束后一键发起评估，多智能体并行打分并生成结构化报告（分数 + 分析 + 循证引用 + 改进建议）
- **循证引用可追溯**：知识核对结论附带引用（来源文献 / 页码 / 标题路径 / 原文片段），且经过引用合法性校验，杜绝幻觉引用
- **实时进度推送**：评估过程通过 WebSocket 推送节点级进度（0-100%）
- **人工复核闭环**：证据不足或立场无法确定时自动转人工复核，复核后基于 LangGraph Checkpoint 从断点恢复评估
- **病例难度自适应推荐**：基于「最近发展区」理论，根据医生历史表现推荐难度略高于当前能力的病例
- **数据统计看板**：个人 / 团队多维度评估数据分析与趋势图表

## 系统架构

### 整体架构

```
[React 前端] <——WebSocket/REST——> [FastAPI 后端] ——> [LangGraph 编排引擎]
                                        |                    |
                                   MySQL / Redis      [5 个评估 Agent 并行]
                                                             |
                                                    [RAG 混合检索系统]
                                                             |
                                              [ChromaDB + BM25 索引 ← 200+ 医学指南 PDF]
```

### Wave DAG 三波次编排

评估流程由 LangGraph 图状态机驱动，按数据依赖分为三个波次，波内并行、波间串行：

| 波次 | 节点 | 说明 |
|---|---|---|
| Wave 1 | inquiry、humanistic、knowledge | 三个基础评估 Agent 并行执行 |
| Wave 2 | diagnosis、treatment | 消费 Wave 1 的知识核对证据后并行执行 |
| Wave 3 | scoring、suggestion | 汇总五维分数，生成总分与改进建议 |

- **证据链传递**：Knowledge Agent 检索到的指南证据（citations）向下游 Diagnosis / Treatment Agent 传递，实现端到端证据可追溯
- **Checkpoint 恢复**：图状态持久化到 Redis，人工复核 / 异常中断后可从断点继续，不重复消耗 Token
- **影子模式**：`LANGGRAPH_SHADOW_MODE` 支持新旧编排引擎并行对比验证

## 多智能体系统

### 五维评估智能体

1. **问诊分析智能体（inquiry）** — 评估病史采集的全面性（主诉 / 现病史 / 既往史 / 过敏史等覆盖度）与问诊技巧
2. **医学知识核对智能体（knowledge）** — 检索医学指南，核对诊断与治疗方案和循证医学证据的一致性
3. **人文关怀智能体（humanistic）** — 评估医患沟通质量、共情表达与人文关怀体现
4. **诊断评估智能体（diagnosis）** — 结合指南证据评估诊断准确性与鉴别诊断逻辑
5. **治疗方案智能体（treatment）** — 评估治疗方案的合理性、安全性（用药禁忌 / 剂量 / 随访计划）

### 知识核对的三种推理模式

知识核对智能体已拆分为 `app/services/agents/knowledge/` 包，内含三种可切换的推理模式：

| 模式 | 入口 | 机制 | 配置开关 |
|---|---|---|---|
| RAG 管线 | `pipeline.run_knowledge_check` | 确定性流程：事实提取 → 查询构建 → 分级检索 → 两阶段重排 → LLM 一致性判断 | 默认兜底 |
| Tool Use | `tool_use.run_knowledge_check_with_tools` | LLM 通过 Function Calling 自主调用检索 / 查询扩展 / HyDE 工具，带调用预算与引用校验（含一次修正重试） | `ENABLE_TOOL_USE` |
| ReAct | `react.run_knowledge_check_react` | 显式 Thought → Action → Observation 推理链，推理过程全程可见（react_trace），达步数上限强制收敛 | `ENABLE_REACT_KNOWLEDGE` |

三种模式返回结构完全兼容；评分映射采用**确定性函数**（consistency × confidence → 分数区间），禁止 LLM 直接给分，保证评分可复现。

### 工具系统

- 统一 ToolRegistry 注册 + ToolExecutor 执行 + ToolBudget 预算控制（限制单次评估的 RAG / MQE / HyDE 调用次数）
- `verify_citation` 引用校验工具：核对 LLM 使用的引用 ID 是否真实存在于检索结果，非法引用触发修正或强制失败

## RAG 检索系统

### 三路融合检索

| 检索路 | 权重 | 实现 |
|---|---|---|
| BM25 关键词 | 0.30 | `bm25s` 引擎（较 `rank_bm25` 索引速度 10x、内存效率 5x），jieba 医学分词 + 760+ 术语词典 |
| Dense 语义向量 | 0.45 | BGE 系列嵌入模型 + ChromaDB |
| Learned Sparse（可选） | 0.25 | BGE-M3 稀疏表示，`BGE_M3_ENABLED` 控制，关闭时自动降级两路融合 |

融合采用 **加权 RRF（Weighted Reciprocal Rank Fusion，k=60）**，权重可通过 `RRF_WEIGHT_*` 环境变量调整。

### 分级检索（L1 → L2 → L3 级联）

- **L1 Base**：三路融合基础召回
- **L2 MQE**：多查询扩展（Multi-Query Expansion），带语义漂移防护，提升召回覆盖率
- **L3 HyDE**：假设性文档嵌入，处理复杂语义查询

### CRAG 置信度闸门

- **HIGH**（多来源、高分、覆盖充分）→ 直接使用
- **MEDIUM**（部分满足）→ 触发 MQE / HyDE 增强检索
- **LOW**（严重不足）→ 拒答并转人工复核，绝不强行给分

### 两阶段重排

1. DashScope `gte-rerank` 专用重排模型粗排（20 → 10）
2. LLM Cross-Encoder 精排（10 → 5），按相关性 / 完整性 / 权威性 / 时效性加权打分（`RERANK_W_*` 可配）

### 其他检索能力

- **医学实体归一化**：313 个 ICD/ATC 实体别名映射（如「心梗」→「急性心肌梗死」）
- **标题层级感知分块**：按指南标题层级切分并注入上下文前缀，表格内容单独抽取
- **检索结果缓存**：Redis 缓存（TTL 24h），以索引版本为键的一部分，索引重建自动失效
- **索引蓝绿发布**：`ACTIVE_INDEX_VERSION` 切换索引版本，支持 A/B 对比与平滑回滚
- **OCR 兜底**（可选）：扫描版 PDF 复用 Qwen-VL 做 OCR 抽取（`ENABLE_OCR`）
- **Small-to-Big / 上下文压缩**（可选）：检索命中后按窗口扩展上下文或抽取式压缩，降低 Token 消耗

## 后端架构

### API 路由（`/api/v1`）

| 路由 | 职责 |
|---|---|
| `/auth` | 登录 / 注册 / JWT 双令牌刷新 / 登出（黑名单） |
| `/patients` | 虚拟患者管理 |
| `/consultations` | 问诊会话与消息交互（含 WebSocket） |
| `/evaluations` | 评估发起 / 查询 / 进度推送 |
| `/stats` | 个人与团队统计 |
| `/knowledge-base` | 知识库文档与索引管理 |
| `/cases` | 病例难度自适应推荐 |
| `/review` | 人工复核工作流 |
| `/admin` | 用户 / 权限 / 系统管理 |
| `/model-versions` | 模型版本注册表 |
| `/data-export` | 评估数据批量导出 |

### 评估可靠性设计

- **四层防重复提交**：前端按钮防抖 → API 幂等校验 → 数据库行级锁 → UNIQUE 约束兜底
- **评估状态机**：`pending → running → completed / needs_review / failed`，失败支持 `retry_pending → retry_running` 重试链
- **LLM 高可用**：
  - 模型路由：按任务重要度分级使用 `MODEL_CRITICAL / STANDARD / LIGHTWEIGHT`
  - 跨 Provider 熔断（LLMFailoverManager）：主 Provider 连续失败自动切换备用 Provider（`LLM_PROVIDERS`）
  - LLM 客户端懒初始化（import 无副作用），并发信号量限流（`LLM_MAX_CONCURRENT`）
  - LLM 响应语义缓存（相似度阈值 0.95，TTL 24h）
- **统一 JSON 解析**：三层策略（标准 JSON → 代码块提取 → 截断修复），保证 LLM 结构化输出稳定落库
- **WebSocket 鉴权**：连接建立后由**首条消息传递 token** 完成认证（避免 token 暴露在 URL / 日志中），超时未认证自动断开

### 异步任务

- Celery worker 承接耗时评估任务，Celery beat 独立服务执行定时任务（数据留存清理、Token 统计汇总等）
- Broker / Backend 使用独立 Redis DB（`CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND`）

## 前端架构

### 页面结构

- **问诊页**：医生与虚拟患者实时对话界面
- **评估页**：五维雷达图、分维度评分卡、循证引用展示、改进建议
- **管理页**：病例 / 用户 / 知识库管理（RBAC 控制可见性）
- **统计页**：个人与团队评估数据趋势分析（Recharts）

### 关键机制

- **WebSocket 进度推送**：实时显示评估进度、当前节点与预计剩余时间
- **评估锁轮询恢复**：页面刷新 / 网络中断后轮询评估锁状态，自动恢复进度展示
- **构建优化**：Vite manualChunks 代码分割 + 路由懒加载，减小首屏体积
- **测试**：Vitest + Testing Library 覆盖关键页面组件（`npm test`）

## 数据库设计

### 核心表

| 表 | 说明 |
|---|---|
| `users` | 用户与 JSON 细粒度权限 |
| `virtual_patients` | 虚拟患者配置（人格 / 病情 / 难度） |
| `consultations` / `consultation_messages` | 问诊会话与消息历史 |
| `evaluations` | 五维评分结果与循证引用 |
| `evaluation_runs` | 评估运行记录（状态机 / 耗时 / Token 用量） |
| `evaluation_locks` | 评估防重锁 |
| `evaluation_checkpoints` | LangGraph 检查点 |
| `review_records` | 人工复核记录 |
| `model_versions` | 模型版本注册表 |
| `audit_logs` | 审计日志 |

### 迁移管理（Alembic）

数据库结构变更统一由 **Alembic** 管理（`backend/alembic/`），禁止手写迁移 SQL：

```bash
cd backend
# 应用全部迁移到最新
venv\Scripts\python.exe -m alembic upgrade head
# 修改 ORM 模型后生成新迁移
venv\Scripts\python.exe -m alembic revision --autogenerate -m "描述"
```

历史手写 SQL（migrate_v2-v10）已归档至 `database/archive/`，仅供追溯；`database/init.sql` + `seed.sql` 用于全新环境一次性建库与种子数据。

## 快速开始

### 环境要求

- Python 3.10+、Node.js 18+、MySQL 8.0、Redis 7.0+
- 阿里云百炼平台 API Key（[DashScope](https://dashscope.aliyuncs.com/)）

### 1. 后端

```bash
cd backend
python -m venv venv
venv\Scripts\activate              # Windows（Linux/macOS: source venv/bin/activate）
pip install -r requirements.txt

copy .env.example .env             # 按注释修改配置（见下）
```

`.env` 必配项（完整清单及默认值见 [.env.example](backend/.env.example)，与 `app/core/config.py` 分组一致）：

```env
# 数据库（建议使用仅授权业务库的专用账户，避免 root）
MYSQL_USER=medical_app
MYSQL_PASSWORD=<你的密码>
MYSQL_DATABASE=medical_ai

# JWT（生产环境必须显式配置，如 openssl rand -hex 32）
SECRET_KEY=<足够长的随机字符串>

# Qwen API（也可通过系统环境变量 DASHSCOPE_API_KEY 提供）
QWEN_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.7-max

# Redis
REDIS_CHECKPOINT_URL=redis://localhost:6379/1
```

### 2. 初始化数据库

```bash
# 建库（MySQL 中执行 database/init.sql + seed.sql），然后应用迁移：
venv\Scripts\python.exe -m alembic upgrade head
# 创建管理员账户
venv\Scripts\python.exe init_admin.py
```

### 3. 构建知识库索引（可选，知识核对功能依赖）

将医学指南 PDF 放入项目根 `data/` 目录后执行索引构建脚本（详见 `backend/scripts/`）。

### 4. 启动服务

```bash
# 后端 API（端口 8000）
venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Celery worker（另开终端）
venv\Scripts\python.exe -m celery -A app.celery_app worker -l info -P solo   # Windows 需 -P solo

# Celery beat 定时任务（另开终端，可选）
venv\Scripts\python.exe -m celery -A app.celery_app beat -l info
```

```bash
# 前端（端口 5173）
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173`，API 文档见 `http://localhost:8000/docs`。

## Docker 部署

`docker-compose.yml` 定义完整服务栈：**mysql、redis、backend、celery-worker、celery-beat、frontend**，以及 monitoring profile 下的 **prometheus、grafana**。

```bash
# 基础服务栈
docker compose up -d

# 含监控（需先在 .env 设置 GRAFANA_ADMIN_PASSWORD）
docker compose --profile monitoring up -d
```

- `docker-compose.override.yml`：本地开发覆盖（热重载 / 端口映射）
- `docker-compose.staging.yml` / `docker-compose.prod.yml`：多环境部署差异配置
- 网络隔离：`backend-net` 与 `frontend-net` 分离；数据卷持久化 MySQL / Redis / ChromaDB / Prometheus / Grafana

## 测试与代码质量

### 后端

```bash
cd backend
venv\Scripts\python.exe -m ruff check .            # lint：零违规（含 B / C90 规则）
venv\Scripts\python.exe -m pytest tests/ -q        # 当前 579 passed / 18 skipped
```

- 覆盖单元 / 集成 / 端到端测试，CI 强制覆盖率门槛 40%
- 重构遵循「先补测试锁定行为，再动代码」

### 前端

```bash
cd frontend
npm run lint     # ESLint 零违规
npm test         # Vitest + Testing Library
npm run build    # tsc -b 严格类型检查 + Vite 构建
```

### 工程规范

- `pre-commit` 钩子提交前自动检查（首次 `pre-commit install`）
- Conventional Commits 提交信息规范；PR 需 CI 全绿 + review 通过，squash 合并
- 详见 [CONTRIBUTING.md](CONTRIBUTING.md)

## 可观测性与监控

- **结构化 JSON 日志**：统一格式，便于集中收集（`LOG_FORMAT=json`）
- **Prometheus 指标**：`/metrics` 端点暴露 HTTP 请求量 / 延迟分布 / 检索命中率等；生产环境需 `METRICS_TOKEN` Bearer 认证
- **Grafana 面板**：`monitoring/` 预置数据源与总览 dashboard，随 monitoring profile 开箱即用
- **Langfuse 链路追踪**（可选）：端到端可视化 Agent 推理链路与 Token 消耗
- **告警**：钉钉 / 企微 Webhook 告警（`ALERT_WEBHOOK_URL`），LLM 错误率超阈值自动通知
- **成本管控**：TokenTracker 按模型 / 用户 / 维度统计用量，每日预算超限告警（`TOKEN_DAILY_LIMIT`）

## 安全设计

- **认证**：JWT Access + Refresh 双令牌；登出令牌进 Redis 黑名单
- **鉴权**：细粒度 RBAC（用户级 JSON 权限配置）
- **WebSocket**：首条消息传 token 鉴权，token 不出现在 URL 与访问日志
- **安全响应头**：全局中间件注入 `X-Content-Type-Options` / `X-Frame-Options` / `Referrer-Policy` 等
- **数据脱敏**：姓名 / 手机号 / 身份证号自动掩码
- **审计日志**：关键操作全量记录（`AUDIT_LOG_ENABLED`），支持留存策略自动清理
- **密钥管理**：`.env` 不入库；生产环境 `SECRET_KEY` 强制显式配置；数据库使用最小权限专用账户
- **传输加密**：HTTPS/TLS

## 项目目录结构

```
medical-ai-platform/
├── backend/
│   ├── app/
│   │   ├── api/v1/               # 11 个 API 路由模块
│   │   ├── core/                 # Settings 配置中心 / 安全 / 中间件
│   │   ├── models/               # SQLAlchemy ORM 模型
│   │   ├── orchestration/        # LangGraph Wave DAG 编排与节点适配器
│   │   ├── services/
│   │   │   ├── agents/           # 评估智能体
│   │   │   │   └── knowledge/    # 知识核对包（facts/queries/scoring/pipeline/tool_use/react）
│   │   │   ├── rag/              # 混合检索 / 重排 / 分块 / 索引
│   │   │   ├── tools/            # 工具注册 / 执行 / 预算
│   │   │   └── prompts/          # Prompt 版本管理
│   │   └── main.py
│   ├── alembic/                  # 数据库迁移（用法见其 README）
│   ├── scripts/                  # 索引构建等运维脚本
│   ├── tests/                    # pytest 测试（579 用例）
│   ├── .env.example              # 环境变量全量模板
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/ components/ layouts/   # 页面与组件
│   │   ├── api/ services/ store/         # 请求封装与状态
│   │   └── test/                         # Vitest 测试
│   └── package.json
├── database/                     # init.sql / seed.sql / archive（历史 SQL 归档）
├── monitoring/                   # prometheus.yml + Grafana provisioning
├── data/                         # 医学指南 / 教材 PDF（200+，不入库）
├── dataset/                      # 虚拟患者病例数据集（150+，不入库）
├── docker-compose*.yml           # 容器编排（base / override / staging / prod）
├── CHANGELOG.md
├── CONTRIBUTING.md
└── README.md
```

## 版本与贡献

- 版本遵循语义化版本（当前 `v1.0.0`），变更记录见 [CHANGELOG.md](CHANGELOG.md)
- 团队协作规范（分支 / 提交 / 质量门槛 / PR 流程）见 [CONTRIBUTING.md](CONTRIBUTING.md)

## 版权声明

Copyright © 2026。本项目为私有软件，保留所有权利（All Rights Reserved）。未经作者书面授权，任何个人或组织不得复制、修改、分发或商用本项目的任何部分。
