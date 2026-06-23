# 基于多智能体的医生临床问诊评估平台 — 技术文档

---

## 目录

1. [项目概述](#1-项目概述)
2. [新功能详解](#2-新功能详解)
3. [部署指南](#3-部署指南)
4. [使用指南](#4-使用指南)
5. [API 文档](#5-api-文档)
6. [故障排除](#6-故障排除)

---

## 1. 项目概述

### 1.1 平台简介

**基于多智能体的医生临床问诊评估平台** 是一套面向医学教育与临床质量控制的智能评估系统。平台通过多个 AI 智能体（Agent）协作，对医生的临床问诊过程进行全方位、多维度的自动化评估，帮助医疗机构提升问诊质量、规范诊疗行为。

平台的核心价值：
- **自动化评估**：替代人工评审，实现对问诊过程的全自动多维度打分
- **循证医学支撑**：通过 RAG（检索增强生成）系统，从权威医学教材与指南中检索证据，确保评估的权威性
- **智能体协作**：采用多智能体架构，各 Agent 专注不同评估维度，实现关注点分离
- **Tool Use 增强**：支持 Function Call / Tool Use 模式，让 Agent 在评估过程中主动调用检索工具获取最新证据

### 1.2 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend (React + Vite + Nginx)             │
│                        端口: 80 / 5173 (开发)                        │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ HTTP / WebSocket
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Backend (FastAPI + Uvicorn)                       │
│                        端口: 8000                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              LangGraph 编排层 (StateGraph)                    │  │
│  │  START → load_context → classify → safety_check →            │  │
│  │  build_route_plan → [Send fan-out] → run_agent × N →         │  │
│  │  [fan-in] → aggregate → scoring → suggestion → END           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐   │
│  │ Safety     │ │ Inquiry    │ │ Diagnosis  │ │ Treatment    │   │
│  │ Agent      │ │ Agent      │ │ Agent      │ │ Agent        │   │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘   │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐   │
│  │ Knowledge Agent  │  │ Humanistic Agent                     │   │
│  │ (RAG/Tool Use)   │  │                                      │   │
│  └──────────────────┘  └──────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Tool Use 系统                                    │  │
│  │  RobustToolExecutor · BudgetManager · HealthChecker          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└───────────┬──────────────────────────────┬─────────────────────────┘
            │                              │
            ▼                              ▼
┌───────────────────────┐   ┌─────────────────────────────────────────┐
│  MySQL 8.0            │   │  Redis 7 (LangGraph Checkpoint + 缓存)  │
│  问诊记录/评估结果    │   │  状态持久化                               │
└───────────────────────┘   └─────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────────┐
│  ChromaDB 向量数据库 + RAG 检索管道                                   │
│  医学教材 PDF → 分块 → Embedding → 向量索引 → Reranker → 引用生成    │
└───────────────────────────────────────────────────────────────────────┘
```

### 1.3 技术栈

| 层次 | 技术 |
|------|------|
| 前端 | React 18 + TypeScript + Vite + Ant Design |
| 后端 | Python 3.10 + FastAPI + SQLAlchemy (async) |
| AI 模型 | 阿里云百炼平台 Qwen (qwen3.7-max / qwen-max) |
| 编排引擎 | LangGraph (StateGraph + Send fan-out/fan-in) |
| 向量数据库 | ChromaDB |
| 关系数据库 | MySQL 8.0 |
| 缓存/状态 | Redis 7 |
| Reranker | gte-rerank (阿里云 DashScope) |
| 容器化 | Docker + Docker Compose |

### 1.4 主要功能模块

#### 五维评估体系

平台通过五个核心 Agent 对问诊质量进行评估：

| 维度 | Agent | 说明 |
|------|-------|------|
| 安全性检查 | `SafetyAgent` | 识别高风险问诊，触发人工复核 |
| 问诊分析 | `InquiryAgent` | 评估问诊的完整性、条理性、针对性 |
| 诊断评估 | `DiagnosisAgent` | 评估诊断的准确性、鉴别诊断的充分性 |
| 治疗评估 | `TreatmentAgent` | 评估治疗方案的合理性、规范性 |
| 医学知识一致性 | `KnowledgeAgent` | 基于 RAG/Tool Use 验证医学知识的准确性 |
| 人文关怀 | `HumanisticAgent` | 评估医患沟通、人文关怀表现 |

#### 其他核心功能

- **虚拟患者系统**：模拟真实患者与医生进行问诊对话
- **实时进度推送**：通过 WebSocket 实时推送评估进度
- **评分策略引擎**：确定性评分 + 策略模式，支持自定义评分规则
- **管理员后台**：全平台问诊记录多维度筛选与统计

---

## 2. 新功能详解

### 2.1 RAG / Tool Use 评估指标体系

为了量化 RAG 检索系统和 Tool Use 功能的质量，平台建立了一套完整的评估指标体系，覆盖检索、引用、拒答、工具使用四大维度。

#### 2.1.1 数据模型

**`RagGoldCase`** — 标准测试用例

```python
class RagGoldCase(BaseModel):
    case_id: str                  # 唯一标识
    split: SplitType              # 数据集分割: dev / test / regression
    department: str               # 科室
    difficulty: DifficultyLevel   # 难度: easy / medium / hard
    
    # 病例信息
    chief_complaint: str          # 主诉
    patient_info: str             # 患者信息
    conversation_text: str        # 对话文本
    doctor_diagnosis: str         # 医生诊断
    treatment_plan: str           # 治疗方案
    
    # 标准答案
    gold_doc_ids: List[str]       # 标准相关文档 ID
    gold_citation_ids: List[str]  # 标准引用 ID
    expected_stance: StanceType   # 期望立场: supports/contradicts/mixed/undetermined
    should_refuse: bool           # 是否应该拒答
    
    # Tool Use 期望
    expected_tool_calls: List[Dict]     # 期望的工具调用
    expected_final_answer_keywords: List[str]  # 期望答案关键词
```

**`RagEvalResult`** — 评估运行结果

```python
class RagEvalResult(BaseModel):
    case_id: str
    mode: str                     # legacy / tooluse
    knowledge_score: Optional[float]  # 知识得分 (0-100)
    evaluation_status: str        # completed / needs_review
    retrieval_status: str         # sufficient / insufficient / error
    evidence_stance: Optional[StanceType]
    tool_trace: List[Dict]        # 工具调用追踪
    latency_ms: Optional[int]     # 延迟
    actual_tool_calls: List[Dict] # 实际工具调用
    system_refused: bool          # 系统是否拒答（计算字段）
    false_acceptance: bool        # 是否错误接受（计算字段）
```

**`RagEvalMetrics`** — 聚合指标模型

汇总所有用例的指标，字段覆盖：
- 检索指标：`recall_at_1/3/5`, `mrr`, `ndcg_at_5`
- 引用指标：`citation_validity`, `citation_hallucination_rate`, `citation_coverage`
- 拒答指标：`refusal_accuracy/precision/recall/f1`, `false_refusal_rate`, `false_acceptance_rate`
- 立场与分数：`stance_accuracy`, `score_range_accuracy`
- Tool Use：`tool_success_rate`, `tool_failure_rate`, `tool_budget_exceeded_rate`, `avg_tool_calls`

#### 2.1.2 指标计算

**检索指标**

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| Recall@K | 前 K 个结果中覆盖的标准文档比例 | `|top_k ∩ gold| / |gold|` |
| MRR | 第一个相关结果的排名倒数 | `1 / rank_of_first_relevant` |
| nDCG@K | 考虑相关性等级的归一化折损累积增益 | `DCG@K / IDCG@K` |
| MAP | 所有查询的平均精度均值 | `mean(AP_i)` |

**引用指标**

| 指标 | 说明 |
|------|------|
| Citation Validity | 引用有效率 = 有效引用数 / 总引用数 |
| Hallucination Rate | 引用幻觉率 = 幻觉引用数 / 总引用数 |
| Coverage | 引用覆盖率 = 覆盖的标准引用数 / 标准引用总数 |

**拒答指标**

基于混淆矩阵（TP=正确拒绝, FP=错误拒绝, FN=错误接受, TN=正确接受）：

| 指标 | 公式 |
|------|------|
| Accuracy | (TP + TN) / Total |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2 × Precision × Recall / (Precision + Recall) |
| False Positive Rate | FP / (FP + TN) |
| False Negative Rate | FN / (TP + FN) |

**Tool Use 指标**

| 指标 | 说明 |
|------|------|
| Tool Success Rate | 工具调用成功率 |
| Tool Failure Rate | 工具调用失败率 (error + timeout) |
| Budget Exceeded Rate | 预算超限率 |
| Avg Tool Calls | 平均每案例工具调用次数 |
| Avg Latency | 平均调用耗时 (ms) |
| Keyword Coverage | 最终答案关键词覆盖率 |
| Tool Call Accuracy | 工具调用准确率（与期望对比） |

#### 2.1.3 评估运行器

平台提供三种评估运行器：

```python
# 传统 RAG 评估
await run_legacy_rag_evaluation(cases_path, split="dev", limit=10)

# Tool Use 评估
await run_tool_use_evaluation(cases_path, split="dev", limit=10)

# 对比批量评估（同时运行两种模式并生成对比报告）
await run_batch_evaluation(cases_path, split="dev", max_concurrency=4)
```

**评估模式说明：**

| 模式 | 说明 |
|------|------|
| `legacy` | 传统 RAG 流程，直接调用 `run_knowledge_check` |
| `tooluse` | Tool Use 流程，调用 `run_knowledge_check_with_tools` |
| `both` | 对同一案例依次运行两种模式 |
| `mock` | 返回模拟结果，用于冒烟测试 |

#### 2.1.4 报告生成

评估完成后，系统自动生成结构化报告：

```python
from evaluation.report import generate_json_report, generate_markdown_report

# 生成 JSON 报告
report = generate_json_report(results, gold_cases, mode="tooluse", ...)

# 生成 Markdown 报告
md_content = generate_markdown_report(report)
```

**报告内容包含：**
- 合规性摘要（P0/P1/P2 三级阈值检查）
- 检索、引用、拒答、Tool Use 各维度指标
- 工具调用明细表
- 按难度/科室分组统计
- 失败样本列表
- 改进建议

**阈值等级定义：**

| 等级 | 说明 | 示例 |
|------|------|------|
| P0（核心门槛） | 必须满足，否则整体不通过 | 引用有效性 ≥ 100%，幻觉率 ≤ 5% |
| P1（重要指标） | 应当满足 | 拒答准确率 ≥ 80%，立场准确率 ≥ 70% |
| P2（参考指标） | 建议满足 | Recall@5 ≥ 50%，工具成功率 ≥ 80% |

#### 2.1.5 Gold Cases 数据集

数据集以 JSONL 格式存储，位于 `backend/evaluation/rag_cases/rag_gold_cases.jsonl`。

每条记录包含一个完整的测试用例，字段涵盖：
- 病例基本信息（主诉、患者信息、对话内容、诊断、治疗方案）
- 标准检索结果（gold_doc_ids, gold_queries）
- 标准引用（gold_citation_ids, gold_citation_keywords）
- 期望评估结果（expected_stance, should_refuse, expected_score_range）
- Tool Use 期望（expected_tool_calls, expected_final_answer_keywords）

数据集支持三种分割（split）：
- `dev`：开发集，用于日常开发调试
- `test`：测试集，用于版本发布前验收
- `regression`：回归集，用于防止功能回退

---

### 2.2 Tool Use 端到端加固

Tool Use 系统允许 Knowledge Agent 在评估过程中主动调用外部工具（如医学知识库检索、查询扩展、HyDE 查询等）。为确保生产环境的稳定性，系统实现了完整的加固机制。

#### 2.2.1 熔断器模式（Circuit Breaker）

每个工具独立维护一个熔断器，防止级联故障：

```
CLOSED（正常）→ 连续失败 ≥ threshold → OPEN（熔断，拒绝请求）
                                            ↓
                                     等待 recovery_timeout
                                            ↓
                                     HALF_OPEN（试探放行）
                                       ↓           ↓
                                   成功 → CLOSED   失败 → OPEN
```

**参数配置：**
- `failure_threshold`: 连续失败次数阈值（默认 5 次）
- `recovery_timeout`: 熔断恢复超时（默认 30 秒）
- `half_open_max_calls`: 半开状态最大试探请求数（默认 1 次）

#### 2.2.2 重试机制

采用指数退避 + 抖动策略，避免雪崩：

```
retry_delay = min(base_delay × 2^attempt, max_delay) + random_jitter
```

**RetryPolicy 参数：**
- `max_retries`: 最大重试次数
- `base_delay`: 基础延迟（秒）
- `max_delay`: 最大延迟上限（秒）
- `jitter`: 随机抖动范围

#### 2.2.3 预算管理（ToolBudgetManager）

控制工具调用的成本与配额：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `session_max_total_calls` | 单次会话最大调用次数 | 50 |
| `session_max_per_tool` | 单工具最大调用次数 | 10 |
| `session_max_cost` | 单次会话最大成本（元） | 0.1 |
| `warning_threshold` | 预警阈值（70%） | 0.7 |
| `critical_threshold` | 严重预警阈值（90%） | 0.9 |

**预警级别：**
- `NORMAL`：正常使用
- `WARNING`：预算使用超过 70%
- `CRITICAL`：预算使用超过 90%
- `EXHAUSTED`：预算已耗尽，拒绝后续调用

#### 2.2.4 健康检查与降级（ToolHealthChecker）

定期探测工具可用性，实现自动降级与恢复：

**健康状态：**
- `HEALTHY`：健康，正常服务
- `DEGRADED`：降级，可用但性能下降
- `UNAVAILABLE`：不可用
- `UNKNOWN`：未检查

**降级策略：**
- 工具不可用时，`DegradedResultBuilder` 返回预设的降级结果
- 关键工具（如 `search_medical_kb`）失败时返回空证据列表，而非抛出异常
- 支持注册自定义降级结果构建器

**自动恢复：**
- 健康检查器定期重新探测不可用工具
- 工具恢复后自动切换回 `HEALTHY` 状态

---

### 2.3 LangGraph Send Fan-out/Fan-in 升级

#### 2.3.1 架构变化

平台的核心编排层从传统的 `asyncio.gather` 并行模式升级为 LangGraph 原生的 **Send fan-out/fan-in** 机制。

**旧模式（dispatch_and_run）：**
```python
# 手动管理并行任务
results = await asyncio.gather(
    *[run_one(name) for name in plan.selected_agents],
    return_exceptions=True,
)
```

**新模式（Send fan-out/fan-in）：**
```python
# LangGraph 原生并行
def route_to_agents(state: EvaluationState) -> list[Send]:
    return [
        Send("run_agent", {"agent_name": name, "context": context, ...})
        for name in plan.selected_agents
    ]

graph.add_conditional_edges("build_route_plan", route_to_agents)
graph.add_edge("run_agent", "aggregate_results")  # fan-in
```

#### 2.3.2 状态图流程

```
START
  → load_context
  → classify_consultation
  → safety_check
  → safety_gate (条件边)
      ├─ continue → build_route_plan
      └─ needs_review → finalize_needs_review → END
  → route_to_agents (Send fan-out)
      ├─ Send → run_agent (inquiry)
      ├─ Send → run_agent (diagnosis)
      ├─ Send → run_agent (treatment)
      ├─ Send → run_agent (knowledge)
      └─ Send → run_agent (humanistic)
  → aggregate_results (fan-in 汇聚)
  → deterministic_scoring
  → review_gate (条件边)
      ├─ completed → generate_suggestion → finalize_completed → END
      └─ needs_review → finalize_needs_review → END
```

#### 2.3.3 优势

| 特性 | 说明 |
|------|------|
| 原生并行 | LangGraph 调度器自动并行执行 Send 分支 |
| 状态一致性 | 通过 reducer（`operator.add`）自动合并并行分支的状态更新 |
| 可检查点 | 支持 Redis Checkpoint，流程中断后可恢复 |
| 可观测性 | 每个节点执行后产生 ProgressEvent，通过 WebSocket 推送 |
| 条件路由 | Safety Gate 和 Review Gate 实现动态流程控制 |
| 向后兼容 | 保留 `dispatch_and_run` 节点用于测试场景 |

#### 2.3.4 状态 Reducer 设计

```python
class EvaluationState(TypedDict, total=False):
    # 使用 Annotated + reducer 支持并行分支合并
    agent_results: Annotated[list[AgentResultEnvelope], add]
    node_errors: Annotated[list[NodeError], add]
    progress_events: Annotated[list[ProgressEvent], add]
```

每个 Send 分支返回的 `agent_results` 列表通过 `operator.add` reducer 自动累积到主状态，无需手动合并。

---

### 2.4 Docker Compose 一键部署

#### 2.4.1 服务编排

平台通过 Docker Compose 编排四个核心服务：

| 服务 | 镜像 | 说明 |
|------|------|------|
| `mysql` | mysql:8.0 | 关系数据库，存储问诊记录与评估结果 |
| `redis` | redis:7-alpine | LangGraph Checkpoint + 缓存 |
| `backend` | 自建 (FastAPI) | 后端 API 服务 |
| `frontend` | 自建 (Nginx + React) | 前端 SPA 应用 |

**服务依赖关系：**
```
frontend → backend → mysql
                    → redis
```

**网络隔离：**
- `backend-net`：MySQL、Redis、Backend 内部通信
- `frontend-net`：Frontend 对外服务 + 反向代理到 Backend

**资源限制：**

| 服务 | 内存 | CPU |
|------|------|-----|
| MySQL | 1 GB | 1.0 |
| Redis | 512 MB | 0.5 |
| Backend | 2 GB | 2.0 |
| Frontend | 256 MB | 0.5 |

#### 2.4.2 环境配置

环境变量通过 `.env` 文件管理，模板位于 `.env.docker`：

**必填配置：**
```bash
DASHSCOPE_API_KEY=sk-your-dashscope-api-key-here   # 阿里云百炼 API Key
MYSQL_PASSWORD=your-secure-password-here             # MySQL 密码
SECRET_KEY=change-this-to-a-secure-random-string     # JWT 密钥
```

**可选配置（有默认值）：**
```bash
# 模型配置
QWEN_MODEL=qwen3.7-max
TOOL_USE_MODEL=qwen-max

# Tool Use 配置
ENABLE_TOOL_USE=false
TOOL_USE_MAX_ROUNDS=4
TOOL_USE_MAX_CALLS=8

# LangGraph 配置
LANGGRAPH_ENABLED=true
LANGGRAPH_SHADOW_MODE=false

# 端口映射
BACKEND_PORT=8000
FRONTEND_PORT=80
```

#### 2.4.3 部署步骤

**生产环境：**
```bash
# 1. 复制环境配置模板
cp .env.docker .env

# 2. 编辑 .env，填写必填配置
# 特别是 DASHSCOPE_API_KEY、MYSQL_PASSWORD、SECRET_KEY

# 3. 启动所有服务
docker compose up -d

# 4. 查看服务状态
docker compose ps

# 5. 查看日志
docker compose logs -f backend
```

**开发环境（热重载）：**
```bash
# 使用 override 文件覆盖为开发配置
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Backend: uvicorn --reload 模式
# Frontend: Vite dev server (端口 5173)
```

---

## 3. 部署指南

### 3.1 环境要求

| 组件 | 最低版本 | 说明 |
|------|----------|------|
| Docker | 20.10+ | 容器运行时 |
| Docker Compose | 2.0+ | 服务编排 |
| 内存 | 4 GB+ | 所有服务合计 |
| 磁盘 | 10 GB+ | 镜像 + 数据卷 |

### 3.2 Docker Compose 部署步骤

#### 步骤 1：准备环境文件

```bash
cd medical-ai-platform
cp .env.docker .env
```

#### 步骤 2：配置必要参数

编辑 `.env` 文件，至少需要设置以下参数：

```bash
# [必填] 阿里云百炼平台 API Key
DASHSCOPE_API_KEY=sk-your-actual-api-key

# [必填] MySQL 密码（生产环境请使用强密码）
MYSQL_PASSWORD=your-strong-password

# [必填] JWT 签名密钥（请使用随机字符串）
SECRET_KEY=$(openssl rand -hex 32)
```

#### 步骤 3：启动服务

```bash
# 生产环境
docker compose up -d

# 开发环境（带热重载）
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

#### 步骤 4：验证部署

```bash
# 检查所有服务状态
docker compose ps

# 验证 Backend 健康检查
curl http://localhost:8000/health

# 验证 Frontend
curl http://localhost:80
```

#### 步骤 5：初始化管理员账户

```bash
# 进入 Backend 容器
docker compose exec backend python backend/init_admin.py
```

### 3.3 配置说明

#### 数据库初始化

MySQL 容器首次启动时自动执行：
- `database/init.sql`：创建表结构
- `database/seed.sql`：插入初始数据

数据库迁移脚本：
```bash
# 执行数据库迁移
docker compose exec backend python backend/migrate_db.py
```

#### 向量数据库初始化

ChromaDB 数据通过 volume 持久化，首次启动时 Backend 会自动：
1. 读取 `data/` 目录下的医学教材 PDF
2. 进行文档分块与 Embedding 生成
3. 建立向量索引

#### Redis 配置

Redis 用于 LangGraph Checkpoint 持久化：
- 使用 `db=1` 避免与应用缓存冲突
- 默认 TTL：86400 秒（24 小时）
- 内存限制：256 MB，淘汰策略：allkeys-lru

---

## 4. 使用指南

### 4.1 如何运行评估

#### 在线评估（通过 Web 界面）

1. 登录平台（默认管理员账户通过 `init_admin.py` 创建）
2. 选择虚拟患者，开始问诊
3. 与患者对话，收集信息
4. 提交诊断和治疗方案
5. 系统自动触发五维评估
6. 通过 WebSocket 实时查看评估进度

#### RAG/Tool Use 离线评估

```python
# 运行 Tool Use 评估
import asyncio
from evaluation.runners import run_tool_use_evaluation
from evaluation.report import generate_json_report, write_json_report

async def main():
    results = await run_tool_use_evaluation(
        cases_path="backend/evaluation/rag_cases/rag_gold_cases.jsonl",
        split="dev",
        limit=10,
    )
    
    # 加载 gold cases 用于报告生成
    from evaluation.datasets import load_gold_cases
    gold_cases = load_gold_cases(
        Path("backend/evaluation/rag_cases/rag_gold_cases.jsonl")
    )
    
    report = generate_json_report(
        results=results,
        gold_cases=gold_cases,
        mode="tooluse",
        dataset_path="rag_gold_cases.jsonl",
        split="dev",
    )
    
    write_json_report(report, Path("backend/evaluation/reports/latest.json"))

asyncio.run(main())
```

#### 对比评估（Legacy vs Tool Use）

```python
from evaluation.runners import run_batch_evaluation

result = await run_batch_evaluation(
    cases_path=Path("backend/evaluation/rag_cases/rag_gold_cases.jsonl"),
    split="dev",
    limit=5,
    max_concurrency=4,
)

# result 包含:
# - result["legacy"]: Legacy 模式结果与报告
# - result["tooluse"]: Tool Use 模式结果与报告
# - result["comparison"]: 两种模式的对比摘要
# - result["query_type_breakdown"]: 按查询类型分组统计
```

### 4.2 如何解读评估报告

#### 报告结构

```json
{
  "timestamp": "2024-01-01T00:00:00+00:00",
  "mode": "tooluse",
  "dataset": {
    "path": "rag_gold_cases.jsonl",
    "split": "dev",
    "total_samples": 10,
    "normal_samples": 7,
    "refusal_samples": 3
  },
  "metrics": {
    "recall_at_5": 0.75,
    "citation_validity": 1.0,
    "refusal_accuracy": 0.9,
    "tool_success_rate": 0.85,
    ...
  },
  "thresholds": {
    "passed": true,
    "violations": [],
    "compliance_rate": 1.0,
    "recommendations": []
  }
}
```

#### 关键指标解读

| 指标 | 理想值 | 说明 |
|------|--------|------|
| Citation Validity | 1.0 (100%) | 所有引用必须来自实际检索到的文档 |
| Hallucination Rate | 0.0 (0%) | 不应存在编造的引用 |
| False Acceptance Rate | ≤ 0.05 (5%) | 应该拒答的案例不能给出正常分数 |
| Refusal Accuracy | ≥ 0.80 (80%) | 拒答判断的准确率 |
| Tool Success Rate | ≥ 0.80 (80%) | 工具调用的成功率 |

#### 阈值违规处理

当 P0 级别指标未通过时，报告整体标记为 `passed: false`，需要优先处理：

```json
{
  "thresholds": {
    "passed": false,
    "violations": [
      {
        "metric": "citation_hallucination_rate",
        "actual": 0.12,
        "threshold": 0.05,
        "level": "P0",
        "description": "引用幻觉率必须 ≤ 5%"
      }
    ],
    "recommendations": [
      "[P0] 引用幻觉率偏高 (12.0% > 5.0%)，需加强引用来源校验"
    ]
  }
}
```

### 4.3 如何扩展和定制

#### 添加新的评估工具（Tool）

1. 在 `backend/app/services/tools/` 下创建新工具文件
2. 继承 `BaseTool` 并实现 `execute` 方法
3. 使用注册函数将工具加入 `ToolRegistry`

```python
from app.services.tools.base import BaseTool, ToolContext

class MyNewTool(BaseTool):
    name = "my_new_tool"
    description = "工具描述"
    
    async def execute(self, ctx: ToolContext, **kwargs) -> dict:
        # 实现工具逻辑
        return {"result": "..."}

# 注册工具
def register_my_tools(registry: ToolRegistry):
    registry.register(MyNewTool())
```

#### 自定义评分策略

```python
from app.services.scoring.policies import ScoringPolicy

class MyPolicy(ScoringPolicy):
    version = "my-policy-v1"
    weights = {
        "inquiry": 0.15,
        "diagnosis": 0.25,
        "treatment": 0.25,
        "knowledge": 0.25,
        "humanistic": 0.10,
    }
```

#### 添加新的 Gold Case

在 `backend/evaluation/rag_cases/rag_gold_cases.jsonl` 中追加新行：

```json
{
  "case_id": "custom_case_001",
  "split": "dev",
  "department": "心内科",
  "difficulty": "medium",
  "chief_complaint": "胸闷、气短",
  "patient_info": "患者，男，65岁...",
  "conversation_text": "医生: 请问您哪里不舒服？\n患者: 最近经常胸闷...",
  "doctor_diagnosis": "冠心病稳定型心绞痛",
  "treatment_plan": "硝酸甘油舌下含服...",
  "gold_doc_ids": ["doc-001", "doc-002"],
  "gold_citation_ids": ["cite-001"],
  "expected_stance": "supports",
  "should_refuse": false,
  "expected_score_range": [70, 90],
  "expected_tool_calls": [{"name": "search_medical_kb", "params": {"query": "冠心病"}}]
}
```

---

## 5. API 文档

### 5.1 认证接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/auth/register` | 用户注册 |
| POST | `/api/v1/auth/login` | 用户登录，返回 JWT Token |

### 5.2 虚拟患者接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/patients/` | 获取虚拟患者列表 |

### 5.3 问诊交互接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/consultations/` | 创建问诊（选择患者开始问诊） |
| GET | `/api/v1/consultations/` | 获取当前医生的问诊列表 |
| GET | `/api/v1/consultations/all` | 管理员：获取全平台问诊记录（支持多维度筛选） |
| GET | `/api/v1/consultations/{id}` | 获取问诊详情（含消息列表） |
| POST | `/api/v1/consultations/{id}/messages` | 医生发送消息，返回患者回复 |
| POST | `/api/v1/consultations/{id}/submit-diagnosis` | 提交诊断和治疗方案 |
| POST | `/api/v1/consultations/{id}/extend` | 延长问诊轮次 |
| POST | `/api/v1/consultations/{id}/end` | 结束问诊 |
| DELETE | `/api/v1/consultations/{id}` | 删除问诊记录 |

### 5.4 评估接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/evaluations/` | 创建评估（触发五维评估流程） |
| GET | `/api/v1/evaluations/{consultation_id}` | 获取问诊的评估结果 |
| WebSocket | `/api/v1/evaluations/ws/{consultation_id}` | 评估进度实时推送 |

### 5.5 数据统计接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/stats/` | 平台统计数据 |

### 5.6 知识库管理接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/knowledge-base/` | 知识库列表 |
| POST | `/api/v1/knowledge-base/reindex` | 触发重新索引 |

### 5.7 系统接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 系统健康检查（含 LangGraph、LLM 状态） |
| GET | `/api/v1/openapi.json` | OpenAPI 规范文档 |

### 5.8 通用说明

**认证方式：** 所有接口（除注册/登录外）需要在请求头中携带 JWT Token：
```
Authorization: Bearer <token>
```

**错误响应格式：**
```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "请求参数不合法",
  "detail": "请求参数不合法",
  "request_id": "abc123"
}
```

**请求 ID：** 每个请求自动分配唯一 `X-Request-ID`，用于日志追踪。

---

## 6. 故障排除

### 6.1 常见问题与解决方案

#### Backend 启动失败

**问题：** `RuntimeError: Redis connection failed`

**原因：** `LANGGRAPH_ENABLED=true` 但 Redis 服务不可用。

**解决方案：**
```bash
# 检查 Redis 服务状态
docker compose ps redis

# 查看 Redis 日志
docker compose logs redis

# 重启 Redis
docker compose restart redis

# 或者临时禁用 LangGraph（回退到旧编排模式）
# 在 .env 中设置:
LANGGRAPH_ENABLED=false
```

---

**问题：** `DASHSCOPE_API_KEY` 未设置

**原因：** Docker 启动时未提供 API Key。

**解决方案：**
```bash
# 确认 .env 文件中设置了 DASHSCOPE_API_KEY
grep DASHSCOPE_API_KEY .env

# 重启 Backend 服务
docker compose restart backend
```

---

#### 评估流程卡住

**问题：** 评估进度停在某个百分比不动

**排查步骤：**
1. 检查 WebSocket 连接是否正常
2. 查看 Backend 日志：
   ```bash
   docker compose logs -f backend | grep "evaluation"
   ```
3. 检查 LLM API 是否可用：
   ```bash
   curl http://localhost:8000/health
   ```
4. 检查 LangGraph Checkpointer 状态（`checkpointer` 字段应为 `available`）

---

#### Tool Use 工具调用失败

**问题：** 工具成功率低于阈值

**排查步骤：**
1. 查看工具调用明细：
   ```bash
   docker compose logs backend | grep "tool_call"
   ```
2. 检查熔断器状态：
   - 日志中搜索 `CircuitBreaker` 相关记录
3. 检查预算是否耗尽：
   - 日志中搜索 `BudgetAlertLevel` 相关记录
4. 临时增加超时时间：
   ```bash
   TOOL_USE_TIMEOUT_SECONDS=60
   ```

---

#### 数据库连接失败

**问题：** `DB_UNAVAILABLE` 错误

**解决方案：**
```bash
# 检查 MySQL 服务状态
docker compose ps mysql

# 查看 MySQL 日志
docker compose logs mysql

# 等待 MySQL 健康检查通过（可能需要 30 秒）
docker compose logs mysql | grep "ready for connections"
```

---

#### 前端无法访问

**问题：** 浏览器打开 `http://localhost` 显示空白或 502

**排查步骤：**
```bash
# 检查 Frontend 容器状态
docker compose ps frontend

# 检查 Nginx 日志
docker compose logs frontend

# 确认 Backend 是否健康（Frontend 依赖 Backend）
docker compose ps backend
```

---

#### RAG 检索效果不佳

**问题：** Recall@5 低于阈值

**优化建议：**
1. 检查 PDF 文档是否正确放入 `data/` 目录
2. 确认向量索引已建立：
   ```bash
   docker compose logs backend | grep "index"
   ```
3. 调整 Reranker 模型：
   ```bash
   RERANK_MODEL=gte-rerank
   ```
4. 增加并发 LLM 调用限制（避免 API 限流）：
   ```bash
   LLM_MAX_CONCURRENT=5
   ```

---

### 6.2 日志查看

```bash
# 查看所有服务日志
docker compose logs -f

# 查看特定服务日志
docker compose logs -f backend
docker compose logs -f mysql
docker compose logs -f redis
docker compose logs -f frontend

# 搜索错误日志
docker compose logs backend | grep "ERROR"
```

### 6.3 服务重启

```bash
# 重启单个服务
docker compose restart backend

# 重启所有服务
docker compose restart

# 完全重建（清除数据卷，谨慎使用！）
docker compose down -v
docker compose up -d
```

### 6.4 性能优化建议

| 场景 | 建议 |
|------|------|
| LLM API 限流 (429) | 降低 `LLM_MAX_CONCURRENT`（如设为 5） |
| 评估速度慢 | 增加 `LLM_MAX_CONCURRENT`（如设为 15） |
| 内存不足 | 降低 Backend 内存限制或减少并发 |
| ChromaDB 检索慢 | 检查向量索引大小，考虑清理旧索引 |

---

## 附录

### A. 项目目录结构

```
medical-ai-platform/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # API 路由
│   │   ├── core/            # 配置、依赖、WebSocket
│   │   ├── db/              # 数据库会话
│   │   ├── models/          # SQLAlchemy 模型
│   │   ├── orchestration/   # LangGraph 编排层
│   │   │   ├── graph.py     # 状态图定义
│   │   │   ├── state.py     # 统一状态模型
│   │   │   ├── checkpointer.py  # Redis Checkpointer
│   │   │   ├── adapters/    # Agent 适配器
│   │   │   └── nodes/       # 图节点
│   │   ├── schemas/         # Pydantic Schema
│   │   └── services/
│   │       ├── agents/      # 五个评估 Agent
│   │       ├── rag/         # RAG 检索管道
│   │       ├── scoring/     # 评分策略
│   │       └── tools/       # Tool Use 系统
│   ├── evaluation/          # RAG/Tool Use 评估框架
│   │   ├── datasets.py      # 数据模型
│   │   ├── metrics.py       # 指标计算
│   │   ├── runners.py       # 评估运行器
│   │   ├── report.py        # 报告生成
│   │   ├── config.py        # 评估配置
│   │   └── rag_cases/       # Gold Cases 数据集
│   └── tests/               # 测试用例
├── frontend/                # React 前端
├── database/                # SQL 初始化与迁移脚本
├── data/                    # 医学教材 PDF
├── docker-compose.yml       # 生产环境编排
├── docker-compose.override.yml  # 开发环境覆盖
├── .env.docker              # 环境变量模板
├── Dockerfile.backend       # Backend 镜像
└── Dockerfile.frontend      # Frontend 镜像
```

### B. 环境变量完整列表

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `DASHSCOPE_API_KEY` | 是 | - | 阿里云百炼 API Key |
| `MYSQL_PASSWORD` | 否 | `qjr3225365` | MySQL 密码 |
| `MYSQL_DATABASE` | 否 | `medical_ai` | 数据库名 |
| `SECRET_KEY` | 否 | `change-this-...` | JWT 签名密钥 |
| `QWEN_MODEL` | 否 | `qwen3.7-max` | 主模型 |
| `TOOL_USE_MODEL` | 否 | `qwen-max` | Tool Use 模型 |
| `ENABLE_TOOL_USE` | 否 | `false` | 启用 Tool Use |
| `LANGGRAPH_ENABLED` | 否 | `true` | 启用 LangGraph 编排 |
| `LANGGRAPH_SHADOW_MODE` | 否 | `false` | 影子模式 |
| `REDIS_CHECKPOINT_URL` | 否 | `redis://redis:6379/1` | Redis Checkpoint URL |
| `REDIS_CHECKPOINT_TTL` | 否 | `86400` | Checkpoint TTL (秒) |
| `ACTIVE_INDEX_VERSION` | 否 | `rag-v1` | RAG 索引版本 |
| `RERANK_MODEL` | 否 | `gte-rerank` | Reranker 模型 |
| `LLM_MAX_CONCURRENT` | 否 | `10` | LLM 最大并发数 |
| `BACKEND_PORT` | 否 | `8000` | Backend 端口 |
| `FRONTEND_PORT` | 否 | `80` | Frontend 端口 |

### C. 相关资源

- **FastAPI 文档**：https://fastapi.tiangolo.com/
- **LangGraph 文档**：https://langchain-ai.github.io/langgraph/
- **ChromaDB 文档**：https://docs.trychroma.com/
- **阿里云百炼平台**：https://dashscope.aliyun.com/
