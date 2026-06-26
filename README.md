# 基于多智能体的医生临床问诊评估平台

## 项目简介

本平台是一个面向临床问诊实训的多智能体自动评估系统。医生可与虚拟患者进行模拟问诊，系统自动调用多个 AI 智能体，从问诊技巧、医学知识、人文关怀、诊断能力和治疗方案五个维度进行加权评估，并生成综合改进建议。

核心技术亮点：
- **Plan-Execute + Send fan-out/fan-in** LangGraph 编排架构
- **患者模拟 ReAct + 滑动窗口记忆 + SSE 实时推送**
- **Knowledge Agent ReAct + RAG Tools** 循证检索
- **Reflection Agent ReAct + Consistency Tools** 评估质量自审
- **RAG V2 分级检索与两阶段重排管线**
- **Tool Use 端到端加固**（熔断器、预算管理器、健康检查器）
- **安全加固体系**（JWT + API 速率限制 + 审计日志 + XSS 防护）
- **LLM 响应语义缓存层**（Redis + SHA256 精确哈希，best-effort 降级）
- **统一 Pydantic 数据契约**、五维加权评分模型（含拒答权重重分配）
- **Docker Compose 一键部署**（MySQL + Redis + 后端 + 前端）

## 系统架构

### 整体架构

```
                          ┌─────────────────────────────────┐
                          │     Plan-Execute 评估主图         │
                          │   (LangGraph StateGraph v5.2)   │
                          └─────────────────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
        ┌─────▼─────┐           ┌───────▼───────┐          ┌──────▼──────┐
        │ Safety 门控 │           │  评估计划生成   │          │ Send fan-out │
        │ (确定性+LLM)│          │ (PlanStep列表) │          │ (并行分发)    │
        └─────┬─────┘           └───────┬───────┘          └──────┬──────┘
              │                          │                          │
              │                          │        ┌─────────────────┼─────────────────┐
              │                          │        │                 │                 │
              │                          │  ┌─────▼─────┐  ┌──────▼──────┐  ┌───────▼───────┐
              │                          │  │ 问诊分析   │  │ 知识核对     │  │ 人文关怀       │
              │                          │  │ Agent     │  │ Agent(ReAct)│  │ Agent         │
              │                          │  └─────┬─────┘  └──────┬──────┘  └───────┬───────┘
              │                          │        │               │                 │
              │                          │  ┌─────▼─────┐  ┌──────▼──────┐         │
              │                          │  │ 诊断评估   │  │ 治疗评估     │         │
              │                          │  │ Agent     │  │ Agent       │         │
              │                          │  └─────┬─────┘  └──────┬──────┘         │
              │                          │        └───────────────┼─────────────────┘
              │                          │                        │
              │                          │              ┌─────────▼─────────┐
              │                          │              │  fan-in 结果聚合   │
              │                          │              │  (Annotated+add)  │
              │                          │              └─────────┬─────────┘
              │                          │                        │
              │                          │              ┌─────────▼─────────┐
              │                          │              │  综合评分 Agent    │
              │                          │              │ (五维加权+权重分配) │
              │                          │              └─────────┬─────────┘
              │                          │                        │
              │                          │              ┌─────────▼─────────┐
              │                          │              │  建议指导 Agent    │
              │                          │              └─────────┬─────────┘
              │                          │                        │
              │                          │              ┌─────────▼─────────┐
              │                          │              │  Reflection Agent │
              │                          │              │ (ReAct+一致性工具) │
              │                          │              └─────────┬─────────┘
              │                          │                        │
              │                          │              ┌─────────▼─────────┐
              └──────────────────────────┴──────────────►    评估报告输出    │
                                                     └───────────────────┘
```

评估流程由 **LangGraph StateGraph** 统一编排，采用 **Plan-Execute** 模式：先由计划节点生成评估步骤列表，再通过 `Send` fan-out 并行分发至各评估 Agent，fan-in 聚合后依次执行综合评分、建议指导和反思验证。所有智能体均由阿里云百炼平台 Qwen API 驱动，全局通过 `asyncio.Semaphore` 控制 LLM 并发调用数，防止 API 限流。

### LangGraph 编排层（v5.2 — Plan-Execute + Send fan-out/fan-in）

**核心特性**：
- **Plan-Execute 模式**：`plan_evaluation` 节点生成 `EvaluationPlan`（含多个 `PlanStep`），`validate_plan` 节点校验完整性，execute 阶段基于步骤列表执行
- **Send fan-out/fan-in**：通过 LangGraph `Send` API 将每个 `PlanStep` 分发给独立工作器并行执行，`Annotated[list, add]` reducer 自动合并结果
- **EvaluationState TypedDict**：所有节点共享状态，含 `evaluation_plan`、`execution_results`、`reflection_result` 等字段
- **Safety 门控**：确定性红旗规则优先 → LLM 语义补充 → fail closed 策略
- **动态路由**：基于咨询类型（initial/follow_up/emergency/communication）和提交状态决定执行哪些 Agent
- **SQLite/Redis Checkpoint**：支持中断恢复和断点续传

**关键文件**：
- `backend/app/orchestration/state.py` — 状态定义（含 `EvaluationPlan`、`PlanStep`、`ExecutionResult`、`ReflectionResult`）
- `backend/app/orchestration/graph.py` — StateGraph 主图（Plan-Execute + Send fan-out/fan-in）
- `backend/app/orchestration/checkpointer.py` — Redis Checkpoint 持久化
- `backend/app/orchestration/adapters/` — Agent 适配器模式（8 个适配器：inquiry/diagnosis/treatment/humanistic/knowledge/reflection/suggestion + base/registry）
- `backend/app/orchestration/routes.py` — 路由矩阵与场景分类

**Feature Flag**：
```bash
LANGGRAPH_ENABLED=true   # 启用 LangGraph 编排（默认 true，设为 false 回退旧编排）
LANGGRAPH_SHADOW_MODE=true  # 影子模式：新旧路径并行对比
```

### 患者模拟：ReAct + 滑动窗口记忆 + SSE 实时推送

虚拟患者模拟采用 **ReAct 模式** 结合 **滑动窗口记忆管理**，通过 **SSE（Server-Sent Events）** 向前端实时推送进度。

**核心特性**：
- **人格驱动**：患者回复严格遵循档案人格特点（焦虑型/沉默型/对抗型/配合型），人格表达优先于语气平实
- **滑动窗口记忆**：完整保留最近 10 轮对话；超过 14 轮时触发 LLM 压缩早期对话为摘要，避免上下文溢出
- **信息量控制**：每次回复严格 1-3 句话，只回答当前问题，禁止主动提供未问到的信息
- **SSE 实时进度**：每个关键步骤（加载历史→保存消息→构建上下文→压缩记忆→生成回复）发送 progress 事件，前端实时展示

**关键文件**：
- `backend/app/services/consultation_service.py` — 患者模拟核心逻辑（`send_doctor_message_stream`）
- `backend/app/api/v1/consultations.py` — SSE 端点（`StreamingResponse` + `text/event-stream`）
- `frontend/src/api/consultation.ts` — 前端 SSE 客户端封装
- `frontend/src/pages/Consultation/index.tsx` — 问诊页面进度展示

### Knowledge Agent：ReAct + RAG Tools

知识 Agent 支持两种模式：**传统 Tool Use**（Function Calling）和 **ReAct 模式**（显式 Thought→Action→Observation 推理链）。

**ReAct 模式**（`ENABLE_REACT_KNOWLEDGE=true`）：
- LLM 每步先输出 `Thought` 解释推理原因，再输出 `Action` 调用工具
- 工具返回 `Observation` 后继续下一轮推理
- 推理过程完整记录在 `react_trace`，支持审计
- 工具白名单：4 个医学检索工具 + 1 个引用校验工具

**传统 Tool Use 模式**（`ENABLE_TOOL_USE=true`）：
- LLM 通过 Function Calling 隐式决定调用工具
- 推理过程不透明，但调用效率更高

**两种模式共享**：
- 确定性边界：总分计算、Safety 门控、拒答规则保持代码控制，LLM 不可干预
- 引用校验后处理：自动检测非法引用 ID，支持一次修正重试
- 预算控制：RAG 调用最多 3 次、MQE 扩展最多 2 次、HyDE 最多 1 次

**关键文件**：
- `backend/app/services/agents/knowledge_agent.py` — `run_knowledge_check()`（传统）+ `run_knowledge_check_with_tools()`（Tool Use）+ `run_knowledge_check_react()`（ReAct）
- `backend/app/services/tools/` — 工具系统（base/registry/executor/budget/medical_retrieval/citation）

### Reflection Agent：ReAct + Consistency Tools

反思 Agent 作为评分后的验证步骤串行执行，使用 ReAct 推理链检查评估结果的一致性、证据充分性和逻辑矛盾。

**核心特性**：
- **4 个一致性工具**：`check_score_consistency`（评分一致性）、`check_evidence_sufficiency`（证据充分性）、`detect_score_contradictions`（矛盾检测）、`summarize_evaluation`（结果汇总）
- **内置矛盾规则**：诊断高分+知识低分、治疗高分+知识低分、病史低分+诊断高分等
- **反思结果辅助性**：不替代原始评分，仅标记需要关注的问题（review flags）
- **Feature Flag**：`ENABLE_REACT_REFLECTION=true`（默认开启）

**关键文件**：
- `backend/app/services/agents/reflection_agent.py` — `run_reflection()` 主函数
- `backend/app/services/tools/consistency.py` — 4 个一致性检查工具
- `backend/app/orchestration/adapters/reflection.py` — LangGraph 适配器

### Tool Use 端到端加固

工具系统除基础执行器外，还包含三层加固组件：

| 组件 | 功能 | 关键文件 |
|------|------|----------|
| `RobustToolExecutor` | 带重试、超时、熔断的执行器 | `tools/robust_tool_executor.py` |
| `ToolBudgetManager` | 全局工具调用预算管理与告警 | `tools/tool_budget_manager.py` |
| `ToolHealthChecker` | 工具健康状态检查与降级结果构建 | `tools/tool_health_checker.py` |
| `CircuitBreaker` | 熔断器（closed→open→half_open） | `tools/robust_tool_executor.py` |

### 安全加固体系

平台内置多层安全机制，保障生产环境部署的安全性：

| 模块 | 功能 | 关键文件 |
|------|------|----------|
| **JWT 认证** | bcrypt 密码哈希 + python-jose JWT 令牌签发与校验 | `core/security.py` |
| **API 速率限制** | slowapi 中间件，基于 IP 的分级限流（认证 5次/分钟、问诊 10次/分钟、评估 5次/小时） | `core/limiter.py` |
| **审计日志** | AuditLog 模型（`audit_logs` 表），记录 login / create_consultation / submit_diagnosis / trigger_evaluation / admin_action 五种操作，支持 IP 和 UA 采集 | `core/audit.py`, `models/audit_log.py` |
| **输入验证** | HTML 标签过滤（`strip_html_tags`）、文本清理（`sanitize_text`）、XSS 检测（`contains_html`） | `core/validation.py` |

**Feature Flag**：
```bash
AUDIT_LOG_ENABLED=true   # 启用审计日志（默认 true）
```

### LLM 响应语义缓存层

基于 Redis 的精确哈希缓存层，对 `temperature=0` 的确定性 LLM 调用进行响应缓存，减少重复 API 调用开销。

**核心特性**：
- **精确哈希匹配**：SHA256 哈希作为缓存键，格式 `llm_cache:{model}:{temperature}:{sha256_hash}`
- **仅缓存确定性调用**：`temperature=0` 时启用，`temperature>0` 具有随机性不缓存
- **TTL 与容量**：默认 24 小时过期，最大 10000 条目
- **静默降级**：Redis 连接失败或任何异常均静默处理（best-effort），不影响正常 LLM 调用
- **统计指标**：`cache_hits`、`cache_misses`、`hit_rate` 进程内计数器

**关键文件**：
- `backend/app/services/llm_cache.py` — 缓存层实现

**Feature Flag**：
```bash
LLM_CACHE_ENABLED=true          # 启用 LLM 响应缓存（默认 true）
LLM_CACHE_TTL=86400             # 缓存过期时间（秒），24小时
LLM_CACHE_MAX_SIZE=10000        # 最大缓存条目数
```

### 评分引擎

评分逻辑从单体拆分为三个独立组件，支持版本化策略和确定性计算。

- **ScoringPolicy**：版本化权重配置（`v1`, `v2`...），支持 A/B 测试
- **ScoreCalculator**：纯代码加权计算，禁止 None 临时权重重分配
- **SummaryGenerator**：LLM 摘要生成 + 五维确定性降级模板

**关键文件**：
- `backend/app/services/scoring/policies.py` — 评分策略
- `backend/app/services/scoring/calculator.py` — 确定性计算器
- `backend/app/services/scoring/summary.py` — 摘要生成器

### RAG 检索系统（V2）

知识核对智能体依托 RAG V2 检索管线，从 80+ 部医学教材与 CSCO/NCCN 指南中检索循证证据。核心特性：

- **统一数据契约（Pydantic Schema）** — `RetrievalQuery`、`EvidenceItem`、`RetrievalBundle`、`Citation`、`KnowledgeAssessment` 等结构化模型。
- **三类独立查询构建** — 从问诊对话中提取结构化病例事实（`ClinicalFacts`），分别构建病例查询（case）、诊断查询（diagnosis）和治疗查询（treatment），消除确认偏误。
- **三级分级检索** — Level 1: BM25 + 向量混合检索 + RRF 融合；Level 2: LLM 多查询扩展（MQE，全局预算 ≤2 次）；Level 3: HyDE（假设文档 embedding，全局预算 ≤1 次）。
- **两阶段重排序** — Stage 1: DashScope gte-rerank 粗排（20→10）；Stage 2: LLM Cross-Encoder 精排（10→5），融合权威性评分和时效性评分。
- **拒答与引用追溯** — `retrieval_status` 与 `evidence_stance` 分离判断；不充分时 `score=None` 直接拒答。`Citation` 模型支持完整审计链。
- **版本化索引管理** — 支持 `rag-v1`、`rag-v2` 等多版本索引共存，热切换。

### RAG / Tool Use 评估指标体系

内置离线评测框架，用于持续监控 RAG 检索质量和 Tool Use 执行效果。

- **评测数据集**：`backend/evaluation/rag_cases/rag_gold_cases.jsonl`，含标准查询-证据对
- **评测运行器**：`backend/evaluation/runners.py`，支持 Recall@K、MRR、nDCG 等指标
- **回归测试**：`backend/tests/evaluation/test_rag_eval_regression.py`，CI 集成

### 评估维度

系统采用五维加权评分模型，各维度权重如下：

| 维度 | 权重 | 智能体 | 说明 |
|------|------|--------|------|
| 问诊技巧（inquiry） | 25% | 问诊分析智能体 | 评估问诊的系统性、完整性与临床规范 |
| 医学知识（knowledge） | 25% | 知识核对智能体 | 基于 RAG 检索对比临床指南，评估知识一致性 |
| 人文关怀（humanistic） | 20% | 人文关怀智能体 | 评估沟通态度、共情能力与患者教育 |
| 诊断能力（diagnosis） | 15% | 诊断评估智能体 | 评估诊断方向与鉴别诊断思路 |
| 治疗方案（treatment） | 15% | 治疗评估智能体 | 评估治疗方案的合理性与指南符合度 |

**权重重分配机制**：当某维度评分为 `None`（拒答/未评估）时，该维度权重自动重分配至其余有效维度。彻底移除了默认 50 分兜底逻辑。

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
| 编排框架 | LangGraph（Plan-Execute + Send fan-out/fan-in） | — |
| 缓存/Checkpoint | Redis（生产环境 Checkpoint 持久化 + LLM 响应缓存） | redis 7.4.1 |
| Checkpoint 持久化 | langgraph-checkpoint-redis | 0.4.1 |
| 速率限制 | slowapi | >=0.1.9 |
| Function Calling | OpenAI SDK（Qwen Function Calling 兼容接口） | — |
| 实时通信 | SSE（Server-Sent Events） | — |
| 容器化 | Docker Compose（MySQL + Redis + Backend + Frontend） | — |

## 项目结构

```
medical-ai-platform/
├── backend/                              # 后端服务（FastAPI）
│   ├── app/
│   │   ├── api/v1/                       # REST API 路由
│   │   │   ├── auth.py                   #   认证（注册/登录）
│   │   │   ├── patients.py               #   虚拟患者管理
│   │   │   ├── consultations.py          #   问诊交互（含 SSE 流式端点）
│   │   │   ├── evaluations.py            #   评估触发与报告
│   │   │   ├── knowledge_base.py         #   知识库管理
│   │   │   └── stats.py                  #   管理员统计
│   │   ├── core/                         # 核心基础设施
│   │   │   ├── config.py                 #   配置管理（含 ReAct/Tool Use/LangGraph/缓存/审计）
│   │   │   ├── security.py               #   密码加密（bcrypt）+ JWT（python-jose）
│   │   │   ├── deps.py                   #   认证依赖注入
│   │   │   ├── audit.py                  #   审计日志记录模块
│   │   │   ├── limiter.py                #   API 速率限制（slowapi）
│   │   │   ├── validation.py             #   输入验证（XSS 防护、HTML 过滤）
│   │   │   └── websocket.py              #   WebSocket 支持
│   │   ├── models/                       # 数据库模型（SQLAlchemy）
│   │   │   ├── audit_log.py              #   审计日志模型
│   │   │   ├── user.py                   #   用户模型
│   │   │   ├── patient.py                #   患者模型
│   │   │   ├── consultation.py           #   问诊模型
│   │   │   ├── evaluation.py             #   评估模型
│   │   │   ├── evaluation_run.py         #   评估运行模型
│   │   │   └── evaluation_node_result.py #   评估节点结果模型
│   │   ├── schemas/                      # 请求/响应模型（Pydantic）
│   │   ├── services/
│   │   │   ├── agents/                   # AI 智能体
│   │   │   │   ├── inquiry_agent.py      #   问诊分析
│   │   │   │   ├── diagnosis_agent.py    #   诊断评估
│   │   │   │   ├── treatment_agent.py    #   治疗评估
│   │   │   │   ├── knowledge_agent.py    #   知识核对（传统/Tool Use/ReAct 三模式）
│   │   │   │   ├── reflection_agent.py   #   反思验证（ReAct + Consistency Tools）
│   │   │   │   ├── humanistic_agent.py   #   人文关怀
│   │   │   │   ├── safety_agent.py       #   安全门控
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
│   │   │   ├── consultation_service.py   #   问诊逻辑（患者模拟 + SSE + 记忆管理）
│   │   │   ├── patient_service.py        #   患者服务
│   │   │   ├── user_service.py           #   用户服务
│   │   │   ├── llm_cache.py              #   LLM 响应缓存层（Redis + SHA256）
│   │   │   ├── qwen_client.py            #   Qwen API 客户端
│   │   │   ├── tools/                    # Function Call 工具系统
│   │   │   │   ├── base.py               #   BaseTool 基类 + ToolContext
│   │   │   │   ├── registry.py           #   工具注册表
│   │   │   │   ├── executor.py           #   统一执行器（校验/预算/超时/截断）
│   │   │   │   ├── robust_tool_executor.py #  熔断器 + 重试 + 健康检查执行器
│   │   │   │   ├── tool_budget_manager.py#   全局工具预算管理
│   │   │   │   ├── tool_health_checker.py#   工具健康状态检查
│   │   │   │   ├── medical_retrieval.py  #   医学检索工具（4 个）
│   │   │   │   ├── citation.py           #   引用校验工具
│   │   │   │   ├── consistency.py        #   一致性检查工具（4 个，供 Reflection 使用）
│   │   │   │   ├── scoring.py            #   评分工具
│   │   │   │   └── budget.py             #   工具预算控制
│   │   │   └── scoring/                  # 评分引擎
│   │   │       ├── policies.py           #   版本化策略
│   │   │       ├── calculator.py         #   确定性计算器
│   │   │       └── summary.py            #   摘要生成器
│   │   ├── orchestration/                # LangGraph 编排层
│   │   │   ├── state.py                  #   EvaluationState + EvaluationPlan + ReflectionResult
│   │   │   ├── graph.py                  #   StateGraph 主图（Plan-Execute + Send fan-out）
│   │   │   ├── checkpointer.py           #   Redis Checkpoint
│   │   │   ├── adapters/                 #   Agent 适配器（8 个）
│   │   │   │   ├── base.py               #     适配器基类
│   │   │   │   ├── registry.py           #     适配器注册表
│   │   │   │   ├── inquiry.py            #     问诊分析适配器
│   │   │   │   ├── diagnosis.py          #     诊断评估适配器
│   │   │   │   ├── treatment.py          #     治疗评估适配器
│   │   │   │   ├── humanistic.py         #     人文关怀适配器
│   │   │   │   ├── knowledge.py          #     知识 Agent 适配器（含 Tool Use/ReAct 切换）
│   │   │   │   ├── reflection.py         #     反思 Agent 适配器
│   │   │   │   └── suggestion.py         #     建议指导适配器
│   │   │   └── routes.py                 #   路由矩阵
│   │   └── db/session.py                 # 数据库连接
│   ├── evaluation/                       # 离线评测框架
│   │   ├── datasets.py                   #   评测数据集加载
│   │   ├── runners.py                    #   评测运行器（Recall/MRR/nDCG）
│   │   └── rag_cases/                    #   RAG 评测用例
│   ├── tests/                            # 测试（30 个测试文件）
│   │   ├── test_auth_error_handling.py   #   认证错误处理测试
│   │   ├── test_auth_password_length.py  #   密码长度校验测试
│   │   ├── test_e2e.py                   #   端到端集成测试
│   │   ├── test_llm_cache.py             #   LLM 缓存层测试
│   │   ├── test_react_upgrade.py         #   ReAct 升级集成测试
│   │   ├── test_security_hardening.py    #   安全加固测试
│   │   ├── agents/                       #   智能体测试
│   │   │   └── test_knowledge_agent_tool_use.py
│   │   ├── evaluation/                   #   评测回归测试（5 个文件）
│   │   │   ├── test_datasets.py
│   │   │   ├── test_metrics.py
│   │   │   ├── test_rag_eval_regression.py
│   │   │   ├── test_rag_eval_smoke.py
│   │   │   └── test_report.py
│   │   ├── orchestration/                #   编排层测试（8 个文件）
│   │   │   ├── test_adapters.py
│   │   │   ├── test_baseline.py
│   │   │   ├── test_graph.py
│   │   │   ├── test_knowledge_adapter_tool_flag.py
│   │   │   ├── test_models.py
│   │   │   ├── test_routes.py
│   │   │   ├── test_safety.py
│   │   │   └── test_service.py
│   │   ├── rag/                          #   RAG 单元测试与离线评测（5 个文件）
│   │   │   ├── eval_offline.py
│   │   │   ├── test_metadata.py
│   │   │   ├── test_query_builder.py
│   │   │   ├── test_retriever.py
│   │   │   └── test_types.py
│   │   ├── services/                     #   服务层测试（3 个文件）
│   │   │   ├── test_consultation_service.py
│   │   │   ├── test_evaluation_service.py
│   │   │   └── test_qwen_client_tools.py
│   │   └── tools/                        #   Tool Use 单元测试（2 个文件）
│   │       ├── test_executor.py
│   │       └── test_registry.py
│   ├── requirements.txt
│   └── .env                              # 环境变量
├── frontend/                             # 前端应用（React + Vite）
│   ├── src/
│   │   ├── api/                          # 后端接口封装（含 SSE 客户端）
│   │   ├── components/                   # 公共组件库
│   │   │   ├── ScoreDisplay.tsx          #   统一分数展示（含 getScoreColor/getScoreLevel）
│   │   │   ├── PersonalityTag.tsx        #   人格标签组件
│   │   │   ├── DimensionRadar.tsx        #   五维雷达图（recharts）
│   │   │   ├── LoadingOverlay.tsx        #   加载遮罩
│   │   │   └── index.ts                  #   统一导出入口
│   │   ├── pages/                        # 页面组件（11 个页面）
│   │   │   ├── Login/                    #   登录
│   │   │   ├── Register/                 #   注册
│   │   │   ├── Dashboard/                #   主页/仪表盘
│   │   │   ├── PatientList/              #   患者列表
│   │   │   ├── Consultation/             #   问诊交互（SSE 流式）
│   │   │   ├── ConsultationList/         #   问诊记录
│   │   │   ├── Evaluation/               #   评估报告
│   │   │   ├── Profile/                  #   个人资料
│   │   │   ├── AdminPatients/            #   管理员 - 患者管理
│   │   │   ├── AdminConsultations/       #   管理员 - 问诊管理
│   │   │   └── AdminStats/               #   管理员 - 统计面板
│   │   ├── store/useAuth.ts              # 状态管理
│   │   └── utils/request.ts              # Axios 封装
│   ├── vite.config.ts
│   └── package.json
├── database/
│   ├── init.sql                          # 建表 SQL（含全部字段）
│   ├── migrate_v2.sql                    # 诊断/治疗字段 + 五维度评估
│   ├── migrate_v3.sql                    # 密码字段扩容
│   ├── migrate_v4.sql                    # RAG 审计字段（幂等迁移）
│   ├── migrate_v5.sql                    # Plan-Execute 字段（evaluation_plan, execution_results）
│   ├── migrate_v6.sql                    # 审计日志表（audit_logs，幂等迁移）
│   ├── migrate_v7.sql                    # consultations 新增 max_rounds 列（INT DEFAULT 20）
│   └── seed.sql                          # 种子数据
├── dataset/                              # 评测数据集（150+ 病例，已 gitignore）
├── data/                                 # 医学教材与指南 PDF（80+ 部）
├── .github/workflows/ci.yml              # CI/CD（GitHub Actions）
├── docker-compose.yml                    # 一键部署（MySQL + Redis + Backend + Frontend）
├── docker-compose.override.yml           # 开发环境覆盖配置
├── Dockerfile.backend                    # 后端容器构建
└── Dockerfile.frontend                   # 前端容器构建
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
| Redis | 6.0+ | Checkpoint 持久化（LangGraph 默认启用，需配置；设为 false 可跳过） |

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
mysql -u root -p medical_ai < database/migrate_v5.sql
mysql -u root -p medical_ai < database/migrate_v6.sql
mysql -u root -p medical_ai < database/migrate_v7.sql
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

# LangGraph 编排默认已启用（如需回退旧编排，取消下行注释）
# $env:LANGGRAPH_ENABLED="false"

# 启用知识 Agent Tool Use（可选）
$env:ENABLE_TOOL_USE="true"

# 启用知识 Agent ReAct 模式（可选，与 Tool Use 二选一）
$env:ENABLE_REACT_KNOWLEDGE="true"

# 启用反思 Agent（可选）
$env:ENABLE_REACT_REFLECTION="true"

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
| `migrate_v5.sql` | Plan-Execute 模式字段：`evaluation_runs.evaluation_plan`（JSON）、`evaluation_runs.execution_results`（JSON） |
| `migrate_v6.sql` | 新建 `audit_logs` 表（幂等迁移），含 user_id / action / resource_id / ip_address / user_agent / detail / created_at |
| `migrate_v7.sql` | `consultations` 表新增 `max_rounds` 列（INT DEFAULT 20），控制最大问诊轮次 |

## 配置说明

所有配置项通过 `backend/.env` 或系统环境变量管理：

### 基础配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `PROJECT_NAME` | 医学问诊评估平台 | 项目名称 |
| `VERSION` | 1.0.0 | 项目版本 |
| `API_V1_PREFIX` | /api/v1 | API v1 路径前缀 |
| `MYSQL_HOST` | localhost | MySQL 地址 |
| `MYSQL_PORT` | 3306 | MySQL 端口 |
| `MYSQL_USER` | root | MySQL 用户 |
| `MYSQL_PASSWORD` | （空） | MySQL 密码 |
| `MYSQL_DATABASE` | medical_ai | 数据库名 |
| `SECRET_KEY` | （内置默认值） | JWT 签名密钥 |
| `ALGORITHM` | HS256 | JWT 签名算法 |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 1440 | Token 有效期（24 小时） |
| `CORS_ORIGINS` | ["http://localhost:5173", "http://localhost:3000"] | CORS 允许源 |
| `DASHSCOPE_API_KEY` | （系统环境变量） | 阿里云百炼 API Key |
| `QWEN_API_BASE_URL` | dashscope 兼容模式 | Qwen API 地址 |
| `QWEN_MODEL` | qwen3.7-max | 默认 LLM 模型 |
| `RERANK_MODEL` | gte-rerank | 专用重排模型 |
| `LLM_MAX_CONCURRENT` | 10 | LLM 最大并发调用数 |
| `LLM_SEMAPHORE_TIMEOUT` | 60 | 信号量等待超时（秒） |
| `ACTIVE_INDEX_VERSION` | rag-v1 | 当前活跃 RAG 索引版本 |

### LangGraph 编排配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LANGGRAPH_ENABLED` | true | 启用 LangGraph 编排（设为 false 回退旧 asyncio.gather 编排） |
| `LANGGRAPH_SHADOW_MODE` | false | 影子模式：新旧路径并行对比 |
| `LANGGRAPH_GRAPH_VERSION` | evaluation-graph-v1 | 图版本标识 |
| `REDIS_CHECKPOINT_URL` | redis://localhost:6379/1 | Redis Checkpoint 连接地址 |
| `REDIS_CHECKPOINT_TTL` | 86400 | Checkpoint 过期时间（秒） |

### Function Call / Tool Use 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ENABLE_TOOL_USE` | true | 启用知识 Agent Tool Use（Function Calling 模式） |
| `TOOL_USE_MODEL` | qwen-max | Tool Use 专用模型 |
| `TOOL_USE_MAX_ROUNDS` | 4 | 最大工具调用轮次 |
| `TOOL_USE_MAX_CALLS` | 8 | 最大工具调用总次数 |
| `TOOL_USE_TIMEOUT_SECONDS` | 30 | 工具调用超时（秒） |
| `TOOL_USE_MAX_RESULT_CHARS` | 6000 | 工具返回结果最大字符数 |
| `KNOWLEDGE_TOOL_MAX_RAG_CALLS` | 3 | RAG 检索最大调用次数 |
| `KNOWLEDGE_TOOL_MAX_MQE_CALLS` | 2 | MQE 扩展最大调用次数 |
| `KNOWLEDGE_TOOL_MAX_HYDE_CALLS` | 1 | HyDE 最大调用次数 |
| `TOOL_USE_FALLBACK_TO_LEGACY` | true | 失败时回退旧路径 |

### ReAct 模式配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ENABLE_REACT_KNOWLEDGE` | true | Knowledge Agent 启用 ReAct 模式（显式 Thought→Action→Observation） |
| `ENABLE_REACT_REFLECTION` | true | Reflection Agent 启用 ReAct 模式（默认开启） |
| `ENABLE_LLM_SUGGESTION` | true | 启用 LLM 建议生成（调用 suggestion_agent 进行对比学习分析，false 时回退规则建议） |
| `REACT_MAX_STEPS` | 6 | ReAct 最大推理步数 |
| `REFLECTION_CONSISTENCY_THRESHOLD` | 0.3 | 评分一致性偏差阈值 |
| `REFLECTION_EVIDENCE_MIN_SCORE` | 60.0 | 证据充足的最低分数 |

### LLM 缓存配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_CACHE_ENABLED` | true | 启用 LLM 响应缓存（仅 temperature=0 的确定性调用） |
| `LLM_CACHE_TTL` | 86400 | 缓存过期时间（秒），24小时 |
| `LLM_CACHE_SIMILARITY_THRESHOLD` | 0.95 | 语义相似度阈值（保留，当前使用精确哈希匹配） |
| `LLM_CACHE_MAX_SIZE` | 10000 | 最大缓存条目数 |

### 安全与审计配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `AUDIT_LOG_ENABLED` | true | 启用审计日志记录 |

### Feature Flags 完整清单

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LANGGRAPH_ENABLED` | true | 启用 LangGraph 编排（设为 false 回退旧 asyncio.gather 编排） |
| `LANGGRAPH_SHADOW_MODE` | false | 影子模式：新旧路径并行对比 |
| `ENABLE_TOOL_USE` | true | 启用知识 Agent Tool Use（Function Calling） |
| `ENABLE_REACT_KNOWLEDGE` | true | Knowledge Agent ReAct 模式 |
| `ENABLE_REACT_REFLECTION` | true | Reflection Agent ReAct 模式（默认开启） |
| `ENABLE_LLM_SUGGESTION` | true | 启用 LLM 建议生成 |
| `LLM_CACHE_ENABLED` | true | 启用 LLM 响应缓存 |
| `AUDIT_LOG_ENABLED` | true | 启用审计日志 |

### 五维评分权重（`scoring/policies.py`）

| 维度 | 权重 |
|------|------|
| inquiry（问诊技巧） | 0.25 |
| knowledge（医学知识） | 0.25 |
| humanistic（人文关怀） | 0.20 |
| diagnosis（诊断能力） | 0.15 |
| treatment（治疗方案） | 0.15 |

### 检索阈值常量（`rag/types.py`）

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

项目包含 30 个测试文件，覆盖全模块单元测试与集成测试。

```powershell
cd backend

# 运行全部单元测试
pytest tests/ -v --tb=short

# 仅运行 RAG 相关测试
pytest tests/rag/ -v

# 仅运行 Tool Use 相关测试
pytest tests/tools/ -v
pytest tests/services/test_qwen_client_tools.py -v
pytest tests/agents/test_knowledge_agent_tool_use.py -v

# ReAct 升级集成测试
pytest tests/test_react_upgrade.py -v

# 安全加固测试
pytest tests/test_security_hardening.py -v

# LLM 缓存层测试
pytest tests/test_llm_cache.py -v

# 编排层测试
pytest tests/orchestration/ -v

# 服务层测试
pytest tests/services/ -v

# 评测回归测试
pytest tests/evaluation/ -v

# 离线评测（评估 RAG 检索质量）
python tests/rag/eval_offline.py
```

**测试目录结构**（30 个测试文件）：

| 目录 | 文件数 | 覆盖范围 |
|------|--------|----------|
| `tests/`（根） | 6 | 认证错误处理、密码长度校验、端到端集成、LLM 缓存、ReAct 升级、安全加固 |
| `tests/agents/` | 1 | 知识 Agent Tool Use 测试 |
| `tests/evaluation/` | 5 | 数据集加载、评测指标、RAG 回归测试、RAG 冒烟测试、报告生成 |
| `tests/orchestration/` | 8 | 适配器、基线、图结构、知识适配器工具标志、模型、路由、安全、服务 |
| `tests/rag/` | 5 | 元数据、查询构建器、检索器、类型、离线评测 |
| `tests/services/` | 3 | 问诊服务、评估服务、Qwen 客户端 |
| `tests/tools/` | 2 | 执行器、注册表 |

**测试覆盖**：
- LangGraph 编排层：图节点测试 + Plan-Execute 测试 + Send fan-out 测试
- Tool Use：Registry 测试 + Executor 测试 + qwen_client 测试
- 知识 Agent：分数映射测试 + Feature Flag 切换测试 + ReAct 模式测试
- Reflection Agent：一致性工具测试 + 反思流程测试
- 适配器：适配器测试 + 路由测试
- 安全加固：JWT 认证测试 + 速率限制测试 + 输入验证测试 + 审计日志测试
- LLM 缓存：精确哈希匹配测试 + TTL 测试 + 降级测试
- RAG 评测：离线回归测试（Recall/MRR/nDCG）

## API 文档

后端启动后，访问自动生成的交互式 API 文档：

| 文档 | 地址 |
|------|------|
| Swagger UI | http://localhost:8000/docs |
| 健康检查 | http://localhost:8000/health |

### 主要接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/auth/register | 注册 |
| POST | /api/v1/auth/login | 登录 |
| GET | /api/v1/auth/me | 获取当前用户 |
| PUT | /api/v1/auth/profile | 更新个人资料 |
| GET | /api/v1/patients/ | 虚拟患者列表 |
| GET | /api/v1/patients/{id} | 患者详情 |
| POST | /api/v1/patients/ | 创建患者 |
| PUT | /api/v1/patients/{id} | 更新患者 |
| DELETE | /api/v1/patients/{id} | 删除患者 |
| POST | /api/v1/consultations/ | 创建问诊 |
| GET | /api/v1/consultations/ | 问诊列表 |
| GET | /api/v1/consultations/all | 管理员全部问诊 |
| GET | /api/v1/consultations/{id} | 问诊详情 |
| POST | /api/v1/consultations/{id}/messages | 发送消息 |
| POST | /api/v1/consultations/{id}/messages/stream | SSE 流式消息（患者回复实时推送） |
| POST | /api/v1/consultations/{id}/extend | 延长问诊 |
| POST | /api/v1/consultations/{id}/submit-diagnosis | 提交诊断 |
| POST | /api/v1/consultations/{id}/end | 结束问诊 |
| DELETE | /api/v1/consultations/{id} | 删除问诊 |
| POST | /api/v1/evaluations/ | 触发评估 |
| GET | /api/v1/evaluations/{id} | 查看评估报告 |
| GET | /api/v1/knowledge-base/stats | 知识库统计信息 |
| POST | /api/v1/knowledge-base/add-pdf | 增量添加 PDF |
| DELETE | /api/v1/knowledge-base/sources/{name} | 删除指定来源 |
| POST | /api/v1/knowledge-base/rebuild | 触发全量重建 |
| GET | /api/v1/knowledge-base/rebuild/status | 查询重建状态 |
| POST | /api/v1/knowledge-base/cache/clear | 清空 Embedding 缓存 |
| GET | /api/v1/stats/ | 管理员统计 |

## CI/CD

项目使用 GitHub Actions 实现持续集成，配置文件为 `.github/workflows/ci.yml`。

### 流水线 Jobs

| Job | 运行环境 | 说明 |
|-----|----------|------|
| `backend-test` | MySQL 8.0 + Redis 7 服务容器 | pytest 含覆盖率（`--cov=app`），上传 Codecov |
| `frontend-build` | Node.js 18 | `npm ci` + `npm run build`，检查 dist 目录输出 |
| `code-quality` | Python 3.10 | flake8 语法检查 + mypy 类型检查 + `compileall` 编译检查 |

**触发条件**：`push` 和 `pull_request` 到 `main` / `master` 分支。
