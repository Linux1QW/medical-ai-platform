# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/lang/zh-CN/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2026-08-10

### Added

- **Interview Coach Agent（opt-in，默认关闭）**：7 节点 LangGraph 图（intent → planner → evidence → draft → critic → finalize → persist），CoachContextView 隔离隐藏字段（`extra="forbid"`），Working Memory 16K 预算，Critic 安全检查，SSE 流式建议 + 幂等重放 + Last-Event-ID 重放
- **三层记忆架构**：Working Memory（对话槽位 + 隐藏事实校验 `HiddenContextViolation`）、Episodic Memory（历史问诊摘要）、Approved Trainee Profile Memory（6 维度、consent 默认 false、PHI 正则校验、管理员审批、最多 5 条进入上下文）
- **Skill Registry + Policy Enforcement**：工具白名单（per-agent least privilege）、预算控制、UNTRUSTED_EVIDENCE 包装、控制指令清洗、结果长度限制
- **MCP Demo Server**：stdio-only JSON-RPC 2.0，两个只读工具（search_teaching_rubric、search_medical_kb），静态 fixture，去标识化，不绑定网络端口
- **LiveKit Voice Beta（optional, NOT ACCEPTED）**：房间令牌 TTL ≤600s、max 4 participants、部分转录仅内存不持久化、原始录音不保存、final transcript 去重 `(room_sid, participant_sid, turn_id)`。默认关闭（VOICE_ENABLED=false），无真实 LiveKit 证据，未纳入验收。
- **Prompt Registry + A/B Rollout**：PromptBundle 生命周期（draft → active → deprecated）+ ExperimentAssignment 实验分配
- **Coach Attribution Flywheel**：trace → eval → attribution candidate → admin review + deidentify → eligible for training
- **Coach API 端点**：`/api/v1/coach/consultations/{id}/suggestions/stream`（SSE）、`/state`、`/feedback`、`/admin/traces`
- **Voice API 端点**：`/api/v1/voice/sessions`、`/sessions/{room_name}/state`、`/sessions/{room_name}/end`
- **Trainee Memory API 端点**：`/api/v1/trainee-memory/memories`、`/me/memories`、`/me/consent`、`/memories/{id}/review`
- **V1.2 数据库迁移**：`4d5e6f7a8b9c` 新增 coach_sessions、coach_decisions、coach_stream_events、trainee_memories、trainee_memory_consents、prompt_bundles、experiment_assignments、agent_trace_events；`5e6f7a8b9c0d` remediation 补充约束和索引
- **V1.2 权限模型**：`coach:use`、`coach:trace:view`、`trainee-memory:manage-self`、`trainee-memory:review`、`voice:use`、`prompt:manage`、`experiment:manage`
- **Agent Telemetry**：12 事件类型、HMAC 签名、隐私安全
- **Coach Frontend UI**：CoachPanel 组件（6 状态：disabled/idle/thinking/suggestion/degraded/error）、useCoachSuggestion hook（SSE + 幂等重放）

### Changed

- README 更新到 v1.2.0，新增 V1.2 功能描述和文档导航
- 文档新增 V1.2 Production Guide、V1.2 Runbook、V1.2 Acceptance Summary

### Security

- Coach 默认关闭（`COACH_ENABLED=false`），所有 coach 端点在禁用时返回 409
- CoachContextView 使用 `extra="forbid"` 拒绝 expected_diagnosis 等隐藏字段
- Working Memory `validate_memory_sources()` 阻止隐藏上下文泄露
- 所有工具输出标记为 UNTRUSTED_EVIDENCE 防止 prompt injection
- Trainee memory consent 默认 false（opt-in），PHI 正则校验拒绝 SSN/信用卡/邮箱/IP
- Voice 不保存原始音频，部分转录仅内存
- MCP Demo Server 不绑定网络端口，stdio-only
- IDOR 防护：Coach 端点验证 consultation 归属，Voice 端点验证问诊访问权

### Note

- Coach 为 opt-in 功能，默认关闭，需要显式配置 `COACH_ENABLED=true` 和 `COACH_HMAC_KEY`
- Voice 为 optional beta（NOT ACCEPTED），需要配置 LiveKit 凭据，不参与评估流程，默认关闭（VOICE_ENABLED=false）
- MCP Demo Server 为演示用途，使用静态 fixture，不连接生产数据库
- 验收指标（Intent F1、hidden-fact leakage、latency 等）需要实测环境运行，当前未宣称通过
- 验收证据由 `final-acceptance.json` 作为唯一事实来源，Markdown 文档由脚本自动生成

## [1.1.0] - 2026-08-10

### Added

- **Transactional Outbox + Dispatcher**：评估任务通过 `evaluation_dispatch_outbox` 表与业务事务原子写入，独立 Dispatcher 进程（`evaluation-dispatcher` 服务）轮询派发至 Celery，保证至少一次投递
- **EvaluationRun 状态机**：`evaluation_run_service.py` 集中管理 run 生命周期（queued → running → completed/needs_review/failed/cancelled），含 retrying 重试路径、协作式取消和 lease 租约机制
- **Reconciliation 对账**：`evaluation_reconciliation.py` 定时扫描过期 lease、stale dispatch、retrying 超时，自动释放/重投/dead letter
- **双 Redis 物理拓扑**：`redis-state`（AOF + noeviction）承载 checkpoint/broker/result/JWT/控制/进度；`redis-cache`（allkeys-LRU）承载 LLM 缓存和检索缓存
- **数据分级保留策略**：dispatch 终态 7 天、无报告 failed/cancelled run 180 天、cache/progress 24h/1h；审计 auto-delete 默认关闭
- **隐私威胁模型**：文档化患者身份泄露、LLM API 泄露、日志泄露等威胁及缓解措施
- **Alembic 迁移服务**：Compose `migrate` 服务在启动时自动执行 V1.1 迁移
- **V1.1 数据库迁移**：`2b3c4d5e6f7a` 新增 `evaluation_dispatch_outbox` 表；`3c4d5e6f7a8b` 新增 review/audit 查询优化索引
- **稳定 Citation ID**：基于 `(kb_version + doc_id + chunk_id + content_hash)` 的 SHA-256 确定性 ID
- **Claim-Evidence Graph**：治疗/诊断 claim 必须附带证据链接，unsupported claim 自动标记需复核
- **五维原子 Rubric 评估体系**：每维度独立评分（pass/partial/fail/unassessed/not_applicable），unassessed ≠ 0 分
- **安全红旗回归集**：高危症状 fail-closed 机制，LLM 失败 + 无规则匹配 → 自动转人工复核
- **人工复核状态机**：pending → in_review → approved/rejected/returned 合法迁移验证
- **并发/Token/成本预算控制**：RunBudget 限制并发 Agent 数 / Token 总量 / 成本上限，安全路径豁免
- **全链路 Trace 与可观测性**：TraceContext 贯穿 Celery 重试，PII 自动脱敏
- **前端证据化报告组件**：RubricItemList + EvidenceTrace + RiskBanner
- **人工复核工作台**：复核队列排序/筛选/详情弹窗/决策表单
- **可版本化临床能力基准集**：BenchmarkManifest 管理 dev/test/regression/safety/benchmark 分组
- **端到端发布验收测试**：25 个 E2E 场景覆盖全部迭代任务
- **Prompt 外置化 + 版本管理**：13 个 agent system prompt 抽离为文件，支持灰度覆盖、变量渲染、缓存与热重载
- **LLM Provider 适配器抽象层**：`ProviderAdapter` + `OpenAICompatibleAdapter` + 注册表

### Changed

- Docker Compose 全面重构：YAML anchor 统一环境变量，新增 `redis-cache`、`evaluation-dispatcher`、`migrate` 服务
- `backend` 服务已通过 `x-backend-environment` 注入正确的容器内 Celery broker/result URL
- RAG 源文件挂载修正为 `./data:/app/data:ro`，与代码 `PDF_DIR` 一致
- 后端测试从 579 用例增至 1093 用例（+100%）
- 前端测试覆盖 64 用例（9 个测试文件）
- `ACCESS_TOKEN_EXPIRE_MINUTES` 默认值调整为 60 分钟（生产不得超过 60）
- 文档全面更新：README quickstart、PROJECT_GUIDE V1.1 组件、technical-document 状态机/Outbox/双 Redis、platform-documentation 临床使用边界

### Fixed

- 修复 failover「半接线」缺陷：熔断切换 Provider 时真实重建底层 LLM 客户端
- 修复 Compose 未注入 Celery broker/result URL 的问题
- 修复 Compose RAG 源文件挂载路径与代码 `PDF_DIR` 不一致的问题
- 修复 `container_name` 限制 Worker 水平扩展的问题

### Security

- staging/production 环境下 `JWT_BLACKLIST_FAIL_CLOSED` 必须为 `true`，防止 Redis 不可用时放行已吊销 token
- staging/production 环境下 `ACCESS_TOKEN_EXPIRE_MINUTES` 不得超过 60 分钟
- staging/production 环境下启用 Langfuse 必须配置 `OBSERVABILITY_HMAC_KEY`（≥32 字节）且 `OBSERVABILITY_CAPTURE_CONTENT=false`
- 可观测性隐私策略：默认 `capture_content=false`，日志/trace 脱敏

### Breaking Changes

- **删除 Celery task status API**：旧 `GET /api/v1/evaluations/task/{task_id}/status` 未校验 task 归属，V1.1 引入 Outbox + run 状态机后应使用 `GET /api/v1/evaluations/runs/{run_id}/status`
- **旧 WebSocket URL 变更**：评估进度 WebSocket 从旧路径迁移到 `/api/v1/evaluations/ws/{consultation_id}`，首消息必须携带 JWT auth
- **双 Redis 分离**：本地开发 `LLM_CACHE_REDIS_URL` 默认从 `localhost:6379` 改为 `localhost:6380`（redis-cache），已有开发环境需更新 Redis 配置
- **Alembic 迁移链新增**：head 从 `1a2b3c4d5e6f` 推进到 `3c4d5e6f7a8b`，升级前必须备份并在副本验证
- **报告关联数据保留策略变更**：run/Evaluation/ReviewRecord/AuditLog 不再由通用 cleanup 删除，保留周期由 `DATA_RETENTION_POLICY_ID` 与审批流程决定

## [1.0.0] - 2026-07-21

### Added
- 评估防重复提交机制 + ORM 模型补全 + Pydantic V2 迁移
- 前端 chunk 优化与评估结果自动展示
- 安全 Agent 确定性红旗规则门控
- RAG 模块 V2 核心架构（混合检索 + Reranker + HyDE）
- 多智能体临床问诊评估平台完整架构
- ReAct 推理链稳定化 + Suggestion Agent 集成
- LLM 缓存层 + 安全加固（限流/审计/密码策略）
- LangGraph 编排重构 + Redis Checkpoint + Function Call
- Tool Use 加固 + 评估指标体系完善
- Docker Compose 部署编排配置
- 前端组件库统一（Ant Design 5.x）
- 代码质量工具链：ruff + pre-commit + pyproject.toml 统一配置

### Changed
- 评估页面交互优化：进入后自动生成评估报告
- 五维评估架构升级：Plan-Execute / Send fan-out / ReAct / Reflection / SSE
- 数据集管理：移除真实医疗数据，保护隐私
- CI/CD 流程完善：测试覆盖率上报 + 健康检查

### Fixed
- 前端 TypeScript 构建错误修复
- test_llm_cache CI mock 三层失效修复
- flake8 F821 前向引用、auth db=None 防御、限流器测试隔离
- RAG 评估测试 CLI 子进程 cwd 路径错误
- 多项 CI 测试稳定性修复
