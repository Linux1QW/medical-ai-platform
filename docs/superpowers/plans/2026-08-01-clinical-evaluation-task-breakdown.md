# 临床问诊评估平台 Task 执行卡 Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

Goal: 将临床问诊评估平台的总体路线图拆分为可以独立开发、独立测试、独立 review 和独立提交的 Task。

Architecture: Task 0 先冻结当前行为；Task 1～4 建立评测可信度；Task 5～8 建立安全和循证；Task 9～11 收敛运行架构；Task 12～15 完成前端、基准集和治理；Task 16 做全链路发布验收。每个 Task 都要求先测试后实现，并且不能跨 Task 偷换数据协议。

Tech Stack: Python、FastAPI、Pydantic v2、SQLAlchemy async、MySQL、Redis、Celery、LangGraph、ChromaDB、pytest、React、Ant Design、Prometheus、Grafana、Langfuse。

## 全局执行规则

- 后端工作目录为 D:\Q123\PyCharm\PycharmProjects\基于多智能体的医生临床问诊评估平台\medical-ai-platform\backend。
- Python 测试命令统一使用 venv\Scripts\python.exe -m pytest。
- 测试中 mock 所有 LLM、真实 Redis、真实外部模型和真实患者数据。
- 每个 Task 的 commit 只包含该 Task 文件范围内的改动。
- 每个 Task 完成后必须记录测试命令、测试结果和未解决风险。
- 任何新字段都必须同时考虑 Pydantic、数据库 JSON、API response、前端 TypeScript 和历史数据兼容。
- 任何安全关键路径必须 fail closed。
- 不允许把 null、unassessed、insufficient 或 error 当成 0 分。
- 不允许 push；只允许本地 commit。

## 依赖总览

Task 0 -> Task 1 -> Task 2
Task 1 -> Task 3 -> Task 4
Task 1 -> Task 5 -> Task 6
Task 1 -> Task 7 -> Task 8
Task 2、Task 3、Task 5 -> Task 9
Task 1、Task 9 -> Task 10 -> Task 11
Task 3、Task 6、Task 8 -> Task 12 -> Task 13
Task 7、Task 8 -> Task 14
Task 6、Task 11、Task 14 -> Task 15
Task 1、Task 5、Task 8、Task 11、Task 12、Task 15 -> Task 16

---

## Task 0：当前系统基线冻结

目标：固定当前 pre-push、评估图、RAG、人工复核和前端状态语义，避免后续优化无法判断是否回归。

前置依赖：无。

输入：
- backend/scripts/eval_regression.py
- backend/scripts/hooks/pre-push
- backend/evaluation/patient_ab_thresholds.json
- backend/app/orchestration/graph.py
- backend/app/services/evaluation_service.py
- backend/evaluation/reports/patient_ab 下已有报告

文件：
- Create: backend/tests/evaluation/test_iteration_baseline.py
- Create: docs/evaluation-baseline.md

实现动作：
1. 记录 3 例报告的实际退出码和 pre-push 行为。
2. 记录 18 例正式报告的阈值检查结果。
3. 记录五维分数的 null、0、insufficient、needs_review 语义。
4. 记录 RAG citation、rag_trace、review_reason 的现有字段。
5. 记录全量后端测试文件数和前端测试文件数。
6. 为当前行为写最小回归测试，不修改生产逻辑。

测试：
- venv\Scripts\python.exe scripts\eval_regression.py
- venv\Scripts\python.exe -m pytest tests\evaluation tests\orchestration tests\rag -q

验收指标：
- 能复现 3 例 smoke 返回 SKIP(2)。
- 能明确指出 pre-push 只阻断退出码 1 的事实。
- docs/evaluation-baseline.md 有完整基线记录。
- 新增基线测试全部通过。

提交：
- Commit message: test(eval): freeze current evaluation baseline

---

## Task 1：ReportManifest 统一报告协议

目标：让每份评测报告知道自己是什么类型、由什么版本生成、包含多少病例。

前置依赖：Task 0。

输入：
- backend/evaluation/report.py
- backend/evaluation/patient_regression.py
- backend/evaluation/patient_ab_thresholds.json

文件：
- Create: backend/evaluation/report_schema.py
- Modify: backend/evaluation/report.py
- Modify: backend/evaluation/patient_regression.py
- Modify: backend/scripts/eval_regression.py
- Modify: backend/evaluation/patient_ab_thresholds.json
- Test: backend/tests/evaluation/test_report_schema.py

核心接口：
- ReportManifest
- ReportKind = smoke | regression | benchmark
- load_report_manifest(report: dict, allow_legacy: bool) -> ReportManifest
- validate_report_manifest(manifest: ReportManifest) -> list[str]

实现动作：
1. 定义 report_kind、report_id、created_at、case_count。
2. 定义 dataset_version、model_version、prompt_version、judge_version、kb_version。
3. 定义 scoring_policy_version 和 seed。
4. 新报告生成时强制写入 manifest。
5. 旧报告只允许兼容读取，不自动升级为正式 regression。
6. 对缺失版本、病例数不一致和非法 kind 返回明确错误。

测试重点：
- 旧 3 例报告兼容读取为 smoke。
- 18 例但没有版本字段的报告不能成为正式 regression。
- manifest.case_count 必须等于 cases 数量。
- report_kind 非法时 Pydantic 校验失败。

命令：
- venv\Scripts\python.exe -m pytest tests\evaluation\test_report_schema.py tests\evaluation\test_report.py -q

验收指标：
- 新报告 100% 带 manifest。
- 旧报告读取不报错。
- 版本字段缺失不会静默通过。

提交：
- Commit message: feat(eval): add versioned report manifest

---

## Task 2：显式回归门禁和统计门禁

目标：彻底隔离 smoke、benchmark 和 regression，避免“最新 JSON 文件”误触发 pre-push。

前置依赖：Task 1。

输入：
- ReportManifest
- 当前 patient_regression.py
- 当前 pre-push hook

文件：
- Create: backend/evaluation/gate.py
- Modify: backend/evaluation/patient_regression.py
- Modify: backend/scripts/eval_regression.py
- Modify: backend/scripts/hooks/pre-push
- Modify: backend/scripts/install_git_hooks.py
- Test: backend/tests/evaluation/test_gate.py
- Test: backend/tests/evaluation/test_install_git_hooks.py

核心接口：
- GateDecision = pass | fail | skip | invalid
- evaluate_report_gate(report: dict, thresholds: dict) -> tuple[GateDecision, list[dict]]
- calculate_bootstrap_ci(values: list[float], seed: int) -> tuple[float, float]
- select_gate_report(report_dir: Path) -> Path | None

实现动作：
1. smoke 永远返回 SKIP。
2. benchmark 只生成报告，不直接阻断 push。
3. regression 必须检查 manifest、病例数和版本完整性。
4. legacy_unknown 返回 INVALID。
5. 指标输出 n、mean、std、CI95、slice 和 critical_case_pass_rate。
6. pre-push 只消费显式退出码：0 PASS、1 FAIL、2 SKIP、3 INVALID。
7. 不再只依赖字典序最新文件；优先读取 manifest.report_kind=regression 的报告。
8. 没有正式 regression 报告时返回 SKIP 或明确提示，不能误判。

测试重点：
- 3、17 例 smoke 不阻断。
- 18 例 regression FAIL 阻断。
- 18 例无 manifest 报告 INVALID。
- benchmark 失败不阻断 pre-push，但报告状态为 FAIL。
- 阈值计算结果固定 seed 后可重复。

命令：
- venv\Scripts\python.exe -m pytest tests\evaluation\test_gate.py tests\evaluation\test_install_git_hooks.py tests\evaluation\test_patient_regression.py -q

验收指标：
- smoke false positive 为 0。
- 报告选择不依赖文件名时间排序。
- pre-push 对退出码 3 能阻断。

提交：
- Commit message: fix(eval): make regression gate report-aware

---

## Task 3：五维原子 Rubric

目标：把问诊分析、医学知识、人文关怀、诊断评估、治疗方案从“单分数”升级为可核查的原子行为项。

前置依赖：Task 1。

输入：
- backend/app/orchestration/state.py
- backend/app/orchestration/adapters
- backend/app/services/scoring
- backend/app/prompts

文件：
- Create: backend/evaluation/rubric.py
- Create: backend/evaluation/rubrics/v1.json
- Modify: backend/app/orchestration/state.py
- Modify: backend/app/schemas/evaluation.py
- Modify: backend/app/models/evaluation_node_result.py
- Test: backend/tests/evaluation/test_rubric.py

核心接口：
- RubricItem
- RubricSet
- aggregate_rubric(items: list[RubricItem]) -> DimensionResult
- validate_rubric_items(items: list[RubricItem]) -> list[str]

实现动作：
1. 为五个维度各设计 5～10 个 rubric item。
2. 每个 item 有 item_id、描述、verdict、score、evidence_spans、citation_ids。
3. 支持 pass、partial、fail、not_applicable、unassessed。
4. unassessed 不得被聚合为 0。
5. high severity item 自动设置 review_required。
6. 将 rubric_items 放入 AgentResultEnvelope 和节点持久化 trace。
7. 保留旧 score 和 analysis 字段，保证历史 API 可读。

五维建议：
- inquiry：主诉、现病史、既往史、用药史、过敏史、关键阴性症状。
- knowledge：医学事实、指南一致性、禁忌识别、证据充分性、引用有效性。
- humanistic：共情、解释清晰、尊重、风险沟通、共同决策。
- diagnosis：诊断合理性、鉴别诊断、证据匹配、不确定性表达。
- treatment：治疗适应证、禁忌证、剂量/疗程完整性、随访、转诊和急救建议。

测试重点：
- unassessed 不变成 0。
- high severity item 自动触发 review_required。
- partial 分数计算可重复。
- 缺少必需 item 时维度为 insufficient。

命令：
- venv\Scripts\python.exe -m pytest tests\evaluation\test_rubric.py tests\orchestration\test_models.py tests\orchestration\test_adapters.py -q

验收指标：
- 五维均有 v1 rubric 文件。
- 每个正式维度报告至少能返回 item-level 结果。
- 旧报告仍能被读取。

提交：
- Commit message: feat(eval): add atomic clinical rubrics

---

## Task 4：Judge 稳定性和人工校准

目标：降低 LLM-as-Judge 的随机性、位置偏差和模型自偏好。

前置依赖：Task 3。

输入：
- backend/evaluation/patient_judge.py
- Task 3 RubricItem
- 50 例人工 calibration 数据

文件：
- Create: backend/evaluation/judge_reliability.py
- Create: backend/evaluation/judge_calibration.jsonl
- Modify: backend/evaluation/patient_judge.py
- Modify: backend/evaluation/report_schema.py
- Test: backend/tests/evaluation/test_judge_reliability.py
- Test: backend/tests/evaluation/test_patient_judge.py

核心接口：
- JudgeRun
- JudgeReliability
- run_repeated_judge(case: dict, judge_model: str, positions: list[str], repeats: int) -> list[JudgeRun]
- evaluate_judge_reliability(runs, human_labels) -> JudgeReliability

实现动作：
1. 对 benchmark 执行原顺序和交换顺序两次评分。
2. 记录 judge_version、model_family、seed、position。
3. 计算 repeat_agreement、position_consistency、score_std。
4. 建立 50 例人工 calibration 集，其中 20 例双专家复核。
5. Judge 与被评估模型尽量使用不同模型族。
6. Judge 置信度低或位置一致性不达标时标记 needs_review。
7. 统计结果写入 ReportManifest。

测试重点：
- 交换顺序导致分数变化过大时触发 review。
- 两次评分一致时 reliability 提升。
- judge 降级调用会被记录。
- 人工标签缺失时不伪造 human_agreement。

命令：
- venv\Scripts\python.exe -m pytest tests\evaluation\test_judge_reliability.py tests\evaluation\test_patient_judge.py -q

验收指标：
- 重复一致率目标 >= 0.85。
- 位置一致率目标 >= 0.90。
- 关键安全项与人工标签召回率目标 >= 0.99。

提交：
- Commit message: feat(eval): add judge reliability calibration

---

## Task 5：风险分类和安全红旗回归集

目标：将安全检查从单一 risk_level 升级为可审计的风险发现和策略动作。

前置依赖：Task 1。

输入：
- backend/app/services/agents/safety_agent.py
- backend/app/orchestration/graph.py
- 既有安全测试

文件：
- Create: backend/evaluation/safety_cases.py
- Create: backend/evaluation/safety_cases/safety_red_flags.jsonl
- Modify: backend/app/services/agents/safety_agent.py
- Modify: backend/app/orchestration/state.py
- Modify: backend/app/orchestration/graph.py
- Test: backend/tests/evaluation/test_safety_cases.py
- Test: backend/tests/orchestration/test_safety.py

核心接口：
- RiskFinding
- SafetyDecision
- evaluate_safety_case(case) -> SafetyDecision
- calculate_safety_metrics(results) -> dict

实现动作：
1. 建立 chest pain、stroke、suicidal ideation、drug allergy 等高危样本。
2. RiskFinding 记录 risk_type、severity、evidence_span、source、policy_action。
3. 规则命中 high/critical 时 LLM 不得降级。
4. LLM 失败且无规则命中时返回 undetermined 并立即复核。
5. 图状态保存 safety policy version。
6. 将安全集分为 emergency、medication、population、privacy、evidence_conflict。
7. 建立离线安全回归命令。

测试重点：
- 高危红旗必须进入 needs_review。
- 规则和 LLM 结果冲突时规则优先。
- LLM 超时时 fail closed。
- 普通低风险病例不会全部误报。

命令：
- venv\Scripts\python.exe -m pytest tests\evaluation\test_safety_cases.py tests\orchestration\test_safety.py -q

验收指标：
- 关键红旗召回率目标 >= 0.99。
- 所有高危 finding 有 evidence_span。
- 安全回归报告能按 risk_type 分层。

提交：
- Commit message: feat(safety): add risk findings and red flag regression

---

## Task 6：人工复核状态机和审计

目标：让人工复核成为可追踪的状态机，而不是覆盖原结果的表单提交。

前置依赖：Task 5、Task 3。

输入：
- backend/app/services/review_service.py
- backend/app/api/v1/review.py
- backend/app/models/review_record.py
- backend/app/models/evaluation.py

文件：
- Modify: backend/app/services/review_service.py
- Modify: backend/app/api/v1/review.py
- Modify: backend/app/models/review_record.py
- Modify: backend/app/models/evaluation.py
- Create: backend/alembic/versions/20260801_review_audit_fields.py
- Test: backend/tests/test_review_admin_endpoints.py
- Test: backend/tests/services/test_review_service.py

核心接口：
- ReviewStatus = pending_review | in_review | approved | rejected | returned
- ReviewDecision
- apply_review_decision(state, decision) -> state
- create_review_snapshot(evaluation) -> dict

实现动作：
1. 保存 original_scores 和 adjusted_scores。
2. 复核调整对象改为 rubric item，不允许只改 total_score。
3. 高风险 approve 必须填写 reason_code 和 feedback。
4. 同一 evaluation 使用乐观锁避免并发复核。
5. 重复提交必须幂等或返回明确冲突。
6. 保存 reviewer_id、时间、source_report_id 和 audit hash。
7. 数据库迁移后补充回滚说明和历史数据默认值。

测试重点：
- 状态迁移非法时拒绝。
- 高风险无理由批准时拒绝。
- 复核记录失败时不把评估状态标记为完成。
- 普通用户不能提交或查看管理员复核详情。

命令：
- venv\Scripts\python.exe -m pytest tests\test_review_admin_endpoints.py tests\services\test_review_service.py -q

验收指标：
- 原始评估不可变。
- 每次调整可还原。
- 高风险复核有完整审计字段。

提交：
- Commit message: feat(review): add auditable review state machine

---

## Task 7：稳定 Citation ID 和来源注册表

目标：让知识证据在 KB 重建、重排和跨版本比较后仍可稳定定位。

前置依赖：Task 1。

输入：
- backend/app/services/tools/medical_retrieval.py
- backend/app/services/tools/citation.py
- backend/app/services/rag/types.py
- backend/app/services/rag/indexing/versioning.py

文件：
- Create: backend/app/services/rag/source_registry.py
- Create: backend/data/source_registry.json
- Modify: backend/app/services/tools/medical_retrieval.py
- Modify: backend/app/services/tools/citation.py
- Modify: backend/app/services/rag/types.py
- Test: backend/tests/rag/test_source_registry.py
- Test: backend/tests/rag/test_citation.py

核心接口：
- SourceMetadata
- stable_citation_id(kb_version, document_id, chunk_id, content) -> str
- resolve_source_metadata(source_id) -> SourceMetadata

实现动作：
1. citation ID 使用 kb_version、document_id、chunk_id、content hash。
2. 来源注册表记录类型、权威等级、发布/生效/失效日期。
3. 指南、教材、论文、数据集来源分级。
4. 旧 citation 保存 legacy_citation_id。
5. 检索结果、rerank 结果和数据库 trace 使用同一 citation ID。
6. source registry 缺失时不把来源标记为高权威。

测试重点：
- 检索顺序改变不影响 citation ID。
- 同一 chunk 在相同 KB 版本下 ID 稳定。
- 不同 KB 版本 ID 可区分。
- 旧 citation 可读取但不能生成新证据链。

命令：
- venv\Scripts\python.exe -m pytest tests\rag\test_source_registry.py tests\rag\test_citation.py tests\rag\test_types.py -q

验收指标：
- citation ID 不依赖列表 index。
- 每条正式证据都能查到 source metadata。
- 前端可显示来源版本和页码。

提交：
- Commit message: feat(rag): add stable citation identity and source registry

---

## Task 8：Claim-Evidence Graph

目标：从“引用 ID 合法”升级到“临床主张被哪条证据支持”。

前置依赖：Task 7、Task 3。

输入：
- knowledge Agent 输出
- citation tool
- RAG trace
- source registry

文件：
- Create: backend/evaluation/rag_claims.py
- Modify: backend/app/services/agents/knowledge/scoring.py
- Modify: backend/app/services/tools/citation.py
- Modify: backend/app/orchestration/state.py
- Modify: backend/app/schemas/evaluation.py
- Test: backend/tests/rag/test_claim_evidence.py
- Test: backend/tests/evaluation/test_rag_claims.py

核心接口：
- ClinicalClaim
- EvidenceLink
- extract_claims(text) -> list[ClinicalClaim]
- validate_claim_evidence(claims) -> dict
- calculate_claim_metrics(claims) -> dict

实现动作：
1. 将知识 Agent 的诊断、治疗、风险和教育结论拆为 claims。
2. 每个 claim 绑定 supports、contradicts 或 insufficient 的 evidence link。
3. 记录 entailment_score、evidence_span 和 verified。
4. 生成 supported、partially_supported、unsupported、conflicting 状态。
5. unsupported 的 treatment/risk claim 自动 needs_review。
6. 扩展 RAG 指标：precision、recall、validity、coverage、entailment、contradiction、authority。
7. 将 claims 写入 evaluation node trace 和 API response。

测试重点：
- 无证据治疗 claim 必须复核。
- 证据冲突不能标记为 supported。
- 有效 citation 但 entailment 不足时仍需 review。
- source authority 低时不能伪造高可信。

命令：
- venv\Scripts\python.exe -m pytest tests\rag tests\evaluation\test_rag_claims.py -q

验收指标：
- 每个治疗主张可定位证据片段。
- claim coverage 和 contradiction rate 可统计。
- RAG 版本变更能比较证据链差异。

提交：
- Commit message: feat(rag): add claim evidence validation

---

## Task 9：通用 PlanStep DAG

目标：把当前手工 Wave 1/Wave 2 依赖收敛为通用 PlanStep DAG。

前置依赖：Task 2、Task 3、Task 5。

输入：
- backend/app/orchestration/routes.py
- backend/app/orchestration/graph.py
- backend/app/orchestration/state.py

文件：
- Modify: backend/app/orchestration/routes.py
- Modify: backend/app/orchestration/graph.py
- Modify: backend/app/orchestration/state.py
- Test: backend/tests/orchestration/test_plan_dependencies.py
- Test: backend/tests/orchestration/test_graph.py

核心接口：
- ready_steps(plan, completed) -> list[PlanStep]
- validate_plan_dag(plan) -> list[str]
- mark_step_status(state, step_id, status) -> state

实现动作：
1. 验证重复 step_id、未知依赖和循环依赖。
2. 根据 completed 集合计算 ready steps。
3. 用依赖关系决定 fan-out，而不是只使用 Agent 名称集合。
4. step 状态使用 pending、running、succeeded、failed、skipped、blocked。
5. 依赖失败传播为 blocked 或 needs_review。
6. 保留旧 route_plan 作为输入兼容层。
7. 确保 knowledge 完成后才允许 diagnosis/treatment 消费 citations。

测试重点：
- diagnosis 依赖 knowledge 时不能提前执行。
- 循环依赖被拒绝。
- 某一 Agent 失败不会让不相关 Agent 重跑。
- reducer 汇聚结果顺序稳定。

命令：
- venv\Scripts\python.exe -m pytest tests\orchestration\test_plan_dependencies.py tests\orchestration\test_graph.py -q

验收指标：
- 新增 Agent 只需新增 PlanStep，不需修改核心路由。
- 每个 step 有明确最终状态。
- 失败原因能追溯到具体依赖。

提交：
- Commit message: refactor(orchestration): execute evaluation plan as DAG

---

## Task 10：并发、Token、成本和缓存预算

目标：防止多 Agent fan-out 在高并发下造成模型限流、成本失控和 Redis 缓存污染。

前置依赖：Task 1、Task 9。

输入：
- backend/app/orchestration/graph.py
- backend/app/services/context_budget.py
- backend/app/services/token_tracker.py
- backend/app/services/llm_cache.py
- backend/app/services/tools/tool_budget_manager.py

文件：
- Create: backend/app/services/run_budget.py
- Modify: backend/app/services/context_budget.py
- Modify: backend/app/services/token_tracker.py
- Modify: backend/app/services/llm_cache.py
- Modify: backend/app/services/tools/tool_budget_manager.py
- Modify: backend/app/orchestration/graph.py
- Test: backend/tests/services/test_run_budget.py
- Test: backend/tests/services/test_llm_cache.py

核心接口：
- RunBudget
- BudgetDecision
- acquire_agent_slot(run_id, agent_name) -> BudgetDecision
- record_usage(run_id, usage) -> None
- release_agent_slot(run_id, agent_name) -> None

实现动作：
1. 定义 max_total_tokens、max_total_cost、max_parallel_agents、max_duration_seconds。
2. 增加 per-run 和 per-model semaphore。
3. 超预算时停止新增非关键 Agent，保留安全和复核路径。
4. cache key 绑定 model_version、prompt_version、kb_version、temperature、tenant。
5. 每个调用记录 cache hit/miss、tokens、cost、latency。
6. 异常、取消和超时必须释放 slot。
7. 把预算超限转成可解释的 degraded/review 状态。

测试重点：
- 并发超过上限时拒绝。
- 任务取消后 slot 可再次使用。
- 不同 Prompt/KB 版本不命中旧 cache。
- 预算超限不会丢失安全结果。

命令：
- venv\Scripts\python.exe -m pytest tests\services\test_run_budget.py tests\services\test_llm_cache.py tests\services\test_token_run_usage.py -q

验收指标：
- 单 run 最大并发可配置。
- 每次评估能算 token 和估算成本。
- cache key 不发生跨版本污染。

提交：
- Commit message: feat(runtime): add run budgets and cost controls

---

## Task 11：全链路 Trace 和观测

目标：从一次评估定位到具体 graph node、Agent、LLM、Tool、RAG、重试和数据库写入。

前置依赖：Task 10。

输入：
- backend/app/core/run_context.py
- backend/app/core/logging.py
- backend/app/tasks/evaluation_task.py
- backend/app/services/observability
- backend/app/services/evaluation_service.py

文件：
- Modify: backend/app/core/run_context.py
- Modify: backend/app/core/logging.py
- Modify: backend/app/tasks/evaluation_task.py
- Modify: backend/app/services/observability/langfuse_client.py
- Modify: backend/app/services/observability/metrics.py
- Modify: backend/app/services/evaluation_service.py
- Test: backend/tests/services/test_trace_propagation.py

核心接口：
- TraceContext
- bind_trace_context(context) -> Token
- serialize_trace_context(context) -> dict
- restore_trace_context(payload) -> TraceContext

实现动作：
1. 统一 trace_id、run_id、consultation_id、celery_task_id、graph_thread_id。
2. Celery 重试复用 run_id，但使用新的 attempt。
3. 所有日志加入 node_name、agent_name、tool_name、status、duration_ms。
4. Langfuse 发送脱敏摘要，不发送未脱敏患者原文。
5. Prometheus 增加评估时延、错误、复核、token、成本、缓存和工具降级指标。
6. evaluation_node_result 保存 trace span 关联 ID。

测试重点：
- Celery payload 传播 trace context。
- 重试后 trace_id 不变、attempt 增加。
- 脱敏函数覆盖 prompt、response、retrieval text。
- trace 缺失时系统生成新 ID，不抛出无关异常。

命令：
- venv\Scripts\python.exe -m pytest tests\services\test_trace_propagation.py tests\tasks\test_evaluation_task_retry.py tests\test_monitoring_endpoints.py -q

验收指标：
- 一次评估可以串起 FastAPI 到数据库。
- Langfuse 不出现未脱敏患者字段。
- Grafana 能区分 LLM 慢、RAG 慢、数据库慢和人工复核堆积。

提交：
- Commit message: feat(observability): propagate evaluation trace context

---

## Task 12：评估报告前端升级

目标：让前端展示“分数、证据、风险和不确定性”，避免把待复核误解成低分。

前置依赖：Task 3、Task 6、Task 8、Task 11。

输入：
- 新 API evaluation schema
- rubric_items
- claims
- risk findings
- trace summary

文件：
- Modify: frontend/src/types/index.ts
- Modify: frontend/src/pages/Evaluation/index.tsx
- Modify: frontend/src/components/DimensionRadar.tsx
- Create: frontend/src/components/RubricItemList.tsx
- Create: frontend/src/components/EvidenceTrace.tsx
- Create: frontend/src/components/RiskBanner.tsx
- Test: frontend/src/pages/Evaluation/index.test.tsx
- Test: frontend/src/components/RubricItemList.test.tsx

实现动作：
1. 前端类型同步 RubricItem、ClinicalClaim、RiskFinding、ReportManifest。
2. 展示每个维度的 rubric item 和 verdict。
3. 点击 item 展示对话证据片段和 citation。
4. 显示 evidence status、risk level 和 review reason。
5. 修复所有 score ?? 0 的误导性渲染。
6. 展示模型、Prompt、Judge、KB、评分策略和生成时间。
7. 将 RAG citation、claim 支持关系和来源版本做成可展开面板。

测试重点：
- unassessed 显示“未评估”，不显示 0。
- needs_review 显示状态，不渲染总分为 0。
- unsupported claim 显示风险。
- 无引用时显示“无可验证证据”，不显示空白。
- manifest 版本信息完整展示。

命令：
- npm test -- --run
- npm run build

验收指标：
- 用户能回答“为什么这个维度得分”。
- 用户能定位“哪句话没有证据”。
- 用户能区分 0 分、未评估、不适用和待复核。

提交：
- Commit message: feat(frontend): add evidence based evaluation report

---

## Task 13：人工复核工作台

目标：将 pending_review 变成有优先级、有证据、有操作审计的管理员工作流。

前置依赖：Task 6、Task 12。

输入：
- review API
- pending evaluations
- risk findings
- rubric adjustments
- trace summary

文件：
- Create: frontend/src/pages/AdminReviews/index.tsx
- Modify: frontend/src/api/evaluation.ts
- Modify: frontend/src/App.tsx
- Modify: frontend/src/types/index.ts
- Create: frontend/src/pages/AdminReviews/index.test.tsx

实现动作：
1. 按 risk_level、priority、created_at 排序。
2. 提供原因、科室、模型版本、报告版本筛选。
3. 展示原始回答、风险红旗、证据链、Agent 分歧和 Judge 置信度。
4. 支持 rubric item 级调整。
5. 强制填写 review reason code 和 feedback。
6. 展示调整前后差异。
7. 普通用户访问返回无权限。

测试重点：
- 高风险优先。
- 重复提交显示冲突。
- 无权限用户不能进入工作台。
- 复核提交后列表状态更新。
- 调整总分必须由 rubric item 聚合产生。

命令：
- npm test -- --run
- npm run build
- venv\Scripts\python.exe -m pytest tests\test_review_admin_endpoints.py -q

验收指标：
- 复核人员无需查数据库即可完成一次判断。
- 每次复核可还原。
- 高风险队列有可观测 SLA。

提交：
- Commit message: feat(frontend): add human review workbench

---

## Task 14：可版本化临床能力基准集

目标：形成可用于 dev、test、regression、safety 和 benchmark 的病例治理体系。

前置依赖：Task 7、Task 8。

输入：
- backend/evaluation/patient_cases
- backend/evaluation/rag_cases
- source registry
- rubric v1

文件：
- Modify: backend/evaluation/patient_eval_set.py
- Modify: backend/evaluation/rag_cases/README.md
- Create: backend/evaluation/benchmark_manifest.json
- Create: backend/scripts/validate_benchmark.py
- Create: backend/tests/evaluation/test_validate_benchmark.py
- Create: docs/benchmark-governance.md

核心接口：
- BenchmarkCase
- validate_benchmark_manifest(path) -> list[str]
- split_cases(cases, split, seed) -> list[BenchmarkCase]

实现动作：
1. 每个病例加入 specialty、difficulty、required_questions、red_flags。
2. 加入 expected_diagnoses、treatment_constraints、gold_citations。
3. 将 dev、test、regression、safety 隔离。
4. 禁止正式 test 集参与阈值调参。
5. 验证 citation 是否存在 source registry。
6. 验证 safety case 必须有 red_flags。
7. 验证 case_id 唯一、split 合法、rubric_version 存在。
8. 记录数据集版本和变更日志。

测试重点：
- 重复 case_id 被拒绝。
- 非法 split 被拒绝。
- treatment case 缺约束被拒绝。
- safety case 缺红旗被拒绝。
- gold citation 不存在时报告错误。

命令：
- venv\Scripts\python.exe scripts\validate_benchmark.py --manifest evaluation\benchmark_manifest.json
- venv\Scripts\python.exe -m pytest tests\evaluation\test_validate_benchmark.py tests\evaluation\test_patient_eval_set.py -q

验收指标：
- 每次 regression 都能重放同一病例集。
- test 集不参与阈值校准。
- 数据集变更有版本和审计记录。

提交：
- Commit message: feat(eval): add versioned clinical benchmark governance

---

## Task 15：数据治理和部署安全

目标：对患者对话、评估 trace、导出和外部模型调用建立数据分级和生命周期控制。

前置依赖：Task 6、Task 11、Task 14。

输入：
- backend/app/core/masking.py
- backend/app/core/audit.py
- backend/app/api/v1/data_export.py
- docker-compose.yml
- docker-compose.prod.yml

文件：
- Modify: backend/app/core/masking.py
- Modify: backend/app/core/audit.py
- Modify: backend/app/api/v1/data_export.py
- Modify: backend/app/core/security.py
- Modify: docker-compose.yml
- Modify: docker-compose.prod.yml
- Create: docs/data-governance.md
- Create: backend/tests/test_data_retention.py

核心接口：
- redact_trace(payload, policy) -> dict
- purge_expired_trace(db, now) -> int
- validate_export_scope(user, scope) -> None

实现动作：
1. 定义 P0 原始对话、P1 脱敏对话、P2 rubric/指标、P3 聚合监控。
2. 每种级别定义保留期限和访问角色。
3. Langfuse、日志和 Prometheus 只接收脱敏数据。
4. 导出必须记录申请人、范围、理由、审批人、字段、时间、hash。
5. Redis cache、checkpoint 和 trace 使用不同 namespace/TTL。
6. 生产 compose 不暴露 MySQL/Redis 公网端口。
7. secret 不写入版本库和 compose 明文。
8. 加入定期清理和备份恢复演练说明。

测试重点：
- 姓名、电话、身份证、地址、病史被脱敏。
- 过期 trace 能被删除。
- 普通用户不能导出 P0 数据。
- 审计日志不包含密码或 token。
- 生产 compose 不存在不必要公网端口。

命令：
- venv\Scripts\python.exe -m pytest tests\test_data_governance.py tests\test_security_hardening.py tests\test_access_control.py tests\test_data_retention.py -q

验收指标：
- P0 数据没有进入外部观测系统。
- 每次导出都有审计记录。
- TTL 和权限策略有自动化测试。

提交：
- Commit message: feat(security): add clinical data lifecycle controls

---

## Task 16：端到端发布验收

目标：验证前 16 个 Task 组合后，平台仍能完成真实的评估、复核、回归和报告展示闭环。

前置依赖：Task 1、Task 5、Task 8、Task 11、Task 12、Task 15。

输入：
- smoke fixture
- regression fixture
- safety fixture
- mock LLM
- mock Redis/MySQL 或测试容器

文件：
- Create: backend/tests/e2e/test_iteration_release.py
- Modify: .github/workflows
- Modify: CONTRIBUTING.md
- Create: docs/evaluation-runbook.md

端到端场景：
1. 3 例 smoke 能运行且 pre-push 允许。
2. 18 例 regression manifest 完整且阈值失败可阻断。
3. 高风险病例进入 pending_review 且没有正式总分。
4. 复核保存原始和调整后 rubric。
5. knowledge Agent 输出 claim、citation、evidence span。
6. diagnosis/treatment 只能消费允许的上游证据。
7. Celery 重试保持 trace_id/run_id 关联。
8. 前端不把未评估渲染成 0。
9. 过期 trace 可清理，导出进入 audit log。

执行命令：
- venv\Scripts\python.exe -m pytest -q
- npm test -- --run
- npm run build
- venv\Scripts\python.exe scripts\eval_regression.py --report evaluation\reports\patient_ab\ab_regression.json
- venv\Scripts\python.exe scripts\validate_benchmark.py --manifest evaluation\benchmark_manifest.json

验收指标：
- 后端全量测试通过。
- 前端测试和构建通过。
- smoke、regression、benchmark 退出码符合协议。
- 安全集关键红旗达标。
- 关键 run 可以从 trace_id 定位到证据和复核记录。
- 无未脱敏患者数据进入日志和 Langfuse。

提交：
- Commit message: test(release): add clinical evaluation release gates

---

## 推荐开发批次

Batch A：Task 0、Task 1、Task 2
- 目标：先解决 pre-push 和报告协议。
- 结束条件：3 例 smoke 永不阻断，正式 regression 可明确阻断。

Batch B：Task 3、Task 4
- 目标：解决评分可信度。
- 结束条件：五维有原子 rubric，Judge 有重复性和人工校准指标。

Batch C：Task 5、Task 6
- 目标：解决医疗安全和复核闭环。
- 结束条件：高危病例 fail closed，人工调整可追踪。

Batch D：Task 7、Task 8
- 目标：解决循证链路。
- 结束条件：治疗主张可追踪到稳定 citation 和 evidence span。

Batch E：Task 9、Task 10、Task 11
- 目标：解决 DAG、成本、并发和观测。
- 结束条件：一次 run 可以定位到具体 Agent、Tool、RAG、成本和重试。

Batch F：Task 12、Task 13
- 目标：解决报告可解释性和人工操作效率。
- 结束条件：前端可解释分数、证据和风险，复核工作台可用。

Batch G：Task 14、Task 15、Task 16
- 目标：完成基准集、治理和发布验收。
- 结束条件：能重放、能审计、能清理、能发布。

## 最优先的首批 5 个 Task

1. Task 1：ReportManifest。
2. Task 2：显式回归门禁。
3. Task 3：五维原子 Rubric。
4. Task 5：安全红旗回归集。
5. Task 7：稳定 Citation ID。

原因：
- Task 1 和 Task 2 直接消除 3 例 smoke 误触发风险。
- Task 3 决定最终评分是否可解释。
- Task 5 决定医疗安全是否可验证。
- Task 7 决定 RAG 证据是否可以长期追踪。

## 每个 Task 的 review 问题

- 是否只改了该 Task 的文件范围？
- 是否先写了失败测试？
- 是否存在真实 LLM 调用？
- 是否把 null、unassessed、insufficient 转成 0？
- 是否破坏历史报告或 API 兼容？
- 是否补充了日志、trace、审计和版本字段？
- 是否说明了失败时的状态和恢复路径？
- 是否有明确测试命令和可量化验收指标？
