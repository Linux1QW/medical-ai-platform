# 临床问诊评估平台设计迭代 Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

Goal: 将当前多智能体医生临床问诊评估平台升级为可复现、可解释、可审计、可持续回归的临床能力评估系统。

Architecture: 以现有 LangGraph Plan-Execute、五维 Agent、RAG、Celery、Redis checkpoint 和人工复核为基础，先建立统一评测报告协议，再引入原子 rubric、Judge 稳定性校准、风险分级、安全红旗回归集和 Claim-Evidence 证据链。编排层保留显式状态与确定性安全门，Legacy 路径逐步收敛为兼容适配层。

Tech Stack: Python 3.10+、FastAPI、Pydantic v2、SQLAlchemy async、MySQL、Redis、Celery、LangGraph、ChromaDB、pytest/pytest-asyncio、React 19、Ant Design、Prometheus、Grafana、Langfuse。

## Global Constraints

- 工作目录：D:\Q123\PyCharm\PycharmProjects\基于多智能体的医生临床问诊评估平台\medical-ai-platform。
- 后端命令均在 backend 下执行，Python 使用 venv\Scripts\python.exe。
- 所有测试必须 mock call_qwen_chat 及其他真实 LLM 调用，不产生真实 API 费用。
- needs_review、insufficient、error 不得静默转换成正常通过或 0 分。
- smoke、regression、benchmark 必须明确区分；smoke 不得阻断 pre-push。
- 报告必须绑定数据集、模型、Prompt、Judge、知识库和评分策略版本。
- 患者对话、病史和评估 trace 写入日志或外部观测系统前必须脱敏。
- 每个任务先写失败测试，再写最小实现，再运行专项回归测试。
- 本计划只包含本地实现、测试、文档和 commit，不包含 push、生产部署或真实患者数据导入。

## File Structure

backend/evaluation/report_schema.py       评测报告及 manifest
backend/evaluation/gate.py                统计门禁和报告类型校验
backend/evaluation/rubric.py              原子 rubric 和维度汇总
backend/evaluation/judge_reliability.py   Judge 稳定性与人工校准
backend/evaluation/safety_cases.py         安全红旗数据集与指标
backend/evaluation/rag_claims.py           Claim-Evidence Graph
backend/app/orchestration/graph.py         图编排和依赖执行
backend/app/services/observability/        trace、成本、指标
backend/app/services/review_service.py     人工复核状态机
frontend/src/pages/Evaluation/index.tsx    评估报告展示
frontend/src/pages/AdminReviews/index.tsx  复核工作台
docs/                                      协议、指标和运行手册

---

## 阶段 0：基线冻结

### Task 0: 建立当前系统基线

Files:
- Create: backend/tests/evaluation/test_iteration_baseline.py
- Create: docs/evaluation-baseline.md
- Inspect: backend/scripts/eval_regression.py
- Inspect: backend/evaluation/patient_ab_thresholds.json
- Inspect: backend/app/orchestration/graph.py
- Inspect: backend/app/services/evaluation_service.py

Produces: smoke/regression 门禁行为表、五维状态表、RAG 引用字段表、人工复核状态表。

- [ ] Step 1: 写失败测试

    def test_three_case_report_is_non_blocking(tmp_path):
        report = {"cases": [{}, {}, {}], "report_kind": "smoke"}
        path = tmp_path / "ab_smoke.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        assert run_gate(path).value == "skip"

- [ ] Step 2: 运行基线测试

Run: venv\Scripts\python.exe -m pytest tests\evaluation\test_iteration_baseline.py -q

记录当前测试失败原因；如果已有等价入口，先在测试中适配现有接口，不改变生产代码。

- [ ] Step 3: 固定当前行为

    venv\Scripts\python.exe scripts\eval_regression.py; Write-Output ('exit=' + $LASTEXITCODE)
    venv\Scripts\python.exe -m pytest tests\evaluation tests\orchestration tests\rag -q

将最新报告名、病例数、返回码、五维字段、测试数量和已知问题写入 docs/evaluation-baseline.md。

- [ ] Step 4: 验收

确认后续修改不能破坏“3 例 smoke 不阻断”的语义；状态字段变更必须同步 schema、数据库、API 和前端类型。

---

## 阶段 1：评测报告协议与门禁

### Task 1: 引入版本化 ReportManifest

Files:
- Create: backend/evaluation/report_schema.py
- Modify: backend/evaluation/report.py
- Modify: backend/evaluation/patient_regression.py
- Modify: backend/scripts/eval_regression.py
- Modify: backend/evaluation/patient_ab_thresholds.json
- Test: backend/tests/evaluation/test_report_schema.py

Interfaces:

    class ReportKind(str, Enum):
        SMOKE = "smoke"
        REGRESSION = "regression"
        BENCHMARK = "benchmark"

    class ReportManifest(BaseModel):
        report_kind: ReportKind
        report_id: str
        created_at: datetime
        case_count: int = Field(ge=0)
        dataset_version: str
        model_version: str
        prompt_version: str
        judge_version: str
        kb_version: str
        scoring_policy_version: str
        seed: int | None = None

- [ ] Step 1: 写兼容测试

    def test_legacy_small_report_is_smoke():
        manifest = load_report_manifest({"cases": [{}, {}]}, allow_legacy=True)
        assert manifest.report_kind is ReportKind.SMOKE

    def test_regression_requires_versions():
        with pytest.raises(ValidationError):
            load_report_manifest({"report_kind": "regression", "cases": [{}] * 18})

- [ ] Step 2: 实现 schema 和旧报告转换

无 manifest 且病例数小于 18 时仅兼容读取为 smoke；无 manifest 且病例数大于等于 18 时标记为 legacy_unknown，不得直接作为正式门禁。

- [ ] Step 3: 让报告生成器写入 manifest

generate_json_report() 接收版本参数并写入完整 manifest，禁止从文件名推断版本。

- [ ] Step 4: 测试并提交

Run: venv\Scripts\python.exe -m pytest tests\evaluation\test_report_schema.py tests\evaluation\test_report.py -q

git add backend/evaluation/report_schema.py backend/evaluation/report.py backend/evaluation/patient_regression.py backend/scripts/eval_regression.py backend/evaluation/patient_ab_thresholds.json backend/tests/evaluation/test_report_schema.py
git commit -m "feat(eval): add versioned report manifest"

### Task 2: 将回归门禁改为显式报告类型和分层统计

Files:
- Create: backend/evaluation/gate.py
- Modify: backend/evaluation/patient_regression.py
- Modify: backend/scripts/eval_regression.py
- Modify: backend/scripts/hooks/pre-push
- Modify: backend/scripts/install_git_hooks.py
- Test: backend/tests/evaluation/test_gate.py
- Test: backend/tests/evaluation/test_install_git_hooks.py

Interfaces:

    class GateDecision(str, Enum):
        PASS = "pass"
        FAIL = "fail"
        SKIP = "skip"
        INVALID = "invalid"

    def evaluate_report_gate(report: dict, thresholds: dict) -> tuple[GateDecision, list[dict]]:
        manifest = load_report_manifest(report, allow_legacy=True)
        if manifest.report_kind is ReportKind.SMOKE:
            return GateDecision.SKIP, []
        if manifest.report_kind is ReportKind.BENCHMARK:
            return GateDecision.PASS, []
        return check_regression_thresholds(report, thresholds)

- [ ] Step 1: 写边界测试

    @pytest.mark.parametrize("kind,n,expected", [
        ("smoke", 3, "skip"),
        ("smoke", 17, "skip"),
    ])
    def test_smoke_never_blocks(kind, n, expected):
        decision, _ = evaluate_report_gate(make_report(n, kind), thresholds())
        assert decision.value == expected

    def test_unknown_large_report_is_invalid():
        decision, _ = evaluate_report_gate(make_report(18, None), thresholds())
        assert decision is GateDecision.INVALID

- [ ] Step 2: 实现门禁状态机

规则：smoke -> SKIP；benchmark -> 只报告不阻断；regression -> 正式统计门禁；legacy_unknown -> INVALID；病例不足 -> SKIP。

- [ ] Step 3: 增加统计字段

每个指标输出 n、mean、std、ci95_low、ci95_high、slice、critical_case_pass_rate 和 status。bootstrap 必须固定 seed。

- [ ] Step 4: 收紧 hook 退出码

约定：0=PASS、1=FAIL、2=SKIP、3=INVALID/基础设施错误。pre-push 只允许 0 和 2，3 必须阻断并提示环境异常。

- [ ] Step 5: 测试并提交

Run: venv\Scripts\python.exe -m pytest tests\evaluation\test_gate.py tests\evaluation\test_install_git_hooks.py tests\evaluation\test_patient_regression.py -q

git add backend/evaluation/gate.py backend/evaluation/patient_regression.py backend/scripts/eval_regression.py backend/scripts/hooks/pre-push backend/scripts/install_git_hooks.py backend/tests/evaluation/test_gate.py backend/tests/evaluation/test_install_git_hooks.py
git commit -m "fix(eval): make regression gate report-aware"

---

## 阶段 2：五维原子 Rubric 与 Judge

### Task 3: 定义五维原子 Rubric

Files:
- Create: backend/evaluation/rubric.py
- Create: backend/evaluation/rubrics/v1.json
- Modify: backend/app/orchestration/state.py
- Modify: backend/app/schemas/evaluation.py
- Modify: backend/app/models/evaluation_node_result.py
- Test: backend/tests/evaluation/test_rubric.py

Interfaces:

    class RubricItem(BaseModel):
        item_id: str
        dimension: str
        description: str
        verdict: Literal["pass", "partial", "fail", "not_applicable", "unassessed"]
        score: float | None = Field(default=None, ge=0, le=100)
        evidence_spans: list[dict] = Field(default_factory=list)
        citation_ids: list[str] = Field(default_factory=list)
        severity: Literal["low", "medium", "high"] = "medium"
        review_required: bool = False

    def aggregate_rubric(items: list[RubricItem]) -> DimensionResult:
        scored = [item for item in items if item.score is not None and item.verdict in {"pass", "partial", "fail"}]
        if not scored or len(scored) != len(items):
            return DimensionResult(dimension=items[0].dimension, status="insufficient", score=None)
        return DimensionResult(dimension=items[0].dimension, status="scored", score=sum(item.score for item in scored) / len(scored))

- [ ] Step 1: 写未评估语义测试

    def test_unassessed_is_not_zero():
        result = aggregate_rubric([
            RubricItem(item_id="inq-1", dimension="inquiry", description="主诉", verdict="pass", score=90),
            RubricItem(item_id="inq-2", dimension="inquiry", description="过敏史", verdict="unassessed"),
        ])
        assert result.status == "insufficient"
        assert result.score is None

- [ ] Step 2: 创建 v1 rubric

五个维度各定义 5～10 个原子项；安全关键项使用 severity=high 和 review_required=true。

- [ ] Step 3: 接入 Agent Envelope 和数据库 trace

新增 rubric_items；旧 Agent 无法提供原子项时标记 unassessed，不得推断为通过。

- [ ] Step 4: 测试并提交

Run: venv\Scripts\python.exe -m pytest tests\evaluation\test_rubric.py tests\orchestration\test_models.py tests\orchestration\test_adapters.py -q

### Task 4: Judge 稳定性和人工校准

Files:
- Create: backend/evaluation/judge_reliability.py
- Create: backend/evaluation/judge_calibration.jsonl
- Modify: backend/evaluation/patient_judge.py
- Modify: backend/evaluation/report_schema.py
- Test: backend/tests/evaluation/test_judge_reliability.py
- Test: backend/tests/evaluation/test_patient_judge.py

Interfaces:

    class JudgeRun(BaseModel):
        judge_version: str
        model_family: str
        seed: int | None
        position: Literal["original", "swapped"]
        scores: dict[str, float]
        overall: float | None
        degraded: bool = False

    class JudgeReliability(BaseModel):
        repeat_agreement: float
        position_consistency: float
        human_agreement: float | None
        score_std: float
        needs_review: bool

- [ ] Step 1: 写位置交换测试

    def test_position_bias_requires_review():
        runs = [
            make_judge_run(position="original", overall=90),
            make_judge_run(position="swapped", overall=70),
        ]
        result = evaluate_judge_reliability(runs, None)
        assert result.position_consistency == 0.0
        assert result.needs_review is True

- [ ] Step 2: 实现重复评分、顺序交换和人工一致性计算

正式 benchmark 启用双运行；regression 至少对安全关键切片启用双运行；生成模型和 Judge 尽量使用不同模型族。

- [ ] Step 3: 建立 50 例 calibration 集

覆盖五维、急症、治疗建议、证据不足、沟通边界；其中 20 例由第二名专家复核。

- [ ] Step 4: 设定初始门槛并测试

初始目标：重复一致率 >=0.85、位置一致率 >=0.90、关键安全项人工召回率 >=0.99。

Run: venv\Scripts\python.exe -m pytest tests\evaluation\test_judge_reliability.py tests\evaluation\test_patient_judge.py -q

---

## 阶段 3：医疗安全和人工复核

### Task 5: 建立风险分类与安全红旗回归集

Files:
- Create: backend/evaluation/safety_cases.py
- Create: backend/evaluation/safety_cases/safety_red_flags.jsonl
- Modify: backend/app/services/agents/safety_agent.py
- Modify: backend/app/orchestration/state.py
- Modify: backend/app/orchestration/graph.py
- Test: backend/tests/evaluation/test_safety_cases.py
- Test: backend/tests/orchestration/test_safety.py

Interfaces:

    class RiskFinding(BaseModel):
        risk_id: str
        risk_type: str
        severity: Literal["low", "medium", "high", "critical"]
        evidence_span: dict
        source: Literal["rule", "llm", "human"]
        policy_action: Literal["allow", "needs_review", "block"]
        rule_version: str

- [ ] Step 1: 写高危红旗测试

    @pytest.mark.parametrize("risk_type", ["chest_pain", "stroke_sign", "suicidal_ideation", "drug_allergy"])
    def test_critical_red_flag_requires_review(risk_type):
        decision = evaluate_safety_case(load_case(risk_type))
        assert decision.immediate_review_required is True

- [ ] Step 2: 统一规则和 LLM Finding

LLM 不得降低确定性规则产生的 high/critical；LLM 失败时保留规则结果，无规则结果时返回 undetermined 并复核。

- [ ] Step 3: 建立安全切片和离线命令

覆盖急症、儿童、孕妇、过敏、药物相互作用、自伤、隐私越权和证据冲突。

命令：venv\Scripts\python.exe -m evaluation.safety_cases --cases evaluation\safety_cases\safety_red_flags.jsonl --fail-on-threshold。

- [ ] Step 4: 测试并提交

Run: venv\Scripts\python.exe -m pytest tests\evaluation\test_safety_cases.py tests\orchestration\test_safety.py -q

### Task 6: 完善人工复核状态机和审计

Files:
- Modify: backend/app/services/review_service.py
- Modify: backend/app/api/v1/review.py
- Modify: backend/app/models/review_record.py
- Modify: backend/app/models/evaluation.py
- Create: backend/alembic/versions/20260801_review_audit_fields.py
- Test: backend/tests/test_review_admin_endpoints.py
- Test: backend/tests/services/test_review_service.py

Interfaces:

    ReviewStatus = Literal["pending_review", "in_review", "approved", "rejected", "returned"]

    class ReviewDecision(BaseModel):
        reviewer_id: str
        status: ReviewStatus
        rubric_adjustments: dict[str, dict]
        reason_code: str
        feedback: str
        source_report_id: str

- [ ] Step 1: 写状态迁移测试

    def test_high_risk_approval_requires_reason():
        state = make_pending_review(risk_level="high")
        decision = make_decision(status="approved", reason_code="", feedback="")
        with pytest.raises(ValueError):
            apply_review_decision(state, decision)

- [ ] Step 2: 保存原始版本和调整版本

复核不得覆盖原始评估；记录原始 rubric、调整后 rubric、理由、reviewer、时间和 source report ID。

- [ ] Step 3: 增加乐观锁和重复提交保护

同一 evaluation 只允许一个活动复核；重复提交返回明确错误。

- [ ] Step 4: 测试

Run: venv\Scripts\python.exe -m pytest tests\test_review_admin_endpoints.py tests\services\test_review_service.py -q

---

## 阶段 4：RAG Claim-Evidence Graph

### Task 7: 稳定 citation ID 和来源注册表

Files:
- Create: backend/app/services/rag/source_registry.py
- Create: backend/data/source_registry.json
- Modify: backend/app/services/tools/medical_retrieval.py
- Modify: backend/app/services/tools/citation.py
- Modify: backend/app/services/rag/types.py
- Test: backend/tests/rag/test_source_registry.py
- Test: backend/tests/rag/test_citation.py

Interfaces:

    class SourceMetadata(BaseModel):
        source_id: str
        source_type: Literal["guideline", "textbook", "paper", "dataset", "other"]
        authority_level: int = Field(ge=0, le=5)
        publisher: str
        publication_date: date | None = None
        effective_date: date | None = None
        expiry_date: date | None = None

    def stable_citation_id(kb_version: str, document_id: str, chunk_id: str, content: str) -> str:
        payload = f"{kb_version}:{document_id}:{chunk_id}:{content}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

- [ ] Step 1: 写稳定性测试

    def test_citation_id_is_order_independent():
        value = stable_citation_id("rag-v1", "doc-1", "chunk-2", "文本")
        assert value == stable_citation_id("rag-v1", "doc-1", "chunk-2", "文本")

- [ ] Step 2: 实现来源 registry

检索结果带 source_id、来源等级、发布日期、生效日期和管辖范围；不再根据文件名临时推断来源类型。

- [ ] Step 3: 兼容旧 citation

旧 ID 保存到 legacy_citation_id，新结果使用稳定 ID。

- [ ] Step 4: 测试

Run: venv\Scripts\python.exe -m pytest tests\rag\test_source_registry.py tests\rag\test_citation.py tests\rag\test_types.py -q

### Task 8: 建立 Claim-Evidence 模型和验证器

Files:
- Create: backend/evaluation/rag_claims.py
- Modify: backend/app/services/agents/knowledge/scoring.py
- Modify: backend/app/services/tools/citation.py
- Modify: backend/app/orchestration/state.py
- Modify: backend/app/schemas/evaluation.py
- Test: backend/tests/rag/test_claim_evidence.py
- Test: backend/tests/evaluation/test_rag_claims.py

Interfaces:

    class EvidenceLink(BaseModel):
        claim_id: str
        citation_id: str
        relation: Literal["supports", "contradicts", "insufficient"]
        entailment_score: float | None = Field(default=None, ge=0, le=1)
        evidence_span: dict
        verified: bool = False

    class ClinicalClaim(BaseModel):
        claim_id: str
        text: str
        claim_type: Literal["finding", "diagnosis", "treatment", "risk", "education"]
        evidence_links: list[EvidenceLink]
        status: Literal["supported", "partially_supported", "unsupported", "conflicting"]

- [ ] Step 1: 写无证据治疗主张测试

    def test_unsupported_treatment_claim_requires_review():
        claim = ClinicalClaim(
            claim_id="c1", text="建议用药", claim_type="treatment",
            evidence_links=[], status="unsupported",
        )
        assert validate_claim_evidence([claim])["needs_review"] is True

- [ ] Step 2: 接入 knowledge Agent

保留现有 citations 兼容字段；正式审计字段为 claims，每个 claim 绑定 evidence span 和 citation。

- [ ] Step 3: 增加 RAG 指标

至少计算 context precision、context recall、citation validity、claim coverage、claim entailment、contradiction rate、source authority score。

- [ ] Step 4: 测试

Run: venv\Scripts\python.exe -m pytest tests\rag tests\evaluation\test_rag_claims.py -q

---

## 阶段 5：编排、性能和可观测性

### Task 9: 将 PlanStep 依赖变成通用 DAG

Files:
- Modify: backend/app/orchestration/routes.py
- Modify: backend/app/orchestration/graph.py
- Modify: backend/app/orchestration/state.py
- Test: backend/tests/orchestration/test_plan_dependencies.py
- Test: backend/tests/orchestration/test_graph.py

Interfaces:

    def ready_steps(plan: EvaluationPlan, completed: set[str]) -> list[PlanStep]:
        return [step for step in plan.steps if step.step_id not in completed and set(step.depends_on).issubset(completed)]

    def validate_plan_dag(plan: EvaluationPlan) -> list[str]:
        errors = validate_step_ids_and_cycles(plan.steps)
        return errors

- [ ] Step 1: 写依赖测试

    def test_diagnosis_waits_for_knowledge():
        plan = make_plan(depends={"diagnosis": ["knowledge"]})
        assert [s.agent_name for s in ready_steps(plan, set())] == ["knowledge"]

- [ ] Step 2: 实现 ready queue 和循环依赖校验

保留当前 Wave 1/Wave 2 作为默认计划，但由依赖函数计算，不再在 graph 中硬编码完整 Agent 列表。

- [ ] Step 3: 增加 step 状态

使用 pending/running/succeeded/failed/skipped/blocked，依赖失败明确传播为 blocked 或 needs_review。

- [ ] Step 4: 测试

Run: venv\Scripts\python.exe -m pytest tests\orchestration\test_plan_dependencies.py tests\orchestration\test_graph.py -q

### Task 10: 增加并发、token、成本和缓存预算

Files:
- Create: backend/app/services/run_budget.py
- Modify: backend/app/services/context_budget.py
- Modify: backend/app/services/token_tracker.py
- Modify: backend/app/services/llm_cache.py
- Modify: backend/app/services/tools/tool_budget_manager.py
- Modify: backend/app/orchestration/graph.py
- Test: backend/tests/services/test_run_budget.py
- Test: backend/tests/services/test_llm_cache.py

Interfaces:

    class RunBudget(BaseModel):
        max_total_tokens: int
        max_total_cost: Decimal
        max_parallel_agents: int
        max_duration_seconds: int

    async def acquire_agent_slot(run_id: str, agent_name: str) -> BudgetDecision:
        return await run_budget_manager.acquire(run_id=run_id, resource=agent_name)

- [ ] Step 1: 写并发限制测试

    async def test_parallel_limit_blocks_extra_agent():
        budget = RunBudget(
            max_total_tokens=1000, max_total_cost=Decimal("1"),
            max_parallel_agents=2, max_duration_seconds=60,
        )
        await reserve_two_slots(budget)
        assert (await acquire_agent_slot("run-1", "diagnosis")).allowed is False

- [ ] Step 2: 实现 per-run/per-model 限流

run_agent 前获取 slot；异常、取消和超时必须释放 slot。

- [ ] Step 3: 绑定 cache key 版本

LLM 和 retrieval cache key 包含 model_version、prompt_version、kb_version、temperature 和 tenant scope。

- [ ] Step 4: 测试

Run: venv\Scripts\python.exe -m pytest tests\services\test_run_budget.py tests\services\test_llm_cache.py tests\services\test_token_run_usage.py -q

### Task 11: 统一 FastAPI-Celery-Graph-LLM trace

Files:
- Modify: backend/app/core/run_context.py
- Modify: backend/app/core/logging.py
- Modify: backend/app/tasks/evaluation_task.py
- Modify: backend/app/services/observability/langfuse_client.py
- Modify: backend/app/services/observability/metrics.py
- Modify: backend/app/services/evaluation_service.py
- Test: backend/tests/services/test_trace_propagation.py

Interfaces:

    class TraceContext(BaseModel):
        trace_id: str
        run_id: str
        consultation_id: int | None = None
        celery_task_id: str | None = None
        graph_thread_id: str | None = None

- [ ] Step 1: 写跨任务传播测试

    def test_task_payload_preserves_trace_context():
        payload = build_task_payload(TraceContext(trace_id="t1", run_id="r1"))
        assert restore_trace_context(payload).trace_id == "t1"

- [ ] Step 2: 实现任务入口绑定

重试复用 run_id，但每次重试记录新的 attempt span；所有 Agent、Tool、RAG 和 Gate 日志包含 trace_id、run_id、node_name、status、duration_ms。

- [ ] Step 3: 脱敏发送观测系统

Langfuse 默认发送脱敏摘要，完整患者对话只保留在受控数据库 trace 中。

- [ ] Step 4: 测试

Run: venv\Scripts\python.exe -m pytest tests\services\test_trace_propagation.py tests\tasks\test_evaluation_task_retry.py tests\test_monitoring_endpoints.py -q

---

## 阶段 6：前端报告和复核工作台

### Task 12: 改造评估报告为“分数 + 证据 + 风险”

Files:
- Modify: frontend/src/types/index.ts
- Modify: frontend/src/pages/Evaluation/index.tsx
- Modify: frontend/src/components/DimensionRadar.tsx
- Create: frontend/src/components/RubricItemList.tsx
- Create: frontend/src/components/EvidenceTrace.tsx
- Create: frontend/src/components/RiskBanner.tsx
- Test: frontend/src/pages/Evaluation/index.test.tsx
- Test: frontend/src/components/RubricItemList.test.tsx

Interfaces:

    export interface RubricItem {
      item_id: string;
      dimension: string;
      description: string;
      verdict: 'pass' | 'partial' | 'fail' | 'not_applicable' | 'unassessed';
      score?: number | null;
      evidence_spans: EvidenceSpan[];
      citation_ids: string[];
      severity: 'low' | 'medium' | 'high';
      review_required: boolean;
    }

- [ ] Step 1: 写 null/未评估测试

    it('does not render unassessed as zero', () => {
      render(<RubricItemList items={[{ item_id: 'x', verdict: 'unassessed', score: null }]} />);
      expect(screen.getByText('未评估')).toBeInTheDocument();
      expect(screen.queryByText('0')).not.toBeInTheDocument();
    });

- [ ] Step 2: 实现 rubric、证据和风险组件

点击 rubric item 展示对话原文片段、引用来源、来源版本和人工确认状态。

- [ ] Step 3: 修正雷达图和分数卡片

null、not_applicable、unassessed、needs_review 不得通过默认值渲染为 0 分。

- [ ] Step 4: 测试并构建

Run: npm test -- --run; npm run build

### Task 13: 实现复核工作台

Files:
- Create: frontend/src/pages/AdminReviews/index.tsx
- Modify: frontend/src/api/evaluation.ts
- Modify: frontend/src/App.tsx
- Modify: frontend/src/types/index.ts
- Create: frontend/src/pages/AdminReviews/index.test.tsx

Interfaces:

    export interface PendingReview {
      evaluation_id: string;
      risk_level: string;
      review_reason: string;
      priority: number;
      created_at: string;
      trace_available: boolean;
    }

- [ ] Step 1: 写复核列表测试

验证高风险优先、重复提交提示、调整前后分数同时展示和普通用户无权限。

- [ ] Step 2: 实现列表、筛选、证据审阅和 rubric 调整

调整对象只能是 rubric item，不允许只修改总分。

- [ ] Step 3: 测试并构建

Run: npm test -- --run; npm run build

---

## 阶段 7：基准集、数据治理和发布

### Task 14: 建设可版本化临床能力基准集

Files:
- Modify: backend/evaluation/patient_eval_set.py
- Modify: backend/evaluation/rag_cases/README.md
- Create: backend/evaluation/benchmark_manifest.json
- Create: backend/scripts/validate_benchmark.py
- Create: backend/tests/evaluation/test_validate_benchmark.py
- Create: docs/benchmark-governance.md

Interfaces:

    class BenchmarkCase(BaseModel):
        case_id: str
        split: Literal["dev", "test", "regression", "safety"]
        specialty: str
        difficulty: Literal["easy", "medium", "hard"]
        facts: list[dict]
        required_questions: list[str]
        red_flags: list[str]
        expected_diagnoses: list[str]
        treatment_constraints: list[str]
        gold_citations: list[str]
        rubric_version: str

- [ ] Step 1: 写数据完整性测试

检查 case_id 唯一、split 合法、safety case 有 red_flags、治疗病例有 treatment_constraints、引用存在于 source registry。

- [ ] Step 2: 实现 validator 和数据切分

命令：venv\Scripts\python.exe scripts\validate_benchmark.py --manifest evaluation\benchmark_manifest.json。禁止用正式 test 集调阈值。

- [ ] Step 3: 测试

Run: venv\Scripts\python.exe -m pytest tests\evaluation\test_validate_benchmark.py tests\evaluation\test_patient_eval_set.py -q

### Task 15: 完善数据治理和部署安全

Files:
- Modify: backend/app/core/masking.py
- Modify: backend/app/core/audit.py
- Modify: backend/app/api/v1/data_export.py
- Modify: backend/app/core/security.py
- Modify: docker-compose.yml
- Modify: docker-compose.prod.yml
- Create: docs/data-governance.md
- Create: backend/tests/test_data_retention.py

Interfaces:

    def redact_trace(payload: dict, policy: RedactionPolicy) -> dict:
        return apply_redaction_policy(payload, policy)

    async def purge_expired_trace(db: AsyncSession, now: datetime) -> int:
        return await delete_expired_trace_rows(db, now)

- [ ] Step 1: 写脱敏、权限和 TTL 测试

覆盖姓名、电话、身份证、地址、原始对话、外部模型请求体、导出审批和过期 trace。

- [ ] Step 2: 定义数据分级

P0 原始对话；P1 脱敏对话；P2 rubric 和指标；P3 聚合监控。不同等级使用不同保留期限和权限。

- [ ] Step 3: 加强导出和生产 compose

导出记录申请人、范围、理由、审批人、字段、时间和文件 hash；生产环境禁止 MySQL/Redis 公网暴露，secret 不写入 compose 文件。

- [ ] Step 4: 测试

Run: venv\Scripts\python.exe -m pytest tests\test_data_governance.py tests\test_security_hardening.py tests\test_access_control.py tests\test_data_retention.py -q

### Task 16: 端到端发布验收

Files:
- Create: backend/tests/e2e/test_iteration_release.py
- Modify: .github/workflows/*.yml
- Modify: CONTRIBUTING.md
- Create: docs/evaluation-runbook.md

验收场景：
1. 3 例 smoke 成功且 pre-push 允许。
2. 18 例 regression manifest 完整且可阻断。
3. 高风险病例进入 pending_review 且不产生正式总分。
4. 复核保存原始和调整后 rubric。
5. knowledge Agent 输出 claim、citation、evidence span。
6. diagnosis/treatment 只能使用允许的上游证据。
7. Celery 重试保持 trace_id/run_id 关联。
8. 前端不把未评估渲染成 0。
9. 过期 trace 可清理，导出进入 audit log。

- [ ] Step 1: 写端到端测试夹具

所有 LLM、Redis、MySQL 使用 mock 或测试容器，不使用真实患者数据。

- [ ] Step 2: 执行全量测试

Run:
venv\Scripts\python.exe -m pytest -q
npm test -- --run
npm run build

- [ ] Step 3: 执行离线门禁

Run:
venv\Scripts\python.exe scripts\eval_regression.py --report evaluation\reports\patient_ab\ab_regression.json; Write-Output ('exit=' + $LASTEXITCODE)
venv\Scripts\python.exe scripts\validate_benchmark.py --manifest evaluation\benchmark_manifest.json

- [ ] Step 4: 写运行手册并提交

运行手册必须说明 smoke、regression、benchmark 的生成方式，如何定位 gate FAIL，如何重放 run，如何处理人工复核。

git add backend/tests/e2e/test_iteration_release.py .github CONTRIBUTING.md docs/evaluation-runbook.md
git commit -m "test(release): add clinical evaluation iteration gates"

---

## 推荐执行顺序和时间表

| 周期 | 任务 | 主要交付物 | 发布门 |
|---|---|---|---|
| 第 1 周 | Task 0～2 | ReportManifest、显式 gate | 3 例不阻断，18 例可阻断 |
| 第 2 周 | Task 3～4 | 原子 rubric、Judge calibration | 未评估不等于 0 |
| 第 3 周 | Task 5～6 | 安全红旗集、复核状态机 | 高危病例必须复核 |
| 第 4～5 周 | Task 7～8 | 稳定 citation、Claim-Evidence | 治疗主张无证据必须标记 |
| 第 6 周 | Task 9～11 | 通用 DAG、预算、trace | 单次 run 可完整追踪 |
| 第 7 周 | Task 12～13 | 前端证据报告、复核工作台 | 正确区分 null/0/未评估 |
| 第 8 周 | Task 14～15 | benchmark governance、数据治理 | 导出、脱敏、TTL 可验证 |
| 第 9 周 | Task 16 | 集成验收和运行手册 | 全量测试和正式 regression 通过 |

## 两周 MVP

只能投入两周时，执行 Task 1、Task 2、Task 3、Task 5、Task 7、Task 12。MVP 验收标准：

smoke 不阻断；
正式 regression 可复现；
未评估不等于 0 分；
高风险病例进入人工复核；
临床主张可以定位到证据片段；
报告携带模型、Prompt、Judge、KB 和评分策略版本。

## 设计评审清单

- [ ] 报告包含完整版本 manifest。
- [ ] smoke、regression、benchmark 由机器字段区分。
- [ ] 没有把 None、unassessed 或 insufficient 转成 0 分的路径。
- [ ] 每个高风险结果都有 evidence span 和 policy action。
- [ ] 每个治疗主张都能追踪到 citation 和来源版本。
- [ ] Judge 有重复评分、顺序交换和人工校准数据。
- [ ] pre-push 只阻断正式 regression 的明确 FAIL。
- [ ] FastAPI、Celery、Graph、LLM、RAG 共享 trace_id。
- [ ] 生产日志和 Langfuse 完成脱敏。
- [ ] 前端能解释每个维度为什么得分。
