# -*- coding: utf-8 -*-
"""Task 16 — 端到端发布验收测试

验证前 16 个 Task 组合后，平台能完成真实的评估、复核、回归和报告展示闭环。
"""

import pytest
from datetime import datetime


# ── 场景 1: ReportManifest 完整性 ─────────────────────────────────────────


class TestE2E_Manifest:
    def test_manifest_round_trip(self):
        """新报告 100% 带 manifest"""
        from evaluation.report_schema import ReportManifest, ReportKind
        manifest = ReportManifest(
            report_kind=ReportKind.REGRESSION,
            report_id="e2e-rpt-001",
            created_at=datetime.utcnow().isoformat(),
            case_count=18,
            dataset_version="v1",
            model_version="gpt-4",
            prompt_version="v1",
            judge_version="v1",
            kb_version="kb-2026",
            scoring_policy_version="v1",
            seed=42,
        )
        assert manifest.report_kind == ReportKind.REGRESSION
        assert manifest.case_count == 18


# ── 场景 2: 门禁退出码协议 ───────────────────────────────────────────────


class TestE2E_Gate:
    def test_smoke_returns_skip(self):
        """3 例 smoke 带完整 manifest 返回 SKIP"""
        from evaluation.gate import evaluate_report_gate, GateDecision
        report = {
            "manifest": {
                "report_kind": "smoke",
                "report_id": "smoke-001",
                "created_at": "2026-08-01T00:00:00",
                "case_count": 3,
                "dataset_version": "v1",
                "model_version": "gpt-4",
                "prompt_version": "v1",
                "judge_version": "v1",
                "kb_version": "kb-2026",
                "scoring_policy_version": "v1",
                "seed": 42,
            },
            "cases": [1, 2, 3],
        }
        decision, _ = evaluate_report_gate(report, {})
        assert decision == GateDecision.SKIP

    def test_legacy_returns_invalid(self):
        """无 manifest 旧报告返回 INVALID"""
        from evaluation.gate import evaluate_report_gate, GateDecision
        report = {"cases": list(range(18))}
        decision, _ = evaluate_report_gate(report, {})
        assert decision == GateDecision.INVALID

    def test_exit_code_mapping(self):
        """退出码 0/1/2/3 完整映射"""
        from evaluation.gate import decision_to_exit_code, GateDecision
        assert decision_to_exit_code(GateDecision.PASS) == 0
        assert decision_to_exit_code(GateDecision.FAIL) == 1
        assert decision_to_exit_code(GateDecision.SKIP) == 2
        assert decision_to_exit_code(GateDecision.INVALID) == 3


# ── 场景 3: 五维 Rubric 语义 ─────────────────────────────────────────────


class TestE2E_Rubric:
    def test_unassessed_not_zero(self):
        """unassessed 不得聚合为 0 分"""
        from evaluation.rubric import RubricItem, RubricVerdict, aggregate_rubric
        items = [
            RubricItem(item_id="A", dimension="inquiry", verdict=RubricVerdict.UNASSESSED,
                       score=None, severity="medium", description=""),
        ]
        result = aggregate_rubric(items, dimension="inquiry")
        # unassessed 不应产生 0 分
        assert result.score is None or result.status == "insufficient"

    def test_high_fail_triggers_review(self):
        """high severity + fail → review_required"""
        from evaluation.rubric import RubricItem, RubricVerdict, aggregate_rubric
        items = [
            RubricItem(item_id="A", dimension="treatment", verdict=RubricVerdict.FAIL,
                       score=0, severity="high", description=""),
            RubricItem(item_id="B", dimension="treatment", verdict=RubricVerdict.PASS,
                       score=80, severity="medium", description=""),
        ]
        result = aggregate_rubric(items, dimension="treatment")
        assert result.review_required is True


# ── 场景 4: 安全红旗 fail closed ─────────────────────────────────────────


class TestE2E_Safety:
    def test_evaluate_safety_case(self):
        """安全评估函数可正常调用"""
        from evaluation.safety_cases import evaluate_safety_case
        result = evaluate_safety_case({"symptoms": "chest_pain"})
        assert hasattr(result, "severity")
        assert hasattr(result, "needs_review")

    def test_llm_failure_fail_closed(self):
        """LLM 失败 + 无规则 → fail closed"""
        from evaluation.safety_cases import evaluate_safety_case
        result = evaluate_safety_case({"symptoms": "unknown"}, llm_failed=True)
        # fail closed: 不确定的结果应标记 needs_review 或 severity 不为 none
        assert result.needs_review is True or result.severity != "none"


# ── 场景 5: 复核状态机 ───────────────────────────────────────────────────


class TestE2E_Review:
    def test_valid_transition(self):
        """pending_review → in_review 合法"""
        from evaluation.review_audit import validate_review_transition
        assert validate_review_transition("pending_review", "in_review") is True
        assert validate_review_transition("in_review", "approved") is True

    def test_invalid_transition_rejected(self):
        """pending_review → approved 非法"""
        from evaluation.review_audit import validate_review_transition
        assert validate_review_transition("pending_review", "approved") is False
        assert validate_review_transition("approved", "pending_review") is False


# ── 场景 6: Citation ID 稳定性 ───────────────────────────────────────────


class TestE2E_Citation:
    def test_same_input_same_id(self):
        """同一 chunk 在相同 KB 版本下 ID 稳定"""
        from evaluation.citation_registry import stable_citation_id
        id1 = stable_citation_id("kb-v1", "doc-1", "chunk-1", "content-hash")
        id2 = stable_citation_id("kb-v1", "doc-1", "chunk-1", "content-hash")
        assert id1 == id2

    def test_different_kb_different_id(self):
        """不同 KB 版本 ID 可区分"""
        from evaluation.citation_registry import stable_citation_id
        id1 = stable_citation_id("kb-v1", "doc-1", "chunk-1", "content-hash")
        id2 = stable_citation_id("kb-v2", "doc-1", "chunk-1", "content-hash")
        assert id1 != id2


# ── 场景 7: Claim-Evidence 验证 ──────────────────────────────────────────


class TestE2E_Claims:
    def test_unsupported_treatment_needs_review(self):
        """无证据治疗 claim 必须复核"""
        from evaluation.rag_claims import validate_claim_evidence, ClinicalClaim, ClaimStatus
        claim = ClinicalClaim(
            claim_id="c1", claim_type="treatment", text="使用阿莫西林",
            status=ClaimStatus.UNSUPPORTED, evidence=[], needs_review=False,
        )
        errors = validate_claim_evidence([claim])
        # unsupported claim 应产生验证错误
        assert len(errors) > 0


# ── 场景 8: PlanStep DAG ─────────────────────────────────────────────────


class TestE2E_DAG:
    def test_cycle_detected(self):
        """循环依赖被拒绝"""
        from evaluation.plan_dag import validate_plan_dag, PlanStep, StepStatus
        steps = [
            PlanStep(step_id="a", depends_on=["b"], status=StepStatus.PENDING),
            PlanStep(step_id="b", depends_on=["a"], status=StepStatus.PENDING),
        ]
        errors = validate_plan_dag(steps)
        assert len(errors) > 0

    def test_ready_steps(self):
        """依赖完成后变为 ready"""
        from evaluation.plan_dag import ready_steps, PlanStep, StepStatus
        steps = [
            PlanStep(step_id="knowledge", depends_on=[], status=StepStatus.SUCCEEDED),
            PlanStep(step_id="diagnosis", depends_on=["knowledge"], status=StepStatus.PENDING),
        ]
        ready = ready_steps(steps, completed={"knowledge"})
        assert any(s.step_id == "diagnosis" for s in ready)


# ── 场景 9: RunBudget 并发控制 ───────────────────────────────────────────


class TestE2E_Budget:
    def test_concurrent_limit_enforced(self):
        """并发超过上限时拒绝"""
        from app.services.run_budget import RunBudget, RunBudgetManager, BudgetDecision
        mgr = RunBudgetManager(RunBudget(max_parallel_agents=2))
        mgr.acquire_agent_slot("run-1", "inquiry")
        mgr.acquire_agent_slot("run-1", "knowledge")
        d = mgr.acquire_agent_slot("run-1", "diagnosis")
        assert d == BudgetDecision.REJECTED

    def test_safety_exempt(self):
        """安全路径豁免并发上限"""
        from app.services.run_budget import RunBudget, RunBudgetManager, BudgetDecision
        mgr = RunBudgetManager(RunBudget(max_parallel_agents=1))
        mgr.acquire_agent_slot("run-1", "inquiry")
        d = mgr.acquire_agent_slot("run-1", "safety")
        assert d == BudgetDecision.ACCEPTED


# ── 场景 10: TraceContext 传播 ───────────────────────────────────────────


class TestE2E_Trace:
    def test_trace_round_trip(self):
        """Celery payload 传播 trace context"""
        from app.services.observability.trace_context import (
            TraceContext, serialize_trace_context, restore_trace_context,
        )
        ctx = TraceContext(trace_id="e2e-trace", run_id="run-e2e", attempt=1)
        payload = serialize_trace_context(ctx)
        restored = restore_trace_context(payload)
        assert restored.trace_id == "e2e-trace"
        assert restored.run_id == "run-e2e"

    def test_retry_preserves_trace_id(self):
        """重试后 trace_id 不变、attempt 增加"""
        from app.services.observability.trace_context import (
            TraceContext, serialize_trace_context, restore_trace_context,
        )
        ctx = TraceContext(trace_id="same-trace", run_id="run-1", attempt=1)
        payload = serialize_trace_context(ctx)
        payload["attempt"] = 2
        restored = restore_trace_context(payload)
        assert restored.trace_id == "same-trace"
        assert restored.attempt == 2

    def test_sanitization(self):
        """脱敏函数覆盖手机号"""
        from app.services.observability.trace_context import sanitize_for_observability
        result = sanitize_for_observability("患者电话13812345678")
        assert "13812345678" not in result


# ── 场景 11: 数据分级和脱敏 ─────────────────────────────────────────────


class TestE2E_DataGovernance:
    def test_redact_pii(self):
        """姓名、电话、身份证被脱敏"""
        from app.evaluation.data_governance import redact_trace, DataClassification
        payload = {"text": "患者张三丰，电话13800001111"}
        result = redact_trace(payload, DataClassification.P1_MASKED)
        assert "13800001111" not in str(result)

    def test_doctor_cannot_export_p0(self):
        """普通用户不能导出 P0 数据"""
        from app.evaluation.data_governance import validate_export_scope, DataClassification
        with pytest.raises(PermissionError):
            validate_export_scope({"role": "doctor"}, DataClassification.P0_RAW)


# ── 场景 12: 基准集校验 ─────────────────────────────────────────────────


class TestE2E_Benchmark:
    def test_duplicate_case_rejected(self):
        """重复 case_id 被拒绝"""
        from app.evaluation.benchmark import (
            BenchmarkCase, BenchmarkManifest, validate_benchmark_manifest,
        )
        manifest = BenchmarkManifest(
            version="v1", rubric_version="v1",
            cases=[
                BenchmarkCase(case_id="dup", split="test", specialty="cardiology", difficulty=3),
                BenchmarkCase(case_id="dup", split="dev", specialty="neurology", difficulty=2),
            ],
        )
        errors = validate_benchmark_manifest(manifest)
        assert len(errors) > 0

    def test_safety_without_red_flags_rejected(self):
        """safety case 缺红旗被拒绝"""
        from app.evaluation.benchmark import (
            BenchmarkCase, BenchmarkManifest, validate_benchmark_manifest,
        )
        manifest = BenchmarkManifest(
            version="v1", rubric_version="v1",
            cases=[
                BenchmarkCase(case_id="s1", split="safety", specialty="emergency",
                              difficulty=5, red_flags=[]),
            ],
        )
        errors = validate_benchmark_manifest(manifest)
        assert any("红旗" in e or "red_flag" in e.lower() for e in errors)

    def test_split_deterministic(self):
        """固定 seed 可重放同一病例集"""
        from app.evaluation.benchmark import BenchmarkCase, split_cases
        cases = [BenchmarkCase(case_id=f"c{i}", split="test", specialty="cardiology", difficulty=3)
                 for i in range(10)]
        r1 = split_cases(cases, "test", seed=42)
        r2 = split_cases(cases, "test", seed=42)
        assert [c.case_id for c in r1] == [c.case_id for c in r2]
