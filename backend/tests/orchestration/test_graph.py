"""LangGraph 主图测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.orchestration.state import (
    EvaluationState,
    EvaluationContext,
    SubmissionFlags,
    RoutePlan,
    SafetyResult,
    AgentResultEnvelope,
    DimensionResult as StateDimensionResult,
)


# ── 图构建测试 ────────────────────────────────────────────────────────────


class TestBuildEvaluationGraph:
    """测试图可以构建和编译"""

    def test_graph_builds_successfully(self):
        """图应该能成功构建"""
        from app.orchestration.graph import build_evaluation_graph

        graph = build_evaluation_graph()
        assert graph is not None
        # 检查关键节点是否存在
        nodes = list(graph.nodes.keys())
        assert "load_context" in nodes
        assert "classify_consultation" in nodes
        assert "safety_check" in nodes
        assert "build_route_plan" in nodes
        assert "dispatch_and_run" in nodes
        assert "aggregate_results" in nodes
        assert "deterministic_scoring" in nodes
        assert "generate_suggestion" in nodes
        assert "finalize_completed" in nodes
        assert "finalize_needs_review" in nodes

    @pytest.mark.asyncio
    async def test_graph_compiles_with_mock_checkpointer(self):
        """图应该能用 mock checkpointer 编译"""
        from app.orchestration.graph import build_evaluation_graph
        from langgraph.checkpoint.memory import MemorySaver

        graph = build_evaluation_graph()
        checkpointer = MemorySaver()
        compiled = graph.compile(checkpointer=checkpointer)
        assert compiled is not None


# ── Safety Gate 条件路由测试 ──────────────────────────────────────────────


class TestSafetyGate:
    """测试 safety_gate 条件路由"""

    def test_safety_gate_continues_on_low_risk(self):
        """低风险应该继续流程"""
        from app.orchestration.graph import safety_gate

        state: EvaluationState = {
            "run_id": "test-1",
            "context": EvaluationContext(conversation_text="正常对话"),
            "consultation_type": "initial",
            "submission_flags": SubmissionFlags(),
            "safety_result": SafetyResult(
                risk_level="low",
                immediate_review_required=False,
            ),
        }
        result = safety_gate(state)
        assert result == "continue"

    def test_safety_gate_continues_on_medium_risk(self):
        """中风险应该继续流程（不触发复核）"""
        from app.orchestration.graph import safety_gate

        state: EvaluationState = {
            "run_id": "test-2",
            "context": EvaluationContext(conversation_text="正常对话"),
            "consultation_type": "initial",
            "submission_flags": SubmissionFlags(),
            "safety_result": SafetyResult(
                risk_level="medium",
                immediate_review_required=False,
            ),
        }
        result = safety_gate(state)
        assert result == "continue"

    def test_safety_gate_needs_review_on_high_risk(self):
        """高风险应该需要复核"""
        from app.orchestration.graph import safety_gate

        state: EvaluationState = {
            "run_id": "test-3",
            "context": EvaluationContext(conversation_text="危险症状"),
            "consultation_type": "initial",
            "submission_flags": SubmissionFlags(),
            "safety_result": SafetyResult(
                risk_level="high",
                immediate_review_required=True,
            ),
        }
        result = safety_gate(state)
        assert result == "needs_review"

    def test_safety_gate_needs_review_on_undetermined(self):
        """不确定风险应该需要复核（fail closed）"""
        from app.orchestration.graph import safety_gate

        state: EvaluationState = {
            "run_id": "test-4",
            "context": EvaluationContext(conversation_text="模糊对话"),
            "consultation_type": "initial",
            "submission_flags": SubmissionFlags(),
            "safety_result": SafetyResult(
                risk_level="undetermined",
                immediate_review_required=True,
            ),
        }
        result = safety_gate(state)
        assert result == "needs_review"

    def test_safety_gate_needs_review_when_none(self):
        """无 safety_result 时应该需要复核（fail closed）"""
        from app.orchestration.graph import safety_gate

        state: EvaluationState = {
            "run_id": "test-5",
            "context": EvaluationContext(conversation_text="对话"),
            "consultation_type": "initial",
            "submission_flags": SubmissionFlags(),
        }
        result = safety_gate(state)
        assert result == "needs_review"


# ── Review Gate 条件路由测试 ──────────────────────────────────────────────


class TestReviewGate:
    """测试 review_gate 条件路由"""

    def test_review_gate_completes_when_all_scored(self):
        """所有维度都 scored 且无需复核时应该完成"""
        from app.orchestration.graph import review_gate

        state: EvaluationState = {
            "run_id": "test-6",
            "context": EvaluationContext(conversation_text="对话"),
            "consultation_type": "initial",
            "submission_flags": SubmissionFlags(),
            "dimension_results": {
                "inquiry": StateDimensionResult(
                    dimension="inquiry",
                    status="scored",
                    score=80.0,
                    analysis="良好",
                ),
                "diagnosis": StateDimensionResult(
                    dimension="diagnosis",
                    status="scored",
                    score=75.0,
                    analysis="良好",
                ),
            },
            "agent_results": [
                AgentResultEnvelope(
                    agent_name="inquiry",  # type: ignore[arg-type]
                    status="success",
                    score=80.0,
                    human_review_needed=False,
                )
            ],
        }
        result = review_gate(state)
        assert result == "completed"

    def test_review_gate_needs_review_on_error_dimension(self):
        """有 error 维度时应该需要复核"""
        from app.orchestration.graph import review_gate

        state: EvaluationState = {
            "run_id": "test-7",
            "context": EvaluationContext(conversation_text="对话"),
            "consultation_type": "initial",
            "submission_flags": SubmissionFlags(),
            "dimension_results": {
                "inquiry": StateDimensionResult(
                    dimension="inquiry",
                    status="error",
                    score=None,
                    analysis="执行异常",
                ),
            },
            "agent_results": [],
        }
        result = review_gate(state)
        assert result == "needs_review"

    def test_review_gate_needs_review_on_insufficient_dimension(self):
        """有 insufficient 维度时应该需要复核"""
        from app.orchestration.graph import review_gate

        state: EvaluationState = {
            "run_id": "test-8",
            "context": EvaluationContext(conversation_text="对话"),
            "consultation_type": "initial",
            "submission_flags": SubmissionFlags(),
            "dimension_results": {
                "inquiry": StateDimensionResult(
                    dimension="inquiry",
                    status="insufficient",
                    score=None,
                    analysis="证据不足",
                ),
            },
            "agent_results": [],
        }
        result = review_gate(state)
        assert result == "needs_review"

    def test_review_gate_needs_review_on_human_review_needed(self):
        """Agent 标记 human_review_needed 时应该需要复核"""
        from app.orchestration.graph import review_gate

        state: EvaluationState = {
            "run_id": "test-9",
            "context": EvaluationContext(conversation_text="对话"),
            "consultation_type": "initial",
            "submission_flags": SubmissionFlags(),
            "dimension_results": {},
            "agent_results": [
                AgentResultEnvelope(
                    agent_name="inquiry",  # type: ignore[arg-type]
                    status="success",
                    score=80.0,
                    human_review_needed=True,
                    review_reason="需要人工确认",
                )
            ],
        }
        result = review_gate(state)
        assert result == "needs_review"


# ── Aggregate Results 测试 ────────────────────────────────────────────────


class TestAggregateResults:
    """测试 aggregate_results 正确转换 agent_results → dimension_results"""

    @pytest.mark.asyncio
    async def test_aggregate_converts_success_agents(self):
        """成功 Agent 应该转换为 scored 维度"""
        from app.orchestration.graph import aggregate_results

        state: EvaluationState = {
            "run_id": "test-10",
            "context": EvaluationContext(conversation_text="对话"),
            "consultation_type": "initial",
            "submission_flags": SubmissionFlags(),
            "route_plan": RoutePlan(
                consultation_type="initial",
                selected_agents=["inquiry", "humanistic"],
                skipped_agents=[],
                skip_reasons={},
            ),
            "agent_results": [
                AgentResultEnvelope(
                    agent_name="inquiry",  # type: ignore[arg-type]
                    status="success",
                    score=80.0,
                    analysis="问诊表现良好",
                ),
                AgentResultEnvelope(
                    agent_name="humanistic",  # type: ignore[arg-type]
                    status="success",
                    score=75.0,
                    analysis="人文关怀良好",
                ),
            ],
        }

        result = await aggregate_results(state)
        dims = result["dimension_results"]

        assert len(dims) == 2
        assert dims["inquiry"].status == "scored"
        assert dims["inquiry"].score == 80.0
        assert dims["humanistic"].status == "scored"
        assert dims["humanistic"].score == 75.0

    @pytest.mark.asyncio
    async def test_aggregate_handles_skipped_agents(self):
        """被跳过的 Agent 应该标记为 not_submitted"""
        from app.orchestration.graph import aggregate_results

        state: EvaluationState = {
            "run_id": "test-11",
            "context": EvaluationContext(conversation_text="对话"),
            "consultation_type": "initial",
            "submission_flags": SubmissionFlags(has_diagnosis=False, has_treatment=False),
            "route_plan": RoutePlan(
                consultation_type="initial",
                selected_agents=["inquiry"],
                skipped_agents=["diagnosis", "treatment"],
                skip_reasons={
                    "diagnosis": "未提交诊断结果",
                    "treatment": "未提交治疗方案",
                },
            ),
            "agent_results": [
                AgentResultEnvelope(
                    agent_name="inquiry",  # type: ignore[arg-type]
                    status="success",
                    score=80.0,
                    analysis="问诊表现良好",
                ),
            ],
        }

        result = await aggregate_results(state)
        dims = result["dimension_results"]

        assert len(dims) == 3
        assert dims["inquiry"].status == "scored"
        assert dims["diagnosis"].status == "not_submitted"
        assert dims["diagnosis"].analysis == "未提交诊断结果"
        assert dims["treatment"].status == "not_submitted"
        assert dims["treatment"].analysis == "未提交治疗方案"

    @pytest.mark.asyncio
    async def test_aggregate_handles_error_agents(self):
        """错误 Agent 应该保持 error 状态"""
        from app.orchestration.graph import aggregate_results

        state: EvaluationState = {
            "run_id": "test-12",
            "context": EvaluationContext(conversation_text="对话"),
            "consultation_type": "initial",
            "submission_flags": SubmissionFlags(),
            "route_plan": RoutePlan(
                consultation_type="initial",
                selected_agents=["inquiry"],
                skipped_agents=[],
                skip_reasons={},
            ),
            "agent_results": [
                AgentResultEnvelope(
                    agent_name="inquiry",  # type: ignore[arg-type]
                    status="error",
                    analysis="执行异常: ConnectionError",
                ),
            ],
        }

        result = await aggregate_results(state)
        dims = result["dimension_results"]

        assert dims["inquiry"].status == "error"
        assert dims["inquiry"].score is None


# ── Dispatch and Run 测试 ─────────────────────────────────────────────────


class TestDispatchAndRun:
    """测试 dispatch_and_run 并行执行 Agent"""

    @pytest.mark.asyncio
    async def test_dispatch_runs_selected_agents(self):
        """应该并行运行选中的 Agent"""
        from app.orchestration.graph import dispatch_and_run

        # Mock adapter
        mock_envelope = AgentResultEnvelope(
            agent_name="inquiry",  # type: ignore[arg-type]
            status="success",
            score=80.0,
            analysis="良好",
        )

        with patch("app.orchestration.adapters.registry.get_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.run = AsyncMock(return_value=mock_envelope)
            mock_get_adapter.return_value = mock_adapter

            state: EvaluationState = {
                "run_id": "test-13",
                "context": EvaluationContext(conversation_text="对话"),
                "consultation_type": "initial",
                "submission_flags": SubmissionFlags(),
                "route_plan": RoutePlan(
                    consultation_type="initial",
                    selected_agents=["inquiry", "humanistic"],
                    skipped_agents=[],
                    skip_reasons={},
                ),
            }

            result = await dispatch_and_run(state)

            assert len(result["agent_results"]) == 2
            assert mock_get_adapter.call_count == 2
            mock_get_adapter.assert_any_call("inquiry")
            mock_get_adapter.assert_any_call("humanistic")

    @pytest.mark.asyncio
    async def test_dispatch_handles_exceptions(self):
        """Agent 异常应该被捕获并标记为 error"""
        from app.orchestration.graph import dispatch_and_run

        with patch("app.orchestration.adapters.registry.get_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.run = AsyncMock(side_effect=Exception("Connection failed"))
            mock_get_adapter.return_value = mock_adapter

            state: EvaluationState = {
                "run_id": "test-14",
                "context": EvaluationContext(conversation_text="对话"),
                "consultation_type": "initial",
                "submission_flags": SubmissionFlags(),
                "route_plan": RoutePlan(
                    consultation_type="initial",
                    selected_agents=["inquiry"],
                    skipped_agents=[],
                    skip_reasons={},
                ),
            }

            result = await dispatch_and_run(state)

            assert len(result["agent_results"]) == 1
            envelope = result["agent_results"][0]
            assert envelope.status == "error"
            assert envelope.human_review_needed is True
            assert "Connection failed" in envelope.review_reason


# ── Deterministic Scoring 测试 ────────────────────────────────────────────


class TestDeterministicScoring:
    """测试确定性评分节点"""

    @pytest.mark.asyncio
    async def test_scoring_calculates_total(self):
        """应该正确计算加权总分"""
        from app.orchestration.graph import deterministic_scoring

        state: EvaluationState = {
            "run_id": "test-15",
            "context": EvaluationContext(conversation_text="对话"),
            "consultation_type": "initial",
            "submission_flags": SubmissionFlags(),
            "dimension_results": {
                "inquiry": StateDimensionResult(
                    dimension="inquiry",
                    status="scored",
                    score=80.0,
                    analysis="良好",
                ),
                "knowledge": StateDimensionResult(
                    dimension="knowledge",
                    status="scored",
                    score=85.0,
                    analysis="良好",
                ),
                "humanistic": StateDimensionResult(
                    dimension="humanistic",
                    status="scored",
                    score=75.0,
                    analysis="良好",
                ),
                "diagnosis": StateDimensionResult(
                    dimension="diagnosis",
                    status="scored",
                    score=70.0,
                    analysis="良好",
                ),
                "treatment": StateDimensionResult(
                    dimension="treatment",
                    status="scored",
                    score=72.0,
                    analysis="良好",
                ),
            },
        }

        result = await deterministic_scoring(state)

        # 验证 total_score 已计算（具体值取决于权重配置）
        assert result["total_score"] is not None
        assert isinstance(result["total_score"], int)
        assert 0 <= result["total_score"] <= 100
