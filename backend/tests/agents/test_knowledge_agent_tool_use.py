"""知识 Agent Tool Use 单元测试

测试 _map_consistency_to_score_v2() 分数映射函数和返回结构兼容性。
"""

import pytest

from app.services.agents.knowledge_agent import _map_consistency_to_score_v2


# ── 分数映射测试 ─────────────────────────────────────────────────────────────────


class TestMapConsistencyToScoreV2:
    """_map_consistency_to_score_v2 分数映射测试"""

    def test_score_mapping_supports(self):
        """consistency=supports, confidence=0.8 → 80~95 分"""
        score = _map_consistency_to_score_v2("supports", 0.8)
        assert score is not None
        assert 80 <= score <= 95, f"Expected 80~95, got {score}"
        # 精确计算: 80 + 0.8 * 15 = 92.0
        assert score == 92.0

    def test_score_mapping_supports_high_confidence(self):
        """consistency=supports, confidence=1.0 → 95 分"""
        score = _map_consistency_to_score_v2("supports", 1.0)
        assert score == 95.0

    def test_score_mapping_supports_low_confidence(self):
        """consistency=supports, confidence=0.0 → 80 分"""
        score = _map_consistency_to_score_v2("supports", 0.0)
        assert score == 80.0

    def test_score_mapping_mixed(self):
        """consistency=mixed, confidence=0.6 → 50~75 分"""
        score = _map_consistency_to_score_v2("mixed", 0.6)
        assert score is not None
        assert 50 <= score <= 75, f"Expected 50~75, got {score}"
        # 精确计算: 50 + 0.6 * 25 = 65.0
        assert score == 65.0

    def test_score_mapping_mixed_high_confidence(self):
        """consistency=mixed, confidence=1.0 → 75 分"""
        score = _map_consistency_to_score_v2("mixed", 1.0)
        assert score == 75.0

    def test_score_mapping_contradicts(self):
        """consistency=contradicts, confidence=0.7 → 0~45 分"""
        score = _map_consistency_to_score_v2("contradicts", 0.7)
        assert score is not None
        assert 0 <= score <= 45, f"Expected 0~45, got {score}"
        # 精确计算: 0.7 * 45 = 31.5
        assert score == 31.5

    def test_score_mapping_contradicts_high_confidence(self):
        """consistency=contradicts, confidence=1.0 → 45 分"""
        score = _map_consistency_to_score_v2("contradicts", 1.0)
        assert score == 45.0

    def test_score_mapping_undetermined(self):
        """consistency=undetermined → score=None"""
        score = _map_consistency_to_score_v2("undetermined", 0.8)
        assert score is None

    def test_score_mapping_undetermined_any_confidence(self):
        """consistency=undetermined 无论 confidence 如何 → score=None"""
        for conf in [0.0, 0.3, 0.5, 0.9, 1.0]:
            score = _map_consistency_to_score_v2("undetermined", conf)
            assert score is None

    def test_score_mapping_insufficient_evidence(self):
        """evidence_sufficiency=insufficient 时，上层逻辑应 score=None, human_review_needed=True

        注意：_map_consistency_to_score_v2 本身不处理 evidence_sufficiency，
        这里测试当 consistency 为有效值但上层判断为 insufficient 时的行为。
        """
        # 如果 consistency 是有效值，函数仍返回分数
        # 但上层 run_knowledge_check_with_tools 会在 evidence_sufficiency=insufficient 时
        # 设置 needs_review=True, score=None
        # 这里只测试函数本身的行为
        score = _map_consistency_to_score_v2("supports", 0.5)
        assert score is not None  # 函数本身返回分数

    def test_score_mapping_unknown_consistency(self):
        """未知 consistency 值 → score=None"""
        score = _map_consistency_to_score_v2("unknown_value", 0.5)
        assert score is None

    def test_no_default_50_score(self):
        """验证不存在任何默认 50 分降级路径

        旧版 _map_consistency_to_score 对 undetermined 返回 50 分左右，
        新版 v2 对 undetermined 和未知值返回 None。
        """
        # undetermined 应返回 None 而非 50
        assert _map_consistency_to_score_v2("undetermined", 1.0) is None
        # 未知值应返回 None
        assert _map_consistency_to_score_v2("garbage", 0.5) is None
        # 所有有效映射都不应恰好等于 50（除非在 mixed 范围内）
        # supports 范围 80~95，不包含 50
        assert _map_consistency_to_score_v2("supports", 0.0) == 80.0
        # contradicts 范围 0~45，不包含 50
        assert _map_consistency_to_score_v2("contradicts", 1.0) == 45.0

    def test_confidence_clamped(self):
        """confidence 超出 [0, 1] 范围时被 clamp"""
        # confidence > 1.0 被 clamp 到 1.0
        score = _map_consistency_to_score_v2("supports", 1.5)
        assert score == 95.0  # 80 + 1.0 * 15

        # confidence < 0.0 被 clamp 到 0.0
        score = _map_consistency_to_score_v2("supports", -0.5)
        assert score == 80.0  # 80 + 0.0 * 15


# ── 返回结构兼容性测试 ────────────────────────────────────────────────────────────


class TestReturnStructure:
    """验证返回结构与旧版 run_knowledge_check() 兼容"""

    def test_return_structure_compatible(self):
        """验证 run_knowledge_check_with_tools 返回的 dict 包含所有旧版必需字段

        旧版 run_knowledge_check 返回的必需字段：
        - raw_response, score, analysis, retrieval_status, evidence_stance,
          citations, human_review_needed, review_reason, confidence, rag_trace, degraded
        """
        # 直接检查 _build_error_result 返回结构（这是最简单的路径）
        from app.services.agents.knowledge_agent import _build_error_result

        result = _build_error_result("test error", [])

        # 所有旧版字段必须存在
        assert "raw_response" in result
        assert "score" in result
        assert "analysis" in result
        assert "retrieval_status" in result
        assert "evidence_stance" in result
        assert "citations" in result
        assert "human_review_needed" in result
        assert "review_reason" in result
        assert "confidence" in result
        assert "rag_trace" in result
        assert "degraded" in result

    def test_error_result_score_is_none(self):
        """错误结果的 score 应为 None"""
        from app.services.agents.knowledge_agent import _build_error_result

        result = _build_error_result("test error", [])
        assert result["score"] is None
        assert result["human_review_needed"] is True
        assert result["degraded"] is True

    def test_tool_trace_included(self):
        """验证 tool_trace 字段存在且为列表"""
        from app.services.agents.knowledge_agent import _build_error_result, _format_tool_trace

        # _format_tool_trace 应返回列表
        traces = _format_tool_trace([])
        assert isinstance(traces, list)

        # 带数据的 trace
        sample_traces = [
            {
                "tool_name": "search_medical_kb",
                "status": "success",
                "elapsed_ms": 100.5,
                "arguments_summary": {"query": "test"},
                "error": None,
            }
        ]
        formatted = _format_tool_trace(sample_traces)
        assert len(formatted) == 1
        assert formatted[0]["tool_name"] == "search_medical_kb"
        assert formatted[0]["status"] == "success"

    def test_citation_failed_result_structure(self):
        """引用校验失败返回结构完整"""
        from app.services.agents.knowledge_agent import _build_citation_failed_result

        result = _build_citation_failed_result("test analysis", [])
        assert result["score"] is None
        assert result["human_review_needed"] is True
        assert result["review_reason"] == "citation_verification_failed"
        assert result["degraded"] is True
        assert isinstance(result["tool_trace"], list)
