# -*- coding: utf-8 -*-
"""Task 13 — 复核反馈 → 候选样本 → 回归门禁 最小数据飞轮测试

覆盖：
- ReviewFeedbackCandidate 模型
- 同一 review 产生稳定 candidate_id
- 手机号/身份证/姓名脱敏
- 无内容模式不含 conversation
- 敏感模式没有双确认则拒绝
- score delta 归因
- CLI 参数：since/until、limit、空结果、原子输出、数据库错误 exit code
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# ── 常量 ──────────────────────────────────────────────────────────────────────

_TEST_HMAC_KEY = "test-feedback-hmac-key-for-unit-tests"


# ── 辅助工厂 ──────────────────────────────────────────────────────────────────


def _make_review_record(
    review_id: str = "rev-001",
    evaluation_id: str = "1",
    reviewer_id: str = "admin-1",
    feedback: str = "诊断依据不充分，需补充鉴别诊断",
    review_reason: str | None = "evidence_insufficient",
    score_adjustments: dict | None = None,
    original_scores: dict | None = None,
    created_at: datetime | None = None,
):
    """构造 mock ReviewRecord（使用 MagicMock 避免 ORM 初始化问题）"""
    from unittest.mock import MagicMock

    rec = MagicMock()
    rec.id = review_id
    rec.evaluation_id = evaluation_id
    rec.reviewer_id = reviewer_id
    rec.feedback = feedback
    rec.review_reason = review_reason
    rec.score_adjustments = score_adjustments
    rec.original_scores = original_scores
    rec.created_at = created_at or datetime(2026, 8, 10, 12, 0, 0)
    return rec


def _make_evaluation(
    eval_id: int = 1,
    consultation_id: int = 42,
    run_id: str = "run-001",
    inquiry_score: float = 70.0,
    knowledge_score: float | None = 60.0,
    humanistic_score: float = 75.0,
    diagnosis_score: float = 50.0,
    treatment_score: float = 65.0,
    retrieval_status: str = "insufficient",
    evidence_stance: str = "undetermined",
    citation_data: list | None = None,
    review_reason: str | None = "evidence_insufficient",
):
    """构造 mock Evaluation（使用 MagicMock 避免 ORM 初始化问题）"""
    from unittest.mock import MagicMock

    ev = MagicMock()
    ev.id = eval_id
    ev.consultation_id = consultation_id
    ev.run_id = run_id
    ev.inquiry_score = inquiry_score
    ev.knowledge_score = knowledge_score
    ev.humanistic_score = humanistic_score
    ev.diagnosis_score = diagnosis_score
    ev.treatment_score = treatment_score
    ev.retrieval_status = retrieval_status
    ev.evidence_stance = evidence_stance
    ev.citation_data = citation_data or []
    ev.review_reason = review_reason
    return ev


# ── 1. 稳定 candidate_id ─────────────────────────────────────────────────────


class TestStableCandidateId:
    """同一 review 产生稳定 candidate_id"""

    def test_same_review_id_produces_same_candidate_id(self):
        from evaluation.feedback_cases import make_candidate_id

        cid1 = make_candidate_id("rev-001")
        cid2 = make_candidate_id("rev-001")
        assert cid1 == cid2

    def test_different_review_id_produces_different_candidate_id(self):
        from evaluation.feedback_cases import make_candidate_id

        cid1 = make_candidate_id("rev-001")
        cid2 = make_candidate_id("rev-002")
        assert cid1 != cid2

    def test_candidate_id_is_deterministic_hex(self):
        from evaluation.feedback_cases import make_candidate_id

        cid = make_candidate_id("rev-abc")
        # HMAC-SHA256 hex digest = 64 chars
        assert len(cid) == 64
        int(cid, 16)  # 合法十六进制


# ── 2. 脱敏 ──────────────────────────────────────────────────────────────────


class TestSanitization:
    """手机号/身份证/姓名脱敏"""

    def test_phone_number_sanitized(self):
        from evaluation.feedback_cases import sanitize_feedback_text

        text = "患者电话13812345678，请尽快联系"
        result = sanitize_feedback_text(text)
        assert "13812345678" not in result
        assert "138****5678" in result

    def test_id_card_sanitized(self):
        from evaluation.feedback_cases import sanitize_feedback_text

        text = "身份证号110101199001011234已核实"
        result = sanitize_feedback_text(text)
        assert "110101199001011234" not in result

    def test_name_sanitized(self):
        from evaluation.feedback_cases import sanitize_feedback_text

        text = "患者姓名张三丰，主诉胸闷"
        result = sanitize_feedback_text(text)
        assert "张三丰" not in result

    def test_clinical_content_preserved(self):
        from evaluation.feedback_cases import sanitize_feedback_text

        text = "主诉：胸闷三天，伴有气短，建议心电图检查"
        result = sanitize_feedback_text(text)
        assert "胸闷" in result
        assert "气短" in result
        assert "心电图" in result


# ── 3. HMAC consultation_id ──────────────────────────────────────────────────


class TestHmacConsultationId:
    """consultation_id 使用 HMAC-SHA256，不可反查"""

    def test_hmac_consultation_id_not_raw(self):
        from evaluation.feedback_cases import hash_consultation_id

        result = hash_consultation_id(42, _TEST_HMAC_KEY)
        assert result != "42"
        assert result != str(42)

    def test_hmac_consultation_id_deterministic(self):
        from evaluation.feedback_cases import hash_consultation_id

        h1 = hash_consultation_id(42, _TEST_HMAC_KEY)
        h2 = hash_consultation_id(42, _TEST_HMAC_KEY)
        assert h1 == h2

    def test_hmac_consultation_id_different_keys(self):
        from evaluation.feedback_cases import hash_consultation_id

        h1 = hash_consultation_id(42, "key-a")
        h2 = hash_consultation_id(42, "key-b")
        assert h1 != h2

    def test_hmac_consultation_id_is_hex(self):
        from evaluation.feedback_cases import hash_consultation_id

        result = hash_consultation_id(42, _TEST_HMAC_KEY)
        assert len(result) == 64
        int(result, 16)


# ── 4. 归因规则 ──────────────────────────────────────────────────────────────


class TestAttribution:
    """score delta 归因"""

    def test_knowledge_scoring_gap(self):
        """知识分调整大于 10 分 → knowledge_scoring_gap"""
        from evaluation.feedback_cases import classify_attribution

        original = {"knowledge_score": 50.0, "diagnosis_score": 60.0}
        adjusted = {"knowledge_score": 65.0, "diagnosis_score": 62.0}
        result = classify_attribution(original, adjusted, review_reason=None)
        assert result == "knowledge_scoring_gap"

    def test_retrieval_insufficient(self):
        """证据不足 → retrieval_insufficient"""
        from evaluation.feedback_cases import classify_attribution

        original = {"knowledge_score": 60.0}
        adjusted = {"knowledge_score": 62.0}
        result = classify_attribution(
            original, adjusted, review_reason="evidence_insufficient"
        )
        assert result == "retrieval_insufficient"

    def test_citation_quality(self):
        """引用不一致 → citation_quality"""
        from evaluation.feedback_cases import classify_attribution

        original = {"knowledge_score": 60.0}
        adjusted = {"knowledge_score": 62.0}
        result = classify_attribution(
            original, adjusted, review_reason="citation_mismatch"
        )
        assert result == "citation_quality"

    def test_default_review_feedback(self):
        """其他 → review_feedback"""
        from evaluation.feedback_cases import classify_attribution

        original = {"knowledge_score": 60.0}
        adjusted = {"knowledge_score": 62.0}
        result = classify_attribution(original, adjusted, review_reason=None)
        assert result == "review_feedback"

    def test_knowledge_gap_takes_priority_over_reason(self):
        """知识分差距 > 10 优先于 review_reason"""
        from evaluation.feedback_cases import classify_attribution

        original = {"knowledge_score": 50.0}
        adjusted = {"knowledge_score": 65.0}
        result = classify_attribution(
            original, adjusted, review_reason="citation_mismatch"
        )
        assert result == "knowledge_scoring_gap"

    def test_knowledge_exactly_10_not_gap(self):
        """知识分调整恰好 10 分不算 gap（需要大于 10）"""
        from evaluation.feedback_cases import classify_attribution

        original = {"knowledge_score": 50.0}
        adjusted = {"knowledge_score": 60.0}
        result = classify_attribution(original, adjusted, review_reason=None)
        assert result == "review_feedback"

    def test_knowledge_negative_large_delta_is_gap(self):
        """知识分大幅下调（>10）也归 knowledge_scoring_gap"""
        from evaluation.feedback_cases import classify_attribution

        original = {"knowledge_score": 70.0}
        adjusted = {"knowledge_score": 55.0}
        # abs(55-70) = 15 > 10
        result = classify_attribution(original, adjusted, review_reason=None)
        assert result == "knowledge_scoring_gap"


# ── 5. build_candidate ───────────────────────────────────────────────────────


class TestBuildCandidate:
    """build_candidate 从 review + evaluation 构造候选"""

    def test_build_candidate_basic(self):
        from evaluation.feedback_cases import build_candidate

        review = _make_review_record(
            original_scores={
                "inquiry_score": 70.0,
                "knowledge_score": 60.0,
                "humanistic_score": 75.0,
                "diagnosis_score": 50.0,
                "treatment_score": 65.0,
            },
            score_adjustments={
                "inquiry_score": 72.0,
                "knowledge_score": 62.0,
                "humanistic_score": 76.0,
                "diagnosis_score": 55.0,
                "treatment_score": 67.0,
            },
        )
        evaluation = _make_evaluation()

        candidate = build_candidate(review, evaluation, hmac_key=_TEST_HMAC_KEY)

        assert candidate.review_id == "rev-001"
        assert candidate.run_id == "run-001"
        assert candidate.consultation_id_hash != "42"
        assert candidate.department is None
        assert candidate.retrieval_status == "insufficient"
        assert candidate.evidence_stance == "undetermined"
        assert candidate.contains_sensitive_content is False

    def test_build_candidate_sanitizes_feedback(self):
        from evaluation.feedback_cases import build_candidate

        review = _make_review_record(
            feedback="患者张三，电话13812345678，诊断有误",
            original_scores={"knowledge_score": 60.0},
            score_adjustments={"knowledge_score": 62.0},
        )
        evaluation = _make_evaluation()
        candidate = build_candidate(review, evaluation, hmac_key=_TEST_HMAC_KEY)

        assert "13812345678" not in candidate.feedback
        assert "张三" not in candidate.feedback

    def test_build_candidate_detects_sensitive_content(self):
        from evaluation.feedback_cases import build_candidate

        review = _make_review_record(
            feedback="患者张三，电话13812345678",
            original_scores={"knowledge_score": 60.0},
            score_adjustments={"knowledge_score": 62.0},
        )
        evaluation = _make_evaluation()
        candidate = build_candidate(review, evaluation, hmac_key=_TEST_HMAC_KEY)
        assert candidate.contains_sensitive_content is True

    def test_build_candidate_no_sensitive_content(self):
        from evaluation.feedback_cases import build_candidate

        review = _make_review_record(
            feedback="诊断依据不充分",
            original_scores={"knowledge_score": 60.0},
            score_adjustments={"knowledge_score": 62.0},
        )
        evaluation = _make_evaluation()
        candidate = build_candidate(review, evaluation, hmac_key=_TEST_HMAC_KEY)
        assert candidate.contains_sensitive_content is False


# ── 6. 无内容模式 vs 敏感模式 ────────────────────────────────────────────────


class TestExportModes:
    """无内容模式不含 conversation；敏感模式需要双确认"""

    def test_default_export_excludes_conversation(self):
        """默认导出模式不含 conversation_text 字段"""
        from evaluation.feedback_cases import build_candidate

        review = _make_review_record(
            feedback="诊断不充分",
            original_scores={"knowledge_score": 60.0},
            score_adjustments={"knowledge_score": 62.0},
        )
        evaluation = _make_evaluation()
        candidate = build_candidate(review, evaluation, hmac_key=_TEST_HMAC_KEY)

        data = candidate.model_dump()
        assert "conversation_text" not in data

    def test_sensitive_mode_requires_double_confirmation(self):
        """敏感模式没有双确认则拒绝"""
        from evaluation.feedback_cases import ExportConfig

        config = ExportConfig(
            hmac_key=_TEST_HMAC_KEY,
            include_content=True,
            acknowledge_sensitive_data=False,
        )
        assert config.can_export_sensitive() is False

    def test_sensitive_mode_with_double_confirmation(self):
        """同时提供两个标志才允许导出"""
        from evaluation.feedback_cases import ExportConfig

        config = ExportConfig(
            hmac_key=_TEST_HMAC_KEY,
            include_content=True,
            acknowledge_sensitive_data=True,
        )
        assert config.can_export_sensitive() is True

    def test_default_mode_allows_export(self):
        """默认模式（不含受限内容）总是允许"""
        from evaluation.feedback_cases import ExportConfig

        config = ExportConfig(
            hmac_key=_TEST_HMAC_KEY,
            include_content=False,
            acknowledge_sensitive_data=False,
        )
        assert config.can_export_sensitive() is True


# ── 7. CLI / 导出逻辑测试 ────────────────────────────────────────────────────


class TestExportLogic:
    """导出逻辑：空结果、原子输出、limit、缺少 HMAC key"""

    def test_missing_hmac_key_exits_nonzero(self):
        """缺少 FEEDBACK_EXPORT_HMAC_KEY 时 CLI 返回非零"""
        from scripts.export_review_feedback import async_main
        import argparse

        args = argparse.Namespace(
            output="/tmp/test.jsonl",
            since=None,
            until=None,
            limit=None,
            include_content=False,
            acknowledge_sensitive_data=False,
        )
        # 临时移除环境变量
        old = os.environ.pop("FEEDBACK_EXPORT_HMAC_KEY", None)
        try:
            result = asyncio_run(async_main(args))
            assert result == 2
        finally:
            if old is not None:
                os.environ["FEEDBACK_EXPORT_HMAC_KEY"] = old

    def test_sensitive_without_acknowledgment_rejected(self):
        """--include-content 没有 --acknowledge-sensitive-data 被拒绝"""
        from scripts.export_review_feedback import async_main
        import argparse

        args = argparse.Namespace(
            output="/tmp/test.jsonl",
            since=None,
            until=None,
            limit=None,
            include_content=True,
            acknowledge_sensitive_data=False,
        )
        os.environ["FEEDBACK_EXPORT_HMAC_KEY"] = _TEST_HMAC_KEY
        result = asyncio_run(async_main(args))
        assert result == 3

    def test_empty_result_writes_empty_file(self):
        """空结果正常退出并写空文件"""
        from scripts.export_review_feedback import async_main
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "candidates.jsonl")
            args = argparse.Namespace(
                output=output_path,
                since=None,
                until=None,
                limit=None,
                include_content=False,
                acknowledge_sensitive_data=False,
            )
            os.environ["FEEDBACK_EXPORT_HMAC_KEY"] = _TEST_HMAC_KEY

            with patch(
                "scripts.export_review_feedback.fetch_review_records",
                new=AsyncMock(return_value=[]),
            ):
                result = asyncio_run(async_main(args))

            assert result == 0
            assert Path(output_path).exists()

    def test_atomic_write_produces_valid_jsonl(self):
        """原子写输出有效 JSONL"""
        from scripts.export_review_feedback import async_main
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "candidates.jsonl")
            args = argparse.Namespace(
                output=output_path,
                since=None,
                until=None,
                limit=None,
                include_content=False,
                acknowledge_sensitive_data=False,
            )
            os.environ["FEEDBACK_EXPORT_HMAC_KEY"] = _TEST_HMAC_KEY

            review = _make_review_record(
                original_scores={"knowledge_score": 60.0},
                score_adjustments={"knowledge_score": 62.0},
            )
            evaluation = _make_evaluation()

            with patch(
                "scripts.export_review_feedback.fetch_review_records",
                new=AsyncMock(return_value=[(review, evaluation)]),
            ):
                result = asyncio_run(async_main(args))

            assert result == 0
            content = Path(output_path).read_text(encoding="utf-8").strip()
            assert content  # 非空
            parsed = json.loads(content.split("\n")[0])
            assert "candidate_id" in parsed
            assert "review_id" in parsed

    def test_limit_caps_output(self):
        """--limit 限制导出数量"""
        from scripts.export_review_feedback import async_main
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "candidates.jsonl")
            args = argparse.Namespace(
                output=output_path,
                since=None,
                until=None,
                limit=2,
                include_content=False,
                acknowledge_sensitive_data=False,
            )
            os.environ["FEEDBACK_EXPORT_HMAC_KEY"] = _TEST_HMAC_KEY

            pairs = []
            for i in range(5):
                review = _make_review_record(
                    review_id=f"rev-{i}",
                    evaluation_id=str(i),
                    original_scores={"knowledge_score": 60.0},
                    score_adjustments={"knowledge_score": 62.0},
                )
                evaluation = _make_evaluation(eval_id=i)
                pairs.append((review, evaluation))

            with patch(
                "scripts.export_review_feedback.fetch_review_records",
                new=AsyncMock(return_value=pairs),
            ):
                result = asyncio_run(async_main(args))

            assert result == 0
            content = Path(output_path).read_text(encoding="utf-8").strip()
            lines = [l for l in content.split("\n") if l.strip()]
            assert len(lines) == 2

    def test_db_error_returns_nonzero(self):
        """数据库错误返回非零退出码"""
        from scripts.export_review_feedback import async_main
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "candidates.jsonl")
            args = argparse.Namespace(
                output=output_path,
                since=None,
                until=None,
                limit=None,
                include_content=False,
                acknowledge_sensitive_data=False,
            )
            os.environ["FEEDBACK_EXPORT_HMAC_KEY"] = _TEST_HMAC_KEY

            with patch(
                "scripts.export_review_feedback.fetch_review_records",
                new=AsyncMock(side_effect=RuntimeError("DB connection failed")),
            ):
                result = asyncio_run(async_main(args))

            assert result == 4


# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def asyncio_run(coro):
    """兼容 pytest-asyncio 的简单 async runner"""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)
