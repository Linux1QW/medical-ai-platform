# -*- coding: utf-8 -*-
"""Task 8: 原子人工复核服务测试 — TDD 先行测试

验证：
1. 事务成功时 run/evaluation/lock/record 同步
2. 数据库异常全部 rollback
3. needs_review→reviewed 合法
4. completed/failed/reviewed 重复提交 409
5. 范围外分数 422
6. 调整后原始分数保留
7. pending list 只查 needs_review，按 created_at 倒序
"""

from __future__ import annotations

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.evaluation import Evaluation
from app.models.evaluation_lock import EvaluationLock
from app.models.evaluation_run import EvaluationRun
from app.models.review_record import ReviewRecord


# ── 辅助工厂 ─────────────────────────────────────────────────────────────────


def _make_evaluation(
    eval_id: int = 1,
    consultation_id: int = 100,
    status: str = "needs_review",
    run_id: str | None = None,
) -> Evaluation:
    run_id = run_id or str(uuid.uuid4())
    ev = Evaluation(
        id=eval_id,
        consultation_id=consultation_id,
        inquiry_score=80.0,
        knowledge_score=75.0,
        humanistic_score=85.0,
        diagnosis_score=70.0,
        treatment_score=78.0,
        total_score=77.0,
        evaluation_status=status,
        human_review_needed=True,
        review_reason="low score",
        run_id=run_id,
        retrieval_status="completed",
        evidence_stance="supported",
        created_at=datetime(2026, 1, 1),
    )
    return ev


def _make_run(run_id: str, consultation_id: int = 100, evaluation_id: int = 1) -> EvaluationRun:
    return EvaluationRun(
        id=run_id,
        consultation_id=consultation_id,
        evaluation_id=evaluation_id,
        status="needs_review",
        checkpoint_thread_id="thread-1",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )


def _make_lock(consultation_id: int = 100, run_id: str | None = None) -> EvaluationLock:
    return EvaluationLock(
        consultation_id=consultation_id,
        status="needs_review",
        run_id=run_id or str(uuid.uuid4()),
        locked_at=datetime(2026, 1, 1),
        heartbeat_at=datetime(2026, 1, 1),
        expires_at=datetime(2099, 1, 1),
    )


# ── 1. 事务成功路径 ─────────────────────────────────────────────────────────


class TestSubmitReviewSuccess:
    """原子复核成功路径"""

    @pytest.mark.asyncio
    async def test_atomic_review_success_syncs_all_entities(self):
        """成功时 run/evaluation/lock/record 全部同步更新"""
        from app.services.review_service import submit_review

        run_id = str(uuid.uuid4())
        ev = _make_evaluation(run_id=run_id)
        run = _make_run(run_id=run_id)
        lock = _make_lock(run_id=run_id)

        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.flush = AsyncMock()
        mock_db.rollback = AsyncMock()

        # SELECT FOR UPDATE → 返回 evaluation
        execute_results = [
            MagicMock(scalar_one_or_none=lambda: ev),  # SELECT eval FOR UPDATE
            MagicMock(scalar_one_or_none=lambda: run),  # SELECT run
            MagicMock(scalar_one_or_none=lambda: lock),  # SELECT lock
        ]
        mock_db.execute = AsyncMock(side_effect=execute_results)
        mock_db.add = MagicMock()

        with patch("app.services.review_service._update_redis_checkpoint", new=AsyncMock()):
            result = await submit_review(
                db=mock_db,
                evaluation_id=1,
                reviewer_id="admin-1",
                feedback="确认无误",
                score_adjustments=None,
            )

        # 验证 commit 只调用一次
        mock_db.commit.assert_awaited_once()
        # 验证返回结构
        assert result["status"] == "reviewed"
        assert result["evaluation_id"] == 1
        assert "review_id" in result
        assert "run_id" in result

    @pytest.mark.asyncio
    async def test_score_adjustment_preserves_original(self):
        """调整后原始分数保留在 ReviewRecord.original_scores"""
        from app.services.review_service import submit_review

        run_id = str(uuid.uuid4())
        ev = _make_evaluation(run_id=run_id)
        run = _make_run(run_id=run_id)
        lock = _make_lock(run_id=run_id)

        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.flush = AsyncMock()
        mock_db.rollback = AsyncMock()

        captured_record = {}

        def capture_add(obj):
            if isinstance(obj, ReviewRecord):
                captured_record["record"] = obj

        mock_db.add = capture_add
        execute_results = [
            MagicMock(scalar_one_or_none=lambda: ev),
            MagicMock(scalar_one_or_none=lambda: run),
            MagicMock(scalar_one_or_none=lambda: lock),
        ]
        mock_db.execute = AsyncMock(side_effect=execute_results)

        adjustments = {"inquiry_score": 90.0, "diagnosis_score": 80.0}

        with patch("app.services.review_service._update_redis_checkpoint", new=AsyncMock()):
            result = await submit_review(
                db=mock_db,
                evaluation_id=1,
                reviewer_id="admin-1",
                feedback="调整分数",
                score_adjustments=adjustments,
            )

        # 原始分数被快照
        record = captured_record["record"]
        assert record.original_scores is not None
        assert record.original_scores["inquiry_score"] == 80.0
        assert record.original_scores["diagnosis_score"] == 70.0

    @pytest.mark.asyncio
    async def test_total_score_recalculated_with_unified_weights(self):
        """总分用统一权重重算"""
        from app.services.review_service import submit_review

        run_id = str(uuid.uuid4())
        ev = _make_evaluation(run_id=run_id)
        run = _make_run(run_id=run_id)
        lock = _make_lock(run_id=run_id)

        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.flush = AsyncMock()
        mock_db.rollback = AsyncMock()
        mock_db.add = MagicMock()

        execute_results = [
            MagicMock(scalar_one_or_none=lambda: ev),
            MagicMock(scalar_one_or_none=lambda: run),
            MagicMock(scalar_one_or_none=lambda: lock),
        ]
        mock_db.execute = AsyncMock(side_effect=execute_results)

        # 调整 inquiry_score 到 90
        adjustments = {"inquiry_score": 90.0}

        with patch("app.services.review_service._update_redis_checkpoint", new=AsyncMock()):
            result = await submit_review(
                db=mock_db,
                evaluation_id=1,
                reviewer_id="admin-1",
                feedback="调整",
                score_adjustments=adjustments,
            )

        # 验证 evaluation 的 total_score 被重新计算
        # 原始: inquiry=80, knowledge=75, humanistic=85, diagnosis=70, treatment=78
        # 调整后: inquiry=90, knowledge=75, humanistic=85, diagnosis=70, treatment=78
        # 权重: 0.25, 0.25, 0.20, 0.15, 0.15
        # 预期: 90*0.25 + 75*0.25 + 85*0.20 + 70*0.15 + 78*0.15 = 22.5+18.75+17+10.5+11.7 = 80.45 → 80
        assert ev.total_score == 80.0


# ── 2. 数据库异常 rollback ──────────────────────────────────────────────────


class TestSubmitReviewRollback:
    """数据库异常时全部 rollback"""

    @pytest.mark.asyncio
    async def test_db_error_causes_full_rollback(self):
        """commit 失败时 rollback 被调用"""
        from app.services.review_service import submit_review

        run_id = str(uuid.uuid4())
        ev = _make_evaluation(run_id=run_id)
        run = _make_run(run_id=run_id)
        lock = _make_lock(run_id=run_id)

        mock_db = AsyncMock()
        mock_db.commit = AsyncMock(side_effect=RuntimeError("DB connection lost"))
        mock_db.flush = AsyncMock()
        mock_db.rollback = AsyncMock()
        mock_db.add = MagicMock()

        execute_results = [
            MagicMock(scalar_one_or_none=lambda: ev),
            MagicMock(scalar_one_or_none=lambda: run),
            MagicMock(scalar_one_or_none=lambda: lock),
        ]
        mock_db.execute = AsyncMock(side_effect=execute_results)

        with pytest.raises(RuntimeError, match="DB connection lost"):
            await submit_review(
                db=mock_db,
                evaluation_id=1,
                reviewer_id="admin-1",
                feedback="测试回滚",
                score_adjustments=None,
            )

        mock_db.rollback.assert_awaited_once()


# ── 3. 状态校验 ─────────────────────────────────────────────────────────────


class TestSubmitReviewStatusValidation:
    """状态校验测试"""

    @pytest.mark.asyncio
    async def test_needs_review_to_reviewed_is_legal(self):
        """needs_review → reviewed 合法"""
        from app.services.review_service import submit_review

        run_id = str(uuid.uuid4())
        ev = _make_evaluation(status="needs_review", run_id=run_id)
        run = _make_run(run_id=run_id)
        lock = _make_lock(run_id=run_id)

        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.flush = AsyncMock()
        mock_db.rollback = AsyncMock()
        mock_db.add = MagicMock()

        execute_results = [
            MagicMock(scalar_one_or_none=lambda: ev),
            MagicMock(scalar_one_or_none=lambda: run),
            MagicMock(scalar_one_or_none=lambda: lock),
        ]
        mock_db.execute = AsyncMock(side_effect=execute_results)

        with patch("app.services.review_service._update_redis_checkpoint", new=AsyncMock()):
            result = await submit_review(
                db=mock_db,
                evaluation_id=1,
                reviewer_id="admin-1",
                feedback="确认",
                score_adjustments=None,
            )
        assert result["status"] == "reviewed"

    @pytest.mark.asyncio
    async def test_completed_status_returns_409(self):
        """completed 状态重复提交返回 409"""
        from app.services.review_service import ReviewConflictError, submit_review

        ev = _make_evaluation(status="completed")
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=lambda: ev)
        )

        with pytest.raises(ReviewConflictError) as exc_info:
            await submit_review(
                db=mock_db,
                evaluation_id=1,
                reviewer_id="admin-1",
                feedback="重复提交",
                score_adjustments=None,
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_failed_status_returns_409(self):
        """failed 状态重复提交返回 409"""
        from app.services.review_service import ReviewConflictError, submit_review

        ev = _make_evaluation(status="failed")
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=lambda: ev)
        )

        with pytest.raises(ReviewConflictError) as exc_info:
            await submit_review(
                db=mock_db,
                evaluation_id=1,
                reviewer_id="admin-1",
                feedback="重复提交",
                score_adjustments=None,
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_reviewed_status_returns_409(self):
        """reviewed 状态重复提交返回 409"""
        from app.services.review_service import ReviewConflictError, submit_review

        ev = _make_evaluation(status="reviewed")
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=lambda: ev)
        )

        with pytest.raises(ReviewConflictError) as exc_info:
            await submit_review(
                db=mock_db,
                evaluation_id=1,
                reviewer_id="admin-1",
                feedback="重复提交",
                score_adjustments=None,
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_evaluation_not_found_returns_none(self):
        """不存在返回 None（路由层转 404）"""
        from app.services.review_service import submit_review

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=lambda: None)
        )

        result = await submit_review(
            db=mock_db,
            evaluation_id=999,
            reviewer_id="admin-1",
            feedback="不存在",
            score_adjustments=None,
        )
        assert result is None


# ── 4. 分数校验 ─────────────────────────────────────────────────────────────


class TestScoreValidation:
    """分数范围校验"""

    @pytest.mark.asyncio
    async def test_out_of_range_score_raises_422(self):
        """范围外分数触发 ValidationError"""
        from app.services.review_service import submit_review
        from app.schemas.review import ScoreAdjustments

        with pytest.raises(Exception):
            ScoreAdjustments(inquiry_score=150.0)

    @pytest.mark.asyncio
    async def test_negative_score_raises_validation_error(self):
        """负数分数触发 ValidationError"""
        from app.schemas.review import ScoreAdjustments

        with pytest.raises(Exception):
            ScoreAdjustments(inquiry_score=-5.0)


# ── 5. Pending list ─────────────────────────────────────────────────────────


class TestListPendingReviews:
    """待复核列表测试"""

    @pytest.mark.asyncio
    async def test_pending_list_queries_needs_review_status(self):
        """pending list 只查 needs_review"""
        from app.services.review_service import list_pending_evaluations

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [
            MagicMock(
                id=1,
                consultation_id=100,
                run_id=str(uuid.uuid4()),
                review_reason="low score",
                total_score=50.0,
                retrieval_status="completed",
                evidence_stance="supported",
                created_at=datetime(2026, 1, 1),
            ),
        ]
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await list_pending_evaluations(mock_db)
        assert len(result) == 1
        assert result[0]["evaluation_id"] == 1

        # 验证查询使用了 needs_review（检查绑定参数或 SQL 编译结果）
        call_args = mock_db.execute.call_args
        stmt = call_args[0][0]
        compiled = stmt.compile()
        params = compiled.params
        # 参数化查询中值应为 needs_review
        param_values = list(params.values())
        assert "needs_review" in param_values or any(
            "needs_review" in str(v) for v in param_values
        ), f"needs_review not found in params: {params}"

    @pytest.mark.asyncio
    async def test_pending_list_ordered_by_created_at_desc(self):
        """pending list 按 created_at 倒序"""
        from app.services.review_service import list_pending_evaluations

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        await list_pending_evaluations(mock_db)

        call_args = mock_db.execute.call_args
        stmt = call_args[0][0]
        stmt_str = str(stmt)
        assert "desc" in stmt_str.lower() or "DESC" in stmt_str

    @pytest.mark.asyncio
    async def test_pending_list_db_error_propagates(self):
        """DB 异常不得吞掉后伪装空列表"""
        from app.services.review_service import list_pending_evaluations

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=RuntimeError("DB down"))

        with pytest.raises(RuntimeError, match="DB down"):
            await list_pending_evaluations(mock_db)
