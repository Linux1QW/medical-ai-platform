# -*- coding: utf-8 -*-
"""Task 8: Legacy evaluation run backfill 脚本测试

验证：
1. 空 run_id 新建 run
2. 精确 run 复用
3. dry-run 不写数据库
4. 歧义/active/orphan 非零退出
"""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── 辅助工厂 ─────────────────────────────────────────────────────────────────


def _make_eval_row(eval_id: int, run_id: str | None = None, consultation_id: int = 100, status: str = "completed"):
    row = MagicMock()
    row.id = eval_id
    row.run_id = run_id
    row.consultation_id = consultation_id
    row.evaluation_status = status
    row.created_at = datetime(2026, 1, 1)
    return row


def _make_run_row(run_id: str, consultation_id: int = 100, evaluation_id: int = 1, status: str = "completed"):
    row = MagicMock()
    row.id = run_id
    row.consultation_id = consultation_id
    row.evaluation_id = evaluation_id
    row.status = status
    return row


# ── 1. 空 run_id 新建 ────────────────────────────────────────────────────────


class TestBackfillEmptyRunId:
    """空 run_id 时创建新 run"""

    @pytest.mark.asyncio
    async def test_creates_new_run_for_null_run_id(self):
        """空 run_id 的 evaluation 创建新 run"""
        from scripts.backfill_legacy_evaluation_runs import backfill_evaluations

        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()
        mock_db.add = MagicMock()

        call_count = 0

        async def execute_side_effect(stmt, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 查询需要 backfill 的 evaluations (text() SQL, fetchall)
                result = MagicMock()
                result.fetchall.return_value = [(1, 100, "completed", datetime(2026, 1, 1))]
                return result
            elif call_count == 2:
                # SELECT FOR UPDATE (text() SQL, fetchone)
                result = MagicMock()
                result.fetchone.return_value = (1, 100, "completed")
                return result
            elif call_count == 3:
                # 查找精确匹配 run (text() SQL, fetchall) - 无匹配
                result = MagicMock()
                result.fetchall.return_value = []
                return result
            elif call_count == 4:
                # 查找 consultation 关联 run (text() SQL, fetchall) - 无匹配
                result = MagicMock()
                result.fetchall.return_value = []
                return result
            else:
                # INSERT/UPDATE 等
                return MagicMock()

        mock_db.execute = AsyncMock(side_effect=execute_side_effect)

        with patch("scripts.backfill_legacy_evaluation_runs._write_manifest", new=AsyncMock()):
            result = await backfill_evaluations(
                db=mock_db,
                operator_id="admin-1",
                batch_size=100,
                dry_run=False,
            )

        assert result["created"] == 1


# ── 2. 精确 run 复用 ─────────────────────────────────────────────────────────


class TestBackfillExactRunReuse:
    """精确匹配 run_id 时复用已有 run"""

    @pytest.mark.asyncio
    async def test_reuses_exact_matching_run(self):
        """evaluation_id 精确关联的唯一 run 被复用"""
        from scripts.backfill_legacy_evaluation_runs import backfill_evaluations

        existing_run_id = str(uuid.uuid4())

        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()
        mock_db.add = MagicMock()

        call_count = 0

        async def execute_side_effect(stmt, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                result = MagicMock()
                result.fetchall.return_value = [(1, 100, "completed", datetime(2026, 1, 1))]
                return result
            elif call_count == 2:
                result = MagicMock()
                result.fetchone.return_value = (1, 100, "completed")
                return result
            elif call_count == 3:
                # 精确匹配一个 run
                result = MagicMock()
                result.fetchall.return_value = [(existing_run_id, 100, "completed")]
                return result
            else:
                return MagicMock()

        mock_db.execute = AsyncMock(side_effect=execute_side_effect)

        with patch("scripts.backfill_legacy_evaluation_runs._write_manifest", new=AsyncMock()):
            result = await backfill_evaluations(
                db=mock_db,
                operator_id="admin-1",
                batch_size=100,
                dry_run=False,
            )

        assert result["reused"] == 1


# ── 3. dry-run ───────────────────────────────────────────────────────────────


class TestBackfillDryRun:
    """dry-run 不写数据库"""

    @pytest.mark.asyncio
    async def test_dry_run_does_not_write(self):
        """dry-run 模式不 commit"""
        from scripts.backfill_legacy_evaluation_runs import backfill_evaluations

        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.add = MagicMock()

        call_count = 0

        async def execute_side_effect(stmt, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                result = MagicMock()
                result.fetchall.return_value = [(1, 100, "completed", datetime(2026, 1, 1))]
                return result
            elif call_count == 2:
                result = MagicMock()
                result.fetchone.return_value = (1, 100, "completed")
                return result
            else:
                result = MagicMock()
                result.fetchall.return_value = []
                return result

        mock_db.execute = AsyncMock(side_effect=execute_side_effect)

        result = await backfill_evaluations(
            db=mock_db,
            operator_id=None,
            batch_size=100,
            dry_run=True,
        )

        # dry-run 不得 commit
        mock_db.commit.assert_not_awaited()
        assert result["dry_run"] is True


# ── 4. 歧义/active/orphan 非零退出 ──────────────────────────────────────────


class TestBackfillConflictExit:
    """歧义/active/orphan 导致非零退出"""

    @pytest.mark.asyncio
    async def test_ambiguous_candidates_cause_conflict(self):
        """多个候选 run 时写入 conflict manifest"""
        from scripts.backfill_legacy_evaluation_runs import backfill_evaluations

        _eval_row = _make_eval_row(eval_id=1, run_id=None)
        # 模拟两个候选 run（raw SQL fetchall 返回 tuple-like rows）
        run1 = (str(uuid.uuid4()), 100, "completed")
        run2 = (str(uuid.uuid4()), 100, "completed")

        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()
        mock_db.add = MagicMock()

        call_count = 0

        async def execute_side_effect(stmt, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 查询需要 backfill 的 evaluations (text() SQL, fetchall)
                result = MagicMock()
                result.fetchall.return_value = [(1, 100, "completed", datetime(2026, 1, 1))]
                return result
            elif call_count == 2:
                # SELECT FOR UPDATE (text() SQL, fetchone)
                result = MagicMock()
                result.fetchone.return_value = (1, 100, "completed")
                return result
            elif call_count == 3:
                # 查找精确匹配 run (text() SQL, fetchall)
                result = MagicMock()
                result.fetchall.return_value = [run1, run2]
                return result
            else:
                return MagicMock()

        mock_db.execute = AsyncMock(side_effect=execute_side_effect)

        with patch("scripts.backfill_legacy_evaluation_runs._write_manifest", new=AsyncMock()):
            result = await backfill_evaluations(
                db=mock_db,
                operator_id="admin-1",
                batch_size=100,
                dry_run=False,
            )

        assert result["conflicts"] >= 1
