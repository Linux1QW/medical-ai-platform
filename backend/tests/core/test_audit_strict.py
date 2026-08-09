"""Tests for audit.py strict mode

TDD Phase 1: These tests should FAIL until the implementation is complete.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Test: default mode flush failure logs safe warning ────────────────────────


@pytest.mark.asyncio
async def test_default_mode_flush_failure_logs_warning():
    """Default mode (strict=False) should log a safe warning on flush failure."""
    from app.core.audit import record_audit_log

    mock_db = AsyncMock()
    mock_db.flush = AsyncMock(side_effect=RuntimeError("connection lost to mysql://root:pw@host/db"))
    mock_db.add = MagicMock()

    with patch("app.core.audit.logger") as mock_logger:
        await record_audit_log(
            db=mock_db,
            user_id=1,
            action="admin_action",
            detail="patient 张三 的诊断结果",
        )

        # Should log a warning
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        # Warning should NOT contain the raw detail (PII)
        assert "张三" not in warning_msg
        # Warning should NOT contain the DB connection string
        assert "mysql://" not in warning_msg
        assert "root:pw" not in warning_msg


# ── Test: strict=True re-raises exception ─────────────────────────────────────


@pytest.mark.asyncio
async def test_strict_mode_reraises_flush_error():
    """strict=True should re-raise the flush error for upper-level rollback."""
    from app.core.audit import record_audit_log

    mock_db = AsyncMock()
    original_error = RuntimeError("flush failed: deadlock detected")
    mock_db.flush = AsyncMock(side_effect=original_error)
    mock_db.add = MagicMock()

    with pytest.raises(RuntimeError, match="deadlock detected"):
        await record_audit_log(
            db=mock_db,
            user_id=1,
            action="admin_action",
            detail="some detail",
            strict=True,
        )


# ── Test: warning message does not contain PII or connection strings ─────────


@pytest.mark.asyncio
async def test_warning_sanitized_no_pii():
    """The warning message should not contain patient text or DB connection details."""
    from app.core.audit import record_audit_log

    mock_db = AsyncMock()
    mock_db.flush = AsyncMock(
        side_effect=RuntimeError("OperationalError: mysql+pymysql://admin:secret@db:3306/medical")
    )
    mock_db.add = MagicMock()

    with patch("app.core.audit.logger") as mock_logger:
        await record_audit_log(
            db=mock_db,
            user_id=1,
            action="create_consultation",
            detail="患者李某胸闷三天",
        )

        warning_msg = mock_logger.warning.call_args[0][0]
        # Should not contain patient name
        assert "李某" not in warning_msg
        # Should not contain connection string
        assert "pymysql" not in warning_msg
        assert "secret" not in warning_msg
        assert "admin" not in warning_msg


# ── Test: normal successful path works ───────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_log_success_path():
    """Normal successful audit log write should work without issues."""
    from app.core.audit import record_audit_log

    mock_db = AsyncMock()
    mock_db.flush = AsyncMock()
    mock_db.add = MagicMock()

    # Should not raise
    await record_audit_log(
        db=mock_db,
        user_id=1,
        action="login",
        detail="User logged in",
    )

    mock_db.add.assert_called_once()
    mock_db.flush.assert_called_once()


# ── Test: db=None is a no-op ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_log_db_none_noop():
    """When db is None, record_audit_log should be a no-op."""
    from app.core.audit import record_audit_log

    # Should not raise
    await record_audit_log(
        db=None,
        user_id=1,
        action="login",
    )


# ── Test: strict=True with no error succeeds normally ────────────────────────


@pytest.mark.asyncio
async def test_strict_mode_success_path():
    """strict=True with successful flush should work normally."""
    from app.core.audit import record_audit_log

    mock_db = AsyncMock()
    mock_db.flush = AsyncMock()
    mock_db.add = MagicMock()

    await record_audit_log(
        db=mock_db,
        user_id=1,
        action="admin_action",
        detail="recovery action",
        strict=True,
    )

    mock_db.add.assert_called_once()
    mock_db.flush.assert_called_once()
