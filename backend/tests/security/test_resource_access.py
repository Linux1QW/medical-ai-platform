"""资源级访问权限测试 — require_evaluation_access / require_evaluation_run_access"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.core.access import require_evaluation_access, require_evaluation_run_access


# ── helpers ──────────────────────────────────────────────────────────────────

def _user(user_id: int, role: str = "doctor"):
    user = MagicMock()
    user.id = user_id
    user.role = role
    return user


def _evaluation(evaluation_id: int, consultation_id: int):
    ev = MagicMock()
    ev.id = evaluation_id
    ev.consultation_id = consultation_id
    return ev


def _run(run_id: str, consultation_id: int):
    r = MagicMock()
    r.id = run_id
    r.consultation_id = consultation_id
    return r


def _consultation(consultation_id: int, doctor_id: int):
    c = MagicMock()
    c.id = consultation_id
    c.doctor_id = doctor_id
    return c


# ── require_evaluation_access ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_eval_owner_can_access(monkeypatch):
    """doctor A 访问自己的 evaluation → 200"""
    db = AsyncMock()
    ev = _evaluation(1, consultation_id=100)
    consultation = _consultation(100, doctor_id=10)
    monkeypatch.setattr(
        "app.core.access._get_evaluation",
        AsyncMock(return_value=ev),
    )
    monkeypatch.setattr(
        "app.core.access.get_consultation",
        AsyncMock(return_value=consultation),
    )
    result = await require_evaluation_access(db, 1, _user(10))
    assert result is ev


@pytest.mark.asyncio
async def test_eval_other_doctor_gets_404(monkeypatch):
    """doctor B 访问 doctor A 的 evaluation → 404（与随机 ID 相同）"""
    db = AsyncMock()
    ev = _evaluation(1, consultation_id=100)
    consultation = _consultation(100, doctor_id=10)
    monkeypatch.setattr(
        "app.core.access._get_evaluation",
        AsyncMock(return_value=ev),
    )
    monkeypatch.setattr(
        "app.core.access.get_consultation",
        AsyncMock(return_value=consultation),
    )
    with pytest.raises(HTTPException) as exc_info:
        await require_evaluation_access(db, 1, _user(99))
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_eval_admin_can_access_any(monkeypatch):
    """admin 访问任意 evaluation → 200"""
    db = AsyncMock()
    ev = _evaluation(1, consultation_id=100)
    consultation = _consultation(100, doctor_id=10)
    monkeypatch.setattr(
        "app.core.access._get_evaluation",
        AsyncMock(return_value=ev),
    )
    monkeypatch.setattr(
        "app.core.access.get_consultation",
        AsyncMock(return_value=consultation),
    )
    result = await require_evaluation_access(db, 1, _user(99, role="admin"))
    assert result is ev


@pytest.mark.asyncio
async def test_eval_not_found_returns_404(monkeypatch):
    """不存在的 evaluation → 404"""
    db = AsyncMock()
    monkeypatch.setattr(
        "app.core.access._get_evaluation",
        AsyncMock(return_value=None),
    )
    with pytest.raises(HTTPException) as exc_info:
        await require_evaluation_access(db, 999, _user(10))
    assert exc_info.value.status_code == 404


# ── require_evaluation_run_access ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_owner_can_access(monkeypatch):
    """doctor A 访问自己的 run → 200"""
    db = AsyncMock()
    run = _run("run-uuid-1", consultation_id=100)
    consultation = _consultation(100, doctor_id=10)
    monkeypatch.setattr(
        "app.core.access._get_evaluation_run",
        AsyncMock(return_value=run),
    )
    monkeypatch.setattr(
        "app.core.access.get_consultation",
        AsyncMock(return_value=consultation),
    )
    result = await require_evaluation_run_access(db, "run-uuid-1", _user(10))
    assert result is run


@pytest.mark.asyncio
async def test_run_other_doctor_gets_404(monkeypatch):
    """doctor B 访问 doctor A 的 run → 404"""
    db = AsyncMock()
    run = _run("run-uuid-1", consultation_id=100)
    consultation = _consultation(100, doctor_id=10)
    monkeypatch.setattr(
        "app.core.access._get_evaluation_run",
        AsyncMock(return_value=run),
    )
    monkeypatch.setattr(
        "app.core.access.get_consultation",
        AsyncMock(return_value=consultation),
    )
    with pytest.raises(HTTPException) as exc_info:
        await require_evaluation_run_access(db, "run-uuid-1", _user(99))
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_run_admin_can_access_any(monkeypatch):
    """admin 访问任意 run → 200"""
    db = AsyncMock()
    run = _run("run-uuid-1", consultation_id=100)
    consultation = _consultation(100, doctor_id=10)
    monkeypatch.setattr(
        "app.core.access._get_evaluation_run",
        AsyncMock(return_value=run),
    )
    monkeypatch.setattr(
        "app.core.access.get_consultation",
        AsyncMock(return_value=consultation),
    )
    result = await require_evaluation_run_access(db, "run-uuid-1", _user(99, role="admin"))
    assert result is run


@pytest.mark.asyncio
async def test_run_not_found_returns_404(monkeypatch):
    """不存在的 run → 404"""
    db = AsyncMock()
    monkeypatch.setattr(
        "app.core.access._get_evaluation_run",
        AsyncMock(return_value=None),
    )
    with pytest.raises(HTTPException) as exc_info:
        await require_evaluation_run_access(db, "nonexistent", _user(10))
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_run_consultation_not_found_returns_404(monkeypatch):
    """run 存在但关联的 consultation 不存在 → 404"""
    db = AsyncMock()
    run = _run("run-uuid-1", consultation_id=999)
    monkeypatch.setattr(
        "app.core.access._get_evaluation_run",
        AsyncMock(return_value=run),
    )
    monkeypatch.setattr(
        "app.core.access.get_consultation",
        AsyncMock(return_value=None),
    )
    with pytest.raises(HTTPException) as exc_info:
        await require_evaluation_run_access(db, "run-uuid-1", _user(10))
    assert exc_info.value.status_code == 404
