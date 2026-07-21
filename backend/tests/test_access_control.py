"""资源访问权限校验测试"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.core.access import require_consultation_access


def _user(user_id: int, role: str = "doctor"):
    user = MagicMock()
    user.id = user_id
    user.role = role
    return user


def _consultation(consultation_id: int, doctor_id: int):
    consultation = MagicMock()
    consultation.id = consultation_id
    consultation.doctor_id = doctor_id
    return consultation


@pytest.mark.asyncio
async def test_owner_can_access_consultation(monkeypatch):
    db = AsyncMock()
    consultation = _consultation(1, 10)
    monkeypatch.setattr(
        "app.core.access.get_consultation",
        AsyncMock(return_value=consultation),
    )
    result = await require_consultation_access(db, 1, _user(10))
    assert result is consultation


@pytest.mark.asyncio
async def test_other_doctor_cannot_access_consultation(monkeypatch):
    db = AsyncMock()
    consultation = _consultation(1, 10)
    monkeypatch.setattr(
        "app.core.access.get_consultation",
        AsyncMock(return_value=consultation),
    )
    with pytest.raises(HTTPException) as exc_info:
        await require_consultation_access(db, 1, _user(99))
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_admin_can_access_any_consultation(monkeypatch):
    db = AsyncMock()
    consultation = _consultation(1, 10)
    monkeypatch.setattr(
        "app.core.access.get_consultation",
        AsyncMock(return_value=consultation),
    )
    result = await require_consultation_access(db, 1, _user(99, role="admin"))
    assert result is consultation


@pytest.mark.asyncio
async def test_missing_consultation_returns_404(monkeypatch):
    db = AsyncMock()
    monkeypatch.setattr(
        "app.core.access.get_consultation",
        AsyncMock(return_value=None),
    )
    with pytest.raises(HTTPException) as exc_info:
        await require_consultation_access(db, 999, _user(10))
    assert exc_info.value.status_code == 404
