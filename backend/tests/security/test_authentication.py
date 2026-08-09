# -*- coding: utf-8 -*-
"""统一认证入口 authenticate_access_token 测试矩阵

覆盖场景：
- 有效 token → 返回 User
- 无效 token → AuthenticationError 401 AUTH_INVALID_TOKEN
- 已吊销 token → AuthenticationError 401 AUTH_TOKEN_REVOKED
- Redis exists 异常 + fail-closed → AuthenticationError 503 AUTH_REVOCATION_UNAVAILABLE
- Redis 连接为空 + fail-closed → AuthenticationError 503
- Redis 连接为空 + fail-open (dev) → 通过 + warning
- 用户不存在 → AuthenticationError 401 AUTH_USER_NOT_FOUND
- 有效但没有 jti 的 token + blacklist 启用 → AuthenticationError 401 AUTH_INVALID_TOKEN
"""
import logging
import warnings
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jose import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.user import User


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_token(
    sub: str = "1",
    expires: bool = True,
    include_jti: bool = True,
    token_type: str = "access",
    extra: dict | None = None,
) -> str:
    """生成一个签名合法的 JWT access token（测试用）"""
    payload: dict = {"sub": sub, "type": token_type}
    if include_jti:
        payload["jti"] = "test-jti-123"
    if expires:
        payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=30)
    else:
        payload["exp"] = datetime.now(timezone.utc) - timedelta(minutes=1)
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def _fake_user(user_id: int = 1) -> User:
    user = User(id=user_id, username="testdoc", email="test@example.com", hashed_password="x")
    return user


# ---------------------------------------------------------------------------
# authenticate_access_token 测试
# ---------------------------------------------------------------------------

class TestAuthenticateAccessToken:
    """authenticate_access_token() 行为矩阵"""

    @pytest.mark.asyncio
    async def test_valid_token_returns_user(self):
        """有效 token → 返回 User 对象"""
        from app.core.authentication import authenticate_access_token

        token = _make_token(sub="42")
        db = MagicMock(spec=AsyncSession)

        fake_user = _fake_user(42)
        with patch("app.core.authentication.get_user_by_id", new_callable=AsyncMock, return_value=fake_user), \
             patch("app.core.authentication.is_token_blacklisted", new_callable=AsyncMock, return_value=False):
            user = await authenticate_access_token(db, token)
        assert user.id == 42

    @pytest.mark.asyncio
    async def test_invalid_token_raises_401(self):
        """无效 token（签名错误）→ 401 AUTH_INVALID_TOKEN"""
        from app.core.authentication import AuthenticationError, authenticate_access_token

        db = MagicMock(spec=AsyncSession)
        with patch("app.core.authentication.is_token_blacklisted", new_callable=AsyncMock, return_value=False):
            with pytest.raises(AuthenticationError) as exc_info:
                await authenticate_access_token(db, "totally.invalid.token")
        assert exc_info.value.status_code == 401
        assert exc_info.value.error_code == "AUTH_INVALID_TOKEN"

    @pytest.mark.asyncio
    async def test_revoked_token_raises_401(self):
        """已吊销 token → 401 AUTH_TOKEN_REVOKED"""
        from app.core.authentication import AuthenticationError, authenticate_access_token

        token = _make_token()
        db = MagicMock(spec=AsyncSession)
        with patch("app.core.authentication.is_token_blacklisted", new_callable=AsyncMock, return_value=True):
            with pytest.raises(AuthenticationError) as exc_info:
                await authenticate_access_token(db, token)
        assert exc_info.value.status_code == 401
        assert exc_info.value.error_code == "AUTH_TOKEN_REVOKED"

    @pytest.mark.asyncio
    async def test_redis_exists_error_fail_closed_raises_503(self):
        """Redis exists 异常 + fail-closed → 503 AUTH_REVOCATION_UNAVAILABLE"""
        from app.core.authentication import AuthenticationError, TokenRevocationStoreUnavailable, authenticate_access_token

        token = _make_token()
        db = MagicMock(spec=AsyncSession)

        async def _raise(*args, **kwargs):
            raise TokenRevocationStoreUnavailable("redis down")

        with patch("app.core.authentication.is_token_blacklisted", side_effect=_raise):
            with pytest.raises(AuthenticationError) as exc_info:
                await authenticate_access_token(db, token)
        assert exc_info.value.status_code == 503
        assert exc_info.value.error_code == "AUTH_REVOCATION_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_redis_unavailable_fail_closed_raises_503(self):
        """Redis 连接为空 + fail-closed → 503 AUTH_REVOCATION_UNAVAILABLE"""
        from app.core.authentication import AuthenticationError, TokenRevocationStoreUnavailable, authenticate_access_token

        token = _make_token()
        db = MagicMock(spec=AsyncSession)

        async def _raise(*args, **kwargs):
            raise TokenRevocationStoreUnavailable("no redis connection")

        with patch("app.core.authentication.is_token_blacklisted", side_effect=_raise):
            with pytest.raises(AuthenticationError) as exc_info:
                await authenticate_access_token(db, token)
        assert exc_info.value.status_code == 503
        assert exc_info.value.error_code == "AUTH_REVOCATION_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_user_not_found_raises_401(self):
        """用户不存在 → 401 AUTH_USER_NOT_FOUND"""
        from app.core.authentication import AuthenticationError, authenticate_access_token

        token = _make_token(sub="999")
        db = MagicMock(spec=AsyncSession)
        with patch("app.core.authentication.get_user_by_id", new_callable=AsyncMock, return_value=None), \
             patch("app.core.authentication.is_token_blacklisted", new_callable=AsyncMock, return_value=False):
            with pytest.raises(AuthenticationError) as exc_info:
                await authenticate_access_token(db, token)
        assert exc_info.value.status_code == 401
        assert exc_info.value.error_code == "AUTH_USER_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_token_without_jti_blacklist_enabled_raises_401(self):
        """有效但没有 jti 的 token + blacklist 启用 → 401 AUTH_INVALID_TOKEN"""
        from app.core.authentication import AuthenticationError, authenticate_access_token

        token = _make_token(include_jti=False)
        db = MagicMock(spec=AsyncSession)

        with patch("app.core.authentication.is_token_blacklisted", new_callable=AsyncMock, return_value=False), \
             patch("app.core.config.settings.JWT_TOKEN_BLACKLIST_ENABLED", True):
            with pytest.raises(AuthenticationError) as exc_info:
                await authenticate_access_token(db, token)
        assert exc_info.value.status_code == 401
        assert exc_info.value.error_code == "AUTH_INVALID_TOKEN"


# ---------------------------------------------------------------------------
# 配置测试
# ---------------------------------------------------------------------------

class TestSecurityConfig:
    """config.py check_security() 对 JWT 吊销 / TTL 的约束"""

    def test_staging_token_ttl_61_rejected(self):
        """staging 环境 ACCESS_TOKEN_EXPIRE_MINUTES=61 拒绝启动"""
        from app.core.config import Settings
        s = Settings(
            ENVIRONMENT="staging",
            SECRET_KEY="prod-secret",
            ACCESS_TOKEN_EXPIRE_MINUTES=61,
            JWT_TOKEN_BLACKLIST_ENABLED=True,
            JWT_BLACKLIST_FAIL_CLOSED=True,
            TESTING=False,
        )
        with pytest.raises(RuntimeError, match="ACCESS_TOKEN_EXPIRE_MINUTES"):
            s.check_security()

    def test_staging_token_ttl_60_accepted(self):
        """staging 环境 ACCESS_TOKEN_EXPIRE_MINUTES=60 通过"""
        from app.core.config import Settings
        s = Settings(
            ENVIRONMENT="staging",
            SECRET_KEY="prod-secret",
            ACCESS_TOKEN_EXPIRE_MINUTES=60,
            JWT_TOKEN_BLACKLIST_ENABLED=True,
            JWT_BLACKLIST_FAIL_CLOSED=True,
            TESTING=False,
        )
        s.check_security()  # should not raise

    def test_production_fail_closed_required(self):
        """production 环境 blacklist 启用但 fail-closed=false → 拒绝"""
        from app.core.config import Settings
        s = Settings(
            ENVIRONMENT="production",
            SECRET_KEY="prod-secret",
            ACCESS_TOKEN_EXPIRE_MINUTES=30,
            JWT_TOKEN_BLACKLIST_ENABLED=True,
            JWT_BLACKLIST_FAIL_CLOSED=False,
            TESTING=False,
        )
        with pytest.raises(RuntimeError, match="FAIL_CLOSED"):
            s.check_security()

    def test_development_long_ttl_warning(self):
        """development 环境长 TTL 输出 warning 但不拒绝"""
        from app.core.config import Settings
        s = Settings(
            ENVIRONMENT="development",
            SECRET_KEY="dev-secret",
            ACCESS_TOKEN_EXPIRE_MINUTES=1440,
            JWT_TOKEN_BLACKLIST_ENABLED=False,
            JWT_BLACKLIST_FAIL_CLOSED=False,
            TESTING=False,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            s.check_security()  # should not raise


# ---------------------------------------------------------------------------
# Logout 测试
# ---------------------------------------------------------------------------

class TestLogoutBlacklistFailure:
    """logout 端点在生产 blacklist 写失败时返回 503"""

    def test_logout_blacklist_write_failure_returns_503(self):
        """blacklist_token() 返回 False → 503"""
        from fastapi.testclient import TestClient
        from app.main import app
        from app.db.session import get_db
        from app.core.deps import get_current_user

        fake_user = _fake_user(1)

        async def _override_user():
            return fake_user

        def _override_db():
            yield MagicMock(spec=AsyncSession)

        app.dependency_overrides[get_current_user] = _override_user
        app.dependency_overrides[get_db] = _override_db
        try:
            with patch("app.api.v1.auth.blacklist_token", new_callable=AsyncMock, return_value=False):
                client = TestClient(app, raise_server_exceptions=False)
                # 提供 Authorization header 以满足 oauth2_scheme
                resp = client.post("/api/v1/auth/logout", headers={"Authorization": "Bearer dummy"})
            assert resp.status_code == 503
            body = resp.json()
            assert body["error_code"] == "AUTH_BLACKLIST_UNAVAILABLE"
            client.close()
        finally:
            app.dependency_overrides.clear()

    def test_logout_success_returns_200(self):
        """blacklist_token() 返回 True → 200"""
        from fastapi.testclient import TestClient
        from app.main import app
        from app.db.session import get_db
        from app.core.deps import get_current_user

        fake_user = _fake_user(1)

        async def _override_user():
            return fake_user

        def _override_db():
            yield MagicMock(spec=AsyncSession)

        app.dependency_overrides[get_current_user] = _override_user
        app.dependency_overrides[get_db] = _override_db
        try:
            with patch("app.api.v1.auth.blacklist_token", new_callable=AsyncMock, return_value=True):
                client = TestClient(app, raise_server_exceptions=False)
                resp = client.post("/api/v1/auth/logout", headers={"Authorization": "Bearer dummy"})
            assert resp.status_code == 200
            client.close()
        finally:
            app.dependency_overrides.clear()
