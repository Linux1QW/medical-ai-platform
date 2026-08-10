# -*- coding: utf-8 -*-
"""V1.2 Production 配置与权限安全测试

验证:
- Coach 功能在 staging/production 环境需要有效的 HMAC key
- Voice 功能需要同时配置三个 LiveKit 值
- Coach 默认关闭
- RBAC 权限最小化：doctor 不能推进实验或审查他人记忆
"""

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.permissions import get_user_permissions

# ── Coach 配置验证 ────────────────────────────────────────────────────────────


class TestCoachConfigValidation:
    """Coach 智能体配置约束"""

    def test_production_rejects_enabled_coach_without_hmac_key(self, monkeypatch):
        """production 环境 COACH_ENABLED=true 但缺少 COACH_HMAC_KEY 应拒绝"""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("COACH_ENABLED", "true")
        monkeypatch.delenv("COACH_HMAC_KEY", raising=False)
        with pytest.raises(ValidationError, match="COACH_HMAC_KEY"):
            Settings()

    def test_staging_rejects_enabled_coach_without_hmac_key(self, monkeypatch):
        """staging 环境 COACH_ENABLED=true 但缺少 COACH_HMAC_KEY 应拒绝"""
        monkeypatch.setenv("ENVIRONMENT", "staging")
        monkeypatch.setenv("COACH_ENABLED", "true")
        monkeypatch.delenv("COACH_HMAC_KEY", raising=False)
        with pytest.raises(ValidationError, match="COACH_HMAC_KEY"):
            Settings()

    def test_production_rejects_short_hmac_key(self, monkeypatch):
        """production 环境 COACH_HMAC_KEY 少于 32 字节应拒绝"""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("COACH_ENABLED", "true")
        monkeypatch.setenv("COACH_HMAC_KEY", "too-short")
        with pytest.raises(ValidationError, match="COACH_HMAC_KEY"):
            Settings()

    def test_production_accepts_coach_with_valid_hmac(self, monkeypatch):
        """production 环境 COACH_ENABLED=true + 有效 HMAC key 应通过"""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("COACH_ENABLED", "true")
        monkeypatch.setenv("COACH_HMAC_KEY", "a" * 32)  # exactly 32 bytes
        s = Settings()
        assert s.COACH_ENABLED is True

    def test_coach_is_disabled_by_default(self):
        """Coach 默认关闭"""
        assert Settings().COACH_ENABLED is False

    def test_development_allows_coach_without_hmac(self, monkeypatch):
        """development 环境不强制 HMAC key 校验"""
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("COACH_ENABLED", "true")
        monkeypatch.delenv("COACH_HMAC_KEY", raising=False)
        s = Settings()
        assert s.COACH_ENABLED is True


# ── Voice / LiveKit 配置验证 ─────────────────────────────────────────────────


class TestVoiceConfigValidation:
    """Voice (LiveKit) 配置约束"""

    def test_partial_livekit_config_rejected(self, monkeypatch):
        """只配置部分 LiveKit 值应拒绝"""
        monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
        monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
        monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
        with pytest.raises(ValidationError, match="LIVEKIT"):
            Settings()

    def test_all_livekit_config_accepted(self, monkeypatch):
        """三个 LiveKit 值都配置时应通过"""
        monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
        monkeypatch.setenv("LIVEKIT_API_KEY", "APIkey123")
        monkeypatch.setenv("LIVEKIT_API_SECRET", "APIsecret456")
        s = Settings()
        assert s.LIVEKIT_URL == "wss://example.livekit.cloud"

    def test_no_livekit_config_is_fine(self, monkeypatch):
        """不配置任何 LiveKit 值是正常的"""
        monkeypatch.delenv("LIVEKIT_URL", raising=False)
        monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
        monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
        s = Settings()
        assert s.LIVEKIT_URL is None


# ── Coach 默认值 ─────────────────────────────────────────────────────────────


class TestCoachDefaults:
    """Coach 配置默认值"""

    def test_coach_context_token_limit_default(self):
        assert Settings().COACH_CONTEXT_TOKEN_LIMIT == 16_000

    def test_coach_output_token_limit_default(self):
        assert Settings().COACH_OUTPUT_TOKEN_LIMIT == 250

    def test_coach_hard_timeout_default(self):
        assert Settings().COACH_HARD_TIMEOUT_SECONDS == 8

    def test_coach_sse_event_ttl_default(self):
        assert Settings().COACH_SSE_EVENT_TTL_SECONDS == 3_600


# ── RBAC 权限最小化 ──────────────────────────────────────────────────────────


class TestRBACLeastPrivilege:
    """V1.2 权限映射测试：doctor 不能越权"""

    def _make_user(self, role: str, permissions=None):
        """构造轻量 User 替身（不需要数据库）"""
        from types import SimpleNamespace
        return SimpleNamespace(role=role, permissions=permissions)

    def test_doctor_cannot_advance_experiments(self):
        """医生不能推进实验"""
        doctor = self._make_user("doctor")
        perms = get_user_permissions(doctor)
        assert "experiment:manage" not in perms

    def test_doctor_cannot_review_others_memory(self):
        """医生不能审查他人的记忆"""
        doctor = self._make_user("doctor")
        perms = get_user_permissions(doctor)
        assert "trainee-memory:review" not in perms

    def test_doctor_has_coach_use(self):
        """医生有 coach:use 权限"""
        doctor = self._make_user("doctor")
        perms = get_user_permissions(doctor)
        assert "coach:use" in perms

    def test_doctor_has_trainee_memory_manage_self(self):
        """医生有 trainee-memory:manage-self 权限"""
        doctor = self._make_user("doctor")
        perms = get_user_permissions(doctor)
        assert "trainee-memory:manage-self" in perms

    def test_doctor_has_voice_use(self):
        """医生有 voice:use 权限"""
        doctor = self._make_user("doctor")
        perms = get_user_permissions(doctor)
        assert "voice:use" in perms

    def test_admin_has_all_doctor_permissions_plus_more(self):
        """管理员拥有医生权限的超集"""
        admin = self._make_user("admin")
        admin_perms = set(get_user_permissions(admin))
        doctor_perms = set(get_user_permissions(self._make_user("doctor")))
        assert doctor_perms.issubset(admin_perms)

    def test_admin_has_experiment_manage(self):
        """管理员有 experiment:manage 权限"""
        admin = self._make_user("admin")
        perms = get_user_permissions(admin)
        assert "experiment:manage" in perms

    def test_admin_has_trainee_memory_review(self):
        """管理员有 trainee-memory:review 权限"""
        admin = self._make_user("admin")
        perms = get_user_permissions(admin)
        assert "trainee-memory:review" in perms

    def test_admin_has_prompt_manage(self):
        """管理员有 prompt:manage 权限"""
        admin = self._make_user("admin")
        perms = get_user_permissions(admin)
        assert "prompt:manage" in perms

    def test_admin_has_coach_trace_view(self):
        """管理员有 coach:trace:view 权限"""
        admin = self._make_user("admin")
        perms = get_user_permissions(admin)
        assert "coach:trace:view" in perms
