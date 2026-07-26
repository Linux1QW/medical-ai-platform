"""ProviderAdapter 抽象层与 failover 真实切换单测。

覆盖：
- ProviderConfig 构造 / identity / from_dict 兼容
- 注册表 register / get_adapter_class（含未知类型回退）/ create_adapter
- OpenAICompatibleAdapter 懒创建底层 AsyncOpenAI
- qwen_client._refresh_active_provider() 在 Provider 变化时真实重建 client，
  相同 identity 时不重建但更新模型（修复 failover "半接线" bug）
"""

import pytest
from openai import AsyncOpenAI

from app.services.llm import (
    OpenAICompatibleAdapter,
    ProviderAdapter,
    ProviderConfig,
    create_adapter,
    get_adapter_class,
    register_adapter,
)

# ── ProviderConfig ───────────────────────────────────────────────────────────

class TestProviderConfig:
    def test_from_dict_full(self):
        cfg = ProviderConfig.from_dict({
            "name": "primary",
            "type": "openai_compatible",
            "api_key": "k",
            "base_url": "https://a/v1",
            "model": "m",
        })
        assert cfg.name == "primary"
        assert cfg.provider_type == "openai_compatible"
        assert cfg.api_key == "k"
        assert cfg.base_url == "https://a/v1"
        assert cfg.model == "m"

    def test_from_dict_defaults(self):
        cfg = ProviderConfig.from_dict({"api_key": "k", "base_url": "u", "model": "m"})
        assert cfg.name == "default"
        assert cfg.provider_type == "openai_compatible"
        assert cfg.timeout == 120.0

    def test_from_dict_provider_type_alias(self):
        """provider_type 与 type 两种键都应识别。"""
        cfg = ProviderConfig.from_dict({"provider_type": "custom", "api_key": "k"})
        assert cfg.provider_type == "custom"

    def test_identity_excludes_model(self):
        base = {"type": "openai_compatible", "api_key": "k", "base_url": "u"}
        c1 = ProviderConfig.from_dict({**base, "model": "m1"})
        c2 = ProviderConfig.from_dict({**base, "model": "m2"})
        assert c1.identity() == c2.identity()  # 模型不影响 identity

    def test_identity_differs_on_key(self):
        c1 = ProviderConfig.from_dict({"api_key": "k1", "base_url": "u", "model": "m"})
        c2 = ProviderConfig.from_dict({"api_key": "k2", "base_url": "u", "model": "m"})
        assert c1.identity() != c2.identity()

    def test_frozen(self):
        import dataclasses

        cfg = ProviderConfig(api_key="k", base_url="u", model="m")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.api_key = "x"  # type: ignore[misc]


# ── 注册表 ───────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_openai_compatible_registered(self):
        assert get_adapter_class("openai_compatible") is OpenAICompatibleAdapter

    def test_unknown_type_falls_back(self):
        assert get_adapter_class("nonexistent-xyz") is OpenAICompatibleAdapter

    def test_create_adapter_returns_instance(self):
        cfg = ProviderConfig.from_dict({"api_key": "k", "base_url": "u", "model": "m"})
        adapter = create_adapter(cfg)
        assert isinstance(adapter, OpenAICompatibleAdapter)
        assert adapter.model == "m"

    def test_register_rejects_base_type(self):
        class Bad(ProviderAdapter):
            provider_type = "base"

            @property
            def client(self):
                return None

        with pytest.raises(ValueError):
            register_adapter(Bad)

    def test_register_custom_adapter(self):
        class Custom(OpenAICompatibleAdapter):
            provider_type = "unit-test-custom"

        try:
            register_adapter(Custom)
            assert get_adapter_class("unit-test-custom") is Custom
        finally:
            from app.services.llm.registry import _REGISTRY
            _REGISTRY.pop("unit-test-custom", None)


# ── OpenAICompatibleAdapter ──────────────────────────────────────────────────

class TestOpenAICompatibleAdapter:
    def test_client_lazy_created(self):
        adapter = OpenAICompatibleAdapter(
            ProviderConfig(api_key="k", base_url="https://a/v1", model="m")
        )
        assert adapter._client is None  # 未访问前不创建
        c = adapter.client
        assert isinstance(c, AsyncOpenAI)
        assert adapter._client is c  # 缓存复用
        assert adapter.client is c

    def test_client_carries_config(self):
        adapter = OpenAICompatibleAdapter(
            ProviderConfig(api_key="secret", base_url="https://a/v1", model="m")
        )
        assert adapter.client.api_key == "secret"
        assert str(adapter.client.base_url).rstrip("/") == "https://a/v1"

    @pytest.mark.asyncio
    async def test_aclose_releases(self):
        adapter = OpenAICompatibleAdapter(
            ProviderConfig(api_key="k", base_url="https://a/v1", model="m")
        )
        _ = adapter.client
        assert adapter._http_client is not None
        await adapter.aclose()
        assert adapter._http_client is None
        assert adapter._client is None


# ── failover 真实切换（回归 "半接线" bug）─────────────────────────────────────

class TestRefreshActiveProvider:
    def _snapshot(self, q):
        # 懒初始化后模块级全局可能为 None，先触发创建以获得可对比的基线
        q.get_active_adapter()
        q._get_client()
        return (q._active_adapter, q._active_model, q.client)

    def _restore(self, q, snap):
        q._active_adapter, q._active_model, q.client = snap

    def test_switch_rebuilds_client(self):
        import app.services.qwen_client as q
        snap = self._snapshot(q)
        try:
            q._refresh_active_provider({
                "name": "backup",
                "type": "openai_compatible",
                "api_key": "backup-key",
                "base_url": "https://backup.example/v1",
                "model": "backup-model",
            })
            # 客户端被真实重建，且携带新端点凭证
            assert q.client is not snap[2]
            assert q.client.api_key == "backup-key"
            assert str(q.client.base_url).rstrip("/") == "https://backup.example/v1"
            assert q._active_model == "backup-model"
            assert q._active_adapter.config.api_key == "backup-key"
        finally:
            self._restore(q, snap)

    def test_same_identity_no_rebuild_but_model_updates(self):
        import app.services.qwen_client as q
        snap = self._snapshot(q)
        try:
            cfg = q._active_adapter.config
            q._refresh_active_provider({
                "type": cfg.provider_type,
                "api_key": cfg.api_key,
                "base_url": cfg.base_url,
                "model": "same-endpoint-new-model",
            })
            # 同一端点：客户端实例不变，仅模型更新
            assert q.client is snap[2]
            assert q._active_model == "same-endpoint-new-model"
        finally:
            self._restore(q, snap)

    def test_failover_manager_switch_drives_real_switch(self):
        """failover_manager.switch_to_next() 的返回可驱动 client 真实切换。"""
        from unittest.mock import patch

        import app.services.qwen_client as q
        from app.services.llm_failover import LLMFailoverManager

        providers = [
            {"name": "p1", "type": "openai_compatible", "api_key": "k1",
             "base_url": "https://p1/v1", "model": "m1"},
            {"name": "p2", "type": "openai_compatible", "api_key": "k2",
             "base_url": "https://p2/v1", "model": "m2"},
        ]
        with patch("app.services.llm_failover.settings") as mock_settings:
            mock_settings.get_llm_providers.return_value = providers
            mock_settings.LLM_CIRCUIT_BREAKER_THRESHOLD = 1
            mock_settings.QWEN_API_BASE_URL = "d"
            mock_settings.QWEN_MODEL = "d"
            mgr = LLMFailoverManager()

        new_provider = mgr.switch_to_next()  # → p2
        assert new_provider["name"] == "p2"

        snap = self._snapshot(q)
        try:
            q._refresh_active_provider(new_provider)
            assert q.client.api_key == "k2"
            assert str(q.client.base_url).rstrip("/") == "https://p2/v1"
            assert q._active_model == "m2"
        finally:
            self._restore(q, snap)


# ── 懒初始化（回归去全局单例 import 副作用）──────────────────────────────

class TestLazyInitialization:
    def _reset(self, q):
        snap = (q._active_adapter, q._active_model, q.client)
        q._active_adapter = None
        q._active_model = None
        q.client = None
        return snap

    def _restore(self, q, snap):
        q._active_adapter, q._active_model, q.client = snap

    def test_get_client_lazy_creates(self):
        """全局为 None 时，首次 _get_client() 才创建适配器与客户端。"""
        import app.services.qwen_client as q
        snap = self._reset(q)
        try:
            assert q._active_adapter is None
            assert q.client is None
            c = q._get_client()
            assert isinstance(c, AsyncOpenAI)
            assert q._active_adapter is not None
            assert q.client is c
            # 二次调用复用同一实例
            assert q._get_client() is c
        finally:
            self._restore(q, snap)

    def test_injected_client_takes_priority(self):
        """测试注入的模块级 client 优先于懒创建（兼容既有 patch 方式）。"""
        import app.services.qwen_client as q
        snap = self._reset(q)
        try:
            sentinel = object()
            q.client = sentinel
            assert q._get_client() is sentinel
            # 未触碰适配器：不会因注入而创建真实连接
            assert q._active_adapter is None
        finally:
            self._restore(q, snap)

    def test_get_active_model_lazy(self):
        """_get_active_model() 在未初始化时触发懒初始化并返回模型名。"""
        import app.services.qwen_client as q
        snap = self._reset(q)
        try:
            model = q._get_active_model()
            assert isinstance(model, str) and model
            assert q._active_adapter is not None
        finally:
            self._restore(q, snap)
