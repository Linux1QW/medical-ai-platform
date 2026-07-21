"""ProviderAdapter 注册表。

按 `provider_type` 字符串注册适配器类，`create_adapter` 根据配置动态实例化。
未知类型回退到 OpenAI 兼容适配器，保证配置错误时仍可运行（degrade-safe）。
"""

from __future__ import annotations

import logging

from app.services.llm.base import ProviderAdapter, ProviderConfig
from app.services.llm.openai_compatible import OpenAICompatibleAdapter

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[ProviderAdapter]] = {}


def register_adapter(adapter_cls: type[ProviderAdapter]) -> type[ProviderAdapter]:
    """注册一个适配器类（按其 `provider_type`）。可用作装饰器。"""
    provider_type = adapter_cls.provider_type
    if not provider_type or provider_type == "base":
        raise ValueError(
            f"适配器 {adapter_cls.__name__} 必须定义非空且非 'base' 的 provider_type"
        )
    _REGISTRY[provider_type] = adapter_cls
    return adapter_cls


def get_adapter_class(provider_type: str) -> type[ProviderAdapter]:
    """按类型获取适配器类；未知类型回退到 OpenAI 兼容适配器。"""
    adapter_cls = _REGISTRY.get(provider_type)
    if adapter_cls is None:
        logger.warning(
            f"未知 provider_type='{provider_type}'，回退使用 OpenAICompatibleAdapter"
        )
        return OpenAICompatibleAdapter
    return adapter_cls


def create_adapter(config: ProviderConfig) -> ProviderAdapter:
    """根据配置创建适配器实例。"""
    adapter_cls = get_adapter_class(config.provider_type)
    return adapter_cls(config)


# ── 内置适配器注册 ──
register_adapter(OpenAICompatibleAdapter)
