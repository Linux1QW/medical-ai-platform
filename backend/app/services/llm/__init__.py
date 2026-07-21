"""LLM Provider 适配器抽象层。

将"具体厂商 / SDK"与"业务调用"解耦：业务侧只依赖 `ProviderAdapter` 接口，
底层可插拔不同 Provider（当前内置 OpenAI 兼容适配器，覆盖阿里云百炼 / OpenAI /
DeepSeek / 任意 OpenAI 兼容端点）。

配合 `llm_failover.failover_manager` 使用时，failover 切换 Provider 后可通过
`create_adapter` 重建真实的底层客户端，实现"真实切换"而非仅记账。
"""

from app.services.llm.base import ProviderAdapter, ProviderConfig
from app.services.llm.openai_compatible import OpenAICompatibleAdapter
from app.services.llm.registry import (
    create_adapter,
    get_adapter_class,
    register_adapter,
)

__all__ = [
    "ProviderConfig",
    "ProviderAdapter",
    "OpenAICompatibleAdapter",
    "register_adapter",
    "get_adapter_class",
    "create_adapter",
]
