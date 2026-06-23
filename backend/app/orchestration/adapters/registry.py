"""Agent 适配器注册表"""

from app.orchestration.adapters.base import BaseAgentAdapter

_REGISTRY: dict[str, BaseAgentAdapter] = {}


def register_adapter(adapter: BaseAgentAdapter):
    _REGISTRY[adapter.agent_name] = adapter


def get_adapter(agent_name: str) -> BaseAgentAdapter:
    if agent_name not in _REGISTRY:
        raise KeyError(f"未注册的适配器: {agent_name}")
    return _REGISTRY[agent_name]


def list_adapters() -> list[str]:
    return list(_REGISTRY.keys())
