"""ProviderAdapter 抽象基类与 Provider 配置数据模型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderConfig:
    """不可变的 Provider 连接配置。

    通过 `identity()` 判断两个配置是否指向同一个底层端点，从而在 failover
    切换后决定是否需要重建底层客户端。
    """

    api_key: str
    base_url: str
    model: str
    name: str = "default"
    provider_type: str = "openai_compatible"
    timeout: float = 120.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderConfig":
        """从 failover_manager / settings 返回的 dict 构造配置。

        兼容缺失字段：仅 api_key / base_url / model 为核心字段，其余取默认值。
        """
        return cls(
            api_key=data.get("api_key", ""),
            base_url=data.get("base_url", ""),
            model=data.get("model", ""),
            name=data.get("name", "default"),
            provider_type=data.get("type") or data.get("provider_type") or "openai_compatible",
            timeout=float(data.get("timeout", 120.0)),
        )

    def identity(self) -> tuple[str, str, str]:
        """返回决定底层客户端身份的三元组：(类型, api_key, base_url)。

        model 变化不需要重建客户端（同一端点可切换模型），故不纳入 identity。
        """
        return (self.provider_type, self.api_key, self.base_url)


class ProviderAdapter(ABC):
    """LLM Provider 适配器抽象基类。

    子类负责根据 `ProviderConfig` 创建并持有具体的底层客户端（如 AsyncOpenAI），
    对上层暴露统一的 `client` / `model` 接口。
    """

    #: 子类覆盖，标识该适配器处理的 provider 类型（注册表按此 key 注册）。
    provider_type: str = "base"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config

    @property
    def config(self) -> ProviderConfig:
        return self._config

    @property
    def model(self) -> str:
        """该 Provider 的默认模型名。"""
        return self._config.model

    @property
    @abstractmethod
    def client(self) -> Any:
        """返回可用于发起请求的底层客户端（懒创建）。"""
        raise NotImplementedError

    def describe(self) -> str:
        """人类可读的简短描述（用于日志）。"""
        return f"{self.provider_type}:{self._config.name}({self._config.base_url})"
