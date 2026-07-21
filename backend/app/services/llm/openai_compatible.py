"""OpenAI 兼容 Provider 适配器。

覆盖任意暴露 OpenAI Chat Completions 协议的端点：阿里云百炼 compatible-mode、
OpenAI 官方、DeepSeek、Moonshot 等。底层复用 `openai.AsyncOpenAI`。
"""

from __future__ import annotations

import httpx
from openai import AsyncOpenAI

from app.services.llm.base import ProviderAdapter, ProviderConfig


class OpenAICompatibleAdapter(ProviderAdapter):
    """基于 `AsyncOpenAI` 的 OpenAI 兼容适配器。

    每个适配器实例持有独立的 `httpx.AsyncClient` 与 `AsyncOpenAI` 客户端，
    均为懒创建：首次访问 `client` 时才建立连接池，避免构造即开销。
    """

    provider_type = "openai_compatible"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._http_client: httpx.AsyncClient | None = None
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.timeout),
                limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
            )
            self._client = AsyncOpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                http_client=self._http_client,
            )
        return self._client

    async def aclose(self) -> None:
        """释放底层连接池（切换 Provider 后可选调用，best-effort）。"""
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            finally:
                self._http_client = None
                self._client = None
