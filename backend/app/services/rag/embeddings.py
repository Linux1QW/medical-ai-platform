# -*- coding: utf-8 -*-
"""DashScope Embedding 调用封装 — 使用 text-embedding-v3 模型生成文本向量"""

import logging
from typing import List

from openai import AsyncOpenAI
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
_embed_client = AsyncOpenAI(
    api_key=settings.QWEN_API_KEY,
    base_url=settings.QWEN_API_BASE_URL,
    http_client=_http_client,
)

EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIM = 1024


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """批量获取文本向量，返回与输入等长的向量列表。

    DashScope text-embedding-v3 单次最多支持 6 条文本，超过时自动分批。
    """
    batch_size = 6
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = await _embed_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
                dimensions=EMBEDDING_DIM,
            )
            for item in resp.data:
                all_embeddings.append(item.embedding)
        except Exception as e:
            logger.error(f"Embedding API 调用失败 (batch {i // batch_size}): {e}")
            raise

    return all_embeddings


async def get_embedding(text: str) -> List[float]:
    """获取单条文本的向量"""
    result = await get_embeddings([text])
    return result[0]
