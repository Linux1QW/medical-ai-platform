# -*- coding: utf-8 -*-
"""DashScope Embedding 调用封装 — 使用 text-embedding-v4 模型生成文本向量"""

import asyncio
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

EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """批量获取文本向量，返回与输入等长的向量列表。

    DashScope text-embedding-v4 单次最多支持 6 条文本，超过时自动分批。
    包含指数退避重试和速率控制，适用于大批量索引构建场景。
    """
    batch_size = 6
    max_retries = 3
    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_idx = i // batch_size
        
        # 指数退避重试
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                resp = await _embed_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                    dimensions=EMBEDDING_DIM,
                )
                for item in resp.data:
                    all_embeddings.append(item.embedding)
                break  # 成功则跳出重试循环
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                if attempt < max_retries:
                    backoff = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    logger.warning(
                        f"Embedding API 调用失败 (batch {batch_idx}/{total_batches}, "
                        f"error_type={error_type}), 第 {attempt + 1} 次重试，等待 {backoff}s: {e}"
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        f"Embedding API 调用失败，已达最大重试次数 (batch {batch_idx}/{total_batches}, "
                        f"error_type={error_type}): {e}"
                    )
                    raise
        
        # 进度日志：每处理 100 个 batch（600 条文本）打印一次
        if (batch_idx + 1) % 100 == 0:
            processed = (batch_idx + 1) * batch_size
            logger.info(f"已处理 {processed}/{len(texts)} 条文本向量")
        
        # 速率控制：每个 batch 之间增加间隔，避免触发限流
        if i + batch_size < len(texts):
            await asyncio.sleep(0.1)

    return all_embeddings


async def get_embedding(text: str) -> List[float]:
    """获取单条文本的向量"""
    result = await get_embeddings([text])
    return result[0]
