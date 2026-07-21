# -*- coding: utf-8 -*-
"""DashScope Embedding 调用封装 — 使用 text-embedding-v4 模型生成文本向量

包含 LRU 内存缓存，对重复查询直接命中缓存，避免重复 API 调用。
"""

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import List, Optional

import httpx
from openai import AsyncOpenAI

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

# ── LRU 缓存配置 ──────────────────────────────────────────────────────────────
# 缓存最近 1000 条查询向量，命中时直接返回，避免重复调用 API
EMBED_CACHE_MAX_SIZE = 1000
_embedding_cache: OrderedDict = OrderedDict()  # hash(text) -> List[float]


def _text_hash(text: str) -> str:
    """计算文本的 MD5 摘要，作为缓存键"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_embed_cache_stats() -> dict:
    """返回缓存统计信息（供监控/调试使用）"""
    return {"size": len(_embedding_cache), "max_size": EMBED_CACHE_MAX_SIZE}


def clear_embed_cache():
    """清空 Embedding 缓存（重建索引前可调用）"""
    _embedding_cache.clear()
    logger.info("Embedding LRU 缓存已清空")


# ── 底层 API 调用（不含缓存逻辑）────────────────────────────────────────────
async def _get_embeddings_from_api(texts: List[str]) -> List[List[float]]:
    """直接调用 DashScope API 批量生成向量，含指数退避重试和速率控制。"""
    batch_size = 6
    max_retries = 3
    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_idx = i // batch_size

        for attempt in range(max_retries + 1):
            try:
                resp = await _embed_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                    dimensions=EMBEDDING_DIM,
                )
                for item in resp.data:
                    all_embeddings.append(item.embedding)
                break
            except Exception as e:
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

        if (batch_idx + 1) % 100 == 0:
            processed = (batch_idx + 1) * batch_size
            logger.info(f"已处理 {processed}/{len(texts)} 条文本向量")

        if i + batch_size < len(texts):
            await asyncio.sleep(0.1)

    return all_embeddings


# ── 公开接口（含 LRU 缓存）───────────────────────────────────────────────────
async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """批量获取文本向量，优先命中 LRU 缓存，未命中部分再调用 API。

    - 缓存容量：EMBED_CACHE_MAX_SIZE 条（LRU 淘汰策略）
    - 适用场景：查询阶段重复词条命中率高；索引构建时缓存意义较低
    """
    if not texts:
        return []

    results: List[Optional[List[float]]] = [None] * len(texts)
    uncached_idx: List[int] = []
    uncached_texts: List[str] = []

    # 1. 检查缓存
    for i, text in enumerate(texts):
        key = _text_hash(text)
        if key in _embedding_cache:
            results[i] = _embedding_cache[key]
            _embedding_cache.move_to_end(key)  # 刷新 LRU 访问顺序
        else:
            uncached_idx.append(i)
            uncached_texts.append(text)

    # 2. 对未命中的文本调用 API
    if uncached_texts:
        cache_hits = len(texts) - len(uncached_texts)
        logger.debug(
            f"Embedding 缓存：命中 {cache_hits}/{len(texts)} 条，"
            f"API 调用 {len(uncached_texts)} 条"
        )
        new_embeddings = await _get_embeddings_from_api(uncached_texts)
        for list_pos, orig_idx in enumerate(uncached_idx):
            emb = new_embeddings[list_pos]
            results[orig_idx] = emb
            # 3. 写入缓存并执行 LRU 淘汰
            key = _text_hash(uncached_texts[list_pos])
            _embedding_cache[key] = emb
            _embedding_cache.move_to_end(key)
            if len(_embedding_cache) > EMBED_CACHE_MAX_SIZE:
                _embedding_cache.popitem(last=False)  # 淘汰最久未访问的条目

    return results  # type: ignore[return-value]


async def get_embedding(text: str) -> List[float]:
    """获取单条文本的向量（走 LRU 缓存）"""
    result = await get_embeddings([text])
    return result[0]
