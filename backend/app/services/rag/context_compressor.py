# -*- coding: utf-8 -*-
"""抽取式上下文压缩 — 喂 LLM 前对长证据做句级降噪

对超过长度阈值的证据，按句切分并用 embedding 计算每句与查询的相似度，
仅保留 top-N 高相关句（保持原文顺序），降低喂给 LLM 的 token 与噪声。

所有能力默认关闭（opt-in），异常时一律降级返回原文，绝不丢证据。
"""

import logging
import math
import re
from typing import List

from app.services.rag.embeddings import get_embeddings
from app.services.rag.types import EvidenceItem

logger = logging.getLogger(__name__)

# 按中英文句末标点/换行切分，且保留分隔符（便于还原原始行文）
_SENT_SPLIT_RE = re.compile(r"[^。！？；!?\n]*[。！？；!?\n]|[^。！？；!?\n]+$")


def split_sentences(text: str) -> List[str]:
    """将文本切分为句子列表（保留句末标点，去除纯空白片段）"""
    if not text:
        return []
    raw = _SENT_SPLIT_RE.findall(text)
    return [s.strip() for s in raw if s and s.strip()]


def _cosine(a: List[float], b: List[float]) -> float:
    """余弦相似度（向量为空或零向量时返回 0）"""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


async def compress_text(
    text: str,
    query: str,
    *,
    min_chars: int,
    top_sentences: int,
    min_score: float,
) -> str:
    """对单条证据文本做句级抽取压缩

    Args:
        text: 原始证据文本
        query: 查询文本（用于计算句子相关性）
        min_chars: 短于该长度不压缩
        top_sentences: 最多保留的句数
        min_score: 句子与查询相似度低于该值则丢弃

    Returns:
        压缩后文本；不满足压缩条件或异常时返回原文。
    """
    if not text or not query:
        return text
    if len(text) < min_chars or top_sentences <= 0:
        return text

    sentences = split_sentences(text)
    if len(sentences) <= top_sentences:
        return text  # 句数本就不多，压缩无收益

    try:
        embeddings = await get_embeddings([query] + sentences)
    except Exception as e:
        logger.warning(f"句级压缩获取 embedding 失败，返回原文: {e}")
        return text
    if not embeddings or len(embeddings) != len(sentences) + 1:
        return text

    q_emb = embeddings[0]
    scored = [
        (i, sent, _cosine(q_emb, emb))
        for i, (sent, emb) in enumerate(zip(sentences, embeddings[1:]))
    ]

    kept = [t for t in scored if t[2] >= min_score]
    if not kept:
        # 全部低于阈值：降级返回原文，避免误删关键证据
        return text

    kept.sort(key=lambda t: t[2], reverse=True)
    kept = kept[:top_sentences]
    kept.sort(key=lambda t: t[0])  # 恢复原文顺序
    return "".join(t[1] for t in kept)


async def compress_evidences(
    items: List[EvidenceItem],
    query: str,
    *,
    min_chars: int,
    top_sentences: int,
    min_score: float,
) -> List[EvidenceItem]:
    """批量对证据列表做句级压缩（就地修改 text 并返回 items）"""
    if not items or not query:
        return items
    for item in items:
        try:
            item.text = await compress_text(
                item.text,
                query,
                min_chars=min_chars,
                top_sentences=top_sentences,
                min_score=min_score,
            )
        except Exception as e:
            logger.warning(f"句级压缩失败（{item.source}#{item.doc_id}），保留原文: {e}")
    return items
