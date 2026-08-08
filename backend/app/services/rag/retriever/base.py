# -*- coding: utf-8 -*-
"""基础检索 — 纯向量检索、metadata 预过滤、证据格式化与上下文扩展"""

import logging
from typing import Dict, List

from app.core.config import settings
from app.services.rag.entity_resolver import extract_entities as _extract_entities
from app.services.rag.medical_store import get_medical_store
from app.services.rag.types import EvidenceItem

logger = logging.getLogger(__name__)


def build_disease_where_document(query: str) -> Dict | None:
    """从查询中抽取疾病实体，构建 ChromaDB where_document 过滤条件（纯函数）

    因 disease_tags 以 JSON 字符串存储，无法用 ChromaDB where 等值匹配，
    改用 where_document 的 $contains 对文档正文做子串过滤。中文子串有效
    （如查询含"肺癌"可命中正文"非小细胞肺癌"）。

    Args:
        query: 原始查询文本

    Returns:
        - 无疾病实体命中时返回 None（表示不过滤）
        - 单个疾病：{"$contains": "<规范名>"}
        - 多个疾病：{"$or": [{"$contains": d1}, {"$contains": d2}, ...]}
    """
    if not query or not query.strip():
        return None
    try:
        entities = _extract_entities(query)
    except Exception as e:  # 抽取失败不应影响检索
        logger.warning(f"疾病实体抽取失败，跳过 metadata 预过滤: {e}")
        return None

    diseases = []
    seen = set()
    for ent in entities:
        if ent.get("type") != "disease":
            continue
        name = ent.get("normalized", "").strip()
        if name and name not in seen:
            seen.add(name)
            diseases.append(name)

    if not diseases:
        return None
    if len(diseases) == 1:
        return {"$contains": diseases[0]}
    return {"$or": [{"$contains": d} for d in diseases]}


async def retrieve_medical_evidence(
    diagnosis: str,
    top_k: int = 5,
    *,
    generation: str | None = None,
) -> List[Dict]:
    """基于诊断结果检索医学指南证据

    Args:
        diagnosis: 医生的诊断文本
        top_k: 返回条数
    Returns:
        医学证据列表 [{"text": ..., "source": ..., "page": ..., "score": ...}, ...]
    """
    store = get_medical_store()
    try:
        where_document = None
        if settings.ENABLE_METADATA_FILTER:
            where_document = build_disease_where_document(diagnosis)
        return await store.search(
            diagnosis,
            top_k=top_k,
            where_document=where_document,
            generation=generation,
        )
    except Exception as e:
        logger.warning(f"医学证据检索失败，降级为无证据模式: {e}")
        return []


def format_evidence_for_verification(evidences: List[Dict]) -> str:
    """为 knowledge_agent 格式化医学证据"""
    if not evidences:
        return "未检索到相关医学证据"
    parts = []
    for i, ev in enumerate(evidences, 1):
        parts.append(
            f"证据{i}（来源: {ev.get('source', '未知')}, 第{ev.get('page', '?')}页）:\n"
            f"{ev.get('text', '')}"
        )
    return "\n\n".join(parts)


def expand_context(
    items: List[EvidenceItem], window: int
) -> List[EvidenceItem]:
    """Small-to-Big 上下文扩展：命中小块后拼接相邻块，提升喂给 LLM 的上下文完整度

    仅拼接同一 source 下 chunk_seq 相邻的块，保持前→中→后顺序。就地修改
    并返回 items（便于链式调用）。缺失 chunk_seq / 无邻居 / 查询失败时该条保持原样。

    Args:
        items: 待扩展的证据列表（通常为精排后的最终小集）
        window: 向前 / 向后各拉取的邻居块数
    """
    if not items or window <= 0:
        return items
    store = get_medical_store()
    for item in items:
        seq = item.chunk_seq
        if seq is None or seq < 0:
            continue
        try:
            if item.generation is None:
                neighbors = store.fetch_neighbors(
                    item.source,
                    seq,
                    window=window,
                )
            else:
                neighbors = store.fetch_neighbors(
                    item.source,
                    seq,
                    window=window,
                    generation=item.generation,
                )
        except Exception as e:
            logger.warning(f"上下文扩展失败（{item.source}#{seq}）: {e}")
            continue
        if not neighbors:
            continue
        before = [
            neighbors[s]["text"]
            for s in sorted(neighbors)
            if s < seq and neighbors[s].get("text")
        ]
        after = [
            neighbors[s]["text"]
            for s in sorted(neighbors)
            if s > seq and neighbors[s].get("text")
        ]
        if before or after:
            item.text = "\n".join(before + [item.text] + after)
    return items
