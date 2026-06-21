# -*- coding: utf-8 -*-
"""两阶段重排序模块 — 专用 reranker 粗排 + LLM 精排

Stage 1: DashScope gte-rerank 专用模型粗排（20 → 10）
Stage 2: LLM Cross-Encoder 精排（10 → 5），仅判断 relevance + completeness
最终排序公式融合 authority_score（代码计算）和 freshness_score（代码计算）
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.qwen_client import call_qwen_chat
from app.services.rag.types import (
    EvidenceItem,
    RerankResult,
    MAX_RERANK_INPUT,
    LLM_RERANK_INPUT,
    RELEVANCE_WEIGHT,
    COMPLETENESS_WEIGHT,
    AUTHORITY_WEIGHT,
    FRESHNESS_WEIGHT,
)

logger = logging.getLogger(__name__)

# ── 权威性分数映射（基于 organization 元数据）──────────────────────────────────
AUTHORITY_MAP = {
    "NCCN": 10,
    "CSCO": 9,
    "CACA": 9,
    "中华医学会": 8,
    "中国医师协会": 8,
}
DEFAULT_AUTHORITY = 5  # 未知来源

# ── LLM 精排 System Prompt ────────────────────────────────────────────────────
LLM_RERANK_PROMPT = """你是一名医学文献相关性评估专家。请判断以下医学证据与查询的相关程度。

对每条证据从两个维度打分（0-10 整数）：
- relevance：证据与查询的语义匹配度（直接回答查询=9-10，部分相关=5-6，不相关=0-2）
- completeness：证据是否包含完整的推荐意见（含剂量、疗程、检查项目等具体信息）

输出要求：严格 JSON 数组，每个元素包含 reference(doc_id)、relevance、completeness、reason。
不要输出任何解释性文字。"""

# 相关性阈值：低于此分数的文档将被过滤（旧接口兼容）
RELEVANCE_THRESHOLD = 4


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: DashScope gte-rerank 专用模型粗排
# ─────────────────────────────────────────────────────────────────────────────

def _sync_dashscope_rerank(
    query: str,
    documents: List[EvidenceItem],
    top_k: int,
) -> List[EvidenceItem]:
    """同步调用 DashScope TextReRank API（需在线程池中运行）"""
    from dashscope import TextReRank

    texts = [doc.text[:500] for doc in documents]
    response = TextReRank.call(
        model=settings.RERANK_MODEL,
        query=query,
        documents=texts,
        top_n=top_k,
        api_key=settings.QWEN_API_KEY,
    )

    if response is None or not hasattr(response, "output") or response.output is None:
        raise RuntimeError("DashScope rerank API 返回空结果")

    # 解析结果，按 relevance_score 降序
    scored = [
        (item.index, item.relevance_score)
        for item in response.output.results
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in scored:
        doc = documents[idx].model_copy()
        doc.rerank_score = round(score, 4)
        results.append(doc)

    return results


async def dedicated_rerank(
    query: str,
    documents: List[EvidenceItem],
    top_k: int = 5,
) -> List[EvidenceItem]:
    """使用 DashScope gte-rerank 专用模型粗排（异步包装）"""
    return await asyncio.to_thread(_sync_dashscope_rerank, query, documents, top_k)


# ─────────────────────────────────────────────────────────────────────────────
# 代码计算 authority_score 和 freshness_score
# ─────────────────────────────────────────────────────────────────────────────

def _compute_authority_score(doc: EvidenceItem) -> float:
    """基于 organization 元数据计算权威性分数（归一化到 0-1）"""
    org = (doc.organization or "").upper()
    for key, score in AUTHORITY_MAP.items():
        if key.upper() in org:
            return score / 10.0
    return DEFAULT_AUTHORITY / 10.0


def _compute_freshness_score(doc: EvidenceItem) -> float:
    """基于 year 元数据计算时效性分数（归一化到 0-1）"""
    if doc.year is None:
        return 0.5  # 未知年份给中等分
    current_year = 2026
    age = current_year - doc.year
    if age <= 1:
        return 1.0
    elif age <= 3:
        return 0.8
    elif age <= 5:
        return 0.6
    else:
        return 0.4


# ─────────────────────────────────────────────────────────────────────────────
# 最终排序公式
# ─────────────────────────────────────────────────────────────────────────────

def _compute_final_score(
    doc: EvidenceItem,
    relevance: int,
    completeness: int,
) -> float:
    """融合 relevance / completeness / authority / freshness 的最终分数"""
    rel_norm = relevance / 10.0
    comp_norm = completeness / 10.0
    auth = doc.authority_score if doc.authority_score is not None else _compute_authority_score(doc)
    fresh = doc.freshness_score if doc.freshness_score is not None else _compute_freshness_score(doc)

    return (
        RELEVANCE_WEIGHT * rel_norm
        + COMPLETENESS_WEIGHT * comp_norm
        + AUTHORITY_WEIGHT * auth
        + FRESHNESS_WEIGHT * fresh
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: LLM 精排
# ─────────────────────────────────────────────────────────────────────────────

async def _llm_fine_rerank(
    query: str,
    documents: List[EvidenceItem],
    top_k: int = 5,
) -> List[EvidenceItem]:
    """LLM 精排：仅对粗排后的前 N 条执行，判断 relevance + completeness"""
    input_docs = documents[:LLM_RERANK_INPUT]

    # 构建评分请求，包含完整上下文
    evidence_parts = []
    for doc in input_docs:
        part = (
            f"【doc_id: {doc.doc_id}】\n"
            f"来源: {doc.source} | 机构: {doc.organization or '未知'} | "
            f"年份: {doc.year or '未知'} | 页码: {doc.page or '?'}\n"
            f"章节: {doc.heading_path or '无'}\n"
            f"内容: {doc.text}\n"
        )
        evidence_parts.append(part)

    user_content = (
        f"【查询】{query}\n\n"
        f"【待评分证据（共{len(input_docs)}条）】\n\n"
        + "\n---\n".join(evidence_parts)
    )

    messages = [
        {"role": "system", "content": LLM_RERANK_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = await call_qwen_chat(
        messages=messages,
        temperature=0.1,
        max_tokens=500,
    )

    rerank_results = _parse_rerank_results(response, input_docs)
    return rerank_results[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# 解析函数：三层策略
# ─────────────────────────────────────────────────────────────────────────────

def _parse_rerank_results(
    response: str,
    documents: List[EvidenceItem],
) -> List[EvidenceItem]:
    """从 LLM 响应中解析 RerankResult 列表，三层解析策略

    1) 直接解析完整 JSON 数组
    2) 正则提取 JSON 数组片段
    3) 降级：给所有文档默认分数
    """
    if not response or not response.strip():
        logger.warning("LLM 精排响应为空")
        return _default_scored_docs(documents)

    expected_ids = {doc.doc_id for doc in documents}

    # ── 策略 1：直接解析 ──────────────────────────────────────────────────────
    parsed = _try_parse_json_array(response.strip(), expected_ids)
    if parsed is not None:
        return _apply_rerank_results(parsed, documents)

    # ── 策略 2：正则提取 JSON 数组 ─────────────────────────────────────────────
    match = re.search(r'\[[\s\S]*?\]', response)
    if match:
        parsed = _try_parse_json_array(match.group(), expected_ids)
        if parsed is not None:
            return _apply_rerank_results(parsed, documents)

    # ── 策略 3：降级默认分数 ──────────────────────────────────────────────────
    logger.warning("LLM 精排响应解析失败，使用默认分数")
    return _default_scored_docs(documents)


def _try_parse_json_array(
    text: str,
    expected_ids: set,
) -> Optional[List[RerankResult]]:
    """尝试将文本解析为 RerankResult 列表"""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, list) or not data:
        return None

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        ref = item.get("reference", "")
        if ref not in expected_ids:
            continue
        relevance = max(0, min(10, int(item.get("relevance", 0))))
        completeness = max(0, min(10, int(item.get("completeness", 0))))
        reason = str(item.get("reason", ""))
        results.append(RerankResult(
            reference=ref,
            relevance=relevance,
            completeness=completeness,
            reason=reason,
        ))

    return results if results else None


def _apply_rerank_results(
    rerank_results: List[RerankResult],
    documents: List[EvidenceItem],
) -> List[EvidenceItem]:
    """将 RerankResult 应用到 EvidenceItem 列表，计算 final_score 并排序"""
    score_map = {r.reference: r for r in rerank_results}
    scored_docs = []

    for doc in documents:
        doc_copy = doc.model_copy()
        result = score_map.get(doc.doc_id)
        if result:
            doc_copy.rerank_score = _compute_final_score(
                doc_copy, result.relevance, result.completeness
            )
        else:
            # 未被 LLM 评分的文档给默认分
            doc_copy.rerank_score = _compute_final_score(doc_copy, 5, 5)
        scored_docs.append(doc_copy)

    scored_docs.sort(key=lambda d: d.rerank_score or 0, reverse=True)
    return scored_docs


def _default_scored_docs(documents: List[EvidenceItem]) -> List[EvidenceItem]:
    """解析全部失败时的降级：给所有文档默认中等分数"""
    result = []
    for doc in documents:
        doc_copy = doc.model_copy()
        doc_copy.rerank_score = _compute_final_score(doc_copy, 5, 5)
        result.append(doc_copy)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 主入口：两阶段重排
# ─────────────────────────────────────────────────────────────────────────────

async def two_stage_rerank(
    query: str,
    documents: List[EvidenceItem],
    top_k: int = 5,
) -> Tuple[List[EvidenceItem], bool]:
    """两阶段重排：专用 reranker 粗排 + LLM 精排

    Args:
        query: 查询文本
        documents: 候选证据列表（EvidenceItem）
        top_k: 最终返回的文档数

    Returns:
        (reranked_documents, degraded)
        degraded: True 表示某阶段降级
    """
    if not documents or not query or not query.strip():
        return documents[:top_k] if documents else [], False

    degraded = False
    candidates = documents[:MAX_RERANK_INPUT]

    # 为每个文档计算 authority 和 freshness 分数
    for doc in candidates:
        doc.authority_score = _compute_authority_score(doc)
        doc.freshness_score = _compute_freshness_score(doc)

    # Stage 1: 专用 reranker 粗排（20 → 10）
    try:
        candidates = await dedicated_rerank(
            query, candidates, top_k=LLM_RERANK_INPUT * 2
        )
    except Exception as e:
        logger.warning(f"专用 reranker 失败，使用 RRF 排序: {e}")
        candidates = candidates[:LLM_RERANK_INPUT]
        degraded = True

    # Stage 2: LLM 精排（10 → 5）
    final: List[EvidenceItem] = []
    try:
        final = await _llm_fine_rerank(
            query, candidates[:LLM_RERANK_INPUT], top_k=top_k
        )
    except Exception as e:
        logger.warning(f"LLM 精排失败，使用专用 reranker 结果: {e}")
        final = candidates[:top_k]
        degraded = True

    # 如果两者都失败
    if degraded and not final:
        final = documents[:top_k]

    logger.info(
        f"两阶段重排完成：{len(documents)} 条候选 → "
        f"粗排 {min(len(candidates), LLM_RERANK_INPUT * 2)} 条 → "
        f"精排返回 {len(final)} 条（degraded={degraded}）"
    )

    return final, degraded


# ─────────────────────────────────────────────────────────────────────────────
# 旧接口兼容（保留原签名，内部转调 two_stage_rerank）
# ─────────────────────────────────────────────────────────────────────────────

async def rerank_documents(
    query: str,
    documents: List[Dict],
    top_k: int = 5,
    threshold: float = RELEVANCE_THRESHOLD,
) -> List[Dict]:
    """使用两阶段重排序对检索结果进行重排序（旧接口兼容）

    内部将 dict 列表转换为 EvidenceItem，调用 two_stage_rerank，
    再将结果转回 dict 格式。

    Args:
        query: 查询文本
        documents: 候选文档列表，每个文档需包含 "text" 字段
        top_k: 返回的最大文档数
        threshold: 最低相关性分数阈值（兼容参数，两阶段模式下不强制过滤）

    Returns:
        重排序后的文档列表（dict 格式），每个文档增加 "rerank_score" 字段
    """
    if not documents or not query or not query.strip():
        return documents[:top_k] if documents else []

    # dict → EvidenceItem
    evidence_items = []
    for doc in documents:
        item = EvidenceItem(
            doc_id=doc.get("doc_id", doc.get("id", str(len(evidence_items)))),
            text=doc.get("text", ""),
            source=doc.get("source", ""),
            page=doc.get("page"),
            heading_path=doc.get("heading_path", ""),
            organization=doc.get("organization"),
            year=doc.get("year"),
            vector_score=doc.get("vector_score"),
            bm25_score=doc.get("bm25_score"),
            rrf_score=doc.get("rrf_score"),
        )
        evidence_items.append(item)

    reranked, degraded = await two_stage_rerank(query, evidence_items, top_k=top_k)

    # EvidenceItem → dict
    result = []
    for item in reranked:
        doc_dict = item.model_dump(exclude_none=False)
        # 保留原始文档中可能携带的额外字段
        for key in ("text", "source", "page", "heading_path", "rerank_score"):
            if key in doc_dict:
                continue
        result.append(doc_dict)

    logger.info(
        f"rerank_documents（兼容接口）完成：{len(documents)} → {len(result)} 条"
        f"（degraded={degraded}）"
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 旧版评分解析（保留供可能的直接使用场景）
# ─────────────────────────────────────────────────────────────────────────────

def _parse_scores(response: str, expected_count: int) -> Optional[List[int]]:
    """从 LLM 响应中解析评分数组（旧版兼容）

    Args:
        response: LLM 原始响应文本
        expected_count: 期望的评分数量

    Returns:
        评分列表，解析失败返回 None
    """
    if not response or not response.strip():
        return None

    # 1. 直接尝试解析
    try:
        scores = json.loads(response.strip())
        if isinstance(scores, list) and len(scores) == expected_count:
            return [max(0, min(10, int(s))) for s in scores]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 2. 尝试提取 JSON 数组
    match = re.search(r'\[[\d\s,]+\]', response)
    if match:
        try:
            scores = json.loads(match.group())
            if isinstance(scores, list) and len(scores) == expected_count:
                return [max(0, min(10, int(s))) for s in scores]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # 3. 尝试提取所有数字
    numbers = re.findall(r'\b(\d+)\b', response)
    if len(numbers) >= expected_count:
        scores = [max(0, min(10, int(n))) for n in numbers[:expected_count]]
        return scores

    return None
