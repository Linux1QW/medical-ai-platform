# -*- coding: utf-8 -*-
"""兼容混合检索入口和 MQE 增强检索。"""

import logging
import time
from typing import Dict, List

from app.services.rag.reranker import rerank_documents
from app.services.rag.retriever.base import retrieve_medical_evidence
from app.services.rag.retriever.fusion import hybrid_recall
from app.services.rag.retriever.tiered import tiered_retrieve
from app.services.rag.types import RetrievalQuery

logger = logging.getLogger(__name__)


async def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    enable_rerank: bool = True,
    rerank_threshold: float = 4,
    enable_hyde: bool = False,
) -> List[Dict]:
    """适配旧参数到统一三路召回，并可选执行 Cross-Encoder 重排序。

    Args:
        query: 查询文本
        top_k: 最终返回条数
        enable_rerank: 是否启用 Cross-Encoder 重排序
        rerank_threshold: 重排序相关性阈值（0-10）
        enable_hyde: 保留的兼容参数；HyDE 由分级检索入口负责

    Returns:
        检索结果列表
    """
    if not query or not query.strip():
        return []

    start_time = time.time()
    recall_k = top_k * 3
    fused_results, fusion_meta = await hybrid_recall(query, top_k=recall_k)
    if not fused_results:
        return []

    if enable_rerank and len(fused_results) > top_k:
        final_results = await rerank_documents(
            query=query,
            documents=fused_results,
            top_k=top_k,
            threshold=rerank_threshold,
        )
        retrieval_fields = (
            "bm25_score",
            "vector_score",
            "sparse_score",
            "rrf_score",
            "generation",
        )
        recalled_by_id = {
            str(document.get("doc_id", document.get("id", ""))): document
            for document in fused_results
        }
        for document in final_results:
            recalled = recalled_by_id.get(str(document.get("doc_id", "")), {})
            for field in retrieval_fields:
                if document.get(field) is None and recalled.get(field) is not None:
                    document[field] = recalled[field]
    else:
        final_results = fused_results[:top_k]

    elapsed_time = time.time() - start_time
    logger.info(
        "兼容混合检索完成：sources=%s → RRF %d → 最终 %d 条，"
        "rerank=%s，enable_hyde=%s（由 tiered 入口处理），耗时 %.3fs",
        fusion_meta.get("sources", {}),
        len(fused_results),
        len(final_results),
        "ON" if enable_rerank else "OFF",
        enable_hyde,
        elapsed_time,
    )

    return final_results


async def retrieve_with_mqe(
    query: str, top_k: int = 5, enable_mqe: bool = True, enable_hybrid: bool = True,
    enable_rerank: bool = True, enable_hyde: bool = False,
) -> List[Dict]:
    """增强版多查询扩展检索（MQE + 混合检索 + Cross-Encoder 重排序）

    检索流程：
    1. LLM 查询扩展（生成 3 条语义变体）
    2. 对每条查询执行混合检索（BM25 + 向量 + RRF）
    3. 合并去重所有结果
    4. Cross-Encoder 重排序取 top_k

    当 enable_mqe=True 且 enable_hybrid=True 时，内部转调 tiered_retrieve 实现分级检索。

    Args:
        query: 原始医学查询文本
        top_k: 返回条数
        enable_mqe: 是否启用多查询扩展（False 则退化为单查询）
        enable_hybrid: 是否启用混合检索（False 则仅用向量检索）
        enable_rerank: 是否启用 Cross-Encoder 重排序
        enable_hyde: 是否启用 HyDE

    Returns:
        合并去重后的医学证据列表
    """
    # 禁用 MQE 或非混合检索时，走原始路径
    if not enable_mqe or not query or not query.strip():
        logger.debug("MQE 已禁用或查询为空")
        if enable_hybrid:
            return await hybrid_retrieve(query, top_k=top_k, enable_rerank=enable_rerank, enable_hyde=enable_hyde)
        return await retrieve_medical_evidence(query, top_k=top_k)

    # 使用分级检索（tiered_retrieve）
    queries = [RetrievalQuery(query_type="diagnosis", text=query, source="clinical_facts")]
    bundle = await tiered_retrieve(queries, top_k_per_query=top_k)

    # 将 EvidenceItem 转回 dict 格式以保持旧接口兼容
    result = []
    for item in bundle.candidates[:top_k]:
        doc_dict = {
            "doc_id": item.doc_id,
            "generation": item.generation,
            "text": item.text,
            "source": item.source,
            "page": item.page,
            "heading_path": item.heading_path,
            "score": item.rrf_score or item.vector_score or item.bm25_score or 0,
            "rrf_score": item.rrf_score,
            "bm25_score": item.bm25_score,
            "vector_score": item.vector_score,
        }
        result.append(doc_dict)

    logger.info(
        f"retrieve_with_mqe（分级检索）：level={bundle.level_used}，"
        f"status={bundle.status}，返回 {len(result)} 条"
    )

    # 如果需要重排序且结果足够，调用重排序
    if enable_rerank and len(result) > 1:
        result = await rerank_documents(
            query=query,
            documents=result,
            top_k=top_k,
            threshold=4,
        )

    return result
