# -*- coding: utf-8 -*-
"""混合检索入口 — BM25+向量+RRF+重排序，以及 MQE 增强检索"""

import asyncio
import logging
import time
from typing import Dict, List

from app.services.rag.bm25_search import get_bm25_index
from app.services.rag.reranker import rerank_documents
from app.services.rag.retriever.base import retrieve_medical_evidence
from app.services.rag.retriever.fusion import reciprocal_rank_fusion
from app.services.rag.retriever.hyde import hyde_retrieve
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
    """混合检索：BM25 关键词检索 + 向量语义检索 + RRF 融合 + Cross-Encoder 重排序

    检索流程：
    1. 并行执行向量检索和 BM25 关键词检索（各取 top_k*2 条粗召回）
       - 当 enable_hyde=True 时，向量检索通道同时执行普通向量检索和 HyDE 检索，
         通过 RRF 融合两路结果后再与 BM25 结果融合
    2. 使用 Reciprocal Rank Fusion (RRF) 融合两路结果
    3. （可选）使用 Cross-Encoder 对融合结果进行精排

    Args:
        query: 查询文本
        top_k: 最终返回条数
        enable_rerank: 是否启用 Cross-Encoder 重排序
        rerank_threshold: 重排序相关性阈值（0-10）
        enable_hyde: 是否启用 HyDE 假设性文档增强

    Returns:
        检索结果列表
    """
    if not query or not query.strip():
        return []

    start_time = time.time()
    recall_k = top_k * 3  # 粗召回数量（给 RRF 和重排序留足候选）

    # ── Step 1: 真并行执行双路检索 ──
    # 向量检索是 I/O 密集型（embedding API），BM25 是 CPU 密集型（内存计算）
    # 使用 asyncio.gather + run_in_executor 实现真正并行
    loop = asyncio.get_event_loop()

    async def vector_search() -> List[Dict]:
        try:
            if enable_hyde:
                # 并行执行普通向量检索和 HyDE 检索
                normal_results, hyde_results = await asyncio.gather(
                    retrieve_medical_evidence(query, top_k=recall_k),
                    hyde_retrieve(query, top_k=recall_k),
                )
                # 两路都有结果时使用 RRF 融合
                if normal_results and hyde_results:
                    fused = reciprocal_rank_fusion(
                        normal_results, hyde_results, top_k=recall_k
                    )
                    logger.info(
                        f"HyDE+向量 RRF 融合：普通 {len(normal_results)} + "
                        f"HyDE {len(hyde_results)} → {len(fused)} 条"
                    )
                    return fused
                elif normal_results:
                    return normal_results
                else:
                    return hyde_results
            else:
                return await retrieve_medical_evidence(query, top_k=recall_k)
        except Exception as e:
            logger.warning(f"混合检索-向量通道失败: {e}")
            return []

    def bm25_search_sync() -> List[Dict]:
        try:
            index = get_bm25_index()
            return index.search(query, top_k=recall_k)
        except Exception as e:
            logger.warning(f"混合检索-BM25通道失败: {e}")
            return []

    # asyncio.gather 同时触发向量检索和 BM25（放入线程池）
    vector_results, bm25_results = await asyncio.gather(
        vector_search(),
        loop.run_in_executor(None, bm25_search_sync),
    )

    logger.info(
        f"混合检索粗召回完成：向量 {len(vector_results)} 条，BM25 {len(bm25_results)} 条"
    )

    # ── Step 2: RRF 融合 ──
    if vector_results and bm25_results:
        # 两路都有结果时使用 RRF 融合
        fused_results = reciprocal_rank_fusion(
            vector_results, bm25_results, top_k=recall_k
        )
        logger.info(f"RRF 融合完成：{len(fused_results)} 条结果")
    elif vector_results:
        # 仅向量通道有结果
        fused_results = vector_results[:recall_k]
    elif bm25_results:
        # 仅 BM25 通道有结果
        fused_results = bm25_results[:recall_k]
    else:
        return []

    # ── Step 3: Cross-Encoder 重排序（可选）──
    if enable_rerank and len(fused_results) > top_k:
        final_results = await rerank_documents(
            query=query,
            documents=fused_results,
            top_k=top_k,
            threshold=rerank_threshold,
        )
    else:
        final_results = fused_results[:top_k]

    elapsed_time = time.time() - start_time
    logger.info(
        f"混合检索完成：向量 {len(vector_results)} + BM25 {len(bm25_results)} "
        f"→ RRF {len(fused_results)} → 最终 {len(final_results)} 条，"
        f"rerank={'ON' if enable_rerank else 'OFF'}，耗时 {elapsed_time:.3f}s"
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
