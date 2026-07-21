# -*- coding: utf-8 -*-
"""RAG 检索接口 — 为评估 Agent 提供相似病例参照

支持三种检索模式：
1. 纯向量检索（默认）
2. 混合检索（BM25 + 向量 + RRF 融合）
3. 混合检索 + Cross-Encoder 重排序（最高精度）
"""

import asyncio
import json
import logging
import math
import re
import time
from typing import Dict, List, Optional

from app.services.qwen_client import call_qwen_chat
from app.services.rag.embeddings import get_embedding, get_embeddings
from app.services.rag.bm25_search import get_bm25_index
from app.services.rag.sparse_search import get_sparse_search
from app.services.rag.hybrid_fusion import weighted_rrf
from app.services.rag.entity_resolver import normalize_query as _entity_normalize_query
from app.services.rag.reranker import rerank_documents
from app.services.rag.types import (
    EvidenceItem, RetrievalBundle, RetrievalQuery, RetrievalConfidence,
    MIN_CANDIDATE_COUNT, MIN_QUERY_TYPE_COVERAGE, MIN_RRF_SCORE, MIN_SOURCE_COUNT,
    MIN_VECTOR_SCORE,
    MAX_MQE_EXPANSIONS, MAX_HYDE_CALLS, MAX_RAG_CANDIDATES,
)
from app.core.config import settings
from app.services.rag.retrieval_cache import get_cached_bundle, set_cached_bundle
from app.services.observability.langfuse_client import get_tracer
from app.services.observability.metrics import RAG_RETRIEVAL_DURATION

logger = logging.getLogger(__name__)

# ── RRF 融合参数 ──
RRF_K = 60  # Reciprocal Rank Fusion 常数，控制排名权重衰减速度

# ── MQE 语义漂移防护阈值 ──
MQE_SIMILARITY_THRESHOLD = 0.7  # 扩展查询与原始查询的最低 embedding 余弦相似度


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


async def _filter_by_embedding_similarity(
    original_query: str,
    expanded_queries: List[str],
    threshold: float = MQE_SIMILARITY_THRESHOLD,
) -> List[str]:
    """通过 embedding 余弦相似度校验，过滤与原始查询语义偏离过大的扩展查询

    防止 MQE 扩展后引入语义漂移：计算每条扩展 query 与原始 query 的向量相似度，
    低于阈值（默认 0.7）的扩展查询视为语义漂移被丢弃。

    Args:
        original_query: 原始医学查询文本
        expanded_queries: LLM 生成的扩展查询列表
        threshold: 相似度阈值（0-1），默认 0.7

    Returns:
        通过相似度校验的扩展查询列表
    """
    if not expanded_queries:
        return []

    try:
        # 并行获取原始查询和所有扩展查询的 embedding
        all_texts = [original_query] + expanded_queries
        all_embeddings = await get_embeddings(all_texts)
        orig_emb = all_embeddings[0]

        filtered = []
        for query, emb in zip(expanded_queries, all_embeddings[1:]):
            sim = _cosine_similarity(orig_emb, emb)
            if sim >= threshold:
                filtered.append(query)
                logger.debug(f"MQE 扩展查询通过语义校验: '{query[:30]}' (sim={sim:.3f})")
            else:
                logger.info(
                    f"MQE 扩展查询因语义漂移被过滤: '{query[:30]}' "
                    f"(sim={sim:.3f} < {threshold})"
                )

        logger.info(
            f"MQE 相似度校验：{len(expanded_queries)} 条扩展查询 "
            f"→ {len(filtered)} 条通过（阈值={threshold}）"
        )
        return filtered

    except Exception as e:
        logger.warning(f"MQE 相似度校验获取 embedding 失败，跳过过滤: {e}")
        return expanded_queries  # 降级：失败时保留全部扩展查询


from app.services.rag.medical_store import get_medical_store


async def retrieve_medical_evidence(
    diagnosis: str, top_k: int = 5
) -> List[Dict]:
    """基于诊断结果检索医学指南证据

    Args:
        diagnosis: 医生的诊断文本
        top_k: 返回条数
    Returns:
        医学证据列表 [{"text": ..., "source": ..., "page": ..., "score": ...}, ...]
    """
    store = get_medical_store()
    if store.collection is None or store.collection.count() == 0:
        logger.debug("医学知识库索引不可用，跳过医学证据检索")
        return []
    try:
        return await store.search(diagnosis, top_k=top_k)
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


# ── HyDE（Hypothetical Document Embeddings）──

HYDE_SYSTEM_PROMPT = """你是一名临床医学指南撰写专家。请根据以下医学查询，生成一段200-300字的理想临床指南段落。

要求：
1. 内容应涵盖该查询涉及疾病的诊断标准、推荐检查项目、一线治疗方案、注意事项等
2. 语气和风格必须模仿正式临床指南/诊疗规范的文体（如NCCN指南、CSCO指南的风格）
3. 使用规范的医学术语，包含具体的药物名称、剂量范围、检查项目名称
4. 不要使用"假设"、"可能"等不确定措辞，应使用"推荐"、"建议"、"首选"等指南性措辞
5. 只输出指南段落文本本身，不要任何前缀说明、标题或解释"""


async def _generate_hypothetical_document(query: str) -> str:
    """使用 LLM 生成假设性理想医学证据段落

    Args:
        query: 原始医学查询文本

    Returns:
        假设性医学指南段落文本；生成失败时返回原始 query 作为降级
    """
    if not query or not query.strip():
        return query

    try:
        messages = [
            {"role": "system", "content": HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": f"请为以下医学查询生成一段理想临床指南段落：\n{query}"},
        ]
        hypothetical_doc = await call_qwen_chat(
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )

        if hypothetical_doc and len(hypothetical_doc.strip()) > 50:
            logger.info(
                f"HyDE 假设文档生成成功：原始查询 '{query[:40]}...' → "
                f"假设文档 {len(hypothetical_doc)} 字符"
            )
            return hypothetical_doc.strip()
        else:
            logger.warning("HyDE 生成的文档过短，降级为原始查询")
            return query

    except Exception as e:
        logger.warning(f"HyDE 文档生成失败，降级为原始查询: {e}")
        return query


async def hyde_retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """HyDE 检索：生成假设性文档 → 用其 embedding 检索真实文档

    流程：
    1. LLM 生成假设性理想医学指南段落
    2. 获取假设文档的 embedding
    3. 用该 embedding 在 ChromaDB 中检索最相似的真实文档

    Args:
        query: 原始查询文本
        top_k: 返回条数

    Returns:
        检索结果列表，与 retrieve_medical_evidence 返回格式一致
    """
    # Step 1: 生成假设性文档
    hyde_doc = await _generate_hypothetical_document(query)

    # Step 2: 获取假设文档的 embedding
    try:
        hyde_embedding = await get_embedding(hyde_doc)
    except Exception as e:
        logger.warning(f"HyDE embedding 获取失败，降级为普通向量检索: {e}")
        return await retrieve_medical_evidence(query, top_k=top_k)

    # Step 3: 用假设文档的 embedding 在 ChromaDB 中检索
    try:
        store = get_medical_store()
        if store.collection is None or store.collection.count() == 0:
            logger.debug("医学知识库为空，HyDE 检索无结果")
            return []

        results = store.collection.query(
            query_embeddings=[hyde_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        evidences = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                doc_text = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = max(0.0, 1.0 - float(distance))

                evidences.append({
                    "doc_id": doc_id,
                    "text": doc_text,
                    "source": metadata.get("source", "未知"),
                    "page": metadata.get("page", 0),
                    "score": round(score, 4),
                    "heading_path": metadata.get("heading_path", ""),
                    "content_type": metadata.get("content_type", ""),
                    "organization": metadata.get("organization"),
                    "year": metadata.get("year"),
                    "version": metadata.get("version"),
                    "document_type": metadata.get("document_type"),
                    "departments": metadata.get("departments"),
                    "disease_tags": metadata.get("disease_tags"),
                    "population": metadata.get("population"),
                    "recommendation_level": metadata.get("recommendation_level"),
                    "evidence_level": metadata.get("evidence_level"),
                    "metadata_source": metadata.get("metadata_source"),
                })

        logger.info(f"HyDE 检索完成：返回 {len(evidences)} 条结果")
        return evidences

    except Exception as e:
        logger.warning(f"HyDE ChromaDB 检索失败: {e}")
        return []


async def expand_queries(original_query: str, n: int = 3) -> List[str]:
    """使用 LLM 将原始医学查询扩展为 n 条语义等价但措辞不同的查询变体

    Args:
        original_query: 原始医学查询文本
        n: 需要扩展的查询数量

    Returns:
        扩展后的查询列表（不包含原始查询）
    """
    system_prompt = f"""你是一个医学查询扩展专家。请将用户的医学查询扩展为 {n} 条语义等价但措辞不同的查询变体。

扩展方向包括：
1. 同义词替换（如"治疗"→"疗法"、"药物"→"药品"）
2. 中英文术语互换（如"非小细胞肺癌"→"NSCLC"、"靶向治疗"→"targeted therapy"）
3. 缩写展开（如"NSCLC"→"非小细胞肺癌"）
4. 上下位概念（如"肺癌"→"肺腺癌/肺鳞癌"、"EGFR突变"→"基因突变"）

要求：
- 扩展后的查询必须与原始查询语义等价，不能改变原意
- 每条扩展查询应该是完整的、可独立用于检索的短语
- 输出必须是严格的 JSON 数组格式，例如：["扩展查询1", "扩展查询2", "扩展查询3"]
- 不要包含任何解释性文字，只输出 JSON 数组"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请扩展以下医学查询：{original_query}"}
    ]

    try:
        response = await call_qwen_chat(
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )

        # 尝试从响应中提取 JSON 数组
        # 先尝试直接解析整个响应
        try:
            expanded = json.loads(response.strip())
            if isinstance(expanded, list):
                # 过滤掉非字符串项和空字符串
                expanded = [str(q).strip() for q in expanded if isinstance(q, (str,)) and str(q).strip()]
                logger.info(f"查询扩展成功：原始查询 '{original_query}' 扩展为 {len(expanded)} 条变体")
                return expanded
        except json.JSONDecodeError:
            pass

        # 尝试从响应中提取 JSON 数组（使用正则表达式）
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                expanded = json.loads(json_match.group())
                if isinstance(expanded, list):
                    expanded = [str(q).strip() for q in expanded if isinstance(q, (str,)) and str(q).strip()]
                    logger.info(f"查询扩展成功：原始查询 '{original_query}' 扩展为 {len(expanded)} 条变体")
                    return expanded
            except json.JSONDecodeError:
                pass

        logger.warning(f"查询扩展解析失败：无法从 LLM 响应中解析 JSON 数组，响应内容：{response[:200]}")
        return []

    except Exception as e:
        logger.warning(f"查询扩展失败：{e}")
        return []


# ── 混合检索核心函数 ──


def reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    top_k: int = 10,
    k: int = RRF_K,
) -> List[Dict]:
    """Reciprocal Rank Fusion（RRF）— 融合向量检索和 BM25 检索结果

    RRF 公式：score(d) = Σ 1 / (k + rank_i(d))
    其中 rank_i(d) 为文档 d 在第 i 个检索列表中的排名（从 1 开始）

    Args:
        vector_results: 向量检索结果列表
        bm25_results: BM25 关键词检索结果列表
        top_k: 返回条数
        k: RRF 常数（默认 60，值越大排名差异的影响越小）

    Returns:
        融合后按 RRF 分数降序排列的文档列表
    """
    # 优先使用稳定的 doc_id 作为去重键；降级到 text[:100]（小概率出现前缀相同但内容不同的情况）
    doc_scores = {}   # dedup_key -> {"doc": Dict, "rrf_score": float}

    # 处理向量检索结果
    for rank, doc in enumerate(vector_results, 1):
        text = doc.get("text", "")
        dedup_key = doc.get("doc_id") or (text[:100] if len(text) > 100 else text)
        if not dedup_key:
            continue

        rrf_score = 1.0 / (k + rank)
        if dedup_key in doc_scores:
            doc_scores[dedup_key]["rrf_score"] += rrf_score
        else:
            doc_scores[dedup_key] = {"doc": doc, "rrf_score": rrf_score}

    # 处理 BM25 检索结果
    for rank, doc in enumerate(bm25_results, 1):
        text = doc.get("text", "")
        dedup_key = doc.get("doc_id") or (text[:100] if len(text) > 100 else text)
        if not dedup_key:
            continue

        rrf_score = 1.0 / (k + rank)
        if dedup_key in doc_scores:
            doc_scores[dedup_key]["rrf_score"] += rrf_score
        else:
            doc_scores[dedup_key] = {"doc": doc, "rrf_score": rrf_score}

    # 按 RRF 分数降序排列
    sorted_results = sorted(
        doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True
    )

    # 返回 top_k，附带 RRF 分数
    final_results = []
    for item in sorted_results[:top_k]:
        doc_copy = dict(item["doc"])
        doc_copy["rrf_score"] = round(item["rrf_score"], 6)
        # 保留原始向量相似度分数（来自向量检索通道的 score）
        if "score" in doc_copy and "bm25_score" not in doc_copy:
            doc_copy["vector_score"] = doc_copy["score"]
        # 用 rrf_score 作为统一的 score 字段（用于后续排序）
        doc_copy["score"] = doc_copy["rrf_score"]
        final_results.append(doc_copy)

    return final_results


async def hybrid_recall(
    query: str,
    top_k: int = 10,
) -> tuple:
    """三路混合检索融合：BM25 + Dense + (可选) Sparse via weighted_rrf

    当 BGE_M3_ENABLED=True 时，并行执行三路检索并通过 weighted_rrf 融合；
    否则降级为 BM25 + Dense 两路融合。

    Args:
        query: 查询文本
        top_k: 每路召回条数

    Returns:
        (fused_results, fusion_meta)
        - fused_results: List[Dict] 融合后的文档列表（含 rrf_score / score 字段）
        - fusion_meta: dict 融合元信息（sources 各路人马数量、weights、fused_count）
    """
    loop = asyncio.get_event_loop()
    recall_k = top_k * 3  # 粗召回数量（给融合留足候选）

    # ── 定义三路检索函数 ──

    async def vector_search() -> List[Dict]:
        try:
            return await retrieve_medical_evidence(query, top_k=recall_k)
        except Exception as e:
            logger.warning(f"hybrid_recall-Dense通道失败: {e}")
            return []

    def bm25_search_sync() -> List[Dict]:
        try:
            index = get_bm25_index()
            return index.search(query, top_k=recall_k)
        except Exception as e:
            logger.warning(f"hybrid_recall-BM25通道失败: {e}")
            return []

    def sparse_search_sync() -> List[Dict]:
        """BGE-M3 Learned Sparse 检索通道（可选降级）"""
        try:
            ss = get_sparse_search()
            if ss is None or not ss.is_indexed:
                return []
            results = ss.search(query, top_k=recall_k)
            return [{"_sparse_idx": idx, "sparse_score": score} for idx, score in results]
        except Exception as e:
            logger.warning(f"hybrid_recall-Sparse通道失败（降级）: {e}")
            return []

    # ── 并行执行三路检索 ──
    if settings.BGE_M3_ENABLED:
        vector_results, bm25_results, sparse_results = await asyncio.gather(
            vector_search(),
            loop.run_in_executor(None, bm25_search_sync),
            loop.run_in_executor(None, sparse_search_sync),
        )
    else:
        vector_results, bm25_results = await asyncio.gather(
            vector_search(),
            loop.run_in_executor(None, bm25_search_sync),
        )
        sparse_results = []

    logger.info(
        f"hybrid_recall 粗召回: BM25={len(bm25_results)}, "
        f"Dense={len(vector_results)}, Sparse={len(sparse_results)}"
    )

    # ── 构建 ranking 格式用于 weighted_rrf ──
    # BM25 ranking: (doc_id, bm25_score)
    bm25_ranking = [
        (doc["doc_id"] if "doc_id" in doc else doc.get("id", f"bm25_{i}"),
         doc.get("bm25_score", 0.0))
        for i, doc in enumerate(bm25_results)
    ]
    # Dense ranking: (doc_id, vector_score)
    dense_ranking = [
        (doc.get("doc_id", f"dense_{i}"),
         doc.get("score", doc.get("vector_score", 0.0)))
        for i, doc in enumerate(vector_results)
    ]
    # Sparse ranking: (_sparse_idx, sparse_score)
    sparse_ranking = [
        (d["_sparse_idx"], d["sparse_score"])
        for d in sparse_results
    ]

    # ── 加权 RRF 融合 ──
    # 构建 doc_id -> 原始 dict 的映射（用于融合后还原完整文档）
    doc_map: Dict = {}
    for doc in bm25_results:
        key = doc.get("doc_id") or doc.get("id", "")
        if key:
            doc_map[key] = doc
    for doc in vector_results:
        key = doc.get("doc_id", "")
        if key and key not in doc_map:
            doc_map[key] = doc
        elif key in doc_map:
            # 合并 vector_score 到已有记录
            doc_map[key]["vector_score"] = doc.get("score", doc.get("vector_score", 0.0))

    if sparse_ranking:
        # 三路融合
        fused = weighted_rrf(
            rankings=[bm25_ranking, dense_ranking, sparse_ranking],
            weights=[0.30, 0.45, 0.25],
            k=35,
        )
        fusion_weights = [0.30, 0.45, 0.25]
    else:
        # 降级为两路融合
        fused = weighted_rrf(
            rankings=[bm25_ranking, dense_ranking],
            weights=[0.40, 0.60],
            k=35,
        )
        fusion_weights = [0.40, 0.60]

    # ── 将融合后的 (doc_id, score) 还原为完整 dict ──
    fused_results: List[Dict] = []
    for doc_id, rrf_score in fused:
        doc = doc_map.get(doc_id)
        if doc is None:
            # sparse-only 文档（无 BM25/Dense 匹配），尝试从 sparse 结果还原
            continue
        doc_copy = dict(doc)
        doc_copy["rrf_score"] = round(rrf_score, 6)
        doc_copy["score"] = doc_copy["rrf_score"]
        fused_results.append(doc_copy)

    fusion_meta = {
        "method": "weighted_rrf",
        "k": 35,
        "sources": {
            "bm25": len(bm25_ranking),
            "dense": len(dense_ranking),
            "sparse": len(sparse_ranking),
        },
        "weights": fusion_weights,
        "fused_count": len(fused_results),
    }

    return fused_results, fusion_meta


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


# ─────────────────────────────────────────────────────────────────────────────
# 分级检索：辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _dict_to_evidence(
    dicts: list,
    query_type: str,
    retrieved_via: str = "base",
) -> list:
    """将 dict 格式的检索结果转为 EvidenceItem"""
    items = []
    for d in dicts:
        item = EvidenceItem(
            doc_id=d.get("doc_id", d.get("id", "")),
            text=d.get("text", ""),
            source=d.get("source", "未知"),
            page=d.get("page"),
            heading_path=d.get("heading_path", ""),
            query_types=[query_type],
            vector_score=d.get("vector_score") or d.get("score"),
            bm25_score=d.get("bm25_score"),
            rrf_score=d.get("rrf_score"),
            # 从 metadata 提取增强字段（如果存在）
            organization=d.get("organization"),
            year=d.get("year") if isinstance(d.get("year"), int) and d.get("year", 0) > 0 else None,
            version=d.get("version"),
            document_type=d.get("document_type"),
            departments=d.get("departments"),
            disease_tags=d.get("disease_tags"),
            population=d.get("population"),
            content_type=d.get("content_type"),
            recommendation_level=d.get("recommendation_level"),
            evidence_level=d.get("evidence_level"),
            retrieved_via=retrieved_via,
        )
        items.append(item)
    return items


def _assess_confidence(
    candidates: list,
    query_types_covered: int,
    source_count: int,
    max_vector_score: float,
    max_rrf_score: float,
) -> RetrievalConfidence:
    """评估检索置信度

    Returns:
        RetrievalConfidence.HIGH: 多来源高分，直接使用
        RetrievalConfidence.MEDIUM: 部分满足，尝试增强
        RetrievalConfidence.LOW: 严重不足，准备拒答
    """
    if not candidates:
        return RetrievalConfidence.LOW

    # HIGH: 充分证据
    if (
        len(candidates) >= 5
        and source_count >= 3
        and max_vector_score >= 0.7
        and query_types_covered >= 2
    ):
        return RetrievalConfidence.HIGH

    # MEDIUM: 有一定证据但不充分
    if (
        len(candidates) >= 3
        and source_count >= 2
        and (max_vector_score >= 0.5 or max_rrf_score >= 0.015)
    ):
        return RetrievalConfidence.MEDIUM

    # LOW: 证据严重不足
    return RetrievalConfidence.LOW


def _assess_retrieval(
    candidates: list,
    query_types_with_hits: set,
    all_query_types: set,
) -> str:
    """兼容包装，返回 'sufficient' 或 'insufficient'"""
    if not candidates:
        return "unavailable"

    sources = set(c.source for c in candidates)
    max_vector = max((c.vector_score or 0) for c in candidates)
    max_rrf = max((c.rrf_score or 0) for c in candidates)

    confidence = _assess_confidence(
        candidates=candidates,
        query_types_covered=len(query_types_with_hits),
        source_count=len(sources),
        max_vector_score=max_vector,
        max_rrf_score=max_rrf,
    )
    return "sufficient" if confidence in (
        RetrievalConfidence.HIGH, RetrievalConfidence.MEDIUM
    ) else "insufficient"


def _merge_evidence(
    *result_lists: list,
) -> list:
    """合并多路证据并去重（按 doc_id），保留最高分数"""
    seen: dict = {}

    for results in result_lists:
        for item in results:
            key = item.doc_id
            if key in seen:
                existing = seen[key]
                # 合并 query_types
                existing.query_types = list(set(existing.query_types + item.query_types))
                # 保留各阶段最高分
                if (item.vector_score or 0) > (existing.vector_score or 0):
                    existing.vector_score = item.vector_score
                if (item.bm25_score or 0) > (existing.bm25_score or 0):
                    existing.bm25_score = item.bm25_score
                if (item.rrf_score or 0) > (existing.rrf_score or 0):
                    existing.rrf_score = item.rrf_score
            else:
                seen[key] = item.model_copy()

    # 合并后重新计算 RRF 分数，确保所有文档都有可比的排序分数
    results = list(seen.values())
    # 先按已有分数做初步排序（rrf_score 优先，vector_score / bm25_score 兜底）
    def _preliminary_sort_key(item: "EvidenceItem") -> float:
        if item.rrf_score is not None:
            return item.rrf_score
        if item.vector_score is not None:
            return item.vector_score
        if item.bm25_score is not None:
            return item.bm25_score / 10.0  # BM25 分数通常较大，粗略归一化
        return 0.0

    results.sort(key=_preliminary_sort_key, reverse=True)

    # 为缺少 rrf_score 的文档补算一个基于排名的 RRF 分数，统一量纲
    for i, item in enumerate(results):
        if item.rrf_score is None:
            item.rrf_score = round(1.0 / (RRF_K + i + 1), 6)

    # 最终按统一后的 rrf_score 排序
    results.sort(key=lambda x: x.rrf_score, reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 分级检索主入口
# ─────────────────────────────────────────────────────────────────────────────

async def tiered_retrieve(
    queries: list,
    top_k_per_query: int = 10,
    candidate_limit: int = MAX_RAG_CANDIDATES,
) -> RetrievalBundle:
    """分级检索：Level 1 (BM25+向量+RRF) → Level 2 (MQE) → Level 3 (HyDE)

    每个级别判断召回是否充分，足够则提前返回。

    Args:
        queries: 检索查询列表（通常包含 case/diagnosis/treatment 三类）
        top_k_per_query: 每个查询的召回条数
        candidate_limit: 最终候选上限

    Returns:
        RetrievalBundle 包含状态、级别、查询和候选证据
    """
    # 空查询防御
    if not queries:
        logger.warning("tiered_retrieve called with empty queries list")
        return RetrievalBundle(
            status="unavailable",
            level_used="base",
            queries=[],
            candidates=[],
            trace={"error": "empty_queries"},
        )

    # ── 缓存检查 ──
    queries_text = "|".join(sorted(q.text for q in queries))
    index_version = settings.ACTIVE_INDEX_VERSION
    cached = await get_cached_bundle(queries_text, index_version)
    if cached is not None:
        logger.info(f"tiered_retrieve 缓存命中: {index_version}")
        return RetrievalBundle.model_validate(cached)

    start = time.monotonic()
    trace = {
        "index_version": settings.ACTIVE_INDEX_VERSION,
        "queries": [{"type": q.query_type, "text": q.text[:100], "source": q.source} for q in queries],
        "levels_attempted": [],
        "retrieval_level": "base",
        "candidate_count": 0,
        "rerank_input_count": 0,
        "llm_rerank_count": 0,
        "retrieval_status": "candidate",
        "timing": {
            "embedding_ms": 0,
            "retrieval_ms": 0,
            "rerank_ms": 0,
            "llm_ms": 0,
        },
        "estimated_cost": 0.0,
        "degraded": False,
        "retrieval": {"fusion": None},
    }

    all_query_types = set(q.query_type for q in queries)
    query_types_with_hits: set = set()
    all_candidates: list = []
    level_used = "base"
    decisions: list = []  # 每级决策记录

    def _compute_confidence(cands: list, qtypes_hits: set) -> RetrievalConfidence:
        """从候选列表计算置信度"""
        if not cands:
            return RetrievalConfidence.LOW
        sources = set(c.source for c in cands)
        max_vec = max((c.vector_score or 0) for c in cands)
        max_rrf = max((c.rrf_score or 0) for c in cands)
        return _assess_confidence(
            candidates=cands,
            query_types_covered=len(qtypes_hits),
            source_count=len(sources),
            max_vector_score=max_vec,
            max_rrf_score=max_rrf,
        )

    def _build_confidence_trace(conf: RetrievalConfidence) -> dict:
        """构建标准化置信度 trace 片段"""
        sources = set(c.source for c in all_candidates) if all_candidates else set()
        max_vec = max((c.vector_score or 0) for c in all_candidates) if all_candidates else 0
        max_rrf = max((c.rrf_score or 0) for c in all_candidates) if all_candidates else 0
        return {
            "confidence": conf.value,
            "scores": {
                "vector": round(max_vec, 4),
                "rrf": round(max_rrf, 4),
                "source_count": len(sources),
                "candidate_count": len(all_candidates),
                "query_types_covered": len(query_types_with_hits),
            },
            "thresholds": {
                "min_vector": 0.5,
                "min_rrf": 0.015,
                "min_candidates": 3,
                "min_sources": 2,
            },
        }

    # ── Level 1: 基础混合召回（三路融合） ──
    trace["levels_attempted"].append("base")
    fusion_info = None  # 记录融合元信息
    for query in queries:
        # 实体归一化：将别名映射为规范名，增强 BM25 匹配
        normalized_text = _entity_normalize_query(query.text)
        fused, fusion_meta = await hybrid_recall(normalized_text, top_k=top_k_per_query)
        fusion_info = fusion_meta

        if not fused:
            continue

        if fused:
            query_types_with_hits.add(query.query_type)

        # 转换并合并
        evidence_items = _dict_to_evidence(fused, query.query_type, retrieved_via="base")
        # 为融合结果设置 rrf_score
        for item in evidence_items:
            if item.rrf_score is None:
                raw = next((f for f in fused if f.get("doc_id") == item.doc_id), None)
                if raw:
                    item.rrf_score = raw.get("rrf_score", raw.get("score", 0))

        all_candidates = _merge_evidence(all_candidates, evidence_items)

    confidence = _compute_confidence(all_candidates, query_types_with_hits)
    decisions.append({"level": "base", "confidence": confidence.value, "candidates": len(all_candidates)})
    if confidence == RetrievalConfidence.HIGH:
        elapsed = time.monotonic() - start
        trace["total_ms"] = round(elapsed * 1000, 1)
        trace["timing"]["retrieval_ms"] = trace["total_ms"]
        trace["retrieval_level"] = "base"
        trace["candidate_count"] = len(all_candidates[:candidate_limit])
        trace["retrieval_status"] = "candidate"
        trace.update(_build_confidence_trace(confidence))
        trace["decisions"] = decisions
        trace["retrieval"]["fusion"] = fusion_info
        result = RetrievalBundle(
            status="candidate",
            level_used="base",
            queries=queries,
            candidates=all_candidates[:candidate_limit],
            confidence=confidence.value,
            trace=trace,
        )
        await set_cached_bundle(queries_text, index_version, result.model_dump())
        return result
    # MEDIUM / LOW → 继续 L2 MQE

    # ── Level 2: MQE ──
    trace["levels_attempted"].append("mqe")
    level_used = "mqe"
    mqe_expansion_count = 0

    for query in queries:
        if mqe_expansion_count >= MAX_MQE_EXPANSIONS:
            break

        expanded = await expand_queries(query.text, n=2)
        if not expanded:
            continue

        # 语义漂移过滤
        expanded = await _filter_by_embedding_similarity(query.text, expanded)

        for eq in expanded:
            mqe_expansion_count += 1
            fused, fusion_meta = await hybrid_recall(eq, top_k=top_k_per_query)
            fusion_info = fusion_meta

            if not fused:
                continue

            if fused:
                query_types_with_hits.add(query.query_type)

            evidence_items = _dict_to_evidence(fused, query.query_type, retrieved_via="mqe")
            all_candidates = _merge_evidence(all_candidates, evidence_items)

            if len(all_candidates) >= candidate_limit:
                break

    confidence = _compute_confidence(all_candidates, query_types_with_hits)
    decisions.append({"level": "mqe", "confidence": confidence.value, "candidates": len(all_candidates)})
    if confidence in (RetrievalConfidence.HIGH, RetrievalConfidence.MEDIUM):
        elapsed = time.monotonic() - start
        trace["total_ms"] = round(elapsed * 1000, 1)
        trace["timing"]["retrieval_ms"] = trace["total_ms"]
        trace["mqe_expansions"] = mqe_expansion_count
        trace["retrieval_level"] = "mqe"
        trace["candidate_count"] = len(all_candidates[:candidate_limit])
        trace["retrieval_status"] = "candidate"
        trace.update(_build_confidence_trace(confidence))
        trace["decisions"] = decisions
        trace["retrieval"]["fusion"] = fusion_info
        result = RetrievalBundle(
            status="candidate",
            level_used="mqe",
            queries=queries,
            candidates=all_candidates[:candidate_limit],
            confidence=confidence.value,
            trace=trace,
        )
        await set_cached_bundle(queries_text, index_version, result.model_dump())
        return result
    # LOW → 继续 L3 HyDE

    # ── Level 3: HyDE（每次评估最多 1 次）──
    trace["levels_attempted"].append("hyde")
    level_used = "hyde"

    # 选择最有价值的查询做 HyDE（优先 case 类型）
    hyde_query = next((q for q in queries if q.query_type == "case"), queries[0])
    hyde_success = False
    try:
        hyde_results = await hyde_retrieve(hyde_query.text, top_k=top_k_per_query)
        if hyde_results:
            query_types_with_hits.add(hyde_query.query_type)
            evidence_items = _dict_to_evidence(hyde_results, hyde_query.query_type, retrieved_via="hyde")
            all_candidates = _merge_evidence(all_candidates, evidence_items)
        hyde_success = True
    except Exception as e:
        logger.warning(f"HyDE 检索失败: {e}")

    # 最终评估
    confidence = _compute_confidence(all_candidates, query_types_with_hits)
    decisions.append({"level": "hyde", "confidence": confidence.value, "candidates": len(all_candidates)})
    elapsed = time.monotonic() - start
    trace["total_ms"] = round(elapsed * 1000, 1)
    trace["timing"]["retrieval_ms"] = trace["total_ms"]
    trace["mqe_expansions"] = mqe_expansion_count
    trace["hyde_calls"] = 1 if hyde_success else 0
    trace["retrieval_level"] = "hyde"
    trace["candidate_count"] = len(all_candidates[:candidate_limit])
    final_status = "candidate" if confidence != RetrievalConfidence.LOW else "insufficient"
    trace["retrieval_status"] = final_status
    trace.update(_build_confidence_trace(confidence))
    trace["decisions"] = decisions
    trace["retrieval"]["fusion"] = fusion_info

    result = RetrievalBundle(
        status=final_status,
        level_used=level_used,
        queries=queries,
        candidates=all_candidates[:candidate_limit],
        confidence=confidence.value,
        trace=trace,
    )

    # ── 写入缓存 ──
    await set_cached_bundle(queries_text, index_version, result.model_dump())

    # ── Langfuse trace + Prometheus 指标 ──
    _elapsed_ms = (time.monotonic() - start) * 1000
    try:
        _query_text = queries[0].text if queries else ""
        get_tracer().trace_rag_retrieval(
            trace_name="rag_tiered_retrieve",
            query=_query_text,
            results=[{"score": c.rrf_score or 0} for c in result.candidates[:5]],
            latency_ms=_elapsed_ms,
        )
    except Exception as e:
        logger.debug(f"Langfuse RAG trace 异常: {e}")
    RAG_RETRIEVAL_DURATION.observe(_elapsed_ms / 1000)

    return result
