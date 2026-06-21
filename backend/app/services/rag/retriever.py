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
from app.services.rag.reranker import rerank_documents

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
        # 用 rrf_score 作为统一的 score 字段
        doc_copy["score"] = doc_copy["rrf_score"]
        final_results.append(doc_copy)

    return final_results


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

    Args:
        query: 原始医学查询文本
        top_k: 返回条数
        enable_mqe: 是否启用多查询扩展（False 则退化为单查询）
        enable_hybrid: 是否启用混合检索（False 则仅用向量检索）
        enable_rerank: 是否启用 Cross-Encoder 重排序

    Returns:
        合并去重后的医学证据列表
    """
    start_time = time.time()

    # 如果禁用 MQE 或查询为空，直接走混合检索
    if not enable_mqe or not query or not query.strip():
        logger.debug("MQE 已禁用或查询为空")
        if enable_hybrid:
            return await hybrid_retrieve(query, top_k=top_k, enable_rerank=enable_rerank, enable_hyde=enable_hyde)
        return await retrieve_medical_evidence(query, top_k=top_k)

    # 1. 并行执行：原始查询的混合检索 + LLM 查询扩展
    async def base_search():
        if enable_hybrid:
            return await hybrid_retrieve(query, top_k=top_k, enable_rerank=False, enable_hyde=enable_hyde)
        return await retrieve_medical_evidence(query, top_k=top_k)

    base_results, expanded_queries = await asyncio.gather(
        base_search(),
        expand_queries(query, n=3)
    )

    # 实体锁定第二层防护：通过 embedding 相似度阈值过滤语义漂移的扩展查询
    # 确保疾病、药物、检查指标等核心实体不被替换，避免语义偏离原始医学意图
    if expanded_queries:
        expanded_queries = await _filter_by_embedding_similarity(query, expanded_queries)

    all_results = list(base_results)

    if expanded_queries:
        logger.info(f"MQE 扩展查询 {len(expanded_queries)} 条，并行检索中...")

        async def safe_retrieve(q: str) -> List[Dict]:
            try:
                if enable_hybrid:
                    return await hybrid_retrieve(q, top_k=top_k, enable_rerank=False, enable_hyde=enable_hyde)
                return await retrieve_medical_evidence(q, top_k=top_k)
            except Exception as e:
                logger.warning(f"扩展查询检索失败: {e}")
                return []

        expanded_results_list = await asyncio.gather(
            *[safe_retrieve(q) for q in expanded_queries]
        )
        for results in expanded_results_list:
            all_results.extend(results)

    total_results = len(all_results)

    # 3. 去重（优先用 doc_id，降级用 text[:100]）
    seen_texts = {}
    for evidence in all_results:
        text = evidence.get("text", "")
        if not text:
            continue
        dedup_key = evidence.get("doc_id") or (text[:100] if len(text) > 100 else text)
        if dedup_key in seen_texts:
            if evidence.get("score", 0) > seen_texts[dedup_key].get("score", 0):
                seen_texts[dedup_key] = evidence
        else:
            seen_texts[dedup_key] = evidence

    deduped_results = list(seen_texts.values())
    deduped_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    # 4. Cross-Encoder 重排序
    if enable_rerank and len(deduped_results) > top_k:
        final_results = await rerank_documents(
            query=query,
            documents=deduped_results,
            top_k=top_k,
            threshold=4,
        )
    else:
        final_results = deduped_results[:top_k]

    elapsed_time = time.time() - start_time
    logger.info(
        f"MQE+混合检索总结：扩展 {len(expanded_queries)} 条，"
        f"原始 {total_results} 条，去重 {len(deduped_results)} 条，"
        f"最终 {len(final_results)} 条，rerank={'ON' if enable_rerank else 'OFF'}，"
        f"耗时 {elapsed_time:.3f}s"
    )

    return final_results
