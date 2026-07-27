# -*- coding: utf-8 -*-
"""HyDE（Hypothetical Document Embeddings）— 假设性文档增强检索"""

import logging
from typing import Dict, List

from app.services.qwen_client import call_qwen_chat
from app.services.rag.embeddings import get_embedding
from app.services.rag.medical_store import get_medical_store
from app.services.rag.retriever.base import retrieve_medical_evidence

logger = logging.getLogger(__name__)

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
                    "chunk_seq": metadata.get("chunk_seq", -1),
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
