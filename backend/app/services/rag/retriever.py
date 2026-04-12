# -*- coding: utf-8 -*-
"""RAG 检索接口 — 为评估 Agent 提供相似病例参照"""

import logging
from typing import Dict, List, Optional

from app.services.rag.embeddings import get_embedding
from app.services.rag.vector_store import get_store

logger = logging.getLogger(__name__)


async def retrieve_similar_cases(
    patient_info: str, top_k: int = 2, exclude_id: Optional[str] = None
) -> List[Dict]:
    """检索与当前患者最相似的病例

    Args:
        patient_info: 患者信息文本（与 Agent 接收的 patient_info 格式一致）
        top_k: 返回条数
        exclude_id: 需要排除的病例 ID（避免检索到自身）

    Returns:
        相似病例元数据列表
    """
    store = get_store()
    if store.index is None or store.index.ntotal == 0:
        logger.debug("RAG 索引不可用，跳过相似病例检索")
        return []

    try:
        query_vec = await get_embedding(patient_info)
        results = store.search(query_vec, top_k=top_k, exclude_id=exclude_id)
        return [meta for meta, score in results]
    except Exception as e:
        logger.warning(f"RAG 检索失败，降级为无参照模式: {e}")
        return []


def format_reference_for_diagnosis(cases: List[Dict]) -> str:
    """为 diagnosis_agent 格式化参照：突出标准诊断"""
    if not cases:
        return ""
    parts = []
    for i, c in enumerate(cases, 1):
        parts.append(
            f"参照病例{i}：{c.get('gender', '')}, {c.get('age', '')}岁, "
            f"主诉: {c.get('chief_complaint', '')}, "
            f"既往史: {c.get('history', '')}, "
            f"检查: {c.get('exams', '无')}\n"
            f"→ 标准诊断: {c.get('diagnosis', '未知')}"
        )
    return "\n\n".join(parts)


def format_reference_for_treatment(cases: List[Dict]) -> str:
    """为 treatment_agent 格式化参照：突出标准处方"""
    if not cases:
        return ""
    parts = []
    for i, c in enumerate(cases, 1):
        parts.append(
            f"参照病例{i}：{c.get('gender', '')}, {c.get('age', '')}岁, "
            f"诊断: {c.get('diagnosis', '')}, "
            f"既往史: {c.get('history', '')}\n"
            f"→ 标准处方:\n{c.get('prescriptions', '无处方')}\n"
            f"→ 注意事项: {c.get('notes', '无')}"
        )
    return "\n\n".join(parts)


def format_reference_for_knowledge(cases: List[Dict]) -> str:
    """为 knowledge_agent 格式化参照：提供完整临床信息"""
    if not cases:
        return ""
    parts = []
    for i, c in enumerate(cases, 1):
        parts.append(
            f"参照病例{i}：{c.get('gender', '')}, {c.get('age', '')}岁, "
            f"主诉: {c.get('chief_complaint', '')}\n"
            f"既往史: {c.get('history', '')}\n"
            f"检查结果: {c.get('exams', '无')}\n"
            f"标准诊断: {c.get('diagnosis', '')}\n"
            f"标准处方: {c.get('prescriptions', '无处方')}"
        )
    return "\n\n".join(parts)


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
