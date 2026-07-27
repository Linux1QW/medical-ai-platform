# -*- coding: utf-8 -*-
"""RAG 检索接口 — 为评估 Agent 提供相似病例参照

支持三种检索模式：
1. 纯向量检索（默认）
2. 混合检索（BM25 + 向量 + RRF 融合）
3. 混合检索 + Cross-Encoder 重排序（最高精度）

本包由原 retriever.py（约 1260 行）拆分而来，按检索层次划分子模块：
- base:       纯向量检索、metadata 预过滤、证据格式化、Small-to-Big 上下文扩展
- similarity: 余弦相似度与 MQE 语义漂移过滤
- hyde:       HyDE 假设性文档增强检索
- mqe:        LLM 多查询扩展
- fusion:     RRF 融合与三路混合召回
- hybrid:     混合检索入口（hybrid_retrieve / retrieve_with_mqe）
- tiered:     分级检索主入口（L1 base → L2 MQE → L3 HyDE）

对外 API 与原 retriever.py 完全兼容：
    from app.services.rag.retriever import tiered_retrieve, hybrid_retrieve, ...
"""

from app.services.rag.medical_store import get_medical_store
from app.services.rag.retriever.base import (
    build_disease_where_document,
    expand_context,
    format_evidence_for_verification,
    retrieve_medical_evidence,
)
from app.services.rag.retriever.fusion import (
    RRF_K,
    hybrid_recall,
    reciprocal_rank_fusion,
)
from app.services.rag.retriever.hybrid import (
    hybrid_retrieve,
    retrieve_with_mqe,
)
from app.services.rag.retriever.hyde import (
    HYDE_SYSTEM_PROMPT,
    _generate_hypothetical_document,
    hyde_retrieve,
)
from app.services.rag.retriever.mqe import expand_queries
from app.services.rag.retriever.similarity import (
    MQE_SIMILARITY_THRESHOLD,
    _cosine_similarity,
    _filter_by_embedding_similarity,
)
from app.services.rag.retriever.tiered import (
    _assess_confidence,
    _assess_retrieval,
    _dict_to_evidence,
    _merge_evidence,
    tiered_retrieve,
)

__all__ = [
    # base
    "build_disease_where_document",
    "retrieve_medical_evidence",
    "format_evidence_for_verification",
    "expand_context",
    "get_medical_store",
    # similarity
    "MQE_SIMILARITY_THRESHOLD",
    "_cosine_similarity",
    "_filter_by_embedding_similarity",
    # hyde
    "HYDE_SYSTEM_PROMPT",
    "_generate_hypothetical_document",
    "hyde_retrieve",
    # mqe
    "expand_queries",
    # fusion
    "RRF_K",
    "reciprocal_rank_fusion",
    "hybrid_recall",
    # hybrid
    "hybrid_retrieve",
    "retrieve_with_mqe",
    # tiered
    "_dict_to_evidence",
    "_assess_confidence",
    "_assess_retrieval",
    "_merge_evidence",
    "tiered_retrieve",
]
