# -*- coding: utf-8 -*-
"""医学检索工具集 — 供 Agent 通过 Function Calling 调用 RAG 检索能力

包含 4 个工具：
1. SearchMedicalKB — 检索临床指南和循证医学证据
2. ExpandQuery — 多查询扩展（MQE）
3. GenerateHydeQuery — HyDE 假设性文档生成
4. RerankEvidence — 两阶段重排序
"""

import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from app.services.tools.base import BaseTool, ToolContext
from app.services.tools.registry import ToolRegistry
from app.services.rag.types import EvidenceItem
from app.services.rag.retriever import (
    hybrid_retrieve,
    expand_queries,
    _generate_hypothetical_document,
)
from app.services.rag.reranker import two_stage_rerank

logger = logging.getLogger(__name__)


# ── Args Schemas ──────────────────────────────────────────────────────────────


class SearchMedicalKBArgs(BaseModel):
    query: str = Field(description="医学查询文本")
    query_type: str = Field(
        default="guideline",
        description="查询类型: case / diagnosis / treatment / guideline",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="返回条数，1-20")


class ExpandQueryArgs(BaseModel):
    original_query: str = Field(description="原始医学查询")
    clinical_context: str = Field(default="", description="临床背景描述")
    max_queries: int = Field(default=3, ge=1, le=10, description="最大扩展数")


class GenerateHydeQueryArgs(BaseModel):
    case_summary: str = Field(description="病例摘要")
    query_type: str = Field(
        description="查询类型: case / diagnosis / treatment",
    )


class RerankEvidenceArgs(BaseModel):
    query: str = Field(description="查询文本")
    candidate_citation_ids: List[str] = Field(
        description="候选证据的 citation_id 列表",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="返回 top K 条")


# ── Tool 1: SearchMedicalKB ──────────────────────────────────────────────────


class SearchMedicalKB(BaseTool):
    name = "search_medical_kb"
    description = "根据医学查询检索临床指南和循证医学证据"
    args_schema = SearchMedicalKBArgs
    timeout_seconds = 60
    critical = True

    async def execute(self, args: SearchMedicalKBArgs, context: ToolContext) -> dict:
        """调用 hybrid_retrieve 检索医学证据，转为统一格式并缓存"""
        try:
            raw_results = await hybrid_retrieve(
                query=args.query,
                top_k=args.top_k,
                enable_rerank=False,  # 重排交给独立工具
            )
        except Exception as e:
            logger.warning(f"search_medical_kb 检索失败: {e}")
            return {
                "evidence": [],
                "retrieval_level": "error",
                "total_found": 0,
                "degraded": True,
            }

        evidence_list = []
        for i, doc in enumerate(raw_results):
            citation_id = _build_citation_id(doc, i)
            snippet = doc.get("text", "")[:300]

            evidence_entry = {
                "citation_id": citation_id,
                "title": doc.get("source", "未知"),
                "section": doc.get("heading_path", ""),
                "snippet": snippet,
                "score": round(doc.get("score", 0) or 0, 4),
                "source_type": _infer_source_type(doc),
                "metadata": {
                    "organization": doc.get("organization"),
                    "year": doc.get("year"),
                    "departments": doc.get("departments"),
                },
            }
            evidence_list.append(evidence_entry)

            # 缓存完整 EvidenceItem 供后续工具使用
            context.evidence_cache[citation_id] = _doc_to_evidence_item(doc, i)

        # 判断检索级别
        retrieval_level = "level1" if len(raw_results) >= args.top_k else "level0"

        return {
            "evidence": evidence_list,
            "retrieval_level": retrieval_level,
            "total_found": len(raw_results),
            "degraded": False,
        }


# ── Tool 2: ExpandQuery ──────────────────────────────────────────────────────


class ExpandQuery(BaseTool):
    name = "expand_query"
    description = "对原始医学查询进行扩展，生成多个相关查询变体"
    args_schema = ExpandQueryArgs
    timeout_seconds = 30

    async def execute(self, args: ExpandQueryArgs, context: ToolContext) -> dict:
        """复用 retriever.expand_queries 进行多查询扩展"""
        # 如果有临床背景，拼接到原始查询中以提升扩展质量
        query_text = args.original_query
        if args.clinical_context:
            query_text = f"{args.clinical_context} {query_text}"

        try:
            expanded = await expand_queries(query_text, n=args.max_queries)
        except Exception as e:
            logger.warning(f"expand_query 失败: {e}")
            expanded = []

        return {
            "expanded_queries": expanded[: args.max_queries],
            "original_query": args.original_query,
        }


# ── Tool 3: GenerateHydeQuery ────────────────────────────────────────────────


class GenerateHydeQuery(BaseTool):
    name = "generate_hyde_query"
    description = "基于病例摘要生成 HyDE 假设性文档查询，用于增强检索"
    args_schema = GenerateHydeQueryArgs
    timeout_seconds = 30

    async def execute(
        self, args: GenerateHydeQueryArgs, context: ToolContext
    ) -> dict:
        """复用 retriever._generate_hypothetical_document 生成 HyDE 查询"""
        # 根据 query_type 构建更有针对性的 HyDE 输入
        hyde_input = args.case_summary
        if args.query_type == "diagnosis":
            hyde_input = f"诊断相关：{args.case_summary}"
        elif args.query_type == "treatment":
            hyde_input = f"治疗方案相关：{args.case_summary}"

        try:
            hyde_doc = await _generate_hypothetical_document(hyde_input)
        except Exception as e:
            logger.warning(f"generate_hyde_query 失败: {e}")
            hyde_doc = args.case_summary  # 降级为原始摘要

        return {
            "hyde_query": hyde_doc,
            "query_type": args.query_type,
        }


# ── Tool 4: RerankEvidence ───────────────────────────────────────────────────


class RerankEvidence(BaseTool):
    name = "rerank_evidence"
    description = "对检索到的医学证据进行重排序，提升相关性"
    args_schema = RerankEvidenceArgs
    timeout_seconds = 60

    async def execute(
        self, args: RerankEvidenceArgs, context: ToolContext
    ) -> dict:
        """从 evidence_cache 查找候选证据，调用 two_stage_rerank 重排"""
        # 从缓存中收集候选 EvidenceItem
        candidates: list[EvidenceItem] = []
        for cid in args.candidate_citation_ids:
            item = context.evidence_cache.get(cid)
            if item is not None and isinstance(item, EvidenceItem):
                candidates.append(item)

        if not candidates:
            return {
                "reranked_evidence": [],
                "total_candidates": 0,
                "returned": 0,
            }

        try:
            reranked, degraded = await two_stage_rerank(
                query=args.query,
                documents=candidates,
                top_k=args.top_k,
            )
        except Exception as e:
            logger.warning(f"rerank_evidence 失败: {e}")
            reranked = candidates[: args.top_k]
            degraded = True

        reranked_list = []
        for item in reranked:
            # 找到对应的 citation_id
            citation_id = _find_citation_id_for_item(item, context)
            reranked_list.append({
                "citation_id": citation_id or item.doc_id,
                "title": item.source,
                "snippet": item.text[:300],
                "score": round(item.rerank_score or 0, 4),
            })

        return {
            "reranked_evidence": reranked_list,
            "total_candidates": len(candidates),
            "returned": len(reranked_list),
        }


# ── 辅助函数 ─────────────────────────────────────────────────────────────────


def _build_citation_id(doc: dict, index: int) -> str:
    """为检索结果构建稳定的 citation_id"""
    doc_id = doc.get("doc_id", "")
    source = doc.get("source", "未知")
    page = doc.get("page", 0)
    if doc_id:
        # 优先使用 doc_id（ChromaDB ID）
        return f"rag:{source}:{page}:{doc_id[-8:]}"
    return f"rag:{source}:{page}:{index}"


def _infer_source_type(doc: dict) -> str:
    """从文档元数据推断来源类型"""
    content_type = doc.get("content_type", "")
    if "recommendation" in content_type:
        return "recommendation"
    doc_type = doc.get("document_type", "")
    if "guideline" in doc_type.lower() or "指南" in doc_type:
        return "guideline"
    source = doc.get("source", "")
    if "指南" in source or "NCCN" in source or "CSCO" in source:
        return "guideline"
    return "textbook"


def _doc_to_evidence_item(doc: dict, index: int) -> EvidenceItem:
    """将检索结果 dict 转为 EvidenceItem"""
    return EvidenceItem(
        doc_id=doc.get("doc_id", f"tmp-{index}"),
        text=doc.get("text", ""),
        source=doc.get("source", "未知"),
        page=doc.get("page"),
        heading_path=doc.get("heading_path", ""),
        vector_score=doc.get("vector_score") or doc.get("score"),
        bm25_score=doc.get("bm25_score"),
        rrf_score=doc.get("rrf_score"),
        organization=doc.get("organization"),
        year=doc.get("year") if isinstance(doc.get("year"), int) else None,
        version=doc.get("version"),
        document_type=doc.get("document_type"),
        departments=doc.get("departments"),
        disease_tags=doc.get("disease_tags"),
        content_type=doc.get("content_type"),
        recommendation_level=doc.get("recommendation_level"),
        evidence_level=doc.get("evidence_level"),
    )


def _find_citation_id_for_item(
    item: EvidenceItem, context: ToolContext
) -> str | None:
    """在 evidence_cache 中反向查找 EvidenceItem 对应的 citation_id"""
    for cid, cached in context.evidence_cache.items():
        if isinstance(cached, EvidenceItem) and cached.doc_id == item.doc_id:
            return cid
    return None


# ── 注册函数 ─────────────────────────────────────────────────────────────────


def register_medical_retrieval_tools(registry: ToolRegistry) -> None:
    """注册所有医学检索工具"""
    tools = [SearchMedicalKB(), ExpandQuery(), GenerateHydeQuery(), RerankEvidence()]
    for tool in tools:
        try:
            registry.register(tool)
        except ValueError:
            pass  # 已注册则跳过
