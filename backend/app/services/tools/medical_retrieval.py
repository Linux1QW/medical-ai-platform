# -*- coding: utf-8 -*-
"""医学检索工具集 — 供 Agent 通过 Function Calling 调用 RAG 检索能力

包含 4 个工具：
1. SearchMedicalKB — 检索临床指南和循证医学证据
2. ExpandQuery — 多查询扩展（MQE）
3. GenerateHydeQuery — HyDE 假设性文档生成
4. RerankEvidence — 两阶段重排序

加固机制：
- 输入安全边界检查（长度限制、注入防护）
- 结果结构验证
- 细粒度错误分类与降级
- 执行耗时监控
"""

import logging
import time
from typing import List

from pydantic import BaseModel, Field

from app.services.rag.reranker import two_stage_rerank
from app.services.rag.retriever import (
    _generate_hypothetical_document,
    expand_queries,
    tiered_retrieve,
)
from app.services.rag.types import EvidenceItem, RetrievalBundle, RetrievalQuery
from app.services.tools.base import BaseTool, ToolContext
from app.services.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ── 安全边界常量 ──────────────────────────────────────────────────────────────

MAX_QUERY_LENGTH = 2000        # 查询最大字符数
MAX_CONTEXT_LENGTH = 5000      # 上下文最大字符数
MAX_CASE_SUMMARY_LENGTH = 5000 # 病例摘要最大字符数
MAX_CITATION_IDS = 100         # 最大候选 citation_id 数量


def _sanitize_query(query: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """输入清洗：截断过长查询、移除潜在注入字符"""
    if not query or not query.strip():
        return ""
    # 截断
    query = query.strip()[:max_length]
    return query


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
        """调用 tiered_retrieve 分级检索医学证据，转为统一格式并缓存"""
        # ── 安全边界检查 ──
        sanitized_query = _sanitize_query(args.query)
        if not sanitized_query:
            return {
                "evidence": [],
                "retrieval_level": "error",
                "total_found": 0,
                "degraded": True,
                "error": "查询为空或无效",
            }

        # 将 query_type 映射为 RetrievalQuery 支持的类型
        rq_type = args.query_type if args.query_type in ("case", "diagnosis", "treatment") else "case"

        start_time = time.monotonic()
        try:
            retrieval_query = RetrievalQuery(
                query_type=rq_type,
                text=sanitized_query,
                source="clinical_facts",
            )
            bundle: RetrievalBundle = await tiered_retrieve(
                queries=[retrieval_query],
                top_k_per_query=args.top_k,
            )
        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000
            logger.warning(f"search_medical_kb 检索失败 ({elapsed:.0f}ms): {e}")
            return {
                "evidence": [],
                "retrieval_level": "error",
                "total_found": 0,
                "degraded": True,
                "error": f"检索异常: {type(e).__name__}",
                "elapsed_ms": round(elapsed, 2),
            }

        elapsed = (time.monotonic() - start_time) * 1000

        # ── 从 RetrievalBundle 提取候选证据 ──
        candidates = bundle.candidates
        if not isinstance(candidates, list):
            logger.warning(f"search_medical_kb candidates 类型异常: {type(candidates)}")
            return {
                "evidence": [],
                "retrieval_level": "error",
                "total_found": 0,
                "degraded": True,
                "error": "检索返回格式异常",
                "elapsed_ms": round(elapsed, 2),
            }

        # 将 rag_trace 记录到 context 供后续使用
        if bundle.trace:
            context.extras["rag_trace"] = bundle.trace

        evidence_list = []
        for i, item in enumerate(candidates):
            citation_id = _build_citation_id_from_evidence(item, i)
            snippet = item.text[:300]

            evidence_entry = {
                "citation_id": citation_id,
                "title": item.source,
                "section": item.heading_path or "",
                "snippet": snippet,
                "score": round(item.rrf_score or item.vector_score or 0, 4),
                "source_type": _infer_source_type_from_evidence(item),
                "metadata": {
                    "organization": item.organization,
                    "year": item.year,
                    "departments": item.departments,
                },
            }
            evidence_list.append(evidence_entry)

            # 缓存完整 EvidenceItem 供后续工具使用
            context.evidence_cache[citation_id] = item

        # 判断检索级别：基于 tiered_retrieve 实际到达的层级
        level_map = {"base": "level1", "mqe": "level2", "hyde": "level3"}
        retrieval_level = level_map.get(bundle.level_used, "level1")

        logger.info(
            f"search_medical_kb: query='{sanitized_query[:50]}...', "
            f"found={len(evidence_list)}, level={retrieval_level}, "
            f"status={bundle.status}, elapsed={elapsed:.0f}ms"
        )

        return {
            "evidence": evidence_list,
            "retrieval_level": retrieval_level,
            "total_found": len(candidates),
            "degraded": bundle.degraded,
            "elapsed_ms": round(elapsed, 2),
            "rag_trace": bundle.trace,
        }


# ── Tool 2: ExpandQuery ──────────────────────────────────────────────────────


class ExpandQuery(BaseTool):
    name = "expand_query"
    description = "对原始医学查询进行扩展，生成多个相关查询变体"
    args_schema = ExpandQueryArgs
    timeout_seconds = 30

    async def execute(self, args: ExpandQueryArgs, context: ToolContext) -> dict:
        """复用 retriever.expand_queries 进行多查询扩展"""
        # ── 安全边界检查 ──
        sanitized_query = _sanitize_query(args.original_query)
        if not sanitized_query:
            return {
                "expanded_queries": [],
                "original_query": args.original_query,
                "degraded": True,
                "error": "原始查询为空或无效",
            }

        # 如果有临床背景，拼接到原始查询中以提升扩展质量
        query_text = sanitized_query
        if args.clinical_context:
            ctx = _sanitize_query(args.clinical_context, MAX_CONTEXT_LENGTH)
            if ctx:
                query_text = f"{ctx} {query_text}"

        try:
            expanded = await expand_queries(query_text, n=args.max_queries)
        except Exception as e:
            logger.warning(f"expand_query 失败: {e}")
            expanded = []

        result_queries = expanded[: args.max_queries]

        # ── 结果验证：无扩展结果时降级为原始查询 ──
        if not result_queries:
            logger.warning("expand_query 未生成扩展查询，使用原始查询作为降级")
            result_queries = [sanitized_query]

        return {
            "expanded_queries": result_queries,
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
        # ── 安全边界检查 ──
        sanitized_summary = _sanitize_query(args.case_summary, MAX_CASE_SUMMARY_LENGTH)
        if not sanitized_summary:
            return {
                "hyde_query": "",
                "query_type": args.query_type,
                "degraded": True,
                "error": "病例摘要为空或无效",
            }

        # 根据 query_type 构建更有针对性的 HyDE 输入
        hyde_input = sanitized_summary
        if args.query_type == "diagnosis":
            hyde_input = f"诊断相关：{sanitized_summary}"
        elif args.query_type == "treatment":
            hyde_input = f"治疗方案相关：{sanitized_summary}"

        try:
            hyde_doc = await _generate_hypothetical_document(hyde_input)
        except Exception as e:
            logger.warning(f"generate_hyde_query 失败: {e}")
            hyde_doc = sanitized_summary  # 降级为原始摘要

        # ── 结果验证 ──
        if not hyde_doc or not hyde_doc.strip():
            logger.warning("generate_hyde_query 返回空结果，降级为原始摘要")
            hyde_doc = sanitized_summary

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
        # ── 安全边界检查 ──
        sanitized_query = _sanitize_query(args.query)
        if not sanitized_query:
            return {
                "reranked_evidence": [],
                "total_candidates": 0,
                "returned": 0,
                "degraded": True,
                "error": "查询为空或无效",
            }

        # 限制候选数量防止过大输入
        candidate_ids = args.candidate_citation_ids[:MAX_CITATION_IDS]

        # 从缓存中收集候选 EvidenceItem
        candidates: list[EvidenceItem] = []
        for cid in candidate_ids:
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
                query=sanitized_query,
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
            "degraded": degraded,
        }


# ── 辅助函数 ─────────────────────────────────────────────────────────────────


def _build_citation_id(doc: dict, index: int) -> str:
    """为检索结果 dict 构建稳定的 citation_id（保留兼容旧路径）"""
    doc_id = doc.get("doc_id", "")
    source = doc.get("source", "未知")
    page = doc.get("page", 0)
    if doc_id:
        return f"rag:{source}:{page}:{doc_id[-8:]}"
    return f"rag:{source}:{page}:{index}"


def _build_citation_id_from_evidence(item: EvidenceItem, index: int) -> str:
    """为 EvidenceItem 构建稳定的 citation_id"""
    source = item.source or "未知"
    page = item.page or 0
    if item.doc_id:
        return f"rag:{source}:{page}:{item.doc_id[-8:]}"
    return f"rag:{source}:{page}:{index}"


def _infer_source_type_from_evidence(item: EvidenceItem) -> str:
    """从 EvidenceItem 元数据推断来源类型"""
    if item.content_type and "recommendation" in item.content_type:
        return "recommendation"
    if item.document_type and ("guideline" in item.document_type.lower() or "指南" in item.document_type):
        return "guideline"
    if "指南" in item.source or "NCCN" in item.source or "CSCO" in item.source:
        return "guideline"
    return "textbook"


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
