# -*- coding: utf-8 -*-
"""RAG 模块统一数据契约

定义检索查询、证据条目、检索结果包和引用信息的 Pydantic 模型，
作为 RAG 子系统（retriever、reranker、metadata、knowledge_agent）之间
的结构化接口，替代原有的 dict + raw_response 模式。
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── 查询类型 ─────────────────────────────────────────────────────────────────

class RetrievalQuery(BaseModel):
    """一条检索查询"""
    query_type: Literal["case", "diagnosis", "treatment"]
    text: str
    source: Literal["clinical_facts", "mqe", "hyde"] = "clinical_facts"


# ── 结构化病例事实 ─────────────────────────────────────────────────────────────

class ClinicalFacts(BaseModel):
    """从问诊对话和患者信息中提取的结构化病例事实

    用于构建三类独立查询（case/diagnosis/treatment），
    避免仅依赖"医生诊断+治疗方案"导致的确认偏误。
    """
    age: Optional[int] = None
    gender: Optional[str] = None
    chief_complaint: str = ""
    symptoms: list[str] = Field(default_factory=list)
    timeline: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    comorbidities: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    doctor_diagnoses: list[str] = Field(default_factory=list)
    treatment_items: list[str] = Field(default_factory=list)


# ── 证据条目 ───────────────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    """单条检索到的医学证据

    保留各阶段分数（vector/bm25/rrf/rerank）互不覆盖，
    并携带增强元数据以支撑权威性/时效性计算。
    """
    doc_id: str
    text: str
    source: str                                    # PDF 来源文件名
    page: Optional[int] = None
    heading_path: str = ""

    # 该证据被哪些查询类型命中
    query_types: list[str] = Field(default_factory=list)

    # 各阶段分数（独立保存，不相互覆盖）
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None

    # 增强元数据（来自 metadata_config）
    organization: Optional[str] = None
    year: Optional[int] = None
    version: Optional[str] = None
    document_type: Optional[str] = None
    departments: Optional[str] = None              # JSON string (ChromaDB 不支持 list)
    disease_tags: Optional[str] = None             # JSON string
    population: Optional[str] = None               # JSON string
    content_type: Optional[str] = None             # "text" | "table" | "recommendation"
    recommendation_level: Optional[str] = None     # 如 "I级推荐"
    evidence_level: Optional[str] = None           # 如 "1A"

    # 由代码计算的分数（禁止让 LLM 猜测）
    authority_score: Optional[float] = None
    freshness_score: Optional[float] = None

    # 标记检索通道
    retrieved_via: Optional[str] = None            # "base" | "mqe" | "hyde"


# ── 检索结果包 ─────────────────────────────────────────────────────────────────

class RetrievalBundle(BaseModel):
    """分级检索的完整结果包

    包含检索状态、使用的最高级别、所有查询、候选证据和追踪信息。
    """
    status: Literal["candidate", "insufficient", "unavailable", "error"]
    level_used: Literal["base", "mqe", "hyde"] = "base"
    queries: list[RetrievalQuery] = Field(default_factory=list)
    candidates: list[EvidenceItem] = Field(default_factory=list)
    degraded: bool = False
    trace: dict = Field(default_factory=dict)


# ── 引用信息 ───────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """单条引用追溯信息

    将 LLM 生成的结论与具体的知识库证据块绑定，
    支持评估报告的可审计性。
    """
    citation_id: str                               # 格式: rag-v2:doc-hash:p{page}:c{seq}
    claim: str                                     # 基于该证据得出的结论
    source: str                                    # 来源 PDF 文件名
    page: Optional[int] = None
    heading_path: str = ""
    text_snippet: str = ""                         # 来自真实 chunk 的文本片段
    rerank_score: Optional[float] = None


# ── 知识评估结果 ────────────────────────────────────────────────────────────────

class KnowledgeAssessment(BaseModel):
    """知识 Agent 的结构化评估结果

    替代原有的 raw_response JSON 字符串模式。
    """
    score: Optional[int] = None                    # None 表示拒答
    confidence: float = 0.5
    retrieval_status: Literal[
        "not_run", "sufficient", "insufficient", "unavailable", "error"
    ] = "not_run"
    evidence_stance: Literal[
        "supports", "contradicts", "mixed", "undetermined"
    ] = "undetermined"
    analysis: str = ""
    citations: list[Citation] = Field(default_factory=list)
    human_review_needed: bool = False
    review_reason: Optional[str] = None
    details: dict = Field(default_factory=dict)    # 兼容性字段


# ── 重排序结果 ─────────────────────────────────────────────────────────────────

class RerankResult(BaseModel):
    """单条证据的重排序结果（LLM 精排输出）"""
    reference: str                                 # doc_id
    relevance: int = 0                             # 0-10
    completeness: int = 0                          # 0-10
    reason: str = ""


# ── 配置常量 ───────────────────────────────────────────────────────────────────

# 调用预算
MAX_MQE_EXPANSIONS = 2
MAX_HYDE_CALLS = 1
MAX_RAG_CANDIDATES = 20

# 召回判断阈值
MIN_CANDIDATE_COUNT = 3                            # 至少 3 个候选
MIN_QUERY_TYPE_COVERAGE = 2                        # 至少覆盖 2 类查询
MIN_RRF_SCORE = 0.015
MIN_VECTOR_SCORE = 0.5                             # 向量相似度独立阈值
MIN_SOURCE_COUNT = 2                               # 至少 2 个不同来源

# 重排序配置
MAX_RERANK_INPUT = 20                              # 专用 reranker 最大输入
LLM_RERANK_INPUT = 5                               # LLM 精排最大输入
DEFAULT_RERANK_MODEL = "gte-rerank"

# 最终排序权重（初始值，由离线评测校准）
RELEVANCE_WEIGHT = 0.4
COMPLETENESS_WEIGHT = 0.3
AUTHORITY_WEIGHT = 0.2
FRESHNESS_WEIGHT = 0.1
