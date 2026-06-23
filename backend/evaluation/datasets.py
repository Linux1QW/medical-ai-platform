"""
Dataset handling for RAG evaluation system.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


logger = logging.getLogger(__name__)


class StanceType(str, Enum):
    """Possible stances for medical evidence."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    MIXED = "mixed"
    UNDETERMINED = "undetermined"


class SplitType(str, Enum):
    """Dataset splits."""
    DEV = "dev"
    TEST = "test"
    REGRESSION = "regression"


class DifficultyLevel(str, Enum):
    """Difficulty levels for cases."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class RagGoldCase(BaseModel):
    """Gold standard case for RAG evaluation."""
    
    # Identification
    case_id: str
    split: SplitType
    department: str
    domain_expertise: Optional[str] = None
    difficulty: DifficultyLevel
    
    # Case information
    chief_complaint: Optional[str] = None
    patient_info: str
    conversation_text: str
    doctor_diagnosis: Optional[str] = None
    treatment_plan: Optional[str] = None
    
    # Expected retrieval results
    gold_queries: Optional[List[str]] = Field(default_factory=list)
    gold_doc_ids: Optional[List[str]] = Field(default_factory=list)
    gold_citation_ids: Optional[List[str]] = Field(default_factory=list)
    gold_relevant_sources: Optional[List[str]] = Field(default_factory=list)
    gold_citation_keywords: Optional[List[str]] = Field(default_factory=list)
    gold_relevance_grades: Optional[Dict[str, int]] = Field(default_factory=dict)
    
    # Expected queries for retrieval evaluation
    expected_queries: Optional[List[str]] = Field(default_factory=list)
    
    # Expected evaluation results
    expected_stance: StanceType
    should_refuse: bool
    expected_score_range: Optional[List[float]] = None
    expected_review_reason: Optional[str] = None
    
    # Tool use expectations (added fields)
    expected_tool_calls: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    expected_tool_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    expected_final_answer_keywords: Optional[List[str]] = Field(default_factory=list)
    
    # Metadata
    notes: Optional[str] = None
    
    class Config:
        use_enum_values = True


class RagEvalResult(BaseModel):
    """Result from running an evaluation case."""
    
    # Identification
    case_id: str
    mode: str  # legacy, tooluse
    
    # System outputs
    knowledge_score: Optional[float] = None
    evaluation_status: str  # completed, needs_review
    human_review_needed: bool
    review_reason: Optional[str] = None
    retrieval_status: str  # sufficient, insufficient, error
    evidence_stance: Optional[StanceType] = None
    citation_data: List[Dict[str, Any]] = Field(default_factory=list)
    rag_trace_data: Dict[str, Any] = Field(default_factory=dict)
    tool_trace: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: Optional[int] = None
    error: Optional[str] = None
    
    # Actual retrieval results (for metric computation)
    actual_stance: Optional[str] = None
    retrieved_doc_ids: Optional[List[str]] = Field(default_factory=list)
    used_citation_ids: Optional[List[str]] = Field(default_factory=list)
    
    # Tool use results (added fields)
    actual_tool_calls: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    final_answer_text: Optional[str] = None
    
    # Internal computed fields
    system_refused: bool = False
    false_acceptance: bool = False
    
    class Config:
        use_enum_values = True


class RagEvalMetrics(BaseModel):
    """评估指标汇总模型

    汇总一次评估运行中所有用例的聚合指标，覆盖检索、引用、
    拒答、立场、分数范围、Tool Use 等维度。
    """

    # ── 检索指标 ──
    recall_at_1: Optional[float] = Field(default=None, description="Recall@1")
    recall_at_3: Optional[float] = Field(default=None, description="Recall@3")
    recall_at_5: Optional[float] = Field(default=None, description="Recall@5")
    mrr: Optional[float] = Field(default=None, description="平均倒数排名 (MRR)")
    ndcg_at_5: Optional[float] = Field(default=None, description="NDCG@5")

    # ── 引用指标 ──
    citation_validity: Optional[float] = Field(default=None, description="引用有效性")
    citation_hallucination_rate: Optional[float] = Field(default=None, description="引用幻觉率")
    citation_coverage: Optional[float] = Field(default=None, description="引用覆盖率")

    # ── 拒答指标 ──
    refusal_accuracy: Optional[float] = Field(default=None, description="拒答准确率")
    refusal_precision: Optional[float] = Field(default=None, description="拒答精确率")
    refusal_recall: Optional[float] = Field(default=None, description="拒答召回率")
    refusal_f1: Optional[float] = Field(default=None, description="拒答 F1 分数")
    false_refusal_rate: Optional[float] = Field(default=None, description="错误拒答率")
    false_acceptance_rate: Optional[float] = Field(default=None, description="错误接受率")

    # ── 立场 & 分数指标 ──
    stance_accuracy: Optional[float] = Field(default=None, description="立场准确率")
    score_range_accuracy: Optional[float] = Field(default=None, description="分数范围准确率")

    # ── Tool Use 指标 ──
    tool_success_rate: Optional[float] = Field(default=None, description="工具成功概率")
    tool_failure_rate: Optional[float] = Field(default=None, description="工具失败概率")
    tool_budget_exceeded_rate: Optional[float] = Field(default=None, description="工具预算超限率")
    avg_tool_calls: Optional[float] = Field(default=None, description="平均工具调用次数")

    # ── 延迟指标 ──
    avg_latency_ms: Optional[float] = Field(default=None, description="平均延迟（毫秒）")


class RagEvalReport(BaseModel):
    """评估报告模型

    包含一次完整评估运行的所有信息：时间戳、模式、数据集描述、
    聚合指标、工具分解、按难度分组统计、失败用例及阈值检查。
    """

    timestamp: str = Field(..., description="报告生成时间戳")
    mode: str = Field(..., description="评估模式 (legacy / tooluse)")
    dataset: Dict[str, Any] = Field(..., description="数据集信息")
    metrics: RagEvalMetrics = Field(..., description="评估指标汇总")

    # ── 可选分解 ──
    tool_breakdown: Optional[Dict[str, Any]] = Field(default=None, description="工具分解")
    breakdown_by_difficulty: Optional[Dict[str, Any]] = Field(default=None, description="按难度分解")
    failed_cases: Optional[List[Dict[str, Any]]] = Field(default=None, description="失败用例")

    thresholds: Dict[str, Any] = Field(..., description="阈值检查")


def load_gold_cases(cases_path: Path) -> List[RagGoldCase]:
    """Load gold cases from JSONL file."""
    if not cases_path.exists():
        raise FileNotFoundError(f"Gold cases file not found: {cases_path}")
    
    cases = []
    with open(cases_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                case = RagGoldCase(**data)
                cases.append(case)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {cases_path}:{line_num}: {e}")
                raise
            except Exception as e:
                logger.error(f"Validation error in {cases_path}:{line_num}: {e}")
                raise
    
    return cases


def save_gold_cases(cases: List[RagGoldCase], cases_path: Path) -> None:
    """Save gold cases to JSONL file."""
    with open(cases_path, 'w', encoding='utf-8') as f:
        for case in cases:
            f.write(json.dumps(case.dict(), ensure_ascii=False) + '\n')
