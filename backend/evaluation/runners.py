"""
Runner functions for different evaluation modes.
"""
import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from .config import EvalConfig, DEFAULT_CONFIG
from .datasets import RagEvalResult, RagGoldCase, RagEvalMetrics, RagEvalReport, load_gold_cases
from .metrics import (
    refusal_metrics_from_results as compute_refusal_metrics,
    tool_metrics as compute_tool_metrics,
    tool_breakdown as compute_tool_breakdown,
    aggregate_metrics_by_dimension,
    final_answer_keyword_coverage,
    tool_call_accuracy,
    retrieval_metrics as compute_retrieval_metrics,
    citation_metrics as compute_citation_metrics,
    score_range_accuracy as compute_score_range_accuracy,
)
from .report import generate_json_report
from ..app.services.agents.knowledge_agent import run_knowledge_check, run_knowledge_check_with_tools


logger = logging.getLogger(__name__)


async def run_case_legacy(case: RagGoldCase) -> RagEvalResult:
    """
    Run a single case using the legacy knowledge check.
    
    Args:
        case: The gold case to evaluate
        
    Returns:
        Evaluation result
    """
    start_time = time.time()
    
    try:
        # Prepare inputs for legacy knowledge check
        conversation_text = case.conversation_text
        patient_info = case.patient_info
        doctor_diagnosis = case.doctor_diagnosis or ""
        treatment_plan = case.treatment_plan or ""
        
        # Run the legacy knowledge check
        result = await run_knowledge_check(
            conversation_text=conversation_text,
            patient_info=patient_info,
            doctor_diagnosis=doctor_diagnosis,
            treatment_plan=treatment_plan,
        )
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Extract relevant fields from the result
        knowledge_score = result.get('score') or result.get('knowledge_score')
        evaluation_status = result.get('evaluation_status', 'completed')
        human_review_needed = result.get('human_review_needed', False)
        review_reason = result.get('review_reason')
        retrieval_status = result.get('retrieval_status', 'sufficient')
        evidence_stance_str = result.get('evidence_stance')
        evidence_stance = evidence_stance_str if evidence_stance_str else None
        
        # Handle citation data
        citations = result.get('citations', [])
        citation_data = []
        if isinstance(citations, list):
            citation_data = citations
        elif isinstance(citations, dict):
            citation_data = [citations]
        
        # Handle rag trace data
        rag_trace_data = result.get('rag_trace', {})
        if not isinstance(rag_trace_data, dict):
            rag_trace_data = {}
        
        # Handle tool trace (for legacy, this should be empty)
        tool_trace = result.get('tool_trace', [])
        if not isinstance(tool_trace, list):
            tool_trace = []
        
        # Extract final answer if available
        final_answer_text = result.get('final_answer', result.get('response', None))
        
        # For legacy mode, actual tool calls would come from rag_trace_data
        actual_tool_calls = []
        if 'tool_calls' in rag_trace_data:
            actual_tool_calls = rag_trace_data['tool_calls']
        
        # Create the evaluation result
        eval_result = RagEvalResult(
            case_id=case.case_id,
            mode="legacy",
            knowledge_score=knowledge_score,
            evaluation_status=evaluation_status,
            human_review_needed=human_review_needed,
            review_reason=review_reason,
            retrieval_status=retrieval_status,
            evidence_stance=evidence_stance,
            citation_data=citation_data,
            rag_trace_data=rag_trace_data,
            tool_trace=tool_trace,
            latency_ms=latency_ms,
            error=None,
            actual_tool_calls=actual_tool_calls,
            final_answer_text=final_answer_text
        )
        
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        eval_result = RagEvalResult(
            case_id=case.case_id,
            mode="legacy",
            knowledge_score=None,
            evaluation_status="error",
            human_review_needed=True,
            review_reason="system_exception",
            retrieval_status="error",
            evidence_stance=None,
            citation_data=[],
            rag_trace_data={},
            tool_trace=[],
            latency_ms=latency_ms,
            error=str(e),
            actual_tool_calls=[],
            final_answer_text=None
        )
    
    return eval_result


async def run_case_tooluse(case: RagGoldCase) -> RagEvalResult:
    """
    Run a single case using the Tool Use knowledge check.
    
    Args:
        case: The gold case to evaluate
        
    Returns:
        Evaluation result
    """
    start_time = time.time()
    
    try:
        # Prepare inputs for Tool Use knowledge check
        consultation = {
            "conversation_text": case.conversation_text,
            "patient_info": case.patient_info,
        }
        
        diagnosis_text = case.doctor_diagnosis or ""
        treatment_text = case.treatment_plan or ""
        
        # Run the Tool Use knowledge check
        result = await run_knowledge_check_with_tools(
            consultation=consultation,
            diagnosis_text=diagnosis_text,
            treatment_text=treatment_text,
        )
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Extract relevant fields from the result
        knowledge_score = result.get('score') or result.get('knowledge_score')
        evaluation_status = result.get('evaluation_status', 'completed')
        human_review_needed = result.get('human_review_needed', False)
        review_reason = result.get('review_reason')
        retrieval_status = result.get('retrieval_status', 'sufficient')
        evidence_stance_str = result.get('evidence_stance')
        evidence_stance = evidence_stance_str if evidence_stance_str else None
        
        # Handle citation data
        citations = result.get('citations', [])
        citation_data = []
        if isinstance(citations, list):
            citation_data = citations
        elif isinstance(citations, dict):
            citation_data = [citations]
        
        # Handle rag trace data
        rag_trace_data = result.get('rag_trace', {})
        if not isinstance(rag_trace_data, dict):
            rag_trace_data = {}
        
        # Handle tool trace
        tool_trace = result.get('tool_trace', [])
        if not isinstance(tool_trace, list):
            tool_trace = []
        
        # Extract final answer if available
        final_answer_text = result.get('final_answer', result.get('response', None))
        
        # Extract actual tool calls
        actual_tool_calls = result.get('actual_tool_calls', [])
        if not isinstance(actual_tool_calls, list):
            actual_tool_calls = []
        
        # Create the evaluation result
        eval_result = RagEvalResult(
            case_id=case.case_id,
            mode="tooluse",
            knowledge_score=knowledge_score,
            evaluation_status=evaluation_status,
            human_review_needed=human_review_needed,
            review_reason=review_reason,
            retrieval_status=retrieval_status,
            evidence_stance=evidence_stance,
            citation_data=citation_data,
            rag_trace_data=rag_trace_data,
            tool_trace=tool_trace,
            latency_ms=latency_ms,
            error=None,
            actual_tool_calls=actual_tool_calls,
            final_answer_text=final_answer_text
        )
        
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        eval_result = RagEvalResult(
            case_id=case.case_id,
            mode="tooluse",
            knowledge_score=None,
            evaluation_status="error",
            human_review_needed=True,
            review_reason="system_exception",
            retrieval_status="error",
            evidence_stance=None,
            citation_data=[],
            rag_trace_data={},
            tool_trace=[],
            latency_ms=latency_ms,
            error=str(e),
            actual_tool_calls=[],
            final_answer_text=None
        )
    
    return eval_result


async def run_case(case: RagGoldCase, mode: str) -> RagEvalResult:
    """
    Run a single case in the specified mode.
    
    Args:
        case: The gold case to evaluate
        mode: Evaluation mode ('legacy', 'tooluse', 'mock')
        
    Returns:
        Evaluation result
    """
    if mode == "legacy":
        return await run_case_legacy(case)
    elif mode == "tooluse":
        return await run_case_tooluse(case)
    elif mode == "mock":
        # Return a mock result for smoke testing
        return RagEvalResult(
            case_id=case.case_id,
            mode="mock",
            knowledge_score=85.0,
            evaluation_status="completed",
            human_review_needed=False,
            review_reason=None,
            retrieval_status="sufficient",
            evidence_stance="supports",
            citation_data=[
                {"id": "mock-citation-1", "text": "Mock citation text", "source": "Mock source"}
            ],
            rag_trace_data={
                "queries": ["mock query"],
                "retrieved_docs": ["mock-doc-1", "mock-doc-2"],
                "processed_at": "2023-01-01T00:00:00Z"
            },
            tool_trace=[
                {
                    "name": "mock_tool",
                    "status": "success",
                    "input": {"query": "mock"},
                    "output": {"result": "mock"},
                    "latency_ms": 100
                }
            ],
            latency_ms=100,
            error=None,
            actual_tool_calls=[{
                "name": "mock_tool",
                "params": {"query": "mock"},
                "result": "mock_result"
            }],
            final_answer_text="This is a mock answer for testing purposes."
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


async def run_evaluation(
    cases: List[RagGoldCase], 
    mode: str, 
    limit: Optional[int] = None
) -> List[RagEvalResult]:
    """
    Run evaluation on a list of cases.
    
    Args:
        cases: List of gold cases to evaluate
        mode: Evaluation mode ('legacy', 'tooluse', 'both', 'mock')
        limit: Maximum number of cases to run (None for all)
        
    Returns:
        List of evaluation results
    """
    if limit is not None:
        cases = cases[:limit]
    
    results = []
    
    if mode == "both":
        # Run both legacy and tooluse modes
        for case in cases:
            # Run legacy
            legacy_result = await run_case(case, "legacy")
            results.append(legacy_result)
            
            # Run tooluse
            tooluse_result = await run_case(case, "tooluse")
            results.append(tooluse_result)
    else:
        # Run single mode
        for case in cases:
            result = await run_case(case, mode)
            results.append(result)
    
    return results


def filter_cases_by_split(cases: List[RagGoldCase], split: str) -> List[RagGoldCase]:
    """
    Filter cases by split type.
    
    Args:
        cases: List of gold cases
        split: Split type to filter by ('dev', 'test', 'regression')
        
    Returns:
        Filtered list of cases
    """
    return [case for case in cases if (case.split.value if hasattr(case.split, 'value') else case.split) == split]


def create_mock_cases(count: int = 5) -> List[RagGoldCase]:
    """
    Create mock cases for smoke testing.
    
    Args:
        count: Number of mock cases to create
        
    Returns:
        List of mock cases
    """
    cases = []
    for i in range(count):
        case = RagGoldCase(
            case_id=f"mock_case_{i+1:03d}",
            split="dev",
            department="内科",
            domain_expertise="通科",
            difficulty="easy",
            chief_complaint="常规检查",
            patient_info=f"患者，性别不详，年龄不详，常规检查。编号: {i+1}",
            conversation_text="医生: 您哪里不舒服？\n患者: 就是想做个常规检查。",
            doctor_diagnosis="健康体检",
            treatment_plan="建议定期体检，保持健康生活方式。",
            gold_queries=["健康体检 常规检查 指南"],
            gold_doc_ids=[f"health-guidelines-{i+1}"],
            gold_citation_ids=[f"mock-citation-{i+1}"],
            gold_relevant_sources=["健康管理指南"],
            gold_citation_keywords=["体检", "常规", "健康"],
            gold_relevance_grades={f"health-guidelines-{i+1}": 3},
            expected_stance="supports",
            should_refuse=False,
            expected_score_range=[80, 100],
            expected_tool_calls=[{
                "name": "search_health_info",
                "params": {"query": f"mock query {i+1}"}
            }],
            expected_tool_params={
                "search_health_info": {"query_contains": ["health", "checkup"]}
            },
            expected_final_answer_keywords=["体检", "常规", "健康"],
            notes="Mock case for smoke testing"
        )
        cases.append(case)
    
    return cases


# ---------------------------------------------------------------------------
# Query-type classification helper
# ---------------------------------------------------------------------------

def classify_query_type(case: RagGoldCase) -> str:
    """根据 gold case 的属性判断查询类型。

    返回以下类别之一:
    - ``"referral"``  : 需要转诊 (should_refuse=True)
    - ``"citation"``  : 需要引用 (gold_citation_ids 非空)
    - ``"information"``: 普通信息查询
    """
    if case.should_refuse:
        return "referral"
    if case.gold_citation_ids:
        return "citation"
    return "information"


def group_cases_by_query_type(
    cases: List[RagGoldCase],
) -> Dict[str, List[RagGoldCase]]:
    """将 gold cases 按查询类型分组。"""
    groups: Dict[str, List[RagGoldCase]] = {
        "information": [],
        "citation": [],
        "referral": [],
    }
    for case in cases:
        qt = classify_query_type(case)
        groups[qt].append(case)
    return groups


# ---------------------------------------------------------------------------
# High-level evaluation runners
# ---------------------------------------------------------------------------

async def run_legacy_rag_evaluation(
    cases_path: Optional[Path] = None,
    split: str = "dev",
    limit: Optional[int] = None,
    config: Optional[EvalConfig] = None,
) -> List[RagEvalResult]:
    """执行传统 RAG 评估流程。

    加载 gold cases 数据集，对每个案例依次运行传统 RAG 系统
    （``run_knowledge_check``），返回评估结果列表。

    Args:
        cases_path: gold cases JSONL 文件路径。为 None 时使用 config 中的路径。
        split: 数据集分割 (``dev`` / ``test`` / ``regression``)。
        limit: 最多评估多少条案例，None 表示全部。
        config: 可选的评估配置，未提供时使用 ``DEFAULT_CONFIG``。

    Returns:
        评估结果列表，每条对应一个 gold case。
    """
    cfg = config or DEFAULT_CONFIG
    path = cases_path or cfg.cases_path

    logger.info("[Legacy RAG] 加载 gold cases: %s (split=%s)", path, split)
    gold_cases = load_gold_cases(path)
    logger.info("[Legacy RAG] 共加载 %d 条 gold cases", len(gold_cases))

    # 按 split 过滤
    gold_cases = filter_cases_by_split(gold_cases, split)
    logger.info("[Legacy RAG] split=%s 过滤后剩余 %d 条", split, len(gold_cases))

    if limit is not None:
        gold_cases = gold_cases[:limit]
        logger.info("[Legacy RAG] limit=%d，最终评估 %d 条", limit, len(gold_cases))

    if not gold_cases:
        logger.warning("[Legacy RAG] 没有可评估的案例")
        return []

    # 按查询类型分组并记录
    groups = group_cases_by_query_type(gold_cases)
    for qt, qt_cases in groups.items():
        logger.info("[Legacy RAG] 查询类型 '%s': %d 条", qt, len(qt_cases))

    results: List[RagEvalResult] = []
    for idx, case in enumerate(gold_cases, 1):
        query_type = classify_query_type(case)
        logger.info(
            "[Legacy RAG] (%d/%d) 评估案例 %s [type=%s]",
            idx, len(gold_cases), case.case_id, query_type,
        )
        try:
            result = await run_case_legacy(case)
            results.append(result)
        except Exception as exc:
            logger.error(
                "[Legacy RAG] 案例 %s 运行异常: %s", case.case_id, exc,
            )
            # 生成错误结果以保证列表完整性
            results.append(RagEvalResult(
                case_id=case.case_id,
                mode="legacy",
                knowledge_score=None,
                evaluation_status="error",
                human_review_needed=True,
                review_reason="system_exception",
                retrieval_status="error",
                evidence_stance=None,
                citation_data=[],
                rag_trace_data={},
                tool_trace=[],
                latency_ms=None,
                error=str(exc),
                actual_tool_calls=[],
                final_answer_text=None,
            ))

    logger.info("[Legacy RAG] 评估完成，共 %d 条结果", len(results))
    return results


async def run_tool_use_evaluation(
    cases_path: Optional[Path] = None,
    split: str = "dev",
    limit: Optional[int] = None,
    config: Optional[EvalConfig] = None,
) -> List[RagEvalResult]:
    """执行 Tool Use RAG 评估流程。

    加载 gold cases 数据集，对每个案例运行 Tool Use 系统
    （``run_knowledge_check_with_tools``），返回评估结果列表。

    Args:
        cases_path: gold cases JSONL 文件路径。为 None 时使用 config 中的路径。
        split: 数据集分割。
        limit: 最多评估多少条案例，None 表示全部。
        config: 可选的评估配置。

    Returns:
        评估结果列表。
    """
    cfg = config or DEFAULT_CONFIG
    path = cases_path or cfg.cases_path

    logger.info("[Tool Use] 加载 gold cases: %s (split=%s)", path, split)
    gold_cases = load_gold_cases(path)
    logger.info("[Tool Use] 共加载 %d 条 gold cases", len(gold_cases))

    gold_cases = filter_cases_by_split(gold_cases, split)
    logger.info("[Tool Use] split=%s 过滤后剩余 %d 条", split, len(gold_cases))

    if limit is not None:
        gold_cases = gold_cases[:limit]
        logger.info("[Tool Use] limit=%d，最终评估 %d 条", limit, len(gold_cases))

    if not gold_cases:
        logger.warning("[Tool Use] 没有可评估的案例")
        return []

    groups = group_cases_by_query_type(gold_cases)
    for qt, qt_cases in groups.items():
        logger.info("[Tool Use] 查询类型 '%s': %d 条", qt, len(qt_cases))

    results: List[RagEvalResult] = []
    for idx, case in enumerate(gold_cases, 1):
        query_type = classify_query_type(case)
        logger.info(
            "[Tool Use] (%d/%d) 评估案例 %s [type=%s]",
            idx, len(gold_cases), case.case_id, query_type,
        )
        try:
            result = await run_case_tooluse(case)
            results.append(result)
        except Exception as exc:
            logger.error(
                "[Tool Use] 案例 %s 运行异常: %s", case.case_id, exc,
            )
            results.append(RagEvalResult(
                case_id=case.case_id,
                mode="tooluse",
                knowledge_score=None,
                evaluation_status="error",
                human_review_needed=True,
                review_reason="system_exception",
                retrieval_status="error",
                evidence_stance=None,
                citation_data=[],
                rag_trace_data={},
                tool_trace=[],
                latency_ms=None,
                error=str(exc),
                actual_tool_calls=[],
                final_answer_text=None,
            ))

    logger.info("[Tool Use] 评估完成，共 %d 条结果", len(results))
    return results


async def run_batch_evaluation(
    cases_path: Optional[Path] = None,
    split: str = "dev",
    limit: Optional[int] = None,
    max_concurrency: int = 4,
    config: Optional[EvalConfig] = None,
) -> Dict[str, Any]:
    """对比 Legacy RAG 与 Tool Use 两种方法的性能。

    对同一组 gold cases 分别运行两种模式，支持通过
    ``asyncio.Semaphore`` 控制并发度，最终生成综合评估报告。

    Args:
        cases_path: gold cases JSONL 文件路径。
        split: 数据集分割。
        limit: 最多评估多少条案例。
        max_concurrency: 最大并发数（控制同时处理的案例数）。
        config: 可选的评估配置。

    Returns:
        综合评估报告字典，包含:
        - ``legacy``: Legacy 模式的结果列表与报告
        - ``tooluse``: Tool Use 模式的结果列表与报告
        - ``comparison``: 两种模式的对比摘要
        - ``query_type_breakdown``: 按查询类型分组的统计
    """
    cfg = config or DEFAULT_CONFIG
    path = cases_path or cfg.cases_path

    logger.info(
        "[Batch] 开始批量评估: path=%s, split=%s, limit=%s, concurrency=%d",
        path, split, limit, max_concurrency,
    )

    # ── 加载 & 预处理 ──
    gold_cases = load_gold_cases(path)
    gold_cases = filter_cases_by_split(gold_cases, split)
    if limit is not None:
        gold_cases = gold_cases[:limit]

    if not gold_cases:
        logger.warning("[Batch] 没有可评估的案例")
        return {
            "legacy": {"results": [], "report": {}, "elapsed_seconds": 0.0},
            "tooluse": {"results": [], "report": {}, "elapsed_seconds": 0.0},
            "comparison": {},
            "query_type_breakdown": {},
        }

    logger.info("[Batch] 共 %d 条案例待评估", len(gold_cases))

    # 按查询类型分组并记录
    groups = group_cases_by_query_type(gold_cases)
    for qt, qt_cases in groups.items():
        logger.info("[Batch] 查询类型 '%s': %d 条", qt, len(qt_cases))

    # ── 并发执行 Legacy ──
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_legacy_limited(case: RagGoldCase) -> RagEvalResult:
        async with semaphore:
            return await run_case_legacy(case)

    logger.info("[Batch] 运行 Legacy RAG (%d 条, 并发=%d)", len(gold_cases), max_concurrency)
    legacy_start = time.time()
    legacy_results: List[RagEvalResult] = await asyncio.gather(
        *[_run_legacy_limited(c) for c in gold_cases],
        return_exceptions=False,
    )
    legacy_elapsed = time.time() - legacy_start
    logger.info("[Batch] Legacy RAG 完成, 耗时 %.2fs", legacy_elapsed)

    # ── 并发执行 Tool Use ──
    async def _run_tooluse_limited(case: RagGoldCase) -> RagEvalResult:
        async with semaphore:
            return await run_case_tooluse(case)

    logger.info("[Batch] 运行 Tool Use (%d 条, 并发=%d)", len(gold_cases), max_concurrency)
    tooluse_start = time.time()
    tooluse_results: List[RagEvalResult] = await asyncio.gather(
        *[_run_tooluse_limited(c) for c in gold_cases],
        return_exceptions=False,
    )
    tooluse_elapsed = time.time() - tooluse_start
    logger.info("[Batch] Tool Use 完成, 耗时 %.2fs", tooluse_elapsed)

    # ── 汇总报告 ──
    legacy_report = generate_json_report(
        results=legacy_results,
        gold_cases=gold_cases,
        mode="legacy",
        dataset_path=str(path),
        split=split,
    )
    tooluse_report = generate_json_report(
        results=tooluse_results,
        gold_cases=gold_cases,
        mode="tooluse",
        dataset_path=str(path),
        split=split,
    )

    # ── 对比摘要 ──
    comparison = _build_comparison_summary(
        legacy_results, tooluse_results, gold_cases,
        legacy_elapsed, tooluse_elapsed,
    )

    # ── 按查询类型分组统计 ──
    query_type_breakdown = _build_query_type_breakdown(
        legacy_results, tooluse_results, gold_cases,
    )

    logger.info("[Batch] 批量评估完成")

    return {
        "legacy": {
            "results": legacy_results,
            "report": legacy_report,
            "elapsed_seconds": legacy_elapsed,
        },
        "tooluse": {
            "results": tooluse_results,
            "report": tooluse_report,
            "elapsed_seconds": tooluse_elapsed,
        },
        "comparison": comparison,
        "query_type_breakdown": query_type_breakdown,
    }


# ---------------------------------------------------------------------------
# Batch evaluation helpers
# ---------------------------------------------------------------------------

def _build_comparison_summary(
    legacy_results: List[RagEvalResult],
    tooluse_results: List[RagEvalResult],
    gold_cases: List[RagGoldCase],
    legacy_elapsed: float,
    tooluse_elapsed: float,
) -> Dict[str, Any]:
    """构建两种模式的对比摘要。"""

    def _avg_score(results: List[RagEvalResult]) -> Optional[float]:
        scores = [r.knowledge_score for r in results if r.knowledge_score is not None]
        return sum(scores) / len(scores) if scores else None

    def _avg_latency(results: List[RagEvalResult]) -> Optional[float]:
        lats = [r.latency_ms for r in results if r.latency_ms is not None]
        return sum(lats) / len(lats) if lats else None

    def _error_count(results: List[RagEvalResult]) -> int:
        return sum(1 for r in results if r.error is not None)

    def _review_count(results: List[RagEvalResult]) -> int:
        return sum(1 for r in results if r.human_review_needed)

    legacy_avg_score = _avg_score(legacy_results)
    tooluse_avg_score = _avg_score(tooluse_results)
    legacy_avg_latency = _avg_latency(legacy_results)
    tooluse_avg_latency = _avg_latency(tooluse_results)

    return {
        "legacy": {
            "avg_knowledge_score": legacy_avg_score,
            "avg_latency_ms": legacy_avg_latency,
            "total_elapsed_seconds": legacy_elapsed,
            "error_count": _error_count(legacy_results),
            "review_needed_count": _review_count(legacy_results),
        },
        "tooluse": {
            "avg_knowledge_score": tooluse_avg_score,
            "avg_latency_ms": tooluse_avg_latency,
            "total_elapsed_seconds": tooluse_elapsed,
            "error_count": _error_count(tooluse_results),
            "review_needed_count": _review_count(tooluse_results),
        },
        "delta": {
            "avg_knowledge_score": (
                (tooluse_avg_score - legacy_avg_score)
                if legacy_avg_score is not None and tooluse_avg_score is not None
                else None
            ),
            "avg_latency_ms": (
                (tooluse_avg_latency - legacy_avg_latency)
                if legacy_avg_latency is not None and tooluse_avg_latency is not None
                else None
            ),
        },
    }


def _build_query_type_breakdown(
    legacy_results: List[RagEvalResult],
    tooluse_results: List[RagEvalResult],
    gold_cases: List[RagGoldCase],
) -> Dict[str, Dict[str, Any]]:
    """按查询类型（information / citation / referral）分组统计。"""
    groups = group_cases_by_query_type(gold_cases)
    case_id_to_idx = {gc.case_id: i for i, gc in enumerate(gold_cases)}

    breakdown: Dict[str, Dict[str, Any]] = {}

    for qt, qt_cases in groups.items():
        if not qt_cases:
            continue

        # 取出该组对应的结果
        qt_indices = [case_id_to_idx[c.case_id] for c in qt_cases]
        qt_legacy = [legacy_results[i] for i in qt_indices]
        qt_tooluse = [tooluse_results[i] for i in qt_indices]

        def _avg(lst: List[Optional[float]]) -> Optional[float]:
            vals = [v for v in lst if v is not None]
            return sum(vals) / len(vals) if vals else None

        breakdown[qt] = {
            "count": len(qt_cases),
            "legacy": {
                "avg_knowledge_score": _avg([r.knowledge_score for r in qt_legacy]),
                "avg_latency_ms": _avg([r.latency_ms for r in qt_legacy]),
                "error_count": sum(1 for r in qt_legacy if r.error),
            },
            "tooluse": {
                "avg_knowledge_score": _avg([r.knowledge_score for r in qt_tooluse]),
                "avg_latency_ms": _avg([r.latency_ms for r in qt_tooluse]),
                "error_count": sum(1 for r in qt_tooluse if r.error),
            },
        }

    return breakdown
