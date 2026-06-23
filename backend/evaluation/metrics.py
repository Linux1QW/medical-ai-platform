"""
Core metrics calculation for RAG evaluation.
"""
import math
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from .datasets import RagEvalResult, RagGoldCase, StanceType


def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    """
    Calculate Recall@K metric.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ranked order)
        gold_ids: List of ground truth relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Recall@K value (0.0 to 1.0)
    """
    if not gold_ids:
        return 0.0  # Cannot calculate recall when there are no gold standards
    
    if len(retrieved_ids) == 0:
        return 0.0
    
    # Take top-k retrieved documents
    top_k_retrieved = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    
    # Count how many gold IDs are in the top-k retrieved
    hits = len(top_k_retrieved.intersection(gold_set))
    
    # Calculate recall
    recall = hits / len(gold_set)
    return min(recall, 1.0)  # Cap at 1.0


def mrr(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved_ids: List of retrieved document IDs (ranked order)
        gold_ids: List of ground truth relevant document IDs
        
    Returns:
        MRR value (0.0 to 1.0)
    """
    if not gold_ids or len(retrieved_ids) == 0:
        return 0.0
    
    gold_set = set(gold_ids)
    
    # Find the rank of the first relevant document
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in gold_set:
            return 1.0 / rank
    
    # No relevant document found in the retrieved list
    return 0.0


def dcg_at_k(retrieved_ids: List[str], relevance_grades: Dict[str, int], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at K.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ranked order)
        relevance_grades: Dictionary mapping doc_id to relevance grade (0-3)
        k: Number of top results to consider
        
    Returns:
        DCG@K value
    """
    if not relevance_grades or len(retrieved_ids) == 0:
        return 0.0
    
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rank = i + 1
        relevance = relevance_grades.get(doc_id, 0)
        
        # DCG formula: rel_1 + sum(rel_i/log2(i+1)) for i=2 to k
        if rank == 1:
            dcg += relevance
        else:
            dcg += relevance / math.log2(rank)
    
    return dcg


def idcg_at_k(relevance_grades: Dict[str, int], k: int) -> float:
    """
    Calculate Ideal Discounted Cumulative Gain at K.
    
    Args:
        relevance_grades: Dictionary mapping doc_id to relevance grade (0-3)
        k: Number of top results to consider
        
    Returns:
        IDCG@K value
    """
    if not relevance_grades:
        return 0.0
    
    # Sort relevance grades in descending order to get ideal ranking
    sorted_relevances = sorted(relevance_grades.values(), reverse=True)[:k]
    
    idcg = 0.0
    for i, relevance in enumerate(sorted_relevances):
        rank = i + 1
        if rank == 1:
            idcg += relevance
        else:
            idcg += relevance / math.log2(rank)
    
    return idcg


def ndcg_at_k(retrieved_ids: List[str], relevance_grades: Dict[str, int], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ranked order)
        relevance_grades: Dictionary mapping doc_id to relevance grade (0-3)
        k: Number of top results to consider
        
    Returns:
        nDCG@K value (0.0 to 1.0)
    """
    if not relevance_grades or len(retrieved_ids) == 0:
        return 0.0
    
    dcg = dcg_at_k(retrieved_ids, relevance_grades, k)
    idcg = idcg_at_k(relevance_grades, k)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def citation_validity(used_ids: List[str], allowed_ids: Set[str]) -> float:
    """
    Calculate citation validity rate.
    
    Args:
        used_ids: List of citation IDs used by the system
        allowed_ids: Set of valid citation IDs that could be used
        
    Returns:
        Citation validity rate (0.0 to 1.0)
    """
    if not used_ids:
        return 1.0  # No citations used means 100% validity
    
    valid_count = sum(1 for cid in used_ids if cid in allowed_ids)
    return valid_count / len(used_ids)


def citation_hallucination_rate(used_ids: List[str], allowed_ids: Set[str]) -> float:
    """
    Calculate citation hallucination rate.
    
    Args:
        used_ids: List of citation IDs used by the system
        allowed_ids: Set of valid citation IDs that could be used
        
    Returns:
        Citation hallucination rate (0.0 to 1.0)
    """
    if not used_ids:
        return 0.0  # No citations used means 0% hallucination
    
    hallucinated_count = sum(1 for cid in used_ids if cid not in allowed_ids)
    return hallucinated_count / len(used_ids)


def citation_coverage(used_ids: List[str], gold_ids: List[str]) -> float:
    """
    Calculate citation coverage of gold keywords.
    
    Args:
        used_ids: List of citation IDs used by the system
        gold_ids: List of expected citation IDs
        
    Returns:
        Citation coverage rate (0.0 to 1.0)
    """
    if not gold_ids:
        return 1.0  # No gold citations expected means 100% coverage
    
    used_set = set(used_ids)
    gold_set = set(gold_ids)
    
    covered = gold_set.intersection(used_set)
    return len(covered) / len(gold_set)


def stance_accuracy(results: List[Tuple[StanceType, StanceType]]) -> float:
    """
    Calculate stance accuracy.
    
    Args:
        results: List of (predicted_stance, expected_stance) tuples
        
    Returns:
        Stance accuracy rate (0.0 to 1.0)
    """
    if not results:
        return 0.0
    
    correct = sum(1 for pred, expected in results if pred == expected)
    return correct / len(results)


def score_range_accuracy(eval_results: List[RagEvalResult], gold_cases: List[RagGoldCase]) -> float:
    """
    Calculate score range accuracy for non-refusal cases.
    
    Args:
        eval_results: List of evaluation results
        gold_cases: List of corresponding gold cases
        
    Returns:
        Score range accuracy rate (0.0 to 1.0)
    """
    pairs_with_ranges = [
        (eval_res, gold_case) 
        for eval_res, gold_case in zip(eval_results, gold_cases) 
        if gold_case.expected_score_range is not None and eval_res.knowledge_score is not None
    ]
    
    if not pairs_with_ranges:
        return 0.0
    
    correct = 0
    for eval_res, gold_case in pairs_with_ranges:
        min_score, max_score = gold_case.expected_score_range
        if min_score <= eval_res.knowledge_score <= max_score:
            correct += 1
    
    return correct / len(pairs_with_ranges)


def compute_system_refused(result: RagEvalResult) -> bool:
    """
    Determine if the system refused based on our unified definition.
    
    Args:
        result: Evaluation result
        
    Returns:
        True if system refused, False otherwise
    """
    # Check if knowledge score is null
    if result.knowledge_score is None:
        return True
    
    # Check if evaluation status indicates need for review
    if result.evaluation_status == "needs_review":
        return True
    
    # Check if human review is needed for specific refusal reasons
    refusal_reasons = [
        "insufficient_evidence",
        "knowledge_undetermined", 
        "citation_verification_failed",
        "retrieval_error",
        "system_exception"
    ]
    
    if (result.human_review_needed and 
        result.review_reason in refusal_reasons):
        return True
    
    return False


def compute_false_acceptance(result: RagEvalResult, gold_case: RagGoldCase) -> bool:
    """
    Determine if this is a false acceptance (should refuse but didn't).
    
    Args:
        result: Evaluation result
        gold_case: Corresponding gold case
        
    Returns:
        True if false acceptance occurred, False otherwise
    """
    return gold_case.should_refuse and not result.system_refused


def refusal_metrics_from_results(results: List[RagEvalResult], gold_cases: List[RagGoldCase]) -> Dict[str, float]:
    """
    Calculate all refusal-related metrics from RagEvalResult objects.

    这是基于模型对象的拒答指标计算函数，适用于已有 RagEvalResult 和
    RagGoldCase 对象的场景。对于简单的布尔预测列表，请使用
    :func:`refusal_metrics`。

    Args:
        results: List of evaluation results
        gold_cases: List of corresponding gold cases
        
    Returns:
        Dictionary containing refusal metrics
    """
    # Precompute system_refused and false_acceptance for each result
    for result in results:
        result.system_refused = compute_system_refused(result)
    
    for result, gold_case in zip(results, gold_cases):
        result.false_acceptance = compute_false_acceptance(result, gold_case)
    
    # Count different types of cases
    total_cases = len(results)
    if total_cases == 0:
        return {
            "refusal_accuracy": 0.0,
            "refusal_precision": 0.0,
            "refusal_recall": 0.0,
            "refusal_f1": 0.0,
            "false_refusal_rate": 0.0,
            "false_acceptance_rate": 0.0
        }
    
    # Count correct refusals (where system refused when it should have)
    correct_refusals = sum(
        1 for result, gold_case in zip(results, gold_cases)
        if result.system_refused and gold_case.should_refuse
    )
    
    # Count incorrect refusals (where system refused when it shouldn't have)
    incorrect_refusals = sum(
        1 for result, gold_case in zip(results, gold_cases)
        if result.system_refused and not gold_case.should_refuse
    )
    
    # Count correct acceptances (where system didn't refuse when it shouldn't have)
    correct_acceptances = sum(
        1 for result, gold_case in zip(results, gold_cases)
        if not result.system_refused and not gold_case.should_refuse
    )
    
    # Count incorrect acceptances (where system didn't refuse when it should have)
    incorrect_acceptances = sum(
        1 for result, gold_case in zip(results, gold_cases)
        if not result.system_refused and gold_case.should_refuse
    )
    
    # Calculate basic metrics
    total_should_refuse = sum(1 for gc in gold_cases if gc.should_refuse)
    total_should_not_refuse = total_cases - total_should_refuse
    total_system_refusals = sum(1 for r in results if r.system_refused)
    
    # Accuracy: (correct_refusals + correct_acceptances) / total
    refusal_accuracy = (correct_refusals + correct_acceptances) / total_cases if total_cases > 0 else 0.0
    
    # Precision: correct_refusals / total_system_refusals
    refusal_precision = correct_refusals / total_system_refusals if total_system_refusals > 0 else 0.0
    
    # Recall: correct_refusals / total_should_refuse
    refusal_recall = correct_refusals / total_should_refuse if total_should_refuse > 0 else 0.0
    
    # F1: harmonic mean of precision and recall
    if refusal_precision + refusal_recall > 0:
        refusal_f1 = 2 * (refusal_precision * refusal_recall) / (refusal_precision + refusal_recall)
    else:
        refusal_f1 = 0.0
    
    # False refusal rate: incorrect_refusals / total_should_not_refuse
    false_refusal_rate = incorrect_refusals / total_should_not_refuse if total_should_not_refuse > 0 else 0.0
    
    # False acceptance rate: incorrect_acceptances / total_should_refuse
    false_acceptance_rate = incorrect_acceptances / total_should_refuse if total_should_refuse > 0 else 0.0
    
    return {
        "refusal_accuracy": refusal_accuracy,
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1": refusal_f1,
        "false_refusal_rate": false_refusal_rate,
        "false_acceptance_rate": false_acceptance_rate
    }


def tool_metrics(results: List[RagEvalResult]) -> Dict[str, float]:
    """
    Calculate all Tool Use-related metrics.
    
    Args:
        results: List of evaluation results containing tool traces
        
    Returns:
        Dictionary containing tool metrics
    """
    if not results:
        return {
            "tool_success_rate": 0.0,
            "tool_failure_rate": 0.0,
            "tool_budget_exceeded_rate": 0.0,
            "avg_tool_calls": 0.0,
            "avg_latency_ms": 0.0
        }
    
    total_calls = 0
    successful_calls = 0
    failed_calls = 0
    budget_exceeded_calls = 0
    total_latency = 0
    samples_with_latency = 0
    
    for result in results:
        # Count tool calls from trace
        sample_calls = len(result.tool_trace)
        total_calls += sample_calls
        
        # Analyze each tool call in the trace
        for tool_call in result.tool_trace:
            status = tool_call.get('status', 'unknown')
            if status == 'success':
                successful_calls += 1
            elif status in ['error', 'timeout']:
                failed_calls += 1
            elif status == 'budget_exceeded':
                budget_exceeded_calls += 1
        
        # Track latency
        if result.latency_ms is not None:
            total_latency += result.latency_ms
            samples_with_latency += 1
    
    # Calculate rates
    tool_success_rate = successful_calls / total_calls if total_calls > 0 else 0.0
    tool_failure_rate = failed_calls / total_calls if total_calls > 0 else 0.0
    tool_budget_exceeded_rate = budget_exceeded_calls / total_calls if total_calls > 0 else 0.0
    avg_tool_calls = total_calls / len(results) if results else 0.0
    avg_latency_ms = total_latency / samples_with_latency if samples_with_latency > 0 else 0.0
    
    return {
        "tool_success_rate": tool_success_rate,
        "tool_failure_rate": tool_failure_rate,
        "tool_budget_exceeded_rate": tool_budget_exceeded_rate,
        "avg_tool_calls": avg_tool_calls,
        "avg_latency_ms": avg_latency_ms
    }


def tool_breakdown(results: List[RagEvalResult]) -> Dict[str, Dict[str, float]]:
    """
    Calculate tool breakdown metrics by tool name.
    
    Args:
        results: List of evaluation results containing tool traces
        
    Returns:
        Dictionary containing tool breakdown metrics
    """
    tool_stats = defaultdict(lambda: {
        "calls": 0,
        "successes": 0,
        "failures": 0,
        "total_latency_ms": 0,
        "call_count": 0
    })
    
    for result in results:
        for tool_call in result.tool_trace:
            tool_name = tool_call.get('name', 'unknown')
            status = tool_call.get('status', 'unknown')
            latency = tool_call.get('latency_ms', 0)
            
            stats = tool_stats[tool_name]
            stats["calls"] += 1
            stats["total_latency_ms"] += latency
            
            if status == 'success':
                stats["successes"] += 1
            elif status in ['error', 'timeout', 'budget_exceeded']:
                stats["failures"] += 1
    
    # Calculate derived metrics
    breakdown = {}
    for tool_name, stats in tool_stats.items():
        total_calls = stats["calls"]
        successes = stats["successes"]
        total_latency = stats["total_latency_ms"]
        
        breakdown[tool_name] = {
            "calls": total_calls,
            "success_rate": successes / total_calls if total_calls > 0 else 0.0,
            "avg_latency_ms": total_latency / total_calls if total_calls > 0 else 0.0
        }
    
    return breakdown


def aggregate_metrics_by_dimension(
    results: List[RagEvalResult], 
    gold_cases: List[RagGoldCase], 
    dimension: str
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics by a specific dimension (difficulty, department, etc.).
    
    Args:
        results: List of evaluation results
        gold_cases: List of corresponding gold cases
        dimension: Dimension to group by ('difficulty', 'department', etc.)
        
    Returns:
        Dictionary mapping dimension values to metrics
    """
    groups = defaultdict(list)
    
    for result, gold_case in zip(results, gold_cases):
        if hasattr(gold_case, dimension):
            dim_value = getattr(gold_case, dimension)
            if dim_value is not None:
                groups[dim_value].append((result, gold_case))
    
    aggregated = {}
    for dim_value, group_pairs in groups.items():
        group_results = [pair[0] for pair in group_pairs]
        group_gold_cases = [pair[1] for pair in group_pairs]
        
        # Calculate basic metrics for this group
        refusal_mets = refusal_metrics_from_results(group_results, group_gold_cases)
        
        # Add other metrics here as needed
        aggregated[dim_value] = refusal_mets
    
    return aggregated


def final_answer_keyword_coverage(eval_results: List[RagEvalResult], gold_cases: List[RagGoldCase]) -> float:
    """
    Calculate coverage of expected keywords in final answers.
    
    Args:
        eval_results: List of evaluation results
        gold_cases: List of corresponding gold cases
        
    Returns:
        Keyword coverage rate (0.0 to 1.0)
    """
    total_expected_keywords = 0
    total_covered_keywords = 0
    
    for result, gold_case in zip(eval_results, gold_cases):
        expected_keywords = gold_case.expected_final_answer_keywords or []
        final_answer = result.final_answer_text or ""
        
        if not expected_keywords:
            continue
            
        total_expected_keywords += len(expected_keywords)
        
        # Count how many expected keywords appear in the final answer
        for keyword in expected_keywords:
            if keyword.lower() in final_answer.lower():
                total_covered_keywords += 1
    
    if total_expected_keywords == 0:
        return 1.0  # No expected keywords means 100% coverage
    
    return total_covered_keywords / total_expected_keywords


def tool_call_accuracy(eval_results: List[RagEvalResult], gold_cases: List[RagGoldCase]) -> float:
    """
    Calculate accuracy of tool calls compared to expected calls.
    
    Args:
        eval_results: List of evaluation results
        gold_cases: List of corresponding gold cases
        
    Returns:
        Tool call accuracy rate (0.0 to 1.0)
    """
    if not gold_cases:
        return 0.0
    
    correct_cases = 0
    
    for result, gold_case in zip(eval_results, gold_cases):
        expected_calls = gold_case.expected_tool_calls or []
        actual_calls = result.actual_tool_calls or []
        
        # Simple check: compare if both lists are empty or both have content
        if len(expected_calls) == 0 and len(actual_calls) == 0:
            correct_cases += 1
        elif len(expected_calls) > 0 and len(actual_calls) > 0:
            # More sophisticated comparison would go here
            # For now, just check if both have tool calls
            correct_cases += 1
    
    return correct_cases / len(gold_cases)


# ============================================================================
# 综合聚合指标函数
# ============================================================================


def retrieval_metrics(
    retrieved_ids_list: List[List[str]],
    gold_ids_list: List[List[str]],
    relevance_grades_list: Optional[List[Dict[str, int]]] = None,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """计算检索指标的综合聚合函数。

    对多组查询的检索结果进行汇总，计算 Recall@K、MRR、nDCG@K、MAP 等指标。

    Args:
        retrieved_ids_list: 每个查询对应的检索文档 ID 列表（按排序顺序），
            外层列表长度等于查询数量。
        gold_ids_list: 每个查询对应的标准相关文档 ID 列表，
            与 retrieved_ids_list 一一对应。
        relevance_grades_list: 可选，每个查询的相关性等级字典
            （doc_id -> grade 0-3）。若提供则计算 nDCG；
            若为 None 则默认所有相关文档 grade=1，不相关文档 grade=0。
        k_values: 需要计算的 K 值列表，默认 [1, 3, 5]。

    Returns:
        包含以下键的字典：
        - ``recall_at_{k}``: 各 K 值的平均 Recall@K
        - ``mrr``: 平均倒数排名 (Mean Reciprocal Rank)
        - ``ndcg_at_{k}``: 各 K 值的平均 nDCG@K
        - ``map``: 平均精度均值 (Mean Average Precision)

    Raises:
        ValueError: 当 retrieved_ids_list 与 gold_ids_list 长度不一致时抛出。

    Examples:
        >>> retrieval_metrics(
        ...     retrieved_ids_list=[["d1", "d2", "d3"], ["d4", "d5"]],
        ...     gold_ids_list=[["d1", "d3"], ["d5"]],
        ... )
        {'recall_at_1': 0.5, 'recall_at_3': 1.0, 'recall_at_5': 1.0,
         'mrr': 0.75, 'ndcg_at_1': ..., 'ndcg_at_3': ..., 'ndcg_at_5': ...,
         'map': ...}
    """
    if len(retrieved_ids_list) != len(gold_ids_list):
        raise ValueError(
            f"retrieved_ids_list ({len(retrieved_ids_list)}) 与 "
            f"gold_ids_list ({len(gold_ids_list)}) 长度不一致"
        )

    if k_values is None:
        k_values = [1, 3, 5]

    n_queries = len(retrieved_ids_list)
    if n_queries == 0:
        result: Dict[str, float] = {"mrr": 0.0, "map": 0.0}
        for k in k_values:
            result[f"recall_at_{k}"] = 0.0
            result[f"ndcg_at_{k}"] = 0.0
        return result

    # 如果没有提供 relevance_grades_list，根据 gold_ids 自动构建
    if relevance_grades_list is None:
        relevance_grades_list = []
        for gold_ids in gold_ids_list:
            grades = {doc_id: 1 for doc_id in gold_ids}
            relevance_grades_list.append(grades)
    else:
        if len(relevance_grades_list) != n_queries:
            raise ValueError(
                f"relevance_grades_list ({len(relevance_grades_list)}) 与 "
                f"retrieved_ids_list ({n_queries}) 长度不一致"
            )

    recall_sums = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    ndcg_sums = {k: 0.0 for k in k_values}
    ap_sum = 0.0

    for retrieved_ids, gold_ids, grades in zip(
        retrieved_ids_list, gold_ids_list, relevance_grades_list
    ):
        # Recall@K
        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved_ids, gold_ids, k)

        # MRR
        mrr_sum += mrr(retrieved_ids, gold_ids)

        # nDCG@K
        for k in k_values:
            ndcg_sums[k] += ndcg_at_k(retrieved_ids, grades, k)

        # Average Precision (AP)
        if gold_ids:
            gold_set = set(gold_ids)
            n_relevant = len(gold_set)
            hits = 0
            precision_sum = 0.0
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in gold_set:
                    hits += 1
                    precision_sum += hits / (i + 1)
            ap = precision_sum / n_relevant if n_relevant > 0 else 0.0
        else:
            ap = 0.0
        ap_sum += ap

    # 取平均
    result = {"mrr": mrr_sum / n_queries, "map": ap_sum / n_queries}
    for k in k_values:
        result[f"recall_at_{k}"] = recall_sums[k] / n_queries
        result[f"ndcg_at_{k}"] = ndcg_sums[k] / n_queries

    return result


def citation_metrics(
    used_citation_ids: List[str],
    allowed_citation_ids: Set[str],
    gold_citation_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """计算引用相关指标的综合聚合函数。

    评估系统生成回答时引用的文档是否有效、是否存在幻觉、覆盖率如何。

    Args:
        used_citation_ids: 系统实际使用的引用 ID 列表（可包含重复，
            表示多次引用同一文档）。
        allowed_citation_ids: 允许引用的有效文档 ID 集合。
        gold_citation_ids: 可选，期望的标准引用 ID 列表，
            用于计算覆盖率。若为 None 则覆盖率返回 None。

    Returns:
        包含以下键的字典：
        - ``citation_validity``: 引用有效率 (0.0-1.0)
        - ``citation_hallucination_rate``: 引用幻觉率 (0.0-1.0)
        - ``citation_coverage``: 引用覆盖率 (0.0-1.0)，若未提供
          gold_citation_ids 则为 None
        - ``unique_citation_count``: 去重后的引用数量
        - ``unique_valid_ratio``: 去重后引用的有效率

    Examples:
        >>> citation_metrics(
        ...     used_citation_ids=["doc1", "doc2", "doc3"],
        ...     allowed_citation_ids={"doc1", "doc2"},
        ...     gold_citation_ids=["doc1", "doc2"],
        ... )
        {'citation_validity': 0.666..., 'citation_hallucination_rate': 0.333...,
         'citation_coverage': 1.0, 'unique_citation_count': 3, 'unique_valid_ratio': 0.666...}
    """
    validity = citation_validity(used_citation_ids, allowed_citation_ids)
    hallucination = citation_hallucination_rate(used_citation_ids, allowed_citation_ids)

    if gold_citation_ids is not None:
        coverage = citation_coverage(used_citation_ids, gold_citation_ids)
    else:
        coverage = None

    # 去重统计
    unique_ids = set(used_citation_ids)
    unique_valid = len(unique_ids.intersection(allowed_citation_ids))
    unique_valid_ratio = unique_valid / len(unique_ids) if unique_ids else 1.0

    return {
        "citation_validity": validity,
        "citation_hallucination_rate": hallucination,
        "citation_coverage": coverage,
        "unique_citation_count": len(unique_ids),
        "unique_valid_ratio": unique_valid_ratio,
    }


def refusal_metrics(
    predictions: List[bool],
    labels: List[bool],
) -> Dict[str, Any]:
    """计算拒绝回答相关指标的综合聚合函数。

    将系统的拒绝预测与标准标签对比，计算准确率、精确率、召回率、F1 等。

    Args:
        predictions: 系统预测结果列表，True 表示拒绝回答，False 表示正常回答。
        labels: 标准标签列表，True 表示应该拒绝，False 表示不应该拒绝。
            长度必须与 predictions 一致。

    Returns:
        包含以下键的字典：
        - ``accuracy``: 准确率 (正确预测数 / 总数)
        - ``precision``: 精确率 (正确拒绝数 / 系统拒绝总数)
        - ``recall``: 召回率 (正确拒绝数 / 应拒绝总数)
        - ``f1``: F1 分数 (精确率和召回率的调和平均)
        - ``false_positive_rate``: 错误拒绝率 (不该拒绝但被拒绝 / 不该拒绝总数)
        - ``false_negative_rate``: 错误接受率 (该拒绝但未被拒绝 / 该拒绝总数)
        - ``support``: 样本总数

    Raises:
        ValueError: 当 predictions 与 labels 长度不一致时抛出。

    Examples:
        >>> refusal_metrics(
        ...     predictions=[True, True, False, False],
        ...     labels=[True, False, False, True],
        ... )
        {'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5,
         'false_positive_rate': 0.5, 'false_negative_rate': 0.5, 'support': 4}
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"predictions ({len(predictions)}) 与 "
            f"labels ({len(labels)}) 长度不一致"
        )

    n = len(predictions)
    if n == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "support": 0,
        }

    # 混淆矩阵
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)   # 正确拒绝
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)  # 错误拒绝
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)  # 错误接受
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)  # 正确接受

    accuracy = (tp + tn) / n

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total_positive = tp + fn  # 应该拒绝的总数
    total_negative = fp + tn  # 不应该拒绝的总数
    false_positive_rate = fp / total_negative if total_negative > 0 else 0.0
    false_negative_rate = fn / total_positive if total_positive > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "support": n,
    }


def tool_use_metrics(
    tool_call_logs: List[Dict[str, Any]],
    expected_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """计算工具使用相关指标的综合聚合函数。

    分析工具调用日志，计算成功率、响应时间、成本效益等指标。
    可选地与预期结果对比，计算工具调用的准确性。

    Args:
        tool_call_logs: 工具调用日志列表，每条记录为字典，支持以下字段：
            - ``name`` (str): 工具名称
            - ``status`` (str): 调用状态，可选值 'success' / 'error' /
              'timeout' / 'budget_exceeded'
            - ``latency_ms`` (float): 调用耗时（毫秒）
            - ``cost`` (float): 调用成本（可选）
            - ``result`` (Any): 调用返回结果（可选，用于与预期对比）
        expected_results: 可选，预期工具调用结果列表，与 tool_call_logs
            一一对应。每条记录可包含：
            - ``expected_tool`` (str): 预期应调用的工具名称
            - ``expected_output`` (Any): 预期输出

    Returns:
        包含以下键的字典：
        - ``total_calls``: 总调用次数
        - ``success_rate``: 成功率
        - ``failure_rate``: 失败率（error + timeout）
        - ``budget_exceeded_rate``: 预算超限率
        - ``avg_latency_ms``: 平均调用耗时（毫秒）
        - ``total_cost``: 总成本（若日志中提供 cost 字段）
        - ``avg_cost_per_call``: 平均每次调用成本
        - ``accuracy``: 工具调用准确率（需提供 expected_results，否则为 None）
        - ``per_tool``: 按工具名称分组的统计子字典

    Raises:
        ValueError: 当 expected_results 长度与 tool_call_logs 不一致时抛出。

    Examples:
        >>> logs = [
        ...     {"name": "search", "status": "success", "latency_ms": 120},
        ...     {"name": "search", "status": "error", "latency_ms": 5000},
        ...     {"name": "calculator", "status": "success", "latency_ms": 30},
        ... ]
        >>> result = tool_use_metrics(logs)
        >>> round(result["success_rate"], 3)
        0.667
    """
    if not tool_call_logs:
        return {
            "total_calls": 0,
            "success_rate": 0.0,
            "failure_rate": 0.0,
            "budget_exceeded_rate": 0.0,
            "avg_latency_ms": 0.0,
            "total_cost": 0.0,
            "avg_cost_per_call": 0.0,
            "accuracy": None,
            "per_tool": {},
        }

    total = len(tool_call_logs)
    successes = 0
    failures = 0
    budget_exceeded = 0
    total_latency = 0.0
    latency_count = 0
    total_cost = 0.0
    cost_count = 0

    # 按工具分组统计
    per_tool: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "calls": 0, "successes": 0, "failures": 0,
        "total_latency_ms": 0.0, "latency_count": 0,
        "total_cost": 0.0,
    })

    for log in tool_call_logs:
        name = log.get("name", "unknown")
        status = log.get("status", "unknown")
        latency = log.get("latency_ms")
        cost = log.get("cost")

        # 全局统计
        if status == "success":
            successes += 1
        elif status in ("error", "timeout"):
            failures += 1
        elif status == "budget_exceeded":
            budget_exceeded += 1

        if latency is not None:
            total_latency += latency
            latency_count += 1

        if cost is not None:
            total_cost += cost
            cost_count += 1

        # 分组统计
        tool_stat = per_tool[name]
        tool_stat["calls"] += 1
        if status == "success":
            tool_stat["successes"] += 1
        elif status in ("error", "timeout", "budget_exceeded"):
            tool_stat["failures"] += 1
        if latency is not None:
            tool_stat["total_latency_ms"] += latency
            tool_stat["latency_count"] += 1
        if cost is not None:
            tool_stat["total_cost"] += cost

    # 计算准确率（需要 expected_results）
    accuracy = None
    if expected_results is not None:
        if len(expected_results) != total:
            raise ValueError(
                f"expected_results ({len(expected_results)}) 与 "
                f"tool_call_logs ({total}) 长度不一致"
            )
        correct = 0
        for log, expected in zip(tool_call_logs, expected_results):
            expected_tool = expected.get("expected_tool")
            expected_output = expected.get("expected_output")
            actual_name = log.get("name")
            actual_result = log.get("result")

            tool_match = (expected_tool is None) or (actual_name == expected_tool)
            output_match = (expected_output is None) or (actual_result == expected_output)
            if tool_match and output_match:
                correct += 1
        accuracy = correct / total

    # 汇总分组指标
    per_tool_summary: Dict[str, Dict[str, float]] = {}
    for name, stat in per_tool.items():
        calls = stat["calls"]
        per_tool_summary[name] = {
            "calls": calls,
            "success_rate": stat["successes"] / calls if calls > 0 else 0.0,
            "avg_latency_ms": (
                stat["total_latency_ms"] / stat["latency_count"]
                if stat["latency_count"] > 0 else 0.0
            ),
            "total_cost": stat["total_cost"],
        }

    return {
        "total_calls": total,
        "success_rate": successes / total if total > 0 else 0.0,
        "failure_rate": failures / total if total > 0 else 0.0,
        "budget_exceeded_rate": budget_exceeded / total if total > 0 else 0.0,
        "avg_latency_ms": total_latency / latency_count if latency_count > 0 else 0.0,
        "total_cost": total_cost,
        "avg_cost_per_call": total_cost / cost_count if cost_count > 0 else 0.0,
        "accuracy": accuracy,
        "per_tool": per_tool_summary,
    }
