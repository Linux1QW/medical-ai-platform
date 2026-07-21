"""
Report generation for RAG evaluation.

Provides JSON / Markdown report generation, threshold checking,
and comparison between Legacy RAG and Tool Use methods.
"""
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .datasets import RagEvalResult, RagGoldCase
from .metrics import (
    aggregate_metrics_by_dimension,
    final_answer_keyword_coverage,
    refusal_metrics_from_results,
    score_range_accuracy,
    tool_breakdown,
    tool_call_accuracy,
    tool_metrics,
)

# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def generate_timestamp() -> str:
    """Generate ISO format timestamp."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Metric calculation helpers
# ---------------------------------------------------------------------------

def calculate_basic_metrics(results: List[RagEvalResult], gold_cases: List[RagGoldCase]) -> Dict[str, float]:
    """Calculate basic metrics for the evaluation."""
    refusal_mets = refusal_metrics_from_results(results, gold_cases)
    tool_mets = tool_metrics(results)

    total_samples = len(results)
    normal_samples = sum(1 for gc in gold_cases if not gc.should_refuse)
    refusal_samples = total_samples - normal_samples

    valid_latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    avg_latency_ms = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0

    # Average knowledge score
    valid_scores = [r.knowledge_score for r in results if r.knowledge_score is not None]
    avg_knowledge_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    basic_metrics = {
        "total_samples": total_samples,
        "normal_samples": normal_samples,
        "refusal_samples": refusal_samples,
        "avg_latency_ms": avg_latency_ms,
        "avg_knowledge_score": avg_knowledge_score,
    }

    all_metrics = {**basic_metrics, **refusal_mets, **tool_mets}
    return all_metrics


def calculate_retrieval_metrics(results: List[RagEvalResult], gold_cases: List[RagGoldCase]) -> Dict[str, float]:
    """Calculate retrieval-specific metrics."""
    recalls_at_k: List[float] = []
    mrr_scores: List[float] = []
    ndcg_scores: List[float] = []

    for result, gold_case in zip(results, gold_cases):
        if not gold_case.gold_doc_ids:
            continue

        rag_trace = result.rag_trace_data or {}
        retrieved_docs = rag_trace.get('retrieved_docs', [])

        if retrieved_docs and gold_case.gold_doc_ids:
            recall_at_1 = recall_at_k(retrieved_docs, gold_case.gold_doc_ids, 1)
            recall_at_3 = recall_at_k(retrieved_docs, gold_case.gold_doc_ids, 3)
            recall_at_5 = recall_at_k(retrieved_docs, gold_case.gold_doc_ids, 5)
            recalls_at_k.extend([recall_at_1, recall_at_3, recall_at_5])

            mrr_score = mrr(retrieved_docs, gold_case.gold_doc_ids)
            mrr_scores.append(mrr_score)

            ndcg_score = ndcg_at_k(retrieved_docs, gold_case.gold_relevance_grades or {}, 5)
            ndcg_scores.append(ndcg_score)

    avg_recall_at_1 = sum(recalls_at_k[::3]) / len(recalls_at_k[::3]) if recalls_at_k else 0
    avg_recall_at_3 = sum(recalls_at_k[1::3]) / len(recalls_at_k[1::3]) if len(recalls_at_k) >= 2 else 0
    avg_recall_at_5 = sum(recalls_at_k[2::3]) / len(recalls_at_k[2::3]) if len(recalls_at_k) >= 3 else 0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0

    return {
        "recall_at_1": avg_recall_at_1,
        "recall_at_3": avg_recall_at_3,
        "recall_at_5": avg_recall_at_5,
        "mrr": avg_mrr,
        "ndcg_at_5": avg_ndcg,
    }


def calculate_citation_metrics(results: List[RagEvalResult], gold_cases: List[RagGoldCase]) -> Dict[str, float]:
    """Calculate citation-specific metrics."""
    all_used_citation_ids: List[str] = []
    all_allowed_citation_ids: set = set()
    all_gold_citation_ids: List[str] = []

    for result, gold_case in zip(results, gold_cases):
        for citation in result.citation_data:
            if isinstance(citation, dict) and 'id' in citation:
                all_used_citation_ids.append(citation['id'])

        rag_trace = result.rag_trace_data or {}
        retrieved_docs = rag_trace.get('retrieved_docs', [])
        for doc_id in retrieved_docs:
            all_allowed_citation_ids.add(doc_id)

        if gold_case.gold_citation_ids:
            all_gold_citation_ids.extend(gold_case.gold_citation_ids)

    citation_validity_val = 1.0
    citation_hallucination_rate_val = 0.0
    citation_coverage_val = 1.0

    if all_used_citation_ids:
        valid_count = sum(1 for cid in all_used_citation_ids if cid in all_allowed_citation_ids)
        citation_validity_val = valid_count / len(all_used_citation_ids)
        hallucinated_count = sum(1 for cid in all_used_citation_ids if cid not in all_allowed_citation_ids)
        citation_hallucination_rate_val = hallucinated_count / len(all_used_citation_ids)

    if all_gold_citation_ids:
        used_set = set(all_used_citation_ids)
        gold_set = set(all_gold_citation_ids)
        covered = gold_set.intersection(used_set)
        citation_coverage_val = len(covered) / len(gold_set) if gold_set else 0.0

    return {
        "citation_validity": citation_validity_val,
        "citation_hallucination_rate": citation_hallucination_rate_val,
        "citation_coverage": citation_coverage_val,
    }


def calculate_stance_metrics(results: List[RagEvalResult], gold_cases: List[RagGoldCase]) -> Dict[str, float]:
    """Calculate stance-related metrics."""
    correct_stances = 0
    total_stance_comparisons = 0

    for result, gold_case in zip(results, gold_cases):
        if result.evidence_stance and gold_case.expected_stance:
            total_stance_comparisons += 1
            if result.evidence_stance.lower() == gold_case.expected_stance.lower():
                correct_stances += 1

    stance_accuracy_val = correct_stances / total_stance_comparisons if total_stance_comparisons > 0 else 0.0

    return {"stance_accuracy": stance_accuracy_val}


def calculate_tool_use_specific_metrics(results: List[RagEvalResult], gold_cases: List[RagGoldCase]) -> Dict[str, float]:
    """Calculate metrics specific to Tool Use functionality."""
    keyword_coverage = final_answer_keyword_coverage(results, gold_cases)
    tool_acc = tool_call_accuracy(results, gold_cases)
    score_acc = score_range_accuracy(results, gold_cases)

    return {
        "final_answer_keyword_coverage": keyword_coverage,
        "tool_call_accuracy": tool_acc,
        "score_range_accuracy": score_acc,
    }


# ---------------------------------------------------------------------------
# Failed-case analysis
# ---------------------------------------------------------------------------

def find_failed_cases(results: List[RagEvalResult], gold_cases: List[RagGoldCase]) -> List[Dict[str, str]]:
    """Find and categorize failed cases."""
    failed_cases: List[Dict[str, str]] = []

    for result, gold_case in zip(results, gold_cases):
        # False acceptances
        if gold_case.should_refuse and result.knowledge_score is not None and result.knowledge_score != 0:
            failed_cases.append({
                "case_id": result.case_id,
                "failure_type": "false_acceptance",
                "message": f"should_refuse=true but knowledge_score={result.knowledge_score}",
            })

        # Hallucinated citations
        for citation in result.citation_data:
            if isinstance(citation, dict) and 'id' in citation:
                citation_id = citation['id']
                rag_trace = result.rag_trace_data or {}
                retrieved_docs = rag_trace.get('retrieved_docs', [])
                if citation_id not in retrieved_docs:
                    failed_cases.append({
                        "case_id": result.case_id,
                        "failure_type": "hallucinated_citation",
                        "message": f"Hallucinated citation: {citation_id}",
                    })

        # System errors
        if result.error:
            failed_cases.append({
                "case_id": result.case_id,
                "failure_type": "system_error",
                "message": f"System error: {result.error}",
            })

    return failed_cases


# ---------------------------------------------------------------------------
# Threshold checking
# ---------------------------------------------------------------------------

# Default threshold definitions — P0 (critical), P1 (important), P2 (advisory)
DEFAULT_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    # P0 — 核心门槛
    "citation_validity": {
        "threshold": 1.0,
        "direction": "gte",
        "level": "P0",
        "description": "引用有效性必须达到 100%",
    },
    "citation_hallucination_rate": {
        "threshold": 0.05,
        "direction": "lte",
        "level": "P0",
        "description": "引用幻觉率必须 ≤ 5%",
    },
    "false_acceptance_rate": {
        "threshold": 0.05,
        "direction": "lte",
        "level": "P0",
        "description": "错误接受率必须 ≤ 5%",
    },
    # P1 — 重要指标
    "refusal_accuracy": {
        "threshold": 0.80,
        "direction": "gte",
        "level": "P1",
        "description": "拒答准确率应 ≥ 80%",
    },
    "stance_accuracy": {
        "threshold": 0.70,
        "direction": "gte",
        "level": "P1",
        "description": "立场准确率应 ≥ 70%",
    },
    "score_range_accuracy": {
        "threshold": 0.60,
        "direction": "gte",
        "level": "P1",
        "description": "分数范围准确率应 ≥ 60%",
    },
    # P2 — 参考指标
    "recall_at_5": {
        "threshold": 0.50,
        "direction": "gte",
        "level": "P2",
        "description": "Recall@5 应 ≥ 50%",
    },
    "tool_success_rate": {
        "threshold": 0.80,
        "direction": "gte",
        "level": "P2",
        "description": "工具调用成功率应 ≥ 80%",
    },
}


def check_thresholds(
    metrics: Dict[str, float],
    thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Check if metrics meet predefined thresholds.

    Args:
        metrics: 实际指标字典。
        thresholds: 自定义阈值定义，为 None 时使用 ``DEFAULT_THRESHOLDS``。

    Returns:
        包含 ``passed`` (bool) 和 ``violations`` (list) 的字典。
    """
    defs = thresholds or DEFAULT_THRESHOLDS

    violations: List[Dict[str, Any]] = []
    passed = True

    for metric_name, defn in defs.items():
        if metric_name not in metrics:
            continue

        actual_val = metrics[metric_name]
        threshold_val = defn["threshold"]
        direction = defn.get("direction", "gte")
        level = defn.get("level", "P1")
        description = defn.get("description", "")

        if direction == "gte":
            is_pass = actual_val >= threshold_val
        elif direction == "lte":
            is_pass = actual_val <= threshold_val
        else:
            is_pass = actual_val >= threshold_val

        if not is_pass:
            violations.append({
                "metric": metric_name,
                "actual": actual_val,
                "threshold": threshold_val,
                "level": level,
                "description": description,
            })
            if level == "P0":
                passed = False

    return {
        "passed": passed,
        "violations": violations,
    }


def threshold_checker(
    metrics: Dict[str, float],
    thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Comprehensive threshold checker with compliance report.

    在 ``check_thresholds`` 基础上增加:
    - 按严重等级 (P0/P1/P2) 分组汇总
    - 生成合规性摘要与改进建议

    Args:
        metrics: 实际指标字典。
        thresholds: 自定义阈值定义，为 None 时使用默认阈值。

    Returns:
        合规性报告字典，包含:
        - ``passed``: 是否全部 P0 通过
        - ``violations``: 违规列表
        - ``summary_by_level``: 按等级分组的统计
        - ``compliance_rate``: 整体合规率
        - ``recommendations``: 改进建议列表
    """
    base_result = check_thresholds(metrics, thresholds)
    violations = base_result["violations"]

    # Group by level
    summary_by_level: Dict[str, Dict[str, int]] = {
        "P0": {"total": 0, "passed": 0, "failed": 0},
        "P1": {"total": 0, "passed": 0, "failed": 0},
        "P2": {"total": 0, "passed": 0, "failed": 0},
    }

    defs = thresholds or DEFAULT_THRESHOLDS
    checked_count = 0

    for metric_name, defn in defs.items():
        if metric_name not in metrics:
            continue
        level = defn.get("level", "P1")
        summary_by_level[level]["total"] += 1
        checked_count += 1

    for v in violations:
        level = v.get("level", "P1")
        summary_by_level[level]["failed"] += 1

    for level_data in summary_by_level.values():
        level_data["passed"] = level_data["total"] - level_data["failed"]

    # Overall compliance rate
    total_checks = sum(d["total"] for d in summary_by_level.values())
    total_passed = sum(d["passed"] for d in summary_by_level.values())
    compliance_rate = total_passed / total_checks if total_checks > 0 else 1.0

    # Generate recommendations
    recommendations: List[str] = []
    for v in violations:
        metric = v["metric"]
        actual = v["actual"]
        threshold = v["threshold"]
        level = v.get("level", "P1")

        if metric == "citation_validity":
            recommendations.append(f"[{level}] 引用有效性不足 ({actual:.1%} < {threshold:.1%})，需检查检索管道与引用映射逻辑")
        elif metric == "citation_hallucination_rate":
            recommendations.append(f"[{level}] 引用幻觉率偏高 ({actual:.1%} > {threshold:.1%})，需加强引用来源校验")
        elif metric == "false_acceptance_rate":
            recommendations.append(f"[{level}] 错误接受率偏高 ({actual:.1%} > {threshold:.1%})，需优化拒答策略")
        elif metric == "refusal_accuracy":
            recommendations.append(f"[{level}] 拒答准确率不足 ({actual:.1%} < {threshold:.1%})，建议调整拒答阈值")
        elif metric == "recall_at_5":
            recommendations.append(f"[{level}] Recall@5 偏低 ({actual:.1%} < {threshold:.1%})，建议优化检索查询或增加召回数量")
        elif metric == "tool_success_rate":
            recommendations.append(f"[{level}] 工具调用成功率不足 ({actual:.1%} < {threshold:.1%})，需检查工具服务稳定性")
        else:
            recommendations.append(f"[{level}] {metric} 未达标: 实际 {actual:.4f}, 阈值 {threshold:.4f}")

    return {
        "passed": base_result["passed"],
        "violations": violations,
        "summary_by_level": summary_by_level,
        "compliance_rate": compliance_rate,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def generate_json_report(
    results: List[RagEvalResult],
    gold_cases: List[RagGoldCase],
    mode: str,
    dataset_path: str,
    split: str,
) -> Dict[str, Any]:
    """Generate JSON report for evaluation.

    Args:
        results: 评估结果列表。
        gold_cases: 对应的 gold case 列表。
        mode: 评估模式 (``legacy`` / ``tooluse``)。
        dataset_path: 数据集文件路径。
        split: 数据集分割。

    Returns:
        结构化报告字典。
    """
    timestamp = generate_timestamp()

    # Calculate all metrics
    basic_metrics = calculate_basic_metrics(results, gold_cases)
    retrieval_metrics = calculate_retrieval_metrics(results, gold_cases)
    citation_metrics = calculate_citation_metrics(results, gold_cases)
    stance_metrics = calculate_stance_metrics(results, gold_cases)
    tool_use_metrics = calculate_tool_use_specific_metrics(results, gold_cases)

    all_metrics = {
        **basic_metrics,
        **retrieval_metrics,
        **citation_metrics,
        **stance_metrics,
        **tool_use_metrics,
    }

    # Breakdowns
    tool_breakdown_data = tool_breakdown(results)
    difficulty_breakdown = aggregate_metrics_by_dimension(results, gold_cases, "difficulty")
    department_breakdown = aggregate_metrics_by_dimension(results, gold_cases, "department")

    # Failed cases
    failed_cases = find_failed_cases(results, gold_cases)

    # Threshold check (comprehensive)
    threshold_result = threshold_checker(all_metrics)

    report = {
        "timestamp": timestamp,
        "mode": mode,
        "dataset": {
            "path": dataset_path,
            "split": split,
            "total_samples": len(results),
            "normal_samples": basic_metrics.get("normal_samples", 0),
            "refusal_samples": basic_metrics.get("refusal_samples", 0),
        },
        "metrics": all_metrics,
        "tool_breakdown": tool_breakdown_data,
        "breakdown_by_difficulty": difficulty_breakdown,
        "breakdown_by_department": department_breakdown,
        "failed_cases": failed_cases,
        "thresholds": threshold_result,
    }

    return report


def write_json_report(report: Dict[str, Any], path: Path) -> None:
    """Write JSON report to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _pct(val: float, digits: int = 1) -> str:
    """Format a 0-1 float as percentage string."""
    return f"{val * 100:.{digits}f}%"


def _delta_str(delta: Optional[float], digits: int = 1, suffix: str = "%") -> str:
    """Format a delta value with sign."""
    if delta is None:
        return "N/A"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.{digits}f}{suffix}"


def generate_markdown_report(
    report: Dict[str, Any],
    comparison_report: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate Markdown report from evaluation data.

    Args:
        report: JSON report dict (from ``generate_json_report``).
        comparison_report: 可选的对比报告 dict (from
            ``generate_comparison_report``)。若提供则追加对比章节。

    Returns:
        Markdown 格式字符串。
    """
    md_lines: List[str] = [
        "# RAG / Tool Use 评估报告",
        "",
        "## 概述",
        "",
        f"- **评估时间**：{report['timestamp']}",
        f"- **模式**：{report['mode']}",
        f"- **数据集**：{report['dataset']['path'].split('/')[-1]}",
        f"- **样本数**：{report['dataset']['total_samples']}",
        f"- **正常案例**：{report['dataset'].get('normal_samples', 'N/A')}",
        f"- **拒答案例**：{report['dataset']['refusal_samples']}",
        "",
    ]

    metrics = report['metrics']
    thresholds = report.get('thresholds', {})

    # ── 合规性摘要 ──
    compliance_rate = thresholds.get('compliance_rate')
    if compliance_rate is not None:
        status_emoji = "✅" if thresholds.get("passed") else "❌"
        md_lines.extend([
            f"**合规率**: {status_emoji} {_pct(compliance_rate, 0)}",
            "",
        ])

    # ── 核心门槛 ──
    md_lines.extend([
        "## 核心门槛",
        "",
        "| 门槛 | 当前值 | 要求 | 等级 | 结果 |",
        "|---|---:|---:|:---:|---|",
    ])

    violations_map = {v["metric"]: v for v in thresholds.get("violations", [])}
    threshold_items = [
        ("citation_validity", "Citation Validity", "100%", "P0"),
        ("citation_hallucination_rate", "Hallucinated Citation Rate", "≤5%", "P0"),
        ("false_acceptance_rate", "False Acceptance Rate", "≤5%", "P0"),
        ("refusal_accuracy", "Refusal Accuracy", "≥80%", "P1"),
        ("stance_accuracy", "Stance Accuracy", "≥70%", "P1"),
        ("score_range_accuracy", "Score Range Accuracy", "≥60%", "P1"),
    ]

    for metric_name, display_name, req_str, level in threshold_items:
        if metric_name in metrics:
            curr_val = _pct(metrics[metric_name])
            is_violated = metric_name in violations_map
            result_str = "❌ 未通过" if is_violated else "✅ 通过"
            md_lines.append(f"| {display_name} | {curr_val} | {req_str} | {level} | {result_str} |")

    # ── 检索指标 ──
    md_lines.extend([
        "",
        "## 检索指标",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
    ])

    for metric_name in ['recall_at_1', 'recall_at_3', 'recall_at_5', 'mrr', 'ndcg_at_5']:
        if metric_name in metrics:
            value = _pct(metrics[metric_name])
            display = metric_name.replace('recall_at_', 'R@').replace('ndcg_at_', 'nDCG@').replace('mrr', 'MRR')
            md_lines.append(f"| {display.upper()} | {value} |")

    # ── 拒答指标 ──
    md_lines.extend([
        "",
        "## 拒答指标",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
    ])

    for metric_name in ['refusal_accuracy', 'refusal_precision', 'refusal_recall',
                        'refusal_f1', 'false_refusal_rate', 'false_acceptance_rate']:
        if metric_name in metrics:
            md_lines.append(f"| {metric_name.replace('_', ' ').title()} | {_pct(metrics[metric_name])} |")

    # ── 引用指标 ──
    md_lines.extend([
        "",
        "## 引用指标",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
    ])

    for metric_name in ['citation_validity', 'citation_hallucination_rate', 'citation_coverage']:
        if metric_name in metrics:
            md_lines.append(f"| {metric_name.replace('_', ' ').title()} | {_pct(metrics[metric_name])} |")

    # ── Tool Use 指标 ──
    avg_tool_calls = metrics.get('avg_tool_calls', 0)
    avg_latency_ms = metrics.get('avg_latency_ms', 0)
    tool_success_rate = metrics.get('tool_success_rate', 0)
    keyword_cov = metrics.get('final_answer_keyword_coverage')
    tool_acc = metrics.get('tool_call_accuracy')
    score_acc = metrics.get('score_range_accuracy')

    md_lines.extend([
        "",
        "## Tool Use 指标",
        "",
        f"- **平均工具调用次数**：{avg_tool_calls:.1f}",
        f"- **平均耗时**：{avg_latency_ms / 1000:.2f}s",
        f"- **工具成功率**：{_pct(tool_success_rate)}",
    ])
    if keyword_cov is not None:
        md_lines.append(f"- **最终回答关键词覆盖率**：{_pct(keyword_cov)}")
    if tool_acc is not None:
        md_lines.append(f"- **工具调用准确率**：{_pct(tool_acc)}")
    if score_acc is not None:
        md_lines.append(f"- **分数范围准确率**：{_pct(score_acc)}")
    md_lines.append("")

    # ── 工具调用明细 ──
    tb = report.get('tool_breakdown', {})
    if tb:
        md_lines.extend([
            "### 工具调用明细",
            "",
            "| 工具名称 | 调用次数 | 成功率 | 平均耗时(ms) |",
            "|---|---:|---:|---:|",
        ])
        for tool_name, stats in tb.items():
            md_lines.append(
                f"| {tool_name} | {stats['calls']} | "
                f"{_pct(stats['success_rate'])} | {stats['avg_latency_ms']:.0f} |"
            )
        md_lines.append("")

    # ── 难度分组 ──
    diff_bd = report.get('breakdown_by_difficulty', {})
    if diff_bd:
        md_lines.extend([
            "## 按难度分组",
            "",
            "| 难度 | 拒答准确率 | 错误接受率 | 错误拒绝率 |",
            "|---|---:|---:|---:|",
        ])
        for diff_name, diff_metrics in diff_bd.items():
            ra = _pct(diff_metrics.get('refusal_accuracy', 0))
            far = _pct(diff_metrics.get('false_acceptance_rate', 0))
            frr = _pct(diff_metrics.get('false_refusal_rate', 0))
            md_lines.append(f"| {diff_name} | {ra} | {far} | {frr} |")
        md_lines.append("")

    # ── 对比分析 ──
    if comparison_report:
        md_lines.extend(_render_comparison_section(comparison_report))

    # ── 失败样本 ──
    fc = report.get('failed_cases', [])
    if fc:
        md_lines.extend([
            "## 失败样本",
            "",
            "| case_id | 类型 | 说明 |",
            "|---|---|---|",
        ])
        for case in fc:
            md_lines.append(f"| {case['case_id']} | {case['failure_type']} | {case['message']} |")
    else:
        md_lines.extend([
            "## 失败样本",
            "",
            "无失败样本 ✅",
        ])

    # ── 改进建议 ──
    recommendations = thresholds.get('recommendations', [])
    if recommendations:
        md_lines.extend([
            "",
            "## 改进建议",
            "",
        ])
        for rec in recommendations:
            md_lines.append(f"- {rec}")

    md_lines.append("")
    return "\n".join(md_lines)


def _render_comparison_section(comp: Dict[str, Any]) -> List[str]:
    """Render comparison section for markdown report."""
    lines: List[str] = [
        "",
        "## Legacy RAG vs Tool Use 对比",
        "",
        "| 指标 | Legacy RAG | Tool Use | 差值 |",
        "|---|---:|---:|---:|",
    ]

    legacy = comp.get('legacy', {})
    tooluse = comp.get('tooluse', {})
    delta = comp.get('delta', {})

    comp_items = [
        ("avg_knowledge_score", "平均知识分数", True),
        ("avg_latency_ms", "平均延迟 (ms)", False),
        ("error_count", "错误数", False, True),
        ("review_needed_count", "需人工审核数", False, True),
    ]

    for item in comp_items:
        key, display = item[0], item[1]
        is_pct = item[2] if len(item) > 2 else False
        is_int = item[3] if len(item) > 3 else False

        l_val = legacy.get(key)
        t_val = tooluse.get(key)
        d_val = delta.get(key)

        if l_val is None and t_val is None:
            continue

        if is_int:
            l_str = str(l_val) if l_val is not None else "N/A"
            t_str = str(t_val) if t_val is not None else "N/A"
            d_str = str(d_val) if d_val is not None else "N/A"
        elif is_pct:
            l_str = _pct(l_val) if l_val is not None else "N/A"
            t_str = _pct(t_val) if t_val is not None else "N/A"
            d_str = _delta_str(d_val) if d_val is not None else "N/A"
        else:
            l_str = f"{l_val:.1f}" if l_val is not None else "N/A"
            t_str = f"{t_val:.1f}" if t_val is not None else "N/A"
            if d_val is not None:
                d_str = f"{d_val:+.1f}"
            else:
                d_str = "N/A"

        lines.append(f"| {display} | {l_str} | {t_str} | {d_str} |")

    # Elapsed time
    l_elapsed = legacy.get('total_elapsed_seconds')
    t_elapsed = tooluse.get('total_elapsed_seconds')
    if l_elapsed is not None and t_elapsed is not None:
        lines.append(f"| 总耗时 (s) | {l_elapsed:.1f} | {t_elapsed:.1f} | {t_elapsed - l_elapsed:+.1f} |")

    lines.append("")

    # Query type breakdown
    qt_bd = comp.get('query_type_breakdown', {})
    if qt_bd:
        lines.extend([
            "### 按查询类型对比",
            "",
            "| 查询类型 | 案例数 | Legacy 分数 | Tool Use 分数 | 差值 |",
            "|---|---:|---:|---:|---:|",
        ])
        for qt, data in qt_bd.items():
            count = data.get('count', 0)
            l_score = data.get('legacy', {}).get('avg_knowledge_score')
            t_score = data.get('tooluse', {}).get('avg_knowledge_score')
            l_str = f"{l_score:.1f}" if l_score is not None else "N/A"
            t_str = f"{t_score:.1f}" if t_score is not None else "N/A"
            if l_score is not None and t_score is not None:
                d_str = f"{t_score - l_score:+.1f}"
            else:
                d_str = "N/A"
            lines.append(f"| {qt} | {count} | {l_str} | {t_str} | {d_str} |")
        lines.append("")

    return lines


def write_markdown_report(report: Dict[str, Any], path: Path) -> None:
    """Write Markdown report to file."""
    markdown_content = generate_markdown_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)


# ---------------------------------------------------------------------------
# Comparison report (Legacy vs Tool Use)
# ---------------------------------------------------------------------------

def generate_comparison_report(
    legacy_results: List[RagEvalResult],
    tooluse_results: List[RagEvalResult],
    gold_cases: List[RagGoldCase],
    legacy_elapsed: float = 0.0,
    tooluse_elapsed: float = 0.0,
) -> Dict[str, Any]:
    """Generate a comparison report between Legacy RAG and Tool Use.

    Args:
        legacy_results: Legacy RAG 评估结果。
        tooluse_results: Tool Use 评估结果。
        gold_cases: 对应的 gold cases（两种模式共用同一组）。
        legacy_elapsed: Legacy 总耗时 (秒)。
        tooluse_elapsed: Tool Use 总耗时 (秒)。

    Returns:
        对比报告字典。
    """
    # Per-mode JSON reports
    legacy_report = generate_json_report(
        legacy_results, gold_cases, "legacy", "", "",
    )
    tooluse_report = generate_json_report(
        tooluse_results, gold_cases, "tooluse", "", "",
    )

    # Summary helpers
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

    comparison = {
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

    # Per-metric comparison
    legacy_metrics = legacy_report.get("metrics", {})
    tooluse_metrics = tooluse_report.get("metrics", {})
    metric_comparison: Dict[str, Dict[str, Any]] = {}

    all_metric_keys = set(legacy_metrics.keys()) | set(tooluse_metrics.keys())
    numeric_keys = {
        k for k in all_metric_keys
        if isinstance(legacy_metrics.get(k), (int, float)) or isinstance(tooluse_metrics.get(k), (int, float))
    }
    # Exclude non-comparable keys
    numeric_keys -= {"total_samples", "normal_samples", "refusal_samples"}

    for key in numeric_keys:
        l_val = legacy_metrics.get(key)
        t_val = tooluse_metrics.get(key)
        if l_val is not None and t_val is not None:
            metric_comparison[key] = {
                "legacy": l_val,
                "tooluse": t_val,
                "delta": t_val - l_val,
            }

    # Query type breakdown
    from .runners import group_cases_by_query_type
    case_id_to_idx = {gc.case_id: i for i, gc in enumerate(gold_cases)}
    groups = group_cases_by_query_type(gold_cases)

    query_type_breakdown: Dict[str, Dict[str, Any]] = {}
    for qt, qt_cases in groups.items():
        if not qt_cases:
            continue
        qt_indices = [case_id_to_idx[c.case_id] for c in qt_cases]
        qt_legacy = [legacy_results[i] for i in qt_indices]
        qt_tooluse = [tooluse_results[i] for i in qt_indices]

        def _avg(lst: List[Optional[float]]) -> Optional[float]:
            vals = [v for v in lst if v is not None]
            return sum(vals) / len(vals) if vals else None

        query_type_breakdown[qt] = {
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

    return {
        "timestamp": generate_timestamp(),
        "comparison": comparison,
        "metric_comparison": metric_comparison,
        "query_type_breakdown": query_type_breakdown,
        "legacy_report": legacy_report,
        "tooluse_report": tooluse_report,
    }


def write_comparison_report(
    comparison: Dict[str, Any],
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
) -> None:
    """Write comparison report to JSON and/or Markdown files.

    Args:
        comparison: 对比报告字典。
        json_path: JSON 输出路径（可选）。
        md_path: Markdown 输出路径（可选）。
    """
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

    if md_path:
        # Build a combined markdown: legacy report + comparison section
        legacy_report = comparison.get("legacy_report", {})
        md_content = generate_markdown_report(
            legacy_report,
            comparison_report=comparison,
        )
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)


# ---------------------------------------------------------------------------
# Late imports to avoid circular dependencies
# ---------------------------------------------------------------------------
from .metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402  # noqa: E402
