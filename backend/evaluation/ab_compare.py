"""
A/B comparison for index versions.

Provides functions to compare retrieval quality between two index versions
by running the same evaluation cases against each version and computing
delta metrics.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .datasets import RagGoldCase, RagEvalResult
from .runners import run_evaluation
from .report import generate_json_report, write_json_report, write_markdown_report

logger = logging.getLogger(__name__)


async def run_ab_comparison(
    version_a: str,
    version_b: str,
    gold_cases: List[RagGoldCase],
    mode: str = "legacy",
) -> Dict[str, Any]:
    """对比两个索引版本的检索/评估质量。

    依次切换 ACTIVE_INDEX_VERSION 到 version_a 和 version_b，
    对同一组 gold_cases 运行评估，计算指标差异。

    Args:
        version_a: 版本 A 标识（如 "rag-v1"）。
        version_b: 版本 B 标识（如 "rag-v2"）。
        gold_cases: 评测用例列表。
        mode: 评估模式 ("legacy" / "tooluse")。

    Returns:
        对比报告字典，包含两版本的指标、差值、推荐操作。
    """
    from app.core.config import settings

    original_version = getattr(settings, "ACTIVE_INDEX_VERSION", "rag-v1")

    # ── 版本 A 评估 ──
    logger.info("[A/B] 评估版本 A: %s", version_a)
    print(f"\n{'=' * 60}")
    print(f"评估版本 A: {version_a}")
    print(f"{'=' * 60}")

    settings.ACTIVE_INDEX_VERSION = version_a
    _clear_caches()

    start_a = time.time()
    results_a = await run_evaluation(gold_cases, mode, limit=None)
    latency_a = time.time() - start_a

    report_a = generate_json_report(
        results=results_a,
        gold_cases=gold_cases,
        mode=mode,
        dataset_path="ab_compare",
        split="dev",
    )

    # ── 版本 B 评估 ──
    logger.info("[A/B] 评估版本 B: %s", version_b)
    print(f"\n{'=' * 60}")
    print(f"评估版本 B: {version_b}")
    print(f"{'=' * 60}")

    settings.ACTIVE_INDEX_VERSION = version_b
    _clear_caches()

    start_b = time.time()
    results_b = await run_evaluation(gold_cases, mode, limit=None)
    latency_b = time.time() - start_b

    report_b = generate_json_report(
        results=results_b,
        gold_cases=gold_cases,
        mode=mode,
        dataset_path="ab_compare",
        split="dev",
    )

    # ── 恢复原始版本 ──
    settings.ACTIVE_INDEX_VERSION = original_version
    _clear_caches()

    # ── 构建对比报告 ──
    comparison = _build_comparison(
        version_a, version_b,
        report_a, report_b,
        results_a, results_b,
        latency_a, latency_b,
    )

    return comparison


def _clear_caches() -> None:
    """清除检索缓存，确保公平对比。"""
    try:
        import asyncio
        from app.services.rag.retrieval_cache import clear_retrieval_cache
        # 如果在事件循环中，需要 await；这里用同步包装
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, clear_retrieval_cache())
                future.result(timeout=5)
    except Exception:
        pass


def _build_comparison(
    version_a: str,
    version_b: str,
    report_a: Dict[str, Any],
    report_b: Dict[str, Any],
    results_a: List[RagEvalResult],
    results_b: List[RagEvalResult],
    latency_a: float,
    latency_b: float,
) -> Dict[str, Any]:
    """构建两版本对比报告。"""

    metrics_a = report_a.get("metrics", {})
    metrics_b = report_b.get("metrics", {})

    # 逐指标对比
    all_keys = set(metrics_a.keys()) | set(metrics_b.keys())
    # 排除非对比指标
    exclude_keys = {"total_samples", "normal_samples", "refusal_samples"}

    metric_comparison: Dict[str, Dict[str, Any]] = {}
    for key in sorted(all_keys - exclude_keys):
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            metric_comparison[key] = {
                "version_a": round(val_a, 4) if isinstance(val_a, float) else val_a,
                "version_b": round(val_b, 4) if isinstance(val_b, float) else val_b,
                "delta": round(delta, 4) if isinstance(delta, float) else delta,
            }

    # 汇总延迟
    def _avg_latency(results: List[RagEvalResult]) -> Optional[float]:
        lats = [r.latency_ms for r in results if r.latency_ms is not None]
        return sum(lats) / len(lats) if lats else None

    def _avg_score(results: List[RagEvalResult]) -> Optional[float]:
        scores = [r.knowledge_score for r in results if r.knowledge_score is not None]
        return sum(scores) / len(scores) if scores else None

    def _error_count(results: List[RagEvalResult]) -> int:
        return sum(1 for r in results if r.error is not None)

    # 生成推荐
    recommendation = _generate_recommendation(metric_comparison)

    return {
        "version_a": version_a,
        "version_b": version_b,
        "total_cases": len(results_a),
        "mode": report_a.get("mode", "legacy"),
        "metrics_comparison": metric_comparison,
        "summary": {
            "version_a": {
                "avg_knowledge_score": _avg_score(results_a),
                "avg_latency_ms": _avg_latency(results_a),
                "total_elapsed_seconds": round(latency_a, 2),
                "error_count": _error_count(results_a),
            },
            "version_b": {
                "avg_knowledge_score": _avg_score(results_b),
                "avg_latency_ms": _avg_latency(results_b),
                "total_elapsed_seconds": round(latency_b, 2),
                "error_count": _error_count(results_b),
            },
        },
        "recommendation": recommendation,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "report_a": report_a,
        "report_b": report_b,
    }


def _generate_recommendation(metric_comparison: Dict[str, Dict[str, Any]]) -> str:
    """根据指标差异生成推荐操作。

    Returns:
        "promote"       — 版本 B 显著优于 A，建议切换
        "reject"        — 版本 B 劣于 A，不建议切换
        "manual_review" — 差异不显著，需人工判断
    """
    # 核心指标权重
    recall_delta = metric_comparison.get("recall_at_5", {}).get("delta", 0) or 0
    citation_val_delta = metric_comparison.get("citation_validity", {}).get("delta", 0) or 0
    score_delta = metric_comparison.get("avg_knowledge_score", {}).get("delta", 0) or 0
    latency_delta = metric_comparison.get("avg_latency_ms", {}).get("delta", 0) or 0

    # 严重退化 → reject
    if citation_val_delta < -0.01:
        return "reject"
    if recall_delta < -0.05:
        return "reject"

    # 显著改善 → promote
    if recall_delta > 0.05 and latency_delta < 500:
        return "promote"
    if score_delta > 5.0 and citation_val_delta >= 0:
        return "promote"

    return "manual_review"


def write_ab_report(report: Dict[str, Any], output_dir: Path) -> None:
    """将 A/B 对比报告写入磁盘。

    生成 JSON 报告和 Markdown 摘要。

    Args:
        report: ``run_ab_comparison`` 返回的对比报告。
        output_dir: 输出目录。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 报告
    json_path = output_dir / "ab_comparison_report.json"
    write_json_report(report, json_path)
    logger.info("A/B JSON report written to: %s", json_path)

    # Markdown 摘要
    md_path = output_dir / "ab_comparison_report.md"
    md_content = _generate_ab_markdown(report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("A/B Markdown report written to: %s", md_path)

    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


def _generate_ab_markdown(report: Dict[str, Any]) -> str:
    """生成 A/B 对比 Markdown 报告。"""
    lines: List[str] = [
        "# A/B 索引版本对比报告",
        "",
        f"- **版本 A**: {report['version_a']}",
        f"- **版本 B**: {report['version_b']}",
        f"- **样本数**: {report['total_cases']}",
        f"- **评估模式**: {report.get('mode', 'legacy')}",
        f"- **时间**: {report['timestamp']}",
        "",
        "## 汇总对比",
        "",
        "| 指标 | 版本 A | 版本 B | Delta |",
        "|---|---:|---:|---:|",
    ]

    summary = report.get("summary", {})
    sa = summary.get("version_a", {})
    sb = summary.get("version_b", {})

    def _fmt(val: Any, digits: int = 2) -> str:
        if val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.{digits}f}"
        return str(val)

    def _delta_fmt(a: Any, b: Any, digits: int = 2) -> str:
        if a is None or b is None:
            return "N/A"
        d = b - a
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.{digits}f}"

    lines.append(f"| 平均知识分数 | {_fmt(sa.get('avg_knowledge_score'))} | {_fmt(sb.get('avg_knowledge_score'))} | {_delta_fmt(sa.get('avg_knowledge_score'), sb.get('avg_knowledge_score'))} |")
    lines.append(f"| 平均延迟 (ms) | {_fmt(sa.get('avg_latency_ms'), 0)} | {_fmt(sb.get('avg_latency_ms'), 0)} | {_delta_fmt(sa.get('avg_latency_ms'), sb.get('avg_latency_ms'), 0)} |")
    lines.append(f"| 总耗时 (s) | {_fmt(sa.get('total_elapsed_seconds'), 1)} | {_fmt(sb.get('total_elapsed_seconds'), 1)} | {_delta_fmt(sa.get('total_elapsed_seconds'), sb.get('total_elapsed_seconds'), 1)} |")
    lines.append(f"| 错误数 | {sa.get('error_count', 0)} | {sb.get('error_count', 0)} | |")

    # 详细指标对比
    mc = report.get("metrics_comparison", {})
    if mc:
        lines.extend([
            "",
            "## 详细指标对比",
            "",
            "| 指标 | 版本 A | 版本 B | Delta |",
            "|---|---:|---:|---:|",
        ])
        for key, vals in sorted(mc.items()):
            va = vals.get("version_a", "N/A")
            vb = vals.get("version_b", "N/A")
            delta = vals.get("delta", "N/A")

            def _fval(v: Any) -> str:
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)

            def _fdelta(v: Any) -> str:
                if isinstance(v, float):
                    sign = "+" if v >= 0 else ""
                    return f"{sign}{v:.4f}"
                return str(v)

            lines.append(f"| {key} | {_fval(va)} | {_fval(vb)} | {_fdelta(delta)} |")

    # 推荐操作
    rec = report.get("recommendation", "manual_review")
    rec_desc = {
        "promote": "版本 B 显著优于 A，建议切换",
        "reject": "版本 B 劣于 A，不建议切换",
        "manual_review": "差异不显著，需人工判断",
    }
    lines.extend([
        "",
        "## 推荐操作",
        "",
        f"**{rec}** — {rec_desc.get(rec, '')}",
        "",
    ])

    return "\n".join(lines)
