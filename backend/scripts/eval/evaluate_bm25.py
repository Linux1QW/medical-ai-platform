#!/usr/bin/env python3
"""Generate a read-only quality and performance baseline for the active BM25 index.

The evaluator never invokes a rebuild or writes to an index.  It only obtains
the active ``BM25Index`` through its public singleton and writes the requested
JSON report to the path supplied by ``--output``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_GOLDEN = Path(__file__).with_name("bm25_golden_set.json")


def _match_group(source: str, groups: Iterable[str]) -> str:
    for group in groups:
        if group and group in source:
            return group
    return ""


def _ranked_and_relevant(results: Sequence[Dict[str, Any]], groups: Sequence[str]) -> Tuple[List[str], set[str]]:
    ranked_ids: List[str] = []
    seen_groups: set[str] = set()
    for position, result in enumerate(results):
        group = _match_group(str(result.get("source", "")), groups)
        if group and group not in seen_groups:
            ranked_ids.append(group)
            seen_groups.add(group)
        elif group:
            ranked_ids.append(f"__duplicate_relevant__:{group}:{position}")
        else:
            ranked_ids.append(f"__irrelevant__:{result.get('doc_id', position)}:{position}")
    return ranked_ids, {group for group in groups if group}


def _recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def _mrr(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, doc_id in enumerate(ranked_ids[:k], start=1) if doc_id in relevant_ids)
    ideal_count = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _case_metrics(ranked_ids: Sequence[str], relevant_ids: set[str], k_values: Sequence[int]) -> Dict[str, float]:
    metrics = {f"recall@{k}": _recall_at_k(ranked_ids, relevant_ids, k) for k in k_values}
    metrics["mrr"] = _mrr(ranked_ids, relevant_ids)
    metrics.update({f"ndcg@{k}": _ndcg_at_k(ranked_ids, relevant_ids, k) for k in k_values})
    return metrics


def _version_from_active_collection(collection_name: str, configured_version: str) -> str:
    """Use the resolved collection version when present, else its configured active version."""
    prefix = "medical_guidelines_"
    if collection_name.startswith(prefix):
        return collection_name[len(prefix) :]
    return configured_version


def _active_index_version(index: Any) -> str:
    """Read the version from the index, or from the active collection and settings."""
    index_version = getattr(index, "index_version", None)
    if index_version:
        return str(index_version)

    from app.core.config import settings
    from app.services.rag.medical_store import _get_collection_name

    return _version_from_active_collection(_get_collection_name(), settings.ACTIVE_INDEX_VERSION)


def _load_golden(path: Path) -> Tuple[List[Dict[str, Any]], Tuple[int, ...]]:
    with path.open(encoding="utf-8") as golden_file:
        golden = json.load(golden_file)
    cases = golden.get("cases", [])
    k_values = tuple(golden.get("k_values", [1, 3, 5, 10]))
    if len(cases) < 40:
        raise ValueError("BM25 golden set must contain at least 40 cases")
    if not k_values or any(not isinstance(k, int) or k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    required_fields = {"id", "category", "query", "relevant_source_contains", "must_preserve_tokens"}
    for case in cases:
        missing = required_fields - case.keys()
        if missing:
            raise ValueError(f"golden case {case.get('id', '<unknown>')} is missing: {sorted(missing)}")
    return cases, k_values


def evaluate(golden_path: Path, top_k: int) -> Dict[str, Any]:
    """Evaluate the active index without rebuilding or changing active sources."""
    started = time.perf_counter()
    from app.services.rag.bm25_search import get_bm25_index, tokenize_medical_text

    index = get_bm25_index()
    cold_load_seconds = time.perf_counter() - started
    cases, k_values = _load_golden(golden_path)
    retrieval_k = max(top_k, max(k_values))
    metrics_by_category: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    latency_ms: List[float] = []
    case_reports: List[Dict[str, Any]] = []

    for case in cases:
        query_started = time.perf_counter()
        results = index.search(case["query"], top_k=retrieval_k)
        elapsed_ms = (time.perf_counter() - query_started) * 1000
        latency_ms.append(elapsed_ms)
        ranked_ids, relevant_ids = _ranked_and_relevant(results, case["relevant_source_contains"])
        case_metrics = _case_metrics(ranked_ids, relevant_ids, k_values)
        metrics_by_category[case["category"]].append(case_metrics)
        tokens = tokenize_medical_text(case["query"])
        case_reports.append(
            {
                "id": case["id"],
                "category": case["category"],
                "latency_ms": round(elapsed_ms, 4),
                "metrics": {key: round(value, 6) for key, value in case_metrics.items()},
                "missing_required_tokens": [token for token in case["must_preserve_tokens"] if token not in tokens],
                "top_sources": [str(result.get("source", "")) for result in results[:5]],
            }
        )

    aggregated = {
        category: {
            "case_count": len(category_metrics),
            **{metric: round(_mean([item[metric] for item in category_metrics]), 6) for metric in category_metrics[0]},
        }
        for category, category_metrics in sorted(metrics_by_category.items())
    }
    token_count = getattr(index, "token_count", sum(len(tokens) for tokens in getattr(index, "doc_tokens", [])))
    return {
        "index_version": _active_index_version(index),
        "document_count": index.doc_count,
        "token_count": token_count,
        "cold_load_seconds": round(cold_load_seconds, 6),
        "latency_ms": {"p50": round(_percentile(latency_ms, 0.50), 4), "p95": round(_percentile(latency_ms, 0.95), 4)},
        "metrics": aggregated,
        "case_count": len(cases),
        "k_values": list(k_values),
        "cases": case_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a read-only BM25 baseline report")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="Path to bm25_golden_set.json")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON report path")
    parser.add_argument("--top-k", type=int, default=10, help="Minimum number of BM25 candidates to retrieve")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    golden_path = args.golden.resolve()
    if not golden_path.is_file():
        parser.error(f"golden set does not exist: {golden_path}")
    report = evaluate(golden_path, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    print(f"BM25 baseline written to {args.output} ({report['case_count']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
