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
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.metrics import (  # noqa: E402
    RAG_K_VALUES,
    RAG_STRATA,
    aggregate_stratified_retrieval_metrics,
    evaluate_rag_quality_gates,
)

DEFAULT_GOLDEN = Path(__file__).with_name("bm25_golden_set.json")
REQUIRED_CATEGORIES = tuple(RAG_STRATA)
REQUIRED_K_VALUES = tuple(RAG_K_VALUES)


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


def _missing_required_tokens(
    query: str,
    required_tokens: Sequence[str],
    tokenizer: Callable[[str], Sequence[str]],
) -> List[str]:
    """Return golden labels whose canonical token sets are absent from a query."""
    query_tokens = set(tokenizer(query))
    missing: List[str] = []
    for required_token in required_tokens:
        canonical_tokens = set(tokenizer(required_token))
        if not canonical_tokens or not canonical_tokens <= query_tokens:
            missing.append(required_token)
    return missing


def _token_preservation_result(case_reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    failed_cases = [case for case in case_reports if case["missing_required_tokens"]]
    missing_count = sum(len(case["missing_required_tokens"]) for case in failed_cases)
    return {
        "passed": not failed_cases,
        "failed_case_count": len(failed_cases),
        "missing_required_token_count": missing_count,
        "failed_case_ids": [case["id"] for case in failed_cases],
    }


def _preservation_exit_code(report: Dict[str, Any], *, fail_on_token_loss: bool) -> int:
    preservation_passed = bool(report["token_preservation"]["passed"])
    return 1 if fail_on_token_loss and not preservation_passed else 0


def compare_reports(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare a candidate generation against a measured BM25 baseline."""
    return evaluate_rag_quality_gates(candidate, baseline)


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


def evaluate(
    golden_path: Path,
    top_k: int,
    *,
    index: Any = None,
    cold_load_seconds: Optional[float] = None,
    consistency: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate an active index without rebuilding or changing active sources.

    ``index`` and ``cold_load_seconds`` are injectable for deterministic unit
    tests.  Production callers leave them unset and obtain the public active
    BM25 singleton exactly once.
    """
    started = time.perf_counter()
    from app.services.rag.bm25_search import get_bm25_index, tokenize_medical_text

    if index is None:
        index = get_bm25_index()
    measured_cold_load = time.perf_counter() - started
    if cold_load_seconds is None:
        cold_load_seconds = measured_cold_load

    cases, _golden_k_values = _load_golden(golden_path)
    retrieval_k = max(top_k, max(REQUIRED_K_VALUES))
    latency_ms: List[float] = []
    case_reports: List[Dict[str, Any]] = []

    for case in cases:
        query_started = time.perf_counter()
        results = index.search(case["query"], top_k=retrieval_k)
        elapsed_ms = (time.perf_counter() - query_started) * 1000
        latency_ms.append(elapsed_ms)
        ranked_ids, relevant_ids = _ranked_and_relevant(
            results, case["relevant_source_contains"]
        )
        case_metrics = _case_metrics(ranked_ids, relevant_ids, REQUIRED_K_VALUES)
        case_reports.append(
            {
                "id": case["id"],
                "category": case["category"],
                "latency_ms": round(elapsed_ms, 4),
                "metrics": {key: round(value, 6) for key, value in case_metrics.items()},
                "missing_required_tokens": _missing_required_tokens(
                    case["query"],
                    case["must_preserve_tokens"],
                    tokenize_medical_text,
                ),
                "top_sources": [
                    str(result.get("source", "")) for result in results[:5]
                ],
            }
        )

    aggregated = aggregate_stratified_retrieval_metrics(case_reports)
    token_count = getattr(
        index,
        "token_count",
        sum(len(tokens) for tokens in getattr(index, "doc_tokens", [])),
    )
    normalized_consistency = {
        "measured": consistency is not None,
        "generation_mismatch_count": None,
        "stale_cache_hit_count": None,
    }
    if consistency is not None:
        normalized_consistency.update(dict(consistency))
    p50_ms = round(_percentile(latency_ms, 0.50), 4)
    p95_ms = round(_percentile(latency_ms, 0.95), 4)
    report = {
        "index_version": _active_index_version(index),
        "document_count": index.doc_count,
        "token_count": token_count,
        "cold_load_seconds": round(float(cold_load_seconds), 6),
        "latency_ms": {"p50": p50_ms, "p95": p95_ms},
        "performance": {
            "cold_load_seconds": round(float(cold_load_seconds), 6),
            "search_p50_ms": p50_ms,
            "search_p95_ms": p95_ms,
        },
        # Keep the historical direct category keys while adding explicit
        # overall/exact-term summaries required by the Task 8 gate.
        "metrics": aggregated,
        "stratified_metrics": {
            category: aggregated[category] for category in REQUIRED_CATEGORIES
        },
        "case_count": len(cases),
        "k_values": list(REQUIRED_K_VALUES),
        "token_preservation": _token_preservation_result(case_reports),
        "consistency": normalized_consistency,
        "cases": case_reports,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a read-only BM25 baseline report")
    parser.add_argument(
        "--golden", type=Path, default=DEFAULT_GOLDEN, help="Path to bm25_golden_set.json"
    )
    parser.add_argument("--output", type=Path, help="Destination JSON report path")
    parser.add_argument("--top-k", type=int, default=10, help="Minimum number of BM25 candidates to retrieve")
    parser.add_argument(
        "--fail-on-token-loss",
        action="store_true",
        help="Exit non-zero after writing the report if a required token is missing",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Measured baseline report used for Task 8 quality/performance gates",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when any Task 8 candidate-generation gate fails",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.fail_on_regression and args.compare is None:
        parser.error("--fail-on-regression requires --compare")
    golden_path = args.golden.resolve()
    if not golden_path.is_file():
        parser.error(f"golden set does not exist: {golden_path}")

    # A missing/empty active index is an honest offline skip.  It is not a
    # fabricated green candidate and leaves the real-gate decision to CI once
    # a generation is available.
    from app.services.rag.bm25_search import get_bm25_index

    active_index = get_bm25_index()
    if not getattr(active_index, "initialized", False) or getattr(active_index, "doc_count", 0) == 0:
        message = "BM25 evaluation unavailable: no initialized active generation."
        if args.fail_on_regression:
            print(message + " Regression gates cannot be skipped.", file=sys.stderr)
            return 1
        print(message + " Offline verification only; measured gates were not run.")
        return 0

    report = evaluate(golden_path, args.top_k, index=active_index)
    gate_result = None
    if args.compare is not None:
        if not args.compare.is_file():
            parser.error(f"baseline report does not exist: {args.compare.resolve()}")
        with args.compare.open(encoding="utf-8") as baseline_file:
            baseline = json.load(baseline_file)
        gate_result = compare_reports(report, baseline)
        report["gates"] = gate_result

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output_file:
            json.dump(report, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        print(f"BM25 report written to {args.output} ({report['case_count']} cases)")
    else:
        print(f"BM25 report evaluated ({report['case_count']} cases)")

    exit_code = _preservation_exit_code(
        report,
        fail_on_token_loss=args.fail_on_token_loss,
    )
    if exit_code:
        preservation = report["token_preservation"]
        print(
            "BM25 required-token preservation failed: "
            f"{preservation['missing_required_token_count']} missing token(s) "
            f"across {preservation['failed_case_count']} case(s)",
            file=sys.stderr,
        )
    if args.fail_on_regression and gate_result is not None and not gate_result["passed"]:
        print(
            "BM25 Task 8 gates failed: " + ", ".join(gate_result["failures"]),
            file=sys.stderr,
        )
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
