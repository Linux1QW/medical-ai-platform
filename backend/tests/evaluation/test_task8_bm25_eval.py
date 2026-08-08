"""Task 8 BM25 report and regression gate contracts."""

import sys
from pathlib import Path

from scripts.eval import evaluate_bm25


def _case(case_id: str, category: str) -> dict:
    return {
        "id": case_id,
        "category": category,
        "query": case_id,
        "relevant_source_contains": ["source"],
        "must_preserve_tokens": [],
    }


def test_evaluate_emits_six_strata_and_overall_metrics(monkeypatch):
    cases = [_case(category, category) for category in evaluate_bm25.REQUIRED_CATEGORIES]

    class FakeIndex:
        doc_count = 6
        token_count = 60
        index_version = "candidate-v2"

        def search(self, query, top_k):
            return [{"doc_id": query, "source": "source.pdf"}]

    monkeypatch.setattr(evaluate_bm25, "_load_golden", lambda _path: (cases, (1, 3, 5, 10)))

    report = evaluate_bm25.evaluate(
        Path("offline-fixture.json"),
        top_k=10,
        index=FakeIndex(),
        cold_load_seconds=1.25,
    )

    assert report["k_values"] == [1, 3, 5, 10]
    assert set(evaluate_bm25.REQUIRED_CATEGORIES).issubset(report["metrics"])
    assert report["metrics"]["overall"]["recall@10"] == 1.0
    assert report["metrics"]["gene_variant"]["ndcg@10"] == 1.0
    assert "exact_term" in report["metrics"]
    assert report["metrics"]["exact_term"]["recall@10"] == 1.0


def test_compare_reports_exposes_all_quality_performance_and_consistency_checks():
    baseline = {
        "metrics": {
            "overall": {"recall@10": 0.70, "ndcg@10": 0.60},
            "exact_term": {"recall@10": 0.50},
        },
        "cold_load_seconds": 2.0,
        "latency_ms": {"p95": 2.0},
        "consistency": {
            "generation_mismatch_count": 0,
            "stale_cache_hit_count": 0,
        },
    }
    candidate = {
        "metrics": {
            "overall": {"recall@10": 0.70, "ndcg@10": 0.61},
            "exact_term": {"recall@10": 0.55},
        },
        "cold_load_seconds": 9.0,
        "latency_ms": {"p95": 5.0},
        "consistency": {
            "generation_mismatch_count": 0,
            "stale_cache_hit_count": 0,
        },
    }

    result = evaluate_bm25.compare_reports(candidate, baseline)

    assert result["passed"] is True
    assert set(result["checks"]) == {
        "overall_recall@10",
        "overall_ndcg@10",
        "exact_term_recall@10",
        "cold_load_seconds",
        "search_p95_ms",
        "generation_mismatch_count",
        "stale_cache_hit_count",
    }


def test_parser_accepts_compare_and_fail_on_regression():
    parser = evaluate_bm25.build_parser()

    args = parser.parse_args(["--compare", "baseline.json", "--fail-on-regression"])

    assert args.compare == Path("baseline.json")
    assert args.fail_on_regression is True


def test_fail_on_regression_rejects_missing_active_generation(monkeypatch):
    class MissingIndex:
        initialized = False
        doc_count = 0

    monkeypatch.setattr(
        "app.services.rag.bm25_search.get_bm25_index",
        lambda: MissingIndex(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_bm25.py",
            "--compare",
            "missing-baseline.json",
            "--fail-on-regression",
        ],
    )

    assert evaluate_bm25.main() == 1
