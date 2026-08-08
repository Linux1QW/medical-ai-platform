"""Task 8 metric and retrieval gate contracts."""

from evaluation.metrics import (
    RAG_K_VALUES,
    RAG_STRATA,
    aggregate_stratified_retrieval_metrics,
    evaluate_rag_quality_gates,
)


def _metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "recall@1": 0.1,
        "recall@3": 0.2,
        "recall@5": 0.3,
        "recall@10": 0.4,
        "mrr": 0.5,
        "ndcg@10": 0.6,
    }
    metrics.update(overrides)
    return metrics


def test_task8_uses_required_strata_and_cutoffs():
    assert RAG_STRATA == (
        "disease_alias",
        "drug_dose",
        "gene_variant",
        "lab_unit",
        "negation",
        "icd_code",
    )
    assert RAG_K_VALUES == (1, 3, 5, 10)


def test_stratified_metrics_always_emits_all_required_metrics():
    records = [
        {"category": "gene_variant", "metrics": _metrics(**{"recall@10": 0.8})},
        {"category": "gene_variant", "metrics": _metrics(**{"recall@10": 0.4})},
    ]

    report = aggregate_stratified_retrieval_metrics(records)

    assert set(report) == set(RAG_STRATA) | {"overall", "exact_term"}
    assert report["gene_variant"]["case_count"] == 2
    assert report["gene_variant"]["recall@10"] == 0.6
    assert report["disease_alias"]["case_count"] == 0
    assert report["disease_alias"]["ndcg@10"] == 0.0
    assert report["exact_term"]["recall@10"] == 0.6


def test_quality_gates_require_baseline_and_consistency_telemetry():
    baseline = {
        "overall": {"recall@10": 0.70, "ndcg@10": 0.60},
        "exact_term": {"recall@10": 0.50},
    }
    candidate = {
        "overall": {"recall@10": 0.70, "ndcg@10": 0.61},
        "exact_term": {"recall@10": 0.55},
        "performance": {"cold_load_seconds": 9.9, "search_p95_ms": 5.0},
        "consistency": {
            "generation_mismatch_count": 0,
            "stale_cache_hit_count": 0,
        },
    }

    gates = evaluate_rag_quality_gates(candidate, baseline)

    assert gates["passed"] is True
    assert all(check["passed"] for check in gates["checks"].values())


def test_quality_gates_do_not_treat_unmeasured_consistency_as_zero():
    baseline = {
        "overall": {"recall@10": 0.70, "ndcg@10": 0.60},
        "exact_term": {"recall@10": 0.50},
    }
    candidate = {
        "overall": {"recall@10": 0.70, "ndcg@10": 0.60},
        "exact_term": {"recall@10": 0.55},
        "performance": {"cold_load_seconds": 1.0, "search_p95_ms": 1.0},
    }

    gates = evaluate_rag_quality_gates(candidate, baseline)

    assert gates["passed"] is False
    assert gates["checks"]["generation_mismatch_count"]["passed"] is False
    assert gates["checks"]["stale_cache_hit_count"]["passed"] is False
    assert "unavailable" in gates["checks"]["generation_mismatch_count"]["reason"]
