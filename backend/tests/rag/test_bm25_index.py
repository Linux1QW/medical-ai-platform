"""Regression tests for the BM25 medical tokenizer."""

from app.services.rag.bm25_search import tokenize_medical_text
from scripts.eval import evaluate_bm25


def test_egfr_and_egfr_renal_are_distinct():
    assert tokenize_medical_text("EGFR") != tokenize_medical_text("eGFR")


def test_bigram_does_not_cross_punctuation():
    assert "林氯" not in tokenize_medical_text("阿司匹林，氯吡格雷")


def test_preserves_compound_dose_and_disease_type():
    tokens = tokenize_medical_text("2型糖尿病 阿司匹林100mg")
    assert "2型糖尿病" in tokens
    assert "100mg" in tokens


def test_relevance_groups_are_counted_once_for_ndcg():
    ranked_ids, relevant_ids = evaluate_bm25._ranked_and_relevant(
        [
            {"doc_id": "one", "source": "非小细胞肺癌指南"},
            {"doc_id": "two", "source": "非小细胞肺癌指南"},
        ],
        ["非小细胞肺癌"],
    )

    assert ranked_ids[0] == "非小细胞肺癌"
    assert ranked_ids[1].startswith("__duplicate_relevant__:")
    assert evaluate_bm25._ndcg_at_k(ranked_ids, relevant_ids, 2) == 1.0


def test_case_metrics_include_ndcg_for_every_configured_k():
    metrics = evaluate_bm25._case_metrics(["source-a"], {"source-a"}, (1, 3, 5))

    assert set(metrics) == {"recall@1", "recall@3", "recall@5", "mrr", "ndcg@1", "ndcg@3", "ndcg@5"}


def test_active_version_uses_resolved_collection_before_settings_fallback():
    version = evaluate_bm25._version_from_active_collection(
        "medical_guidelines_rag-20260801",
        "rag-v1",
    )

    assert version == "rag-20260801"
