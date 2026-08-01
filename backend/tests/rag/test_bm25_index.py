"""Regression tests for the BM25 medical tokenizer and registry."""

import pytest

from app.services.rag import bm25_search
from app.services.rag.bm25_search import BM25Index, tokenize_medical_text
from app.services.rag.lexical.artifacts import (
    BM25ArtifactMismatch,
    BM25ArtifactNotFound,
)
from scripts.eval import evaluate_bm25


@pytest.fixture(autouse=True)
def reset_bm25_registry(monkeypatch):
    monkeypatch.setattr(bm25_search, "_bm25_index", None)
    monkeypatch.setattr(bm25_search, "_bm25_index_generation", None)
    monkeypatch.setattr(bm25_search, "_bm25_indexes", {})


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


def test_rebuild_validation_failure_does_not_replace_active_index(
    tmp_path, monkeypatch
):
    active = BM25Index()
    active.build([{"id": "old", "text": "EGFR肺癌"}])
    bm25_search._bm25_index = active
    bm25_search._bm25_index_generation = "g-old"
    bm25_search._bm25_indexes["g-old"] = active

    monkeypatch.setattr(
        "app.services.rag.lexical.artifacts.load_bm25_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            BM25ArtifactMismatch("bad manifest")
        ),
    )
    legacy_called = False

    def legacy_build():
        nonlocal legacy_called
        legacy_called = True
        return BM25Index()

    monkeypatch.setattr(bm25_search, "_build_legacy_bm25_index", legacy_build)

    bm25_search.rebuild_bm25_index("g-bad", tmp_path)

    assert bm25_search._bm25_index is active
    assert "g-bad" not in bm25_search._bm25_indexes
    assert not legacy_called


def test_get_validation_failure_keeps_serving_active_index(tmp_path, monkeypatch):
    active = BM25Index()
    active.build([{"id": "old", "text": "EGFR肺癌"}])
    bm25_search._bm25_index = active
    bm25_search._bm25_index_generation = "g-old"
    bm25_search._bm25_indexes["g-old"] = active
    monkeypatch.setattr(bm25_search.settings, "ACTIVE_INDEX_VERSION", "g-bad")
    monkeypatch.setattr(
        "app.services.rag.lexical.artifacts.load_bm25_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            BM25ArtifactMismatch("bad manifest")
        ),
    )

    returned = bm25_search.get_bm25_index(artifact_root=tmp_path)

    assert returned is active
    assert "g-bad" not in bm25_search._bm25_indexes


def test_get_failed_legacy_build_keeps_serving_active_index(tmp_path, monkeypatch):
    active = BM25Index()
    active.build([{"id": "old", "text": "EGFR肺癌"}])
    bm25_search._bm25_index = active
    bm25_search._bm25_index_generation = "g-old"
    bm25_search._bm25_indexes["g-old"] = active
    monkeypatch.setattr(bm25_search.settings, "ACTIVE_INDEX_VERSION", "g-new")
    monkeypatch.setattr(
        "app.services.rag.lexical.artifacts.load_bm25_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            BM25ArtifactNotFound("missing")
        ),
    )
    monkeypatch.setattr(bm25_search, "_build_legacy_bm25_index", BM25Index)

    returned = bm25_search.get_bm25_index(artifact_root=tmp_path)

    assert returned is active
    assert "g-new" not in bm25_search._bm25_indexes


def test_rebuild_atomically_swaps_in_valid_generation(tmp_path, monkeypatch):
    from app.services.rag.lexical.artifacts import build_bm25_artifact

    active = BM25Index()
    active.build([{"id": "old", "text": "高血压"}])
    bm25_search._bm25_index = active
    bm25_search._bm25_index_generation = "g-old"
    bm25_search._bm25_indexes["g-old"] = active
    build_bm25_artifact(
        "g-new",
        [{"id": "new", "text": "EGFR肺癌", "source": "new.pdf"}],
        tmp_path,
    )
    monkeypatch.setattr(bm25_search.settings, "ACTIVE_INDEX_VERSION", "g-new")

    bm25_search.rebuild_bm25_index("g-new", tmp_path)

    installed = bm25_search._bm25_indexes["g-new"]
    assert installed is bm25_search._bm25_index
    assert installed is not active
    assert installed.search("EGFR", 1)[0]["id"] == "new"


def test_default_get_promotes_preloaded_generation_to_active_alias(monkeypatch):
    old = BM25Index()
    old.build([{"id": "old", "text": "高血压"}])
    preloaded = BM25Index()
    preloaded.build([{"id": "new", "text": "EGFR肺癌"}])
    bm25_search._bm25_index = old
    bm25_search._bm25_index_generation = "g-old"
    bm25_search._bm25_indexes.update({"g-old": old, "g-new": preloaded})
    monkeypatch.setattr(bm25_search.settings, "ACTIVE_INDEX_VERSION", "g-new")

    returned = bm25_search.get_bm25_index()

    assert returned is preloaded
    assert bm25_search._bm25_index is preloaded
    assert bm25_search._bm25_index_generation == "g-new"
