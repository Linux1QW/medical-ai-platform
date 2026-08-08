"""Generation-aware retrieval cache contracts."""

from app.services.rag.retrieval_cache import (
    build_retrieval_cache_key,
    compact_cached_bundle,
    hydrate_cached_bundle,
)
from app.services.rag.types import RetrievalQuery


def _queries():
    return [
        RetrievalQuery(query_type="case", source="clinical_facts", text="持续胸痛"),
        RetrievalQuery(query_type="treatment", source="mqe", text="急性冠脉治疗"),
    ]


def test_cache_key_changes_after_source_update():
    old = build_retrieval_cache_key(_queries(), generation="g1", top_k=8)
    new = build_retrieval_cache_key(_queries(), generation="g2", top_k=8)

    assert old != new


def test_cache_key_includes_query_shape_and_retrieval_settings(monkeypatch):
    original = build_retrieval_cache_key(_queries(), generation="g1", top_k=8)
    changed_top_k = build_retrieval_cache_key(_queries(), generation="g1", top_k=9)
    monkeypatch.setattr(
        "app.services.rag.retrieval_cache.settings.RRF_WEIGHT_BM25",
        0.99,
    )
    changed_rrf = build_retrieval_cache_key(_queries(), generation="g1", top_k=8)

    assert len({original, changed_top_k, changed_rrf}) == 3


def test_cached_bundle_excludes_document_text():
    compact = compact_cached_bundle(
        {
            "status": "candidate",
            "candidates": [
                {
                    "doc_id": "d1",
                    "generation": "g1",
                    "text": "large medical document body",
                    "source": "guide.pdf",
                    "rrf_score": 0.03,
                    "private_debug_payload": {"prompt": "must not persist"},
                }
            ],
        }
    )

    assert "text" not in compact["candidates"][0]
    assert compact["candidates"][0]["doc_id"] == "d1"
    assert compact["candidates"][0]["rrf_score"] == 0.03
    assert "private_debug_payload" not in compact["candidates"][0]
    assert set(compact["candidates"][0]) == {
        "doc_id",
        "generation",
        "source",
        "rrf_score",
    }


def test_cache_hydration_reads_text_from_requested_generation():
    class Store:
        def get_documents_by_ids(self, doc_ids, *, generation):
            assert doc_ids == ["d1"]
            assert generation == "g2"
            return {"d1": {"text": "current generation body"}}

    hydrated = hydrate_cached_bundle(
        {
            "status": "candidate",
            "candidates": [
                {
                    "doc_id": "d1",
                    "generation": "g2",
                    "source": "guide.pdf",
                    "rrf_score": 0.03,
                }
            ],
        },
        generation="g2",
        store=Store(),
    )

    assert hydrated["candidates"][0]["text"] == "current generation body"


def test_cache_hydration_drops_stale_generation_candidates():
    class Store:
        def get_documents_by_ids(self, doc_ids, *, generation):
            return {}

    hydrated = hydrate_cached_bundle(
        {
            "status": "candidate",
            "candidates": [
                {
                    "doc_id": "d1",
                    "generation": "g-old",
                    "source": "guide.pdf",
                }
            ],
        },
        generation="g-new",
        store=Store(),
    )

    assert hydrated["candidates"] == []
