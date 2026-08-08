"""Versioned learned-sparse artifact contracts."""

from unittest.mock import Mock

import numpy as np

from app.services.rag import sparse_search


def test_sparse_artifact_round_trip_is_generation_scoped(tmp_path, monkeypatch):
    encoder = Mock()
    encoder.encode_corpus.return_value = {
        "dense": np.array([[0.1], [0.2]]),
        "sparse": [{1: 1.0}, {1: 2.0}],
    }
    encoder.encode_query.return_value = {"sparse": {1: 1.0}}
    monkeypatch.setattr(sparse_search, "get_dual_encoder", lambda: encoder)
    monkeypatch.setattr(sparse_search.settings, "BGE_M3_ENABLED", True)

    manifest = sparse_search.build_sparse_artifact(
        "g-sparse",
        [
            {
                "id": "d1",
                "text": "first",
                "source": "a.pdf",
                "generation": "g-sparse",
            },
            {
                "id": "d2",
                "text": "second",
                "source": "b.pdf",
                "generation": "g-sparse",
            },
        ],
        tmp_path,
    )
    loaded = sparse_search.load_sparse_artifact(
        "g-sparse", tmp_path, install=False
    )

    assert manifest.index_generation == "g-sparse"
    assert loaded.search("query", top_k=1)[0] == {
        "doc_id": "d2",
        "text": "second",
        "source": "b.pdf",
        "generation": "g-sparse",
        "sparse_score": 2.0,
    }


def test_sparse_registry_never_returns_another_generation(tmp_path, monkeypatch):
    old = sparse_search.LearnedSparseSearch()
    sparse_search._sparse_searches = {"g-old": old}
    monkeypatch.setattr(sparse_search.settings, "BGE_M3_ENABLED", True)
    monkeypatch.setattr(
        sparse_search,
        "load_sparse_artifact",
        Mock(side_effect=sparse_search.SparseArtifactNotFound("missing")),
    )

    try:
        sparse_search.get_sparse_search("g-new", tmp_path)
    except sparse_search.SparseArtifactNotFound:
        pass
    else:
        raise AssertionError("missing generation must not return g-old")
