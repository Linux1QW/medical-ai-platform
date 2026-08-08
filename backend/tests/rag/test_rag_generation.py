"""Unified RAG generation and atomic activation contracts."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.rag.indexing import builder, versioning
from app.services.rag.indexing.builder import build_candidate_snapshot
from app.services.rag.indexing.manifest import (
    IndexGenerationMismatch,
    RAGComponentManifest,
    RAGIndexManifest,
    build_index_generation,
    validate_candidate_manifest,
)
from app.services.rag.indexing.versioning import (
    ActiveGenerationConflict,
    compare_and_set_active_generation,
)
from app.services.rag.medical_store import (
    COLLECTION_NAME,
    IndexGenerationUnavailable,
    _get_collection_name,
    _reset_collection_cache,
)


@pytest.fixture
def candidate_manifest() -> RAGIndexManifest:
    return RAGIndexManifest(
        index_generation="rag-20260808112233-01234567",
        corpus_sha256="0123456789abcdef" * 4,
        source_count=2,
        chunk_count=3,
        parser_version="document-extractors-v1",
        chunker_version="medical-chunker-v1",
        tokenizer_version="medical-lexical-v3",
        embedding_model="qwen3.7-text-embedding",
        embedding_dimension=1024,
        chroma_collection="medical_guidelines_rag-20260808112233-01234567",
        bm25_artifact="rag-20260808112233-01234567/bm25",
        sparse_artifact=None,
        created_at=datetime(2026, 8, 8, 11, 22, 33, tzinfo=timezone.utc),
    )


def test_manifest_populates_component_generations(candidate_manifest):
    assert candidate_manifest.chroma.index_generation == candidate_manifest.index_generation
    assert candidate_manifest.bm25.index_generation == candidate_manifest.index_generation
    assert candidate_manifest.sparse is None


def test_switch_rejects_mismatched_components(candidate_manifest):
    candidate_manifest.bm25.index_generation = "g-old"

    with pytest.raises(IndexGenerationMismatch, match="bm25"):
        validate_candidate_manifest(candidate_manifest)


def test_switch_rejects_collection_for_another_generation(candidate_manifest):
    candidate_manifest.chroma_collection = "medical_guidelines_g-old"

    with pytest.raises(IndexGenerationMismatch, match="chroma"):
        validate_candidate_manifest(candidate_manifest)


def test_manifest_rejects_prefixed_bm25_artifact_path(candidate_manifest):
    candidate_manifest.bm25_artifact = (
        f"unexpected/{candidate_manifest.index_generation}/bm25"
    )

    with pytest.raises(IndexGenerationMismatch, match="bm25"):
        validate_candidate_manifest(candidate_manifest)


def test_manifest_requires_exact_sparse_artifact_path(candidate_manifest, monkeypatch):
    generation = candidate_manifest.index_generation
    monkeypatch.setattr(builder.settings, "BGE_M3_ENABLED", True)
    candidate_manifest.sparse_artifact = f"unexpected/{generation}/sparse"
    candidate_manifest.sparse = RAGComponentManifest(index_generation=generation)

    with pytest.raises(IndexGenerationMismatch, match="sparse"):
        validate_candidate_manifest(candidate_manifest)


def test_generation_uses_fixed_timestamp_and_corpus_prefix():
    created_at = datetime(2026, 8, 8, 11, 22, 33, tzinfo=timezone.utc)

    assert build_index_generation("abcdef0123456789" * 4, created_at) == (
        "rag-20260808112233-abcdef01"
    )


@pytest.mark.asyncio
async def test_active_pointer_uses_atomic_compare_and_set():
    redis = AsyncMock()
    redis.eval.return_value = 1

    switched = await compare_and_set_active_generation(
        expected_generation="g-old",
        candidate_generation="g-new",
        redis=redis,
    )

    assert switched is True
    redis.eval.assert_awaited_once()
    args = redis.eval.await_args.args
    assert args[1:] == (1, "rag:active_generation", "g-old", "g-new")


@pytest.mark.asyncio
async def test_active_pointer_rejects_concurrent_switch():
    redis = AsyncMock()
    redis.eval.return_value = 0
    redis.get.return_value = "g-other"

    with pytest.raises(ActiveGenerationConflict, match="g-other"):
        await compare_and_set_active_generation(
            expected_generation="g-old",
            candidate_generation="g-new",
            redis=redis,
        )


def test_missing_generation_does_not_fall_back_without_migration_flag(monkeypatch):
    client = Mock()
    client.list_collections.return_value = [Mock(name=COLLECTION_NAME)]
    monkeypatch.setattr(
        "app.services.rag.medical_store.chromadb.PersistentClient",
        lambda **_: client,
    )
    monkeypatch.setattr(
        "app.services.rag.medical_store.settings.RAG_LEGACY_COLLECTION_FALLBACK",
        False,
    )
    _reset_collection_cache()

    with pytest.raises(IndexGenerationUnavailable, match="rag-missing"):
        _get_collection_name(generation="rag-missing", use_cache=False)


def test_legacy_fallback_requires_explicit_migration_flag(monkeypatch):
    client = Mock()
    legacy_collection = Mock()
    legacy_collection.name = COLLECTION_NAME
    client.list_collections.return_value = [legacy_collection]
    monkeypatch.setattr(
        "app.services.rag.medical_store.chromadb.PersistentClient",
        lambda **_: client,
    )
    monkeypatch.setattr(
        "app.services.rag.medical_store.settings.RAG_LEGACY_COLLECTION_FALLBACK",
        True,
    )
    _reset_collection_cache()

    assert _get_collection_name(generation="rag-missing", use_cache=False) == COLLECTION_NAME


def test_incremental_replacement_builds_snapshot_without_mutating_active():
    active_documents = [
        {"id": "old-a", "text": "old", "metadata": {"source": "a.pdf"}},
        {"id": "keep-b", "text": "keep", "metadata": {"source": "b.pdf"}},
    ]
    replacement = [
        {"id": "new-a", "text": "new", "metadata": {"source": "a.pdf"}}
    ]

    candidate = build_candidate_snapshot(
        active_documents,
        replacement,
        source_name="a.pdf",
        force_replace=True,
    )

    assert [document["id"] for document in candidate] == ["keep-b", "new-a"]
    assert [document["id"] for document in active_documents] == ["old-a", "keep-b"]


def test_candidate_builder_publishes_sparse_artifact_when_enabled(
    tmp_path, monkeypatch
):
    generation_time = datetime(2026, 8, 8, 11, 22, 33, tzinfo=timezone.utc)
    records = [
        {
            "id": "d1",
            "text": "evidence",
            "metadata": {"source": "guide.pdf"},
            "embedding": [0.1],
        }
    ]
    build_sparse = Mock()
    monkeypatch.setattr(builder.settings, "BGE_M3_ENABLED", True)
    monkeypatch.setattr(builder, "build_bm25_artifact", Mock())
    monkeypatch.setattr(builder, "build_sparse_artifact", build_sparse)
    monkeypatch.setattr(
        builder,
        "_publish_chroma_candidate",
        lambda generation, records: f"medical_guidelines_{generation}",
    )
    monkeypatch.setattr(builder, "write_rag_index_manifest", Mock())

    manifest = builder._publish_candidate_generation(
        records,
        artifact_root=tmp_path,
        created_at=generation_time,
    )

    build_sparse.assert_called_once_with(
        manifest.index_generation,
        builder._bm25_documents(records),
        tmp_path,
    )
    assert manifest.sparse_artifact == f"{manifest.index_generation}/sparse"


@pytest.mark.asyncio
async def test_activation_validates_and_loads_enabled_sparse_artifact(
    candidate_manifest, tmp_path, monkeypatch
):
    generation = candidate_manifest.index_generation
    candidate_manifest.sparse_artifact = f"{generation}/sparse"
    candidate_manifest.sparse = RAGComponentManifest(index_generation=generation)
    collection = Mock()
    collection.count.return_value = candidate_manifest.chunk_count
    store = Mock()
    store.get_collection_for_generation.return_value = collection
    load_bm25 = Mock()
    load_sparse = Mock()
    cas = AsyncMock(return_value=True)
    monkeypatch.setattr(versioning.settings, "BGE_M3_ENABLED", True)
    monkeypatch.setattr(versioning, "get_medical_store", lambda: store)
    monkeypatch.setattr(versioning, "load_bm25_artifact", load_bm25)
    monkeypatch.setattr(versioning, "load_sparse_artifact", load_sparse)
    monkeypatch.setattr(versioning, "compare_and_set_active_generation", cas)

    await versioning.activate_candidate_generation(
        candidate_manifest,
        expected_generation="g-old",
        artifact_root=tmp_path,
    )

    load_bm25.assert_called_once_with(generation, tmp_path)
    load_sparse.assert_called_once_with(generation, tmp_path, install=True)
