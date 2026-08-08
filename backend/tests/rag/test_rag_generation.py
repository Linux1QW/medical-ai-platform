"""Unified RAG generation and atomic activation contracts."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.rag.indexing.builder import build_candidate_snapshot
from app.services.rag.indexing.manifest import (
    IndexGenerationMismatch,
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
