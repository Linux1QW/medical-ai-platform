"""Unified RAG generation and atomic activation contracts."""

import threading
from datetime import datetime, timezone
from types import SimpleNamespace
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
async def test_candidate_builder_reports_actual_work_boundaries(
    tmp_path, monkeypatch
):
    document = tmp_path / "guide.pdf"
    document.write_bytes(b"%PDF-1.7")
    events = []

    def extract_document(_path):
        events.append("work:parse")
        return [{"source": "guide.pdf", "page": 1, "content_type": "text"}]

    async def apply_ocr(_pages):
        return None

    def extract_chunks(_page):
        events.append("work:chunk")
        return [{"text": "evidence", "heading_path": "heading"}]

    async def get_embeddings(_texts):
        events.append("work:embed")
        return [[0.1]]

    metadata = SimpleNamespace(
        organization="",
        year=0,
        version="",
        document_type="guideline",
        title="Guide",
        departments=[],
        disease_tags=[],
        population=[],
    )
    monkeypatch.setattr(builder, "extract_document", extract_document)
    monkeypatch.setattr(builder, "apply_ocr_to_pages", apply_ocr)
    monkeypatch.setattr(builder, "_extract_chunks_from_page", extract_chunks)
    monkeypatch.setattr(builder, "get_embeddings", get_embeddings)
    monkeypatch.setattr(builder, "get_enriched_metadata", lambda _source: metadata)
    monkeypatch.setattr(builder, "extract_entities", lambda _text: [])

    records = await builder._extract_candidate_records(
        [document], phase_callback=lambda phase: events.append(f"phase:{phase}")
    )

    assert records[0]["embedding"] == [0.1]
    assert events == [
        "phase:parse",
        "work:parse",
        "phase:chunk",
        "work:chunk",
        "phase:embed",
        "work:embed",
    ]

    events.clear()
    monkeypatch.setattr(builder.settings, "BGE_M3_ENABLED", True)
    monkeypatch.setattr(
        builder,
        "_publish_chroma_candidate",
        lambda generation, _records: events.append("work:chroma")
        or f"medical_guidelines_{generation}",
    )
    monkeypatch.setattr(
        builder,
        "build_bm25_artifact",
        lambda *_args: events.append("work:bm25"),
    )
    monkeypatch.setattr(
        builder,
        "build_sparse_artifact",
        lambda *_args: events.append("work:sparse"),
    )
    monkeypatch.setattr(builder, "write_rag_index_manifest", Mock())

    builder._publish_candidate_generation(
        records,
        artifact_root=tmp_path,
        phase_callback=lambda phase: events.append(f"phase:{phase}"),
    )

    assert events == [
        "phase:chroma",
        "work:chroma",
        "phase:bm25",
        "work:bm25",
        "phase:sparse",
        "work:sparse",
    ]


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
    load_sparse.assert_called_once_with(generation, tmp_path, install=False)


def test_task_uses_fixed_index_build_phases():
    from app.tasks.rag_index_task import INDEX_BUILD_PHASES

    assert INDEX_BUILD_PHASES == (
        "snapshot",
        "parse",
        "chunk",
        "embed",
        "chroma",
        "bm25",
        "sparse",
        "validate",
        "switch",
        "publish",
    )


def test_redis_index_lock_uses_task_id_and_renewable_ttl():
    from app.tasks.rag_index_task import (
        RAG_INDEX_BUILD_LOCK,
        RedisIndexBuildLock,
    )

    redis = Mock()
    redis.set.return_value = True
    redis.eval.return_value = 1
    lock = RedisIndexBuildLock(redis, ttl=90)

    assert lock.acquire("task-1") is True
    assert redis.set.call_args.args[:2] == (RAG_INDEX_BUILD_LOCK, "task-1")
    assert redis.set.call_args.kwargs == {"nx": True, "ex": 90}
    assert lock.renew("task-1") is True
    assert lock.release("task-1") is True
    assert redis.eval.call_count == 2


def test_failed_lock_renewal_sets_lost_lock_signal():
    from app.tasks.rag_index_task import _heartbeat

    lock = Mock(ttl=3)
    lock.renew.return_value = False
    stop_event = Mock()
    stop_event.wait.return_value = False
    lost_lock_event = threading.Event()

    _heartbeat(lock, "task-1", stop_event, lost_lock_event)

    assert lost_lock_event.is_set()


@pytest.mark.asyncio
async def test_task_reports_phases_only_when_builder_reaches_them(
    candidate_manifest, monkeypatch, tmp_path
):
    from app.services.rag.indexing import versioning
    from app.tasks import rag_index_task

    task = Mock()
    phases = []
    task.update_state.side_effect = lambda **call: phases.append(call["meta"]["phase"])
    lock = Mock()
    lock.is_owned_by.return_value = True
    lost_lock_event = threading.Event()

    async def build_candidate(*, phase_callback, **_kwargs):
        assert phases == ["snapshot"]
        for phase in ("parse", "chunk", "embed", "chroma", "bm25", "sparse"):
            phase_callback(phase)
        return candidate_manifest

    generation_redis = AsyncMock()
    monkeypatch.setattr(builder, "build_full_index_candidate", build_candidate)
    monkeypatch.setattr(versioning, "_get_generation_redis", AsyncMock(return_value=generation_redis))
    monkeypatch.setattr(
        versioning, "get_active_index_generation", AsyncMock(return_value="g-old")
    )
    monkeypatch.setattr(
        versioning,
        "activate_candidate_generation",
        AsyncMock(return_value={"previous": "g-old"}),
    )
    monkeypatch.setattr(versioning, "publish_index_switched", Mock())
    monkeypatch.setattr(
        rag_index_task, "_manifest_sha256", Mock(return_value="a" * 64)
    )
    monkeypatch.setattr(
        "app.services.rag.indexing.manifest.validate_candidate_manifest", Mock()
    )

    await rag_index_task._build_candidate(
        task,
        operation="rebuild",
        redis_client=Mock(),
        lock=lock,
        task_id="task-1",
        lost_lock_event=lost_lock_event,
    )

    assert phases == list(rag_index_task.INDEX_BUILD_PHASES)


@pytest.mark.asyncio
async def test_task_rechecks_lock_owner_before_switch_and_publish(
    candidate_manifest, monkeypatch
):
    from app.services.rag.indexing import versioning
    from app.tasks import rag_index_task

    async def build_candidate(*, phase_callback, **_kwargs):
        for phase in ("parse", "chunk", "embed", "chroma", "bm25", "sparse"):
            phase_callback(phase)
        return candidate_manifest

    generation_redis = AsyncMock()
    activate = AsyncMock(return_value={"previous": "g-old"})
    publish = Mock()
    lock = Mock()
    lock.is_owned_by.side_effect = [True, False]
    monkeypatch.setattr(builder, "build_full_index_candidate", build_candidate)
    monkeypatch.setattr(versioning, "_get_generation_redis", AsyncMock(return_value=generation_redis))
    monkeypatch.setattr(
        versioning, "get_active_index_generation", AsyncMock(return_value="g-old")
    )
    monkeypatch.setattr(versioning, "activate_candidate_generation", activate)
    monkeypatch.setattr(versioning, "publish_index_switched", publish)
    monkeypatch.setattr(
        rag_index_task, "_manifest_sha256", Mock(return_value="a" * 64)
    )
    monkeypatch.setattr(
        "app.services.rag.indexing.manifest.validate_candidate_manifest", Mock()
    )

    with pytest.raises(rag_index_task.LostIndexBuildLock):
        await rag_index_task._build_candidate(
            Mock(),
            operation="rebuild",
            redis_client=Mock(),
            lock=lock,
            task_id="task-1",
            lost_lock_event=threading.Event(),
        )

    activate.assert_awaited_once()
    publish.assert_not_called()


def test_switch_event_contains_generation_previous_and_manifest_digest():
    import json

    from app.services.rag.indexing.versioning import publish_index_switched

    redis = Mock()
    redis.publish.return_value = 1

    assert publish_index_switched(
        "rag-new", "rag-old", "a" * 64, redis=redis
    ) is True

    channel, raw_payload = redis.publish.call_args.args
    assert channel == "rag:index-switched"
    assert json.loads(raw_payload) == {
        "generation": "rag-new",
        "previous": "rag-old",
        "manifest_sha256": "a" * 64,
    }


def test_worker_loads_and_installs_generation_atomically(
    candidate_manifest, monkeypatch, tmp_path
):
    from app.services.rag import bm25_search, sparse_search
    from app.services.rag.indexing import versioning

    generation = candidate_manifest.index_generation
    candidate_manifest.sparse_artifact = f"{generation}/sparse"
    candidate_manifest.sparse = RAGComponentManifest(index_generation=generation)
    collection = Mock()
    collection.count.return_value = candidate_manifest.chunk_count
    store = Mock()
    old_collection = object()
    store.collection = old_collection
    store.get_collection_for_generation.return_value = collection
    loaded_bm25 = Mock(initialized=True)
    old_bm25 = object()
    old_sparse = object()
    loaded_sparse = object()
    monkeypatch.setattr(bm25_search, "_bm25_index", old_bm25)
    monkeypatch.setattr(bm25_search, "_bm25_index_generation", "g-old")
    monkeypatch.setattr(bm25_search, "_bm25_indexes", {"g-old": old_bm25})
    monkeypatch.setattr(sparse_search, "_sparse_search", old_sparse)
    monkeypatch.setattr(sparse_search, "_sparse_searches", {"g-old": old_sparse})
    monkeypatch.setattr(versioning.settings, "BGE_M3_ENABLED", True)
    monkeypatch.setattr(versioning.settings, "ACTIVE_INDEX_VERSION", "g-old")
    monkeypatch.setattr(
        versioning,
        "load_rag_index_manifest",
        lambda *_args, **_kwargs: candidate_manifest,
        raising=False,
    )
    monkeypatch.setattr(versioning, "get_medical_store", lambda: store)
    monkeypatch.setattr(versioning, "load_bm25_artifact", lambda *args, **kwargs: loaded_bm25)
    load_sparse = Mock(return_value=loaded_sparse)
    monkeypatch.setattr(versioning, "load_sparse_artifact", load_sparse)

    result = versioning.load_generation_for_worker(
        generation,
        artifact_root=tmp_path,
    )

    assert result == generation
    load_sparse.assert_called_once_with(generation, tmp_path, install=False)
    assert bm25_search._bm25_index is loaded_bm25
    assert bm25_search._bm25_index_generation == generation
    assert bm25_search._bm25_indexes[generation] is loaded_bm25
    assert sparse_search._sparse_search is loaded_sparse
    assert sparse_search._sparse_searches[generation] is loaded_sparse
    assert store.collection is collection
    assert versioning.settings.ACTIVE_INDEX_VERSION == generation


def test_worker_load_failure_keeps_old_local_reference(
    candidate_manifest, monkeypatch, tmp_path
):
    from app.services.rag import bm25_search, sparse_search
    from app.services.rag.indexing import versioning

    generation = candidate_manifest.index_generation
    candidate_manifest.sparse_artifact = f"{generation}/sparse"
    candidate_manifest.sparse = RAGComponentManifest(index_generation=generation)
    old_collection = object()
    old_bm25 = object()
    old_sparse = object()
    collection = Mock()
    collection.count.return_value = candidate_manifest.chunk_count
    store = Mock()
    store.collection = old_collection
    store.get_collection_for_generation.return_value = collection
    monkeypatch.setattr(
        versioning,
        "load_rag_index_manifest",
        lambda *_args, **_kwargs: candidate_manifest,
        raising=False,
    )
    monkeypatch.setattr(versioning, "get_medical_store", lambda: store)
    monkeypatch.setattr(versioning, "load_bm25_artifact", Mock(return_value=Mock(initialized=True)))
    monkeypatch.setattr(
        versioning, "load_sparse_artifact", Mock(side_effect=RuntimeError("bad sparse"))
    )
    monkeypatch.setattr(versioning.settings, "BGE_M3_ENABLED", True)
    monkeypatch.setattr(versioning.settings, "ACTIVE_INDEX_VERSION", "g-old")
    monkeypatch.setattr(bm25_search, "_bm25_index", old_bm25)
    monkeypatch.setattr(bm25_search, "_bm25_index_generation", "g-old")
    monkeypatch.setattr(bm25_search, "_bm25_indexes", {"g-old": old_bm25})
    monkeypatch.setattr(sparse_search, "_sparse_search", old_sparse)
    monkeypatch.setattr(sparse_search, "_sparse_searches", {"g-old": old_sparse})

    with pytest.raises(RuntimeError, match="bad sparse"):
        versioning.load_generation_for_worker(
            generation,
            artifact_root=tmp_path,
        )

    assert bm25_search._bm25_index is old_bm25
    assert bm25_search._bm25_index_generation == "g-old"
    assert generation not in bm25_search._bm25_indexes
    assert sparse_search._sparse_search is old_sparse
    assert generation not in sparse_search._sparse_searches
    assert store.collection is old_collection
    assert versioning.settings.ACTIVE_INDEX_VERSION == "g-old"


def test_worker_install_failure_rolls_back_all_local_references(
    candidate_manifest, monkeypatch, tmp_path
):
    from app.services.rag import bm25_search, sparse_search
    from app.services.rag.indexing import versioning

    generation = candidate_manifest.index_generation
    candidate_manifest.sparse_artifact = f"{generation}/sparse"
    candidate_manifest.sparse = RAGComponentManifest(index_generation=generation)
    old_collection = object()
    old_bm25 = object()
    old_sparse = object()
    collection = Mock()
    collection.count.return_value = candidate_manifest.chunk_count
    store = Mock(collection=old_collection)
    store.get_collection_for_generation.return_value = collection
    loaded_bm25 = Mock(initialized=True)
    loaded_sparse = object()
    monkeypatch.setattr(
        versioning,
        "load_rag_index_manifest",
        lambda *_args, **_kwargs: candidate_manifest,
    )
    monkeypatch.setattr(versioning, "get_medical_store", lambda: store)
    monkeypatch.setattr(versioning, "load_bm25_artifact", Mock(return_value=loaded_bm25))
    monkeypatch.setattr(
        versioning, "load_sparse_artifact", Mock(return_value=loaded_sparse)
    )
    monkeypatch.setattr(versioning.settings, "BGE_M3_ENABLED", True)
    monkeypatch.setattr(versioning.settings, "ACTIVE_INDEX_VERSION", "g-old")
    monkeypatch.setattr(bm25_search, "_bm25_index", old_bm25)
    monkeypatch.setattr(bm25_search, "_bm25_index_generation", "g-old")
    monkeypatch.setattr(bm25_search, "_bm25_indexes", {"g-old": old_bm25})
    monkeypatch.setattr(sparse_search, "_sparse_search", old_sparse)
    monkeypatch.setattr(sparse_search, "_sparse_searches", {"g-old": old_sparse})

    def fail_after_partial_bm25_install(selected_generation, candidate):
        bm25_search._bm25_index = candidate
        bm25_search._bm25_index_generation = selected_generation
        bm25_search._bm25_indexes[selected_generation] = candidate
        raise RuntimeError("install failed")

    monkeypatch.setattr(versioning, "install_bm25_index", fail_after_partial_bm25_install)

    with pytest.raises(RuntimeError, match="install failed"):
        versioning.load_generation_for_worker(generation, artifact_root=tmp_path)

    assert bm25_search._bm25_index is old_bm25
    assert bm25_search._bm25_index_generation == "g-old"
    assert bm25_search._bm25_indexes == {"g-old": old_bm25}
    assert sparse_search._sparse_search is old_sparse
    assert sparse_search._sparse_searches == {"g-old": old_sparse}
    assert store.collection is old_collection
    assert versioning.settings.ACTIVE_INDEX_VERSION == "g-old"
