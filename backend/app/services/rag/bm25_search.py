# -*- coding: utf-8 -*-
"""Medical BM25 retrieval backed by bm25s and versioned artifacts."""

import logging
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

import bm25s

from app.core.config import settings
from app.services.rag.lexical.query_expansion import expand_lexical_query
from app.services.rag.lexical.tokenizer import tokenize_medical_text

logger = logging.getLogger(__name__)


def _field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value)
    return str(value)


def build_document_tokens(
    doc: Dict[str, Any], *, text_field: str = "text"
) -> List[str]:
    """Tokenize a document and apply bounded heading/entity field boosts."""
    body = tokenize_medical_text(
        _field_text(doc.get(text_field, "")), mode="document"
    )
    heading = tokenize_medical_text(
        _field_text(doc.get("heading_path", "")), mode="document"
    )
    entities = tokenize_medical_text(
        _field_text(doc.get("entity_names", "")), mode="document"
    )
    return (
        body
        + heading * settings.BM25_HEADING_BOOST
        + entities * settings.BM25_ENTITY_BOOST
    )


class BM25Index:
    """Compatibility wrapper around a bm25s index."""

    def __init__(self) -> None:
        self.documents: Any = []
        self.doc_count: int = 0
        self.token_count: int = 0
        self._bm25: Optional[bm25s.BM25] = None
        self.initialized: bool = False

    @classmethod
    def _from_loaded(
        cls,
        engine: bm25s.BM25,
        documents: Any,
        *,
        token_count: int,
    ) -> "BM25Index":
        index = cls()
        index.documents = documents
        index.doc_count = len(documents)
        index.token_count = int(token_count)
        index._bm25 = engine
        engine.corpus = None
        index.initialized = index.doc_count > 0
        return index

    def build(self, documents: List[Dict], text_field: str = "text") -> None:
        """Build locally, then update this wrapper only after bm25s succeeds."""
        documents_snapshot = list(documents)
        doc_count = len(documents_snapshot)
        if doc_count == 0:
            self.documents = []
            self.doc_count = 0
            self.token_count = 0
            self._bm25 = None
            self.initialized = False
            return

        tokenized = [
            build_document_tokens(doc, text_field=text_field)
            for doc in documents_snapshot
        ]
        token_count = sum(len(tokens) for tokens in tokenized)
        engine = bm25s.BM25(
            method=settings.BM25_METHOD,
            k1=settings.BM25_K1,
            b=settings.BM25_B,
        )
        engine.index(tokenized, show_progress=False)

        self.documents = documents_snapshot
        self.doc_count = doc_count
        self.token_count = token_count
        self._bm25 = engine
        self.initialized = True
        logger.info(
            "BM25 index built with bm25s: %d documents, %.0f average tokens",
            self.doc_count,
            self.token_count / max(self.doc_count, 1),
        )

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Return the existing BM25 result dictionary shape."""
        if (
            not self.initialized
            or self._bm25 is None
            or top_k <= 0
            or not query
            or not query.strip()
        ):
            return []

        query_tokens = tokenize_medical_text(query, mode="query")
        query_tokens = expand_lexical_query(query, query_tokens)
        if not query_tokens:
            return []

        results, scores = self._bm25.retrieve(
            [query_tokens],
            k=min(top_k, self.doc_count),
            show_progress=False,
        )

        final_results: List[Dict] = []
        for idx, score in zip(results[0], scores[0], strict=False):
            doc_index = int(idx)
            numeric_score = float(score)
            if numeric_score <= 0:
                continue
            doc_copy = dict(self.documents[doc_index])
            doc_copy["bm25_score"] = round(numeric_score, 4)
            doc_copy["doc_id"] = doc_copy.get("id", "")
            final_results.append(doc_copy)
            if len(final_results) >= top_k:
                break

        return final_results


# The dictionary is generation-aware; the alias preserves compatibility for callers
# and tests that have historically treated BM25 as one process-wide singleton.
_registry_lock = RLock()
_bm25_indexes: Dict[str, BM25Index] = {}
_bm25_index: Optional[BM25Index] = None
_bm25_index_generation: Optional[str] = None


def _active_generation() -> str:
    configured_generation = getattr(settings, "ACTIVE_INDEX_VERSION", None)
    if configured_generation:
        return str(configured_generation)
    with _registry_lock:
        if _bm25_index_generation:
            return _bm25_index_generation
    return str(getattr(settings, "ACTIVE_INDEX_VERSION", "rag-v1"))


def _artifact_root(artifact_root: Optional[Path]) -> Path:
    if artifact_root is not None:
        return Path(artifact_root)
    return Path(settings.BM25_ARTIFACT_ROOT)


def _build_legacy_bm25_index() -> BM25Index:
    """Build a candidate from Chroma without mutating the active registry."""
    candidate = BM25Index()
    try:
        from app.services.rag.medical_store import (
            _get_collection_name,
            get_medical_store,
        )

        store = get_medical_store()
        if store.client is None:
            store._init_client()
        assert store.client is not None

        collection_name = _get_collection_name()
        try:
            collection = store.client.get_collection(collection_name)
        except Exception:
            logger.warning("BM25 legacy collection does not exist: %s", collection_name)
            return candidate

        count = collection.count()
        if count == 0:
            logger.warning("BM25 legacy collection is empty: %s", collection_name)
            return candidate

        all_docs: List[Dict] = []
        batch_size = 1000
        for offset in range(0, count, batch_size):
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            for position, doc_id in enumerate(result.get("ids") or []):
                documents = result.get("documents") or []
                metadatas = result.get("metadatas") or []
                doc_text = documents[position] if position < len(documents) else ""
                metadata = metadatas[position] if position < len(metadatas) else {}
                all_docs.append(
                    {
                        "id": doc_id,
                        "text": doc_text,
                        "source": metadata.get("source", "未知"),
                        "page": metadata.get("page", 0),
                        "heading_path": metadata.get("heading_path", ""),
                        "entity_names": metadata.get("entity_names", ""),
                        "chunk_seq": metadata.get("chunk_seq", -1),
                        "content_type": metadata.get("content_type", ""),
                        "organization": metadata.get("organization"),
                        "year": metadata.get("year"),
                        "version": metadata.get("version"),
                        "document_type": metadata.get("document_type"),
                        "departments": metadata.get("departments"),
                        "disease_tags": metadata.get("disease_tags"),
                        "population": metadata.get("population"),
                        "recommendation_level": metadata.get(
                            "recommendation_level"
                        ),
                        "evidence_level": metadata.get("evidence_level"),
                        "metadata_source": metadata.get("metadata_source"),
                    }
                )

        candidate.build(all_docs)
        logger.warning(
            "BM25 loaded through explicit legacy Chroma fallback: %s (%d documents)",
            collection_name,
            len(all_docs),
        )
    except Exception:
        logger.exception("BM25 legacy fallback build failed")
    return candidate


def _load_generation_candidate(
    generation: str, artifact_root: Optional[Path]
) -> BM25Index:
    from app.services.rag.lexical.artifacts import (
        BM25ArtifactNotFound,
        load_bm25_artifact,
    )

    try:
        return load_bm25_artifact(
            generation,
            _artifact_root(artifact_root),
            mmap=True,
        )
    except BM25ArtifactNotFound:
        if not settings.RAG_LEGACY_COLLECTION_FALLBACK:
            raise
        logger.warning(
            "BM25 artifact for generation %s is absent; using explicit legacy "
            "Chroma fallback",
            generation,
        )
        return _build_legacy_bm25_index()


def _cached_index(generation: str) -> Optional[BM25Index]:
    with _registry_lock:
        cached = _bm25_indexes.get(generation)
        if cached is not None:
            return cached
        if (
            _bm25_index is not None
            and _bm25_index_generation in (None, generation)
        ):
            _bm25_indexes[generation] = _bm25_index
            return _bm25_index
    return None


def get_bm25_index(
    generation: Optional[str] = None,
    artifact_root: Optional[Path] = None,
) -> BM25Index:
    """Get a generation index, loading and atomically publishing a candidate."""
    global _bm25_index, _bm25_index_generation

    selected_generation = generation or _active_generation()
    explicit_generation = generation is not None
    cached = (
        _bm25_indexes.get(selected_generation)
        if explicit_generation
        else _cached_index(selected_generation)
    )
    if cached is not None:
        if generation is None:
            with _registry_lock:
                _bm25_index = cached
                _bm25_index_generation = selected_generation
        return cached

    try:
        candidate = _load_generation_candidate(selected_generation, artifact_root)
    except Exception:
        logger.exception(
            "BM25 artifact validation/load failed for generation %s; active index "
            "was not changed",
            selected_generation,
        )
        if explicit_generation:
            raise
        with _registry_lock:
            return _bm25_index if _bm25_index is not None else BM25Index()

    if not candidate.initialized:
        logger.warning(
            "BM25 candidate for generation %s is uninitialized; active index was "
            "not changed",
            selected_generation,
        )
        if explicit_generation:
            raise RuntimeError(
                f"BM25 generation {selected_generation!r} is uninitialized"
            )
        with _registry_lock:
            return _bm25_index if _bm25_index is not None else candidate

    with _registry_lock:
        installed = _bm25_indexes.setdefault(selected_generation, candidate)
        if generation is None:
            _bm25_index = installed
            _bm25_index_generation = selected_generation
        return installed


def _try_load_documents() -> None:
    """Compatibility helper that atomically publishes a successful legacy build."""
    global _bm25_index, _bm25_index_generation
    candidate = _build_legacy_bm25_index()
    if not candidate.initialized:
        return
    generation = _active_generation()
    with _registry_lock:
        _bm25_indexes[generation] = candidate
        _bm25_index = candidate
        _bm25_index_generation = generation


def install_bm25_index(generation: str, index: BM25Index) -> BM25Index:
    """Atomically install a validated generation for this Worker.

    Loading and validation happen before this function is called.  Keeping the
    final registry update in one lock-protected operation means readers either
    retain the old index or observe the complete new generation.
    """
    global _bm25_index, _bm25_index_generation

    if not index.initialized:
        raise ValueError(f"cannot install an uninitialized BM25 generation: {generation}")
    with _registry_lock:
        _bm25_indexes[generation] = index
        _bm25_index = index
        _bm25_index_generation = generation
    return index


def rebuild_bm25_index(
    generation: Optional[str] = None,
    artifact_root: Optional[Path] = None,
) -> None:
    """Load/build a candidate and swap it under the registry lock when valid."""
    global _bm25_index, _bm25_index_generation

    selected_generation = generation or _active_generation()
    try:
        candidate = _load_generation_candidate(selected_generation, artifact_root)
    except Exception:
        logger.exception(
            "BM25 rebuild rejected generation %s; active index was not changed",
            selected_generation,
        )
        return

    if not candidate.initialized:
        logger.warning(
            "BM25 rebuild produced no initialized candidate for generation %s; "
            "active index was not changed",
            selected_generation,
        )
        return

    if generation is None or selected_generation == _active_generation():
        install_bm25_index(selected_generation, candidate)
    else:
        with _registry_lock:
            _bm25_indexes[selected_generation] = candidate
