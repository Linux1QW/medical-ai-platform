"""Contracts for the persistent V1.2 Coach E2E RAG seed."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.fixtures import seed_v12_coach_e2e as seed


def test_low_evidence_seed_persists_all_rag_components_and_active_pointer() -> None:
    redis_client = MagicMock()
    redis_client.get.return_value = seed.E2E_INDEX_VERSION

    with (
        patch(
            "app.services.rag.indexing.manifest.manifest_path",
            return_value=Path("missing/manifest.json"),
        ),
        patch(
            "app.services.rag.indexing.builder._publish_chroma_candidate",
            return_value=f"medical_guidelines_{seed.E2E_INDEX_VERSION}",
        ) as publish_chroma,
        patch(
            "app.services.rag.lexical.artifacts.build_bm25_artifact"
        ) as build_bm25,
        patch(
            "app.services.rag.indexing.manifest.write_rag_index_manifest"
        ) as write_manifest,
        patch("redis.Redis.from_url", return_value=redis_client),
    ):
        chunk, manifest = seed.seed_low_evidence_index()

    assert chunk["metadata"]["is_synthetic"] is True
    assert manifest["chunk_count"] == 1
    assert manifest["index_generation"].endswith(manifest["corpus_sha256"][:8])
    publish_chroma.assert_called_once()
    build_bm25.assert_called_once()
    write_manifest.assert_called_once()
    redis_client.set.assert_called_once_with(
        "rag:active_generation", seed.E2E_INDEX_VERSION
    )
