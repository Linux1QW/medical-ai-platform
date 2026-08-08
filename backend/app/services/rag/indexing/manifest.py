"""Unified identity and persistence contract for a RAG index generation."""

import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field, model_validator

from app.core.config import settings

_GENERATION_PATTERN = re.compile(r"^rag-\d{14}-[0-9a-f]{8}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class IndexGenerationMismatch(RuntimeError):
    """Raised when candidate components do not share one generation."""


class RAGComponentManifest(BaseModel):
    """Identity carried by one component of a RAG generation."""

    index_generation: str


class RAGIndexManifest(BaseModel):
    """Top-level immutable identity for dense, lexical, and sparse indexes."""

    index_generation: str
    corpus_sha256: str
    source_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    parser_version: str
    chunker_version: str
    tokenizer_version: str
    embedding_model: str
    embedding_dimension: int = Field(gt=0)
    chroma_collection: str
    bm25_artifact: str
    sparse_artifact: Optional[str]
    created_at: datetime
    chroma: Optional[RAGComponentManifest] = None
    bm25: Optional[RAGComponentManifest] = None
    sparse: Optional[RAGComponentManifest] = None

    @model_validator(mode="after")
    def populate_component_identities(self) -> "RAGIndexManifest":
        if self.chroma is None:
            self.chroma = RAGComponentManifest(
                index_generation=self.index_generation
            )
        if self.bm25 is None:
            self.bm25 = RAGComponentManifest(index_generation=self.index_generation)
        if self.sparse_artifact is not None and self.sparse is None:
            self.sparse = RAGComponentManifest(index_generation=self.index_generation)
        return self


def build_index_generation(
    corpus_sha256: str,
    created_at: Optional[datetime] = None,
) -> str:
    """Build the fixed ``rag-YYYYMMDDHHMMSS-<sha8>`` generation name."""
    if not isinstance(corpus_sha256, str) or not _SHA256_PATTERN.fullmatch(
        corpus_sha256
    ):
        raise ValueError("corpus_sha256 must be a lowercase SHA-256 digest")
    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    return f"rag-{timestamp:%Y%m%d%H%M%S}-{corpus_sha256[:8]}"


def compute_corpus_sha256(documents: Iterable[dict[str, Any]]) -> str:
    """Hash a deterministic document snapshot independent of input ordering."""
    canonical = [dict(document) for document in documents]
    canonical.sort(key=lambda document: str(document.get("id", "")))
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_candidate_manifest(manifest: RAGIndexManifest) -> None:
    """Reject a candidate unless every component has exactly one identity."""
    generation = manifest.index_generation
    if not _GENERATION_PATTERN.fullmatch(generation):
        raise IndexGenerationMismatch(
            f"invalid RAG index generation format: {generation!r}"
        )
    if not _SHA256_PATTERN.fullmatch(manifest.corpus_sha256):
        raise IndexGenerationMismatch("manifest corpus_sha256 is invalid")

    expected_collection = f"medical_guidelines_{generation}"
    if manifest.chroma_collection != expected_collection:
        raise IndexGenerationMismatch(
            "chroma collection does not match candidate generation"
        )

    expected_bm25 = f"{generation}/bm25"
    normalized_bm25 = manifest.bm25_artifact.replace("\\", "/").strip("/")
    if normalized_bm25 != expected_bm25:
        raise IndexGenerationMismatch(
            "bm25 artifact does not match candidate generation"
        )

    expected_sparse = f"{generation}/sparse"
    normalized_sparse = (
        manifest.sparse_artifact.replace("\\", "/").strip("/")
        if manifest.sparse_artifact is not None
        else None
    )
    if settings.BGE_M3_ENABLED:
        if normalized_sparse != expected_sparse:
            raise IndexGenerationMismatch(
                "sparse artifact does not match candidate generation"
            )
    elif normalized_sparse is not None:
        raise IndexGenerationMismatch(
            "sparse artifact must be absent when learned sparse retrieval is disabled"
        )

    components = {
        "chroma": manifest.chroma,
        "bm25": manifest.bm25,
        "sparse": manifest.sparse,
    }
    for name, component in components.items():
        if component is not None and component.index_generation != generation:
            raise IndexGenerationMismatch(
                f"{name} generation {component.index_generation!r} does not match "
                f"candidate {generation!r}"
            )
    if manifest.sparse_artifact is None and manifest.sparse is not None:
        raise IndexGenerationMismatch(
            "sparse component identity exists without a sparse artifact"
        )
    if manifest.sparse_artifact is not None and manifest.sparse is None:
        raise IndexGenerationMismatch(
            "sparse artifact exists without a sparse component identity"
        )


def manifest_path(
    generation: str,
    artifact_root: Optional[Path] = None,
) -> Path:
    root = Path(artifact_root or settings.BM25_ARTIFACT_ROOT)
    return root / generation / "manifest.json"


def write_rag_index_manifest(
    manifest: RAGIndexManifest,
    artifact_root: Optional[Path] = None,
) -> Path:
    """Validate and atomically publish a candidate manifest last."""
    validate_candidate_manifest(manifest)
    destination = manifest_path(manifest.index_generation, artifact_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"RAG manifest already exists for {manifest.index_generation!r}"
        )

    handle, temporary_name = tempfile.mkstemp(
        prefix=".manifest.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with open(handle, "w", encoding="utf-8", closefd=True) as output:
            output.write(
                json.dumps(
                    manifest.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_rag_index_manifest(
    generation: str,
    artifact_root: Optional[Path] = None,
) -> RAGIndexManifest:
    path = manifest_path(generation, artifact_root)
    try:
        manifest = RAGIndexManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise IndexGenerationMismatch(
            f"cannot load RAG manifest for {generation!r}"
        ) from exc
    if manifest.index_generation != generation:
        raise IndexGenerationMismatch(
            "RAG manifest generation does not match its directory"
        )
    validate_candidate_manifest(manifest)
    return manifest
