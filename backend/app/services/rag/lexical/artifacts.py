"""Versioned, validated persistence for native bm25s artifacts."""

import hashlib
import json
import math
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import bm25s

from app.core.config import settings
from app.services.rag.lexical.tokenizer import TOKENIZER_VERSION

_GENERATION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_NATIVE_FILES = (
    "data.csc.index.npy",
    "indices.csc.index.npy",
    "indptr.csc.index.npy",
    "vocab.index.json",
    "params.index.json",
    "corpus.jsonl",
    "corpus.mmindex.json",
)


class BM25ArtifactError(RuntimeError):
    """Base class for artifact persistence and validation failures."""


class BM25ArtifactNotFound(BM25ArtifactError):
    """Raised when no artifact exists for a generation."""


class BM25ArtifactMismatch(BM25ArtifactError):
    """Raised when artifact identity, config, or integrity does not match."""


class BM25ArtifactNotReady(BM25ArtifactMismatch):
    """Raised when an artifact was not fully published."""


class BM25ArtifactAlreadyExists(BM25ArtifactError):
    """Raised when attempting to overwrite an immutable generation artifact."""


@dataclass(frozen=True)
class BM25ArtifactManifest:
    index_generation: str
    corpus_sha256: str
    document_count: int
    tokenizer_version: str
    bm25s_version: str
    method: str
    k1: float
    b: float
    created_at: str
    token_count: int = 0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BM25ArtifactManifest":
        required = {
            "index_generation",
            "corpus_sha256",
            "document_count",
            "tokenizer_version",
            "bm25s_version",
            "method",
            "k1",
            "b",
            "created_at",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise BM25ArtifactMismatch(
                f"manifest missing required fields: {', '.join(missing)}"
            )
        try:
            return cls(
                index_generation=payload["index_generation"],
                corpus_sha256=payload["corpus_sha256"],
                document_count=payload["document_count"],
                tokenizer_version=payload["tokenizer_version"],
                bm25s_version=payload["bm25s_version"],
                method=payload["method"],
                k1=payload["k1"],
                b=payload["b"],
                created_at=payload["created_at"],
                token_count=payload.get("token_count", 0),
            )
        except (TypeError, ValueError) as exc:
            raise BM25ArtifactMismatch("manifest contains invalid values") from exc


def _validate_generation(generation: str) -> str:
    if not isinstance(generation, str) or not _GENERATION_PATTERN.fullmatch(
        generation
    ):
        raise ValueError(
            "generation must contain only letters, digits, dot, underscore, and dash"
        )
    return generation


def _resolve_root(artifact_root: Optional[Path]) -> Path:
    return Path(artifact_root or settings.BM25_ARTIFACT_ROOT)


def _artifact_dir(generation: str, artifact_root: Optional[Path]) -> Path:
    return _resolve_root(artifact_root) / generation / "bm25"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> BM25ArtifactManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BM25ArtifactMismatch(f"cannot read manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise BM25ArtifactMismatch("manifest root must be an object")
    return BM25ArtifactManifest.from_dict(payload)


def _validate_manifest(
    manifest: BM25ArtifactManifest,
    requested_generation: str,
) -> None:
    mismatches = []
    if manifest.index_generation != requested_generation:
        mismatches.append(
            f"generation={manifest.index_generation!r}, expected {requested_generation!r}"
        )
    if (
        manifest.tokenizer_version != TOKENIZER_VERSION
        or manifest.tokenizer_version != settings.BM25_TOKENIZER_VERSION
    ):
        mismatches.append(
            f"tokenizer={manifest.tokenizer_version!r}, expected "
            f"{TOKENIZER_VERSION!r}"
        )
    if manifest.bm25s_version != bm25s.__version__:
        mismatches.append(
            f"bm25s={manifest.bm25s_version!r}, expected {bm25s.__version__!r}"
        )
    if manifest.method != settings.BM25_METHOD:
        mismatches.append(
            f"method={manifest.method!r}, expected {settings.BM25_METHOD!r}"
        )
    try:
        if not math.isclose(float(manifest.k1), settings.BM25_K1):
            mismatches.append(f"k1={manifest.k1!r}, expected {settings.BM25_K1!r}")
    except (TypeError, ValueError):
        mismatches.append("k1 must be numeric")
    try:
        if not math.isclose(float(manifest.b), settings.BM25_B):
            mismatches.append(f"b={manifest.b!r}, expected {settings.BM25_B!r}")
    except (TypeError, ValueError):
        mismatches.append("b must be numeric")
    if not isinstance(manifest.document_count, int) or manifest.document_count <= 0:
        mismatches.append("document_count must be a positive integer")
    if not isinstance(manifest.token_count, int) or manifest.token_count < 0:
        mismatches.append("token_count must be a non-negative integer")
    try:
        datetime.fromisoformat(manifest.created_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        mismatches.append("created_at must be an ISO-8601 timestamp")
    if mismatches:
        raise BM25ArtifactMismatch("manifest mismatch: " + "; ".join(mismatches))


def _validate_files(artifact_dir: Path) -> None:
    missing = [name for name in _NATIVE_FILES if not (artifact_dir / name).is_file()]
    if not (artifact_dir / "manifest.json").is_file():
        missing.append("manifest.json")
    if missing:
        raise BM25ArtifactMismatch(
            "artifact missing required files: " + ", ".join(missing)
        )


def _validate_corpus(
    artifact_dir: Path,
    manifest: BM25ArtifactManifest,
) -> None:
    corpus_path = artifact_dir / "corpus.jsonl"
    actual_sha256 = _sha256_file(corpus_path)
    if actual_sha256 != manifest.corpus_sha256:
        raise BM25ArtifactMismatch(
            "corpus SHA-256 mismatch: "
            f"{actual_sha256}, expected {manifest.corpus_sha256}"
        )
    with corpus_path.open("rb") as corpus_file:
        document_count = sum(1 for line in corpus_file if line.strip())
    if document_count != manifest.document_count:
        raise BM25ArtifactMismatch(
            "corpus document count mismatch: "
            f"{document_count}, expected {manifest.document_count}"
        )


def _validate_loaded_engine(
    engine: bm25s.BM25,
    documents: Any,
    manifest: BM25ArtifactManifest,
) -> None:
    if len(documents) != manifest.document_count:
        raise BM25ArtifactMismatch(
            "loaded corpus document count does not match manifest"
        )
    if int(engine.scores.get("num_docs", -1)) != manifest.document_count:
        raise BM25ArtifactMismatch("bm25s index document count does not match manifest")
    if engine.method != manifest.method:
        raise BM25ArtifactMismatch("bm25s method does not match manifest")
    if not math.isclose(float(engine.k1), float(manifest.k1)):
        raise BM25ArtifactMismatch("bm25s k1 does not match manifest")
    if not math.isclose(float(engine.b), float(manifest.b)):
        raise BM25ArtifactMismatch("bm25s b does not match manifest")


def _validate_sample_query(index: Any) -> None:
    from app.services.rag.bm25_search import build_document_tokens

    for position in range(index.doc_count):
        tokens = build_document_tokens(dict(index.documents[position]))
        if not tokens:
            continue
        results, _scores = index._bm25.retrieve(
            [tokens[:8]], k=1, show_progress=False
        )
        result_position = int(results[0][0])
        if 0 <= result_position < index.doc_count:
            return
        break
    raise BM25ArtifactMismatch("bm25s sample query validation failed")


def _load_from_dir(
    generation: str,
    artifact_dir: Path,
    *,
    mmap: bool,
    require_ready: bool,
):
    from app.services.rag.bm25_search import BM25Index

    if require_ready and not (artifact_dir / "READY").is_file():
        raise BM25ArtifactNotReady(
            f"BM25 artifact is not ready for generation {generation!r}"
        )
    _validate_files(artifact_dir)
    manifest = _read_manifest(artifact_dir / "manifest.json")
    _validate_manifest(manifest, generation)
    _validate_corpus(artifact_dir, manifest)

    try:
        engine = bm25s.BM25.load(
            artifact_dir,
            load_corpus=True,
            mmap=mmap,
        )
    except Exception as exc:
        raise BM25ArtifactMismatch("bm25s native artifact load failed") from exc
    documents = getattr(engine, "corpus", None)
    if documents is None:
        raise BM25ArtifactMismatch("bm25s artifact did not load its corpus")
    _validate_loaded_engine(engine, documents, manifest)
    index = BM25Index._from_loaded(
        engine,
        documents,
        token_count=manifest.token_count,
    )
    _validate_sample_query(index)
    return manifest, index


def build_bm25_artifact(
    generation: str,
    documents: Iterable[Dict[str, Any]],
    artifact_root: Optional[Path] = None,
) -> BM25ArtifactManifest:
    """Build and validate a generation artifact before publishing READY."""
    from app.services.rag.bm25_search import BM25Index

    selected_generation = _validate_generation(generation)
    target_dir = _artifact_dir(selected_generation, artifact_root)
    if target_dir.exists():
        raise BM25ArtifactAlreadyExists(
            f"BM25 artifact already exists for generation {selected_generation!r}"
        )

    documents_snapshot = [dict(document) for document in documents]
    candidate = BM25Index()
    candidate.build(documents_snapshot)
    if not candidate.initialized or candidate._bm25 is None:
        raise ValueError("cannot persist an empty BM25 corpus")

    generation_dir = target_dir.parent
    generation_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".bm25.staging-", dir=generation_dir)
    )
    try:
        candidate._bm25.save(
            staging_dir,
            corpus=documents_snapshot,
            show_progress=False,
        )
        corpus_sha256 = _sha256_file(staging_dir / "corpus.jsonl")
        manifest = BM25ArtifactManifest(
            index_generation=selected_generation,
            corpus_sha256=corpus_sha256,
            document_count=candidate.doc_count,
            tokenizer_version=TOKENIZER_VERSION,
            bm25s_version=bm25s.__version__,
            method=settings.BM25_METHOD,
            k1=settings.BM25_K1,
            b=settings.BM25_B,
            created_at=datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            token_count=candidate.token_count,
        )
        (staging_dir / "manifest.json").write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

        _load_from_dir(
            selected_generation,
            staging_dir,
            mmap=False,
            require_ready=False,
        )
        (staging_dir / "READY").write_text(
            f"{manifest.corpus_sha256}\n",
            encoding="ascii",
        )
        if target_dir.exists():
            raise BM25ArtifactAlreadyExists(
                f"BM25 artifact concurrently published for {selected_generation!r}"
            )
        staging_dir.rename(target_dir)
        return manifest
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def load_bm25_artifact(
    generation: str,
    artifact_root: Optional[Path] = None,
    *,
    mmap: bool = True,
):
    """Load a validated generation artifact without touching process registry state."""
    selected_generation = _validate_generation(generation)
    artifact_dir = _artifact_dir(selected_generation, artifact_root)
    if not artifact_dir.is_dir():
        raise BM25ArtifactNotFound(
            f"BM25 artifact not found for generation {selected_generation!r}"
        )
    _manifest, index = _load_from_dir(
        selected_generation,
        artifact_dir,
        mmap=mmap,
        require_ready=True,
    )
    return index
