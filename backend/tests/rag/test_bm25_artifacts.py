"""Tests for versioned persistent BM25 artifacts."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.services.rag.bm25_search import BM25Index, build_document_tokens
from app.services.rag.lexical.artifacts import (
    BM25ArtifactMismatch,
    BM25ArtifactNotReady,
    build_bm25_artifact,
    load_bm25_artifact,
)

NATIVE_ARTIFACT_FILES = (
    "corpus.jsonl",
    "corpus.mmindex.json",
    "data.csc.index.npy",
    "indices.csc.index.npy",
    "indptr.csc.index.npy",
    "params.index.json",
    "vocab.index.json",
)


@pytest.fixture
def documents():
    return [
        {
            "id": "d1",
            "text": "EGFR突变肺癌患者可考虑靶向治疗",
            "source": "a.pdf",
            "heading_path": "肺癌治疗",
            "entity_names": "EGFR",
        },
        {
            "id": "d2",
            "text": "eGFR 35mL/min/1.73m2提示肾功能下降",
            "source": "b.pdf",
            "heading_path": "肾功能评估",
            "entity_names": "eGFR",
        },
    ]


def _artifact_dir(root, generation="g-test"):
    return root / generation / "bm25"


def _rewrite_manifest(root, **changes):
    path = _artifact_dir(root) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _native_file_hashes(root):
    artifact_dir = _artifact_dir(root)
    return {
        filename: hashlib.sha256((artifact_dir / filename).read_bytes()).hexdigest()
        for filename in NATIVE_ARTIFACT_FILES
    }


def test_artifact_round_trip_uses_native_mmap_and_preserves_result_shape(
    tmp_path, documents
):
    manifest = build_bm25_artifact("g-test", documents, tmp_path)
    loaded = load_bm25_artifact("g-test", tmp_path, mmap=True)

    assert manifest.document_count == 2
    assert manifest.token_count > 0
    assert list(manifest.file_sha256) == list(NATIVE_ARTIFACT_FILES)
    assert manifest.file_sha256 == _native_file_hashes(tmp_path)
    assert manifest.file_sha256["corpus.jsonl"] == manifest.corpus_sha256
    assert manifest.enable_cjk_bigram is settings.BM25_ENABLE_CJK_BIGRAM
    assert manifest.heading_boost == settings.BM25_HEADING_BOOST
    assert manifest.entity_boost == settings.BM25_ENTITY_BOOST
    assert set(asdict(manifest)) >= {
        "index_generation",
        "corpus_sha256",
        "document_count",
        "tokenizer_version",
        "bm25s_version",
        "method",
        "k1",
        "b",
        "enable_cjk_bigram",
        "heading_boost",
        "entity_boost",
        "created_at",
        "file_sha256",
    }
    assert isinstance(loaded._bm25.scores["data"], np.memmap)
    assert loaded.token_count == manifest.token_count
    assert not hasattr(loaded, "doc_tokens")

    result = loaded.search("EGFR", 1)[0]
    assert result["id"] == "d1"
    assert result["doc_id"] == "d1"
    assert result["source"] == "a.pdf"
    assert isinstance(result["bm25_score"], float)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("index_generation", "other-generation"),
        ("tokenizer_version", "medical-lexical-v1"),
        ("bm25s_version", "0.0.0"),
        ("method", "robertson"),
        ("k1", 9.0),
        ("b", 0.1),
        ("enable_cjk_bigram", True),
        ("heading_boost", 1),
        ("entity_boost", 1),
    ],
)
def test_load_rejects_manifest_identity_and_config_mismatch(
    tmp_path, documents, field, value
):
    build_bm25_artifact("g-test", documents, tmp_path)
    _rewrite_manifest(tmp_path, **{field: value})

    with pytest.raises(BM25ArtifactMismatch):
        load_bm25_artifact("g-test", tmp_path)


def test_load_rejects_corpus_integrity_mismatch(tmp_path, documents):
    build_bm25_artifact("g-test", documents, tmp_path)
    corpus_path = _artifact_dir(tmp_path) / "corpus.jsonl"
    corpus_path.write_text(
        corpus_path.read_text(encoding="utf-8") + '{"id":"tampered"}\n',
        encoding="utf-8",
    )

    with pytest.raises(BM25ArtifactMismatch, match="corpus"):
        load_bm25_artifact("g-test", tmp_path)


def test_load_rejects_native_index_byte_tampering(
    tmp_path, documents, monkeypatch
):
    build_bm25_artifact("g-test", documents, tmp_path)
    native_path = _artifact_dir(tmp_path) / "data.csc.index.npy"
    payload = bytearray(native_path.read_bytes())
    payload[-1] ^= 0x01
    native_path.write_bytes(payload)
    native_load_called = False

    def unexpected_native_load(*args, **kwargs):
        nonlocal native_load_called
        native_load_called = True
        raise AssertionError("bm25s.load must not run before hash validation")

    monkeypatch.setattr("bm25s.BM25.load", unexpected_native_load)

    with pytest.raises(
        BM25ArtifactMismatch,
        match=r"data\.csc\.index\.npy SHA-256 mismatch",
    ):
        load_bm25_artifact("g-test", tmp_path)
    assert not native_load_called


@pytest.mark.parametrize("inventory_change", ["missing", "extra"])
def test_load_rejects_native_hash_inventory_mismatch(
    tmp_path, documents, inventory_change
):
    build_bm25_artifact("g-test", documents, tmp_path)
    file_sha256 = _native_file_hashes(tmp_path)
    if inventory_change == "missing":
        file_sha256.pop("data.csc.index.npy")
    else:
        file_sha256["unused.bin"] = "0" * 64
    _rewrite_manifest(tmp_path, file_sha256=file_sha256)

    with pytest.raises(BM25ArtifactMismatch, match="file SHA-256 inventory"):
        load_bm25_artifact("g-test", tmp_path)


def test_load_rejects_missing_ready_marker(tmp_path, documents):
    build_bm25_artifact("g-test", documents, tmp_path)
    (_artifact_dir(tmp_path) / "READY").unlink()

    with pytest.raises(BM25ArtifactNotReady):
        load_bm25_artifact("g-test", tmp_path)


def test_load_rejects_modified_ready_content(tmp_path, documents):
    build_bm25_artifact("g-test", documents, tmp_path)
    (_artifact_dir(tmp_path) / "READY").write_text(
        "tampered\n",
        encoding="ascii",
    )

    with pytest.raises(BM25ArtifactMismatch, match="READY"):
        load_bm25_artifact("g-test", tmp_path)


def test_failed_native_save_never_publishes_partial_artifact(
    tmp_path, documents, monkeypatch
):
    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("bm25s.BM25.save", fail_save)

    with pytest.raises(OSError, match="disk full"):
        build_bm25_artifact("g-test", documents, tmp_path)

    generation_dir = tmp_path / "g-test"
    assert not _artifact_dir(tmp_path).exists()
    assert not list(generation_dir.glob(".bm25.staging-*"))


def test_build_releases_document_tokens_and_boosts_bounded_fields(documents):
    index = BM25Index()
    index.build(documents)

    tokens = build_document_tokens(documents[0])
    assert tokens.count("肺癌") == 1 + 2
    assert tokens.count("gene:EGFR") == 1 + 3
    assert "renal:eGFR" not in tokens
    assert index.token_count == sum(len(build_document_tokens(doc)) for doc in documents)
    assert not hasattr(index, "doc_tokens")


def test_egfr_and_egfr_remain_distinct_after_artifact_round_trip(
    tmp_path, documents
):
    build_bm25_artifact("g-test", documents, tmp_path)
    loaded = load_bm25_artifact("g-test", tmp_path)

    assert loaded.search("EGFR", 1)[0]["id"] == "d1"
    assert loaded.search("eGFR", 1)[0]["id"] == "d2"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("BM25_HEADING_BOOST", 0),
        ("BM25_HEADING_BOOST", 4),
        ("BM25_ENTITY_BOOST", 0),
        ("BM25_ENTITY_BOOST", 4),
    ],
)
def test_field_boost_configuration_is_bounded_to_one_through_three(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value})
