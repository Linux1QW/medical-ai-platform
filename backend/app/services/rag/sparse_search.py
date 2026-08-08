# -*- coding: utf-8 -*-
"""基于 BGE-M3 Learned Sparse 表示的检索

BGE-M3 在训练时联合优化了 learned sparse 表示（类似 SPLADE），
每个 token 都被赋予一个权重，形成词表级的稀疏向量。
相比传统 BM25（基于统计频率），learned sparse 能捕捉语义相关性。

本模块提供：
- LearnedSparseSearch：基于 learned sparse 的检索器
- 与现有 BM25 + Dense 两路检索互补，为三路融合检索提供基础

降级策略：
- 当 DualEncoder 不可用时（BGE_M3_ENABLED=False 或模型加载失败），
  search() 返回空列表，系统自动降级为现有 BM25 + Dense 方案。
"""

import hashlib
import hmac
import json
import logging
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.core.config import settings
from app.services.rag.dual_encoder import DualEncoder, get_dual_encoder

logger = logging.getLogger(__name__)

_GENERATION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SPARSE_FILES = ("documents.jsonl", "sparse.json")


class SparseArtifactError(RuntimeError):
    """Base error for versioned learned-sparse artifacts."""


class SparseArtifactNotFound(SparseArtifactError):
    """Raised when an exact sparse generation is absent."""


class SparseArtifactMismatch(SparseArtifactError):
    """Raised when sparse artifact identity or integrity is invalid."""


class SparseArtifactAlreadyExists(SparseArtifactError):
    """Raised when an immutable sparse generation already exists."""


@dataclass(frozen=True)
class SparseArtifactManifest:
    index_generation: str
    document_count: int
    model_path: str
    created_at: str
    file_sha256: dict[str, str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SparseArtifactManifest":
        try:
            return cls(
                index_generation=str(payload["index_generation"]),
                document_count=int(payload["document_count"]),
                model_path=str(payload["model_path"]),
                created_at=str(payload["created_at"]),
                file_sha256=dict(payload["file_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SparseArtifactMismatch("invalid sparse artifact manifest") from exc


class LearnedSparseSearch:
    """
    基于 BGE-M3 learned sparse 表示的检索器

    类似 SPLADE，但由 BGE-M3 统一提供 dense + sparse 双表示。
    语料和查询均编码为 {token_id: weight} 稀疏向量，
    通过点积计算相似度。

    降级行为：
    - 未设置编码器时，search() 返回空列表
    - 索引未构建时，search() 返回空列表
    """

    def __init__(self):
        self._encoder: Optional[DualEncoder] = None
        self._corpus_sparse: list = []   # list of {token_id: weight}
        self._corpus_dense: Optional[np.ndarray] = None  # (N, 1024)
        self._corpus_doc_ids: list[str] = []
        self._corpus_documents: list[dict] = []

    def set_encoder(self, encoder: DualEncoder) -> None:
        """设置双表示编码器

        Args:
            encoder: DualEncoder 实例（来自 get_dual_encoder()）
        """
        self._encoder = encoder
        logger.info("LearnedSparseSearch: 编码器已设置")

    def build_index(self, documents: list[dict]) -> None:
        """构建稀疏索引（编码所有文档）

        Args:
            documents: 包含真实 doc_id、文本和最小元数据的语料列表

        Raises:
            RuntimeError: 编码器未设置时
        """
        self._corpus_sparse = []
        self._corpus_dense = None
        self._corpus_doc_ids = []
        self._corpus_documents = []

        if not self._encoder:
            logger.warning("LearnedSparseSearch.build_index: 编码器未设置，跳过索引构建")
            return

        try:
            self._corpus_documents = [
                {
                    "doc_id": str(document["doc_id"]),
                    "text": document.get("text", ""),
                    **{
                        key: document[key]
                        for key in ("source", "page", "heading_path", "generation")
                        if key in document
                    },
                }
                for document in documents
                if document.get("doc_id") not in (None, "")
            ]
            self._corpus_doc_ids = [
                document["doc_id"] for document in self._corpus_documents
            ]
            texts = [document["text"] for document in self._corpus_documents]
            if not texts:
                return
            result = self._encoder.encode_corpus(texts)
            self._corpus_dense = result["dense"]
            self._corpus_sparse = result["sparse"]
            logger.info(
                f"LearnedSparseSearch 索引构建完成：{len(texts)} 条文档，"
                f"dense shape={self._corpus_dense.shape}"
            )
        except Exception as e:
            logger.error(f"LearnedSparseSearch 索引构建失败: {e}")
            self._corpus_sparse = []
            self._corpus_dense = None
            self._corpus_doc_ids = []
            self._corpus_documents = []

    def search(self, query: str, top_k: int = 30) -> list:
        """检索：使用 learned sparse 表示计算相似度

        Args:
            query: 查询文本
            top_k: 返回条数

        Returns:
            带真实字符串 doc_id、最小元数据和 sparse_score 的字典列表
            当编码器不可用或索引未构建时返回空列表（降级）
        """
        if not self._encoder:
            logger.debug("LearnedSparseSearch.search: 编码器未设置，返回空结果（降级）")
            return []

        if not self._corpus_sparse:
            logger.debug("LearnedSparseSearch.search: 索引为空，返回空结果")
            return []

        try:
            query_enc = self._encoder.encode_query(query)
            query_sparse = query_enc["sparse"]

            # 计算查询与所有文档的 sparse 点积相似度
            scores = []
            for i, doc_sparse in enumerate(self._corpus_sparse):
                score = self._sparse_dot(query_sparse, doc_sparse)
                scores.append((i, score))

            # 按分数降序排列，取 top_k
            scores.sort(key=lambda x: x[1], reverse=True)
            result = [
                {
                    **self._corpus_documents[index],
                    "doc_id": self._corpus_doc_ids[index],
                    "sparse_score": score,
                }
                for index, score in scores[:top_k]
            ]

            logger.debug(
                f"LearnedSparseSearch.search: query='{query[:40]}...' "
                f"→ {len(result)} 条结果，top score={result[0]['sparse_score']:.4f}" if result else "→ 0 条结果"
            )
            return result

        except Exception as e:
            logger.warning(f"LearnedSparseSearch.search 失败，降级返回空结果: {e}")
            return []

    @staticmethod
    def _sparse_dot(vec_a: dict, vec_b: dict) -> float:
        """计算两个稀疏向量的点积

        优化：遍历较小的向量，在较大向量中查找匹配项。

        Args:
            vec_a: {token_id: weight}
            vec_b: {token_id: weight}

        Returns:
            点积分数
        """
        # 遍历较小的向量以提升效率
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a

        score = 0.0
        for token_id, weight_a in vec_a.items():
            if token_id in vec_b:
                score += weight_a * vec_b[token_id]
        return score

    @property
    def is_indexed(self) -> bool:
        """索引是否已构建"""
        return bool(self._corpus_sparse)


def _validate_generation(generation: str) -> str:
    if not isinstance(generation, str) or not _GENERATION_PATTERN.fullmatch(
        generation
    ):
        raise ValueError("invalid sparse artifact generation")
    return generation


def _artifact_dir(generation: str, artifact_root: Optional[Path]) -> Path:
    root = Path(artifact_root or settings.BM25_ARTIFACT_ROOT)
    return root / generation / "sparse"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_documents(
    generation: str,
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    for document in documents:
        document_id = document.get("doc_id", document.get("id"))
        if document_id in (None, ""):
            continue
        normalized.append(
            {
                "doc_id": str(document_id),
                "text": str(document.get("text", "")),
                **{
                    key: document[key]
                    for key in ("source", "page", "heading_path")
                    if key in document
                },
                "generation": generation,
            }
        )
    return normalized


def build_sparse_artifact(
    generation: str,
    documents: list[dict[str, Any]],
    artifact_root: Optional[Path] = None,
) -> SparseArtifactManifest:
    """Build and atomically publish one immutable learned-sparse generation."""
    selected_generation = _validate_generation(generation)
    target = _artifact_dir(selected_generation, artifact_root)
    if target.exists():
        raise SparseArtifactAlreadyExists(
            f"sparse artifact already exists for {selected_generation!r}"
        )
    encoder = get_dual_encoder()
    if encoder is None:
        raise SparseArtifactError(
            "learned sparse retrieval is enabled but its encoder is unavailable"
        )
    normalized = _normalized_documents(selected_generation, documents)
    candidate = LearnedSparseSearch()
    candidate.set_encoder(encoder)
    candidate.build_index(normalized)
    if not candidate.is_indexed:
        raise SparseArtifactError("cannot persist an empty sparse artifact")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sparse.staging-", dir=target.parent))
    try:
        with (staging / "documents.jsonl").open("w", encoding="utf-8") as output:
            for document in candidate._corpus_documents:
                output.write(
                    json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n"
                )
        sparse_payload = [
            {str(token_id): float(weight) for token_id, weight in vector.items()}
            for vector in candidate._corpus_sparse
        ]
        (staging / "sparse.json").write_text(
            json.dumps(sparse_payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        file_sha256 = {
            filename: _sha256_file(staging / filename)
            for filename in _SPARSE_FILES
        }
        manifest = SparseArtifactManifest(
            index_generation=selected_generation,
            document_count=len(candidate._corpus_documents),
            model_path=settings.BGE_M3_MODEL_PATH,
            created_at=datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            file_sha256=file_sha256,
        )
        (staging / "manifest.json").write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        (staging / "READY").write_text(
            f"{selected_generation}\n", encoding="ascii"
        )
        if target.exists():
            raise SparseArtifactAlreadyExists(
                f"sparse artifact concurrently published for {selected_generation!r}"
            )
        staging.rename(target)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_sparse_artifact(
    generation: str,
    artifact_root: Optional[Path] = None,
    *,
    install: bool = True,
) -> LearnedSparseSearch:
    """Validate and load one exact sparse generation."""
    global _sparse_search
    selected_generation = _validate_generation(generation)
    artifact_dir = _artifact_dir(selected_generation, artifact_root)
    if not artifact_dir.is_dir():
        raise SparseArtifactNotFound(
            f"sparse artifact not found for {selected_generation!r}"
        )
    required = {*_SPARSE_FILES, "manifest.json", "READY"}
    if {path.name for path in artifact_dir.iterdir() if path.is_file()} != required:
        raise SparseArtifactMismatch("sparse artifact file inventory mismatch")
    try:
        payload = json.loads(
            (artifact_dir / "manifest.json").read_text(encoding="utf-8")
        )
        manifest = SparseArtifactManifest.from_dict(payload)
        ready = (artifact_dir / "READY").read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SparseArtifactMismatch("cannot read sparse artifact") from exc
    if manifest.index_generation != selected_generation or ready != selected_generation:
        raise SparseArtifactMismatch("sparse artifact generation mismatch")
    if manifest.model_path != settings.BGE_M3_MODEL_PATH:
        raise SparseArtifactMismatch("sparse artifact model mismatch")
    if set(manifest.file_sha256) != set(_SPARSE_FILES):
        raise SparseArtifactMismatch("sparse artifact hash inventory mismatch")
    for filename in _SPARSE_FILES:
        actual = _sha256_file(artifact_dir / filename)
        if not hmac.compare_digest(actual, manifest.file_sha256[filename]):
            raise SparseArtifactMismatch(
                f"sparse artifact SHA-256 mismatch: {filename}"
            )
    try:
        documents = [
            json.loads(line)
            for line in (artifact_dir / "documents.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        sparse_payload = json.loads(
            (artifact_dir / "sparse.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SparseArtifactMismatch("invalid sparse artifact payload") from exc
    if len(documents) != manifest.document_count or len(sparse_payload) != len(
        documents
    ):
        raise SparseArtifactMismatch("sparse artifact document count mismatch")
    encoder = get_dual_encoder()
    if encoder is None:
        raise SparseArtifactError("sparse encoder is unavailable")
    loaded = LearnedSparseSearch()
    loaded.set_encoder(encoder)
    loaded._corpus_documents = documents
    loaded._corpus_doc_ids = [document["doc_id"] for document in documents]
    loaded._corpus_sparse = [
        {int(token_id): float(weight) for token_id, weight in vector.items()}
        for vector in sparse_payload
    ]
    if install:
        _sparse_searches[selected_generation] = loaded
        _sparse_search = loaded
    return loaded


# ── 模块级单例 ─────────────────────────────────────────────────────────────────

_sparse_search: Optional[LearnedSparseSearch] = None
_sparse_searches: dict[str, LearnedSparseSearch] = {}


def get_sparse_search(
    generation: Optional[str] = None,
    artifact_root: Optional[Path] = None,
) -> Optional[LearnedSparseSearch]:
    """获取全局 LearnedSparseSearch 实例

    当 BGE_M3_ENABLED=False 或 DualEncoder 不可用时返回 None，
    调用方应降级为现有 BM25 + Dense 两路检索。
    """
    global _sparse_search

    if not settings.BGE_M3_ENABLED:
        return None

    if generation is not None:
        cached = _sparse_searches.get(generation)
        if cached is not None:
            return cached
        return load_sparse_artifact(generation, artifact_root, install=True)

    if not _sparse_search:
        encoder = get_dual_encoder()
        if encoder is None:
            return None

        _sparse_search = LearnedSparseSearch()
        _sparse_search.set_encoder(encoder)

    return _sparse_search


def rebuild_sparse_index(
    generation: Optional[str] = None,
    artifact_root: Optional[Path] = None,
) -> bool:
    """重建 Sparse 索引（在索引版本切换后调用）

    从 ChromaDB 加载文档并构建 learned sparse 索引。
    当 BGE_M3_ENABLED=False 或编码器不可用时跳过。

    Returns:
        True 表示索引构建成功，False 表示失败或跳过
    """
    global _sparse_search

    encoder = get_dual_encoder()
    if encoder is None:
        logger.info("Sparse 索引重建: BGE-M3 未启用，跳过")
        return False

    if generation is not None:
        try:
            load_sparse_artifact(generation, artifact_root, install=True)
            return True
        except SparseArtifactError as exc:
            logger.error("Sparse generation load failed: %s", exc)
            return False

    try:
        from app.services.rag.medical_store import _get_collection_name, get_medical_store

        store = get_medical_store()
        client = store._ensure_client()

        collection_name = _get_collection_name()
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            logger.warning(f"Sparse 索引: collection '{collection_name}' 不存在")
            return False

        if collection.count() == 0:
            logger.warning(f"Sparse 索引: collection '{collection_name}' 为空")
            return False

        # 从 collection 获取所有文档文本
        count = collection.count()
        all_documents: list[dict] = []
        batch_size = 1000
        for offset in range(0, count, batch_size):
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            if result["documents"]:
                for document_id, text, metadata in zip(
                    result["ids"],
                    result["documents"],
                    result.get("metadatas") or [None] * len(result["ids"]),
                    strict=False,
                ):
                    all_documents.append(
                        {
                            **(metadata or {}),
                            "doc_id": str(document_id),
                            "text": text,
                            "generation": str(settings.ACTIVE_INDEX_VERSION),
                        }
                    )

        if not all_documents:
            logger.warning("Sparse 索引: 无文档可索引")
            return False

        # 重建实例
        _sparse_search = LearnedSparseSearch()
        _sparse_search.set_encoder(encoder)
        _sparse_search.build_index(all_documents)

        logger.info(f"Sparse 索引已重建: {len(all_documents)} 条文档")
        return True

    except Exception as e:
        logger.error(f"Sparse 索引重建失败: {e}")
        return False
