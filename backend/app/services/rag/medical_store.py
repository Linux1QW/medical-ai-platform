# -*- coding: utf-8 -*-
"""ChromaDB 医学知识存储 — 基于向量检索的医学指南管理"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

from app.core.config import settings
from app.services.rag.embeddings import EMBEDDING_DIM, get_embedding

logger = logging.getLogger(__name__)

# 持久化目录: backend/data/medical_kb/
PERSIST_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "medical_kb"
)

COLLECTION_NAME = "medical_guidelines"  # 保留用于向后兼容

# ChromaDB 1.5.7 已知缺陷：一旦集合规模越过 hnsw:sync_threshold（默认 1000），
# HNSW 索引会被落盘为独立段，而该版本自身的段读取器无法再把它加载回来，
# 跨进程冷读报 "Error loading hnsw index"。将阈值设为极大值可让索引始终留在
# WAL（向量真值持久化于 chroma.sqlite3），进程首次查询时从 WAL 内存重建，
# 规避坏段读取路径。数据零丢失，仅首查有一次性重建开销。
# 注意：该 metadata 仅在集合【创建时】生效；get_or_create 对已存在集合会忽略
# metadata。存量旧集合仍是默认阈值，需经 rebuild_kb_from_cache.py --fresh
# 删除重建才真正继承此配置。
_HNSW_SYNC_THRESHOLD = 1_000_000
COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:sync_threshold": _HNSW_SYNC_THRESHOLD,
    "embedding_dim": EMBEDDING_DIM,
}

# 懒加载缓存：避免每次调用都查询 ChromaDB
_resolved_collection_name: Optional[str] = None

# 构建模式标志：True 时跳过回退逻辑，让 ChromaDB 自动创建 collection
_build_mode: bool = False


class IndexGenerationUnavailable(RuntimeError):
    """Raised when an explicitly requested generation cannot be served."""

    status = "unavailable"


def set_build_mode(enabled: bool) -> None:
    """设置构建模式标志（构建索引时调用，禁用 collection 回退逻辑）"""
    global _build_mode
    _build_mode = enabled


def _get_collection_name(
    use_cache: bool = True,
    *,
    generation: Optional[str] = None,
) -> str:
    """根据指定 generation 返回 collection 名称。

    生产默认不允许回退到旧名称。只有显式启用
    ``RAG_LEGACY_COLLECTION_FALLBACK`` 的迁移窗口可以使用 legacy collection。

    构建模式下直接返回版本化名称，让 ChromaDB 的 get_or_create 自动创建。
    """
    global _resolved_collection_name

    if generation is None and use_cache and _resolved_collection_name is not None:
        return _resolved_collection_name

    version = generation or getattr(settings, 'ACTIVE_INDEX_VERSION', 'rag-v1')
    versioned_name = f"medical_guidelines_{version}"

    if _build_mode:
        # 构建模式：直接返回版本化名称，让 ChromaDB 自动创建
        if generation is None:
            _resolved_collection_name = versioned_name
        return versioned_name

    # 检索模式：保留回退逻辑
    # 检查版本化 collection 是否存在
    try:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        collections = client.list_collections()
        collection_names = [c.name for c in collections] if collections else []

        if versioned_name in collection_names:
            if generation is None:
                _resolved_collection_name = versioned_name
            return versioned_name

        if (
            settings.RAG_LEGACY_COLLECTION_FALLBACK
            and COLLECTION_NAME in collection_names
        ):
            logger.warning(
                f"版本化 collection '{versioned_name}' 不存在，"
                f"显式迁移开关允许回退到旧 collection '{COLLECTION_NAME}'。"
                f"建议运行索引重建并切换到新版本。"
            )
            if generation is None:
                _resolved_collection_name = COLLECTION_NAME
            return COLLECTION_NAME
    except Exception as e:
        if isinstance(e, IndexGenerationUnavailable):
            raise
        logger.debug(f"检查 collection 存在性失败: {e}")

    logger.error(
        "RAG generation unavailable: generation=%s collection=%s; alert required",
        version,
        versioned_name,
    )
    raise IndexGenerationUnavailable(
        f"RAG generation {version!r} is unavailable: "
        f"collection {versioned_name!r} does not exist"
    )


def _reset_collection_cache() -> None:
    """重置 collection 名称缓存并刷新单例的 collection 引用（版本切换后调用）"""
    global _resolved_collection_name
    _resolved_collection_name = None
    # 如果单例已存在，刷新其 collection 引用
    if _medical_store is not None and _medical_store.client is not None:
        _medical_store.refresh_collection()


class MedicalKnowledgeStore:
    """基于 ChromaDB 的医学指南向量存储"""

    def __init__(self) -> None:
        self.client: Optional[ClientAPI] = None
        self.collection: Optional[chromadb.Collection] = None

    def _init_client(self) -> None:
        """初始化 ChromaDB 客户端（持久化模式）"""
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        collection_name = _get_collection_name()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=dict(COLLECTION_METADATA),
        )
        logger.info(f"ChromaDB 医学知识库已初始化: {PERSIST_DIR} (collection={collection_name})")

    def _ensure_client(self) -> ClientAPI:
        """返回已初始化的 client（用于类型收窄）"""
        if self.client is None:
            self._init_client()
        assert self.client is not None
        return self.client

    def _ensure_collection(self) -> chromadb.Collection:
        """返回已初始化的 collection（用于类型收窄）"""
        if self.collection is None:
            self._init_client()
        assert self.collection is not None
        return self.collection

    def refresh_collection(self) -> None:
        """重新解析并更新 collection 引用（版本切换后调用）"""
        if self.client is None:
            self._init_client()
            return
        collection_name = _get_collection_name(use_cache=False)
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=dict(COLLECTION_METADATA),
            )
            logger.info(f"Collection 引用已刷新: {collection_name}")
        except Exception as e:
            logger.error(f"刷新 collection 引用失败: {e}")

    def get_collection_for_generation(
        self,
        generation: str,
        *,
        create: bool = False,
    ) -> chromadb.Collection:
        """Return an exact generation collection without changing active state."""
        client = self._ensure_client()
        name = f"medical_guidelines_{generation}"
        if create:
            return client.get_or_create_collection(
                name=name,
                metadata=dict(COLLECTION_METADATA),
            )
        try:
            return client.get_collection(name)
        except Exception as exc:
            logger.error(
                "RAG generation unavailable: generation=%s collection=%s; "
                "alert required",
                generation,
                name,
            )
            raise IndexGenerationUnavailable(
                f"RAG generation {generation!r} is unavailable"
            ) from exc

    def get_documents_by_ids(
        self,
        doc_ids: List[str],
        *,
        generation: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Hydrate cache records from one exact generation by document ID."""
        if not doc_ids:
            return {}
        collection = self.get_collection_for_generation(generation)
        result = collection.get(ids=doc_ids, include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        hydrated: Dict[str, Dict[str, Any]] = {}
        for position, doc_id in enumerate(ids):
            hydrated[str(doc_id)] = {
                "text": documents[position] if position < len(documents) else "",
                "metadata": (
                    metadatas[position] if position < len(metadatas) else {}
                ),
            }
        return hydrated

    def export_generation_documents(self, generation: str) -> List[Dict[str, Any]]:
        """Read a complete immutable snapshot for incremental candidate builds."""
        collection = self.get_collection_for_generation(generation)
        result = collection.get(include=["documents", "metadatas", "embeddings"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        embeddings = result.get("embeddings")
        snapshot = []
        for position, doc_id in enumerate(ids):
            snapshot.append(
                {
                    "id": str(doc_id),
                    "text": documents[position] if position < len(documents) else "",
                    "metadata": (
                        dict(metadatas[position])
                        if position < len(metadatas)
                        else {}
                    ),
                    "embedding": (
                        list(embeddings[position])
                        if embeddings is not None and position < len(embeddings)
                        else None
                    ),
                }
            )
        return snapshot

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
    ) -> None:
        """添加文档块到集合

        Args:
            ids: 文档唯一标识列表
            documents: 文档文本列表
            embeddings: 文档向量列表
            metadatas: 文档元数据列表（包含 source, page 等）
        """
        collection = self._ensure_collection()

        # ChromaDB 单次添加上限约 5000 条，分批处理
        batch_size = 1000
        total = len(ids)

        for i in range(0, total, batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]

            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=cast(Any, batch_embs),
                metadatas=cast(Any, batch_metas),
            )
            logger.debug(
                f"已添加批次 {i // batch_size + 1}/{(total - 1) // batch_size + 1}: "
                f"{len(batch_ids)} 条文档"
            )

        logger.info(f"共添加 {total} 条文档到医学知识库")

    async def search(
        self,
        query_text: str,
        top_k: int = 5,
        where_document: Optional[Dict] = None,
        *,
        generation: Optional[str] = None,
    ) -> List[Dict]:
        """检索相关医学证据

        Args:
            query_text: 查询文本（如诊断结果）
            top_k: 返回条数
            where_document: 可选的 ChromaDB 文档内容过滤条件（如
                {"$contains": "肺癌"}）。传入时先做带过滤查询，若命中
                不足 METADATA_FILTER_MIN_RESULTS 则自动回退为无过滤查询，
                避免过滤过度导致漏召。

        Returns:
            医学证据列表 [{"text": ..., "source": ..., "page": ..., "score": ...}, ...]
        """
        collection = (
            self.get_collection_for_generation(generation)
            if generation is not None
            else self._ensure_collection()
        )

        if collection.count() == 0:
            logger.debug("医学知识库为空，无检索结果")
            return []

        # 1. 异步获取查询向量
        query_embedding = await get_embedding(query_text)

        # 2. ChromaDB 查询为同步阻塞调用，放入线程池避免阻塞事件循环
        #    （与 retriever 中 BM25 / sparse 检索的 run_in_executor 模式一致）
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._query_with_fallback(
                collection,
                query_embedding,
                top_k,
                where_document,
            ),
        )

        # 3. 格式化结果
        evidences = self._format_results(results, generation=generation)
        logger.debug(f"医学知识库检索返回 {len(evidences)} 条证据")
        return evidences

    def _query_with_fallback(
        self,
        collection: chromadb.Collection,
        query_embedding: List[float],
        top_k: int,
        where_document: Optional[Dict],
    ) -> Any:
        """执行 ChromaDB 查询；带过滤时命中不足则回退无过滤查询"""
        include: Any = ["documents", "metadatas", "distances"]
        if where_document:
            try:
                filtered = collection.query(
                    query_embeddings=cast(Any, [query_embedding]),
                    n_results=top_k,
                    where_document=where_document,
                    include=include,
                )
                hit_count = (
                    len(filtered["ids"][0])
                    if filtered.get("ids") and filtered["ids"]
                    else 0
                )
                if hit_count >= settings.METADATA_FILTER_MIN_RESULTS:
                    logger.debug(
                        f"metadata 预过滤命中 {hit_count} 条（filter={where_document}）"
                    )
                    return filtered
                logger.debug(
                    f"metadata 预过滤仅命中 {hit_count} 条（<"
                    f"{settings.METADATA_FILTER_MIN_RESULTS}），回退无过滤查询"
                )
            except Exception as e:
                logger.warning(f"metadata 预过滤查询失败，回退无过滤：{e}")

        return collection.query(
            query_embeddings=cast(Any, [query_embedding]),
            n_results=top_k,
            include=include,
        )

    def _format_results(
        self,
        results: Dict,
        *,
        generation: Optional[str] = None,
    ) -> List[Dict]:
        """将 ChromaDB 查询结果格式化为证据列表"""
        evidences = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                doc_text = (
                    results["documents"][0][i]
                    if results["documents"]
                    else ""
                )
                metadata = (
                    results["metadatas"][0][i]
                    if results["metadatas"]
                    else {}
                )
                distance = (
                    results["distances"][0][i]
                    if results["distances"]
                    else 0.0
                )
                score = 1.0 - float(distance)

                evidences.append(
                    {
                        "doc_id": doc_id,
                        "generation": generation or metadata.get(
                            "index_generation"
                        ),
                        "text": doc_text,
                        "source": metadata.get("source", "未知"),
                        "page": metadata.get("page", 0),
                        "score": round(score, 4),
                        "heading_path": metadata.get("heading_path", ""),
                        "chunk_seq": metadata.get("chunk_seq", -1),
                        "content_type": metadata.get("content_type", ""),
                        "organization": metadata.get("organization"),
                        "year": metadata.get("year"),
                        "version": metadata.get("version"),
                        "document_type": metadata.get("document_type"),
                        "departments": metadata.get("departments"),
                        "disease_tags": metadata.get("disease_tags"),
                        "population": metadata.get("population"),
                        "recommendation_level": metadata.get("recommendation_level"),
                        "evidence_level": metadata.get("evidence_level"),
                        "metadata_source": metadata.get("metadata_source"),
                    }
                )
        return evidences

    async def search_by_embedding(
        self,
        query_embedding: List[float],
        *,
        top_k: int,
        generation: str,
    ) -> List[Dict]:
        """Search the exact generation with a caller-provided embedding."""
        collection = self.get_collection_for_generation(generation)
        if collection.count() == 0:
            return []
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: collection.query(
                query_embeddings=cast(Any, [query_embedding]),
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            ),
        )
        return self._format_results(results, generation=generation)

    def count(self) -> int:
        """返回知识库中文档总数"""
        if self.collection is None:
            return 0
        return self.collection.count()

    def get_all_sources(self) -> List[str]:
        """返回知识库中所有已索引的来源文件名（去重列表）"""
        collection = self._ensure_collection()
        if collection.count() == 0:
            return []
        result = collection.get(include=["metadatas"])
        sources: set[str] = set()
        for meta in result.get("metadatas") or []:
            src = meta.get("source", "")
            if isinstance(src, str) and src:
                sources.add(src)
        return sorted(sources)

    def get_source_doc_count(self, source: str) -> int:
        """返回指定来源的文档块数量"""
        collection = self._ensure_collection()
        result = collection.get(
            where={"source": source},
            include=[],
        )
        return len(result.get("ids") or [])

    def fetch_neighbors(
        self,
        source: str,
        chunk_seq: int,
        window: int = 1,
        *,
        generation: Optional[str] = None,
    ) -> Dict[int, Dict]:
        """按 source + chunk_seq 拉取相邻文本块（Small-to-Big 上下文扩展）

        Args:
            source: 来源文件名
            chunk_seq: 中心块在该来源内的全局序号
            window: 向前 / 向后各拉取的邻居块数

        Returns:
            {chunk_seq: {"text": ..., "page": ...}} 映射（仅邻居，不含中心块）。
            无邻居或查询失败时返回空 dict。
        """
        collection = (
            self.get_collection_for_generation(generation)
            if generation is not None
            else self._ensure_collection()
        )
        if window <= 0 or chunk_seq is None or chunk_seq < 0:
            return {}
        wanted = [
            s
            for s in range(chunk_seq - window, chunk_seq + window + 1)
            if s >= 0 and s != chunk_seq
        ]
        if not wanted:
            return {}
        try:
            result = collection.get(
                where=cast(Any, {
                    "$and": [
                        {"source": source},
                        {"chunk_seq": {"$in": wanted}},
                    ]
                }),
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning(f"邻居块查询失败（source={source}, seq={chunk_seq}）: {e}")
            return {}

        neighbors: Dict[int, Dict] = {}
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        for i in range(len(ids)):
            meta = metas[i] if i < len(metas) else {}
            seq = meta.get("chunk_seq", -1)
            if not isinstance(seq, int):
                continue
            neighbors[seq] = {
                "text": docs[i] if i < len(docs) else "",
                "page": meta.get("page", 0),
            }
        return neighbors

    def delete_by_source(self, source: str) -> int:
        """删除指定来源的全部文档块，返回删除条数"""
        collection = self._ensure_collection()
        # 先查出 IDs，ChromaDB 的 delete(where=...) 并非所有版本都稳定，
        # 以 ID 列表删除最为可靠
        result = collection.get(
            where={"source": source},
            include=[],
        )
        ids = result.get("ids") or []
        if ids:
            collection.delete(ids=ids)
            logger.info(f"已删除来源 '{source}' 的 {len(ids)} 条文档")
        return len(ids)


# 全局单例
_medical_store: Optional[MedicalKnowledgeStore] = None


def get_medical_store() -> MedicalKnowledgeStore:
    """获取全局医学知识库单例（懒加载）"""
    global _medical_store
    if _medical_store is None:
        _medical_store = MedicalKnowledgeStore()
        _medical_store._init_client()
    return _medical_store


def list_index_versions() -> list[str]:
    """列出所有存在的索引版本"""
    store = get_medical_store()
    client = store._ensure_client()
    collections = client.list_collections()
    versions = []
    for col in collections:
        if col.name.startswith("medical_guidelines_"):
            version = col.name.replace("medical_guidelines_", "")
            versions.append(version)
    return versions


def get_collection_count(collection_name: str = None) -> int:
    """获取指定 collection 的文档数"""
    store = get_medical_store()
    client = store._ensure_client()
    name = collection_name or _get_collection_name()
    try:
        col = client.get_collection(name)
        return col.count()
    except Exception:
        return 0
