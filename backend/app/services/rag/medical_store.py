# -*- coding: utf-8 -*-
"""ChromaDB 医学知识存储 — 基于向量检索的医学指南管理"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from app.core.config import settings
from app.services.rag.embeddings import EMBEDDING_DIM, get_embedding

logger = logging.getLogger(__name__)

# 持久化目录: backend/data/medical_kb/
PERSIST_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "medical_kb"
)

COLLECTION_NAME = "medical_guidelines"  # 保留用于向后兼容

# 懒加载缓存：避免每次调用都查询 ChromaDB
_resolved_collection_name: Optional[str] = None

# 构建模式标志：True 时跳过回退逻辑，让 ChromaDB 自动创建 collection
_build_mode: bool = False


def set_build_mode(enabled: bool) -> None:
    """设置构建模式标志（构建索引时调用，禁用 collection 回退逻辑）"""
    global _build_mode
    _build_mode = enabled


def _get_collection_name(use_cache: bool = True) -> str:
    """根据活跃索引版本返回 collection 名称，带向后兼容回退

    首次调用时检查 ChromaDB 中实际存在的 collection，
    如果版本化 collection 不存在则回退到旧名称 'medical_guidelines'。

    构建模式下直接返回版本化名称，让 ChromaDB 的 get_or_create 自动创建。
    """
    global _resolved_collection_name

    if use_cache and _resolved_collection_name is not None:
        return _resolved_collection_name

    version = getattr(settings, 'ACTIVE_INDEX_VERSION', 'rag-v1')
    versioned_name = f"medical_guidelines_{version}"

    if _build_mode:
        # 构建模式：直接返回版本化名称，让 ChromaDB 自动创建
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
            _resolved_collection_name = versioned_name
            return versioned_name

        # 回退到旧名称
        if "medical_guidelines" in collection_names:
            logger.warning(
                f"版本化 collection '{versioned_name}' 不存在，"
                f"回退到旧 collection 'medical_guidelines'。"
                f"建议运行索引重建并切换到新版本。"
            )
            _resolved_collection_name = "medical_guidelines"
            return "medical_guidelines"
    except Exception as e:
        logger.debug(f"检查 collection 存在性失败: {e}")

    # 默认返回版本化名称（首次部署场景）
    _resolved_collection_name = versioned_name
    return versioned_name


def _reset_collection_cache() -> None:
    """重置 collection 名称缓存并刷新单例的 collection 引用（版本切换后调用）"""
    global _resolved_collection_name
    _resolved_collection_name = None
    # 如果单例已存在，刷新其 collection 引用
    if _medical_store is not None and _medical_store.client is not None:
        _medical_store.refresh_collection()


class MedicalKnowledgeStore:
    """基于 ChromaDB 的医学指南向量存储"""

    def __init__(self):
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None

    def _init_client(self):
        """初始化 ChromaDB 客户端（持久化模式）"""
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        collection_name = _get_collection_name()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_dim": EMBEDDING_DIM},
        )
        logger.info(f"ChromaDB 医学知识库已初始化: {PERSIST_DIR} (collection={collection_name})")

    def refresh_collection(self):
        """重新解析并更新 collection 引用（版本切换后调用）"""
        if self.client is None:
            self._init_client()
            return
        collection_name = _get_collection_name(use_cache=False)
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "embedding_dim": EMBEDDING_DIM},
            )
            logger.info(f"Collection 引用已刷新: {collection_name}")
        except Exception as e:
            logger.error(f"刷新 collection 引用失败: {e}")

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
    ):
        """添加文档块到集合

        Args:
            ids: 文档唯一标识列表
            documents: 文档文本列表
            embeddings: 文档向量列表
            metadatas: 文档元数据列表（包含 source, page 等）
        """
        if self.collection is None:
            self._init_client()

        # ChromaDB 单次添加上限约 5000 条，分批处理
        batch_size = 1000
        total = len(ids)

        for i in range(0, total, batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]

            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embs,
                metadatas=batch_metas,
            )
            logger.debug(
                f"已添加批次 {i // batch_size + 1}/{(total - 1) // batch_size + 1}: "
                f"{len(batch_ids)} 条文档"
            )

        logger.info(f"共添加 {total} 条文档到医学知识库")

    async def search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """检索相关医学证据

        Args:
            query_text: 查询文本（如诊断结果）
            top_k: 返回条数

        Returns:
            医学证据列表 [{"text": ..., "source": ..., "page": ..., "score": ...}, ...]
        """
        if self.collection is None:
            self._init_client()

        if self.collection.count() == 0:
            logger.debug("医学知识库为空，无检索结果")
            return []

        # 1. 异步获取查询向量
        query_embedding = await get_embedding(query_text)

        # 2. 同步执行 ChromaDB 查询
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # 3. 格式化结果
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
                        "text": doc_text,
                        "source": metadata.get("source", "未知"),
                        "page": metadata.get("page", 0),
                        "score": round(score, 4),
                        "heading_path": metadata.get("heading_path", ""),
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

        logger.debug(f"医学知识库检索返回 {len(evidences)} 条证据")
        return evidences

    def count(self) -> int:
        """返回知识库中文档总数"""
        if self.collection is None:
            return 0
        return self.collection.count()

    def get_all_sources(self) -> List[str]:
        """返回知识库中所有已索引的来源文件名（去重列表）"""
        if self.collection is None:
            self._init_client()
        if self.collection.count() == 0:
            return []
        result = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in result.get("metadatas") or []:
            src = meta.get("source", "")
            if src:
                sources.add(src)
        return sorted(sources)

    def get_source_doc_count(self, source: str) -> int:
        """返回指定来源的文档块数量"""
        if self.collection is None:
            self._init_client()
        result = self.collection.get(
            where={"source": source},
            include=[],
        )
        return len(result.get("ids") or [])

    def delete_by_source(self, source: str) -> int:
        """删除指定来源的全部文档块，返回删除条数"""
        if self.collection is None:
            self._init_client()
        # 先查出 IDs，ChromaDB 的 delete(where=...) 并非所有版本都稳定，
        # 以 ID 列表删除最为可靠
        result = self.collection.get(
            where={"source": source},
            include=[],
        )
        ids = result.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
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
    if store.client is None:
        store._init_client()
    collections = store.client.list_collections()
    versions = []
    for col in collections:
        if col.name.startswith("medical_guidelines_"):
            version = col.name.replace("medical_guidelines_", "")
            versions.append(version)
    return versions


def get_collection_count(collection_name: str = None) -> int:
    """获取指定 collection 的文档数"""
    store = get_medical_store()
    if store.client is None:
        store._init_client()
    name = collection_name or _get_collection_name()
    try:
        col = store.client.get_collection(name)
        return col.count()
    except Exception:
        return 0
