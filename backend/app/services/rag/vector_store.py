# -*- coding: utf-8 -*-
"""FAISS 向量索引管理 — 构建、保存、加载、检索"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.services.rag.embeddings import EMBEDDING_DIM

logger = logging.getLogger(__name__)

# 索引文件存放目录：backend/data/rag_index/
INDEX_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "rag_index"
FAISS_INDEX_FILE = INDEX_DIR / "cases.faiss"
METADATA_FILE = INDEX_DIR / "cases_meta.pkl"


class CaseVectorStore:
    """基于 FAISS 的病例向量存储"""

    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict] = []  # 与向量一一对应的病例元数据

    def build(self, embeddings: List[List[float]], metadata: List[Dict]):
        """从向量和元数据构建索引"""
        vectors = np.array(embeddings, dtype=np.float32)
        # L2 归一化后使用内积，等效于余弦相似度
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(vectors)
        self.metadata = metadata
        logger.info(f"FAISS 索引构建完成: {self.index.ntotal} 条病例")

    def save(self):
        """持久化索引和元数据到磁盘"""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        # 用 serialize_index + Python 写入，规避 FAISS C++ 层不支持中文路径的问题
        index_bytes = faiss.serialize_index(self.index)
        with open(FAISS_INDEX_FILE, "wb") as f:
            f.write(faiss.serialize_index(self.index))
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"索引已保存至 {INDEX_DIR}")

    def load(self) -> bool:
        """从磁盘加载索引，成功返回 True"""
        if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
            logger.warning(f"索引文件不存在: {INDEX_DIR}")
            return False
        with open(FAISS_INDEX_FILE, "rb") as f:
            index_bytes = f.read()
        self.index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
        with open(METADATA_FILE, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info(f"FAISS 索引已加载: {self.index.ntotal} 条病例")
        return True

    def search(
        self, query_vector: List[float], top_k: int = 3, exclude_id: Optional[str] = None
    ) -> List[Tuple[Dict, float]]:
        """检索最相似的 top_k 条病例，返回 [(metadata, score), ...]"""
        if self.index is None or self.index.ntotal == 0:
            return []

        q = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(q)

        # 多检索几条以便排除后仍有足够结果
        search_k = top_k + 3 if exclude_id else top_k
        scores, indices = self.index.search(q, min(search_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            if exclude_id and meta.get("patient_id") == exclude_id:
                continue
            results.append((meta, float(score)))
            if len(results) >= top_k:
                break

        return results


# 全局单例
_store: Optional[CaseVectorStore] = None


def get_store() -> CaseVectorStore:
    """获取全局向量存储单例（懒加载）"""
    global _store
    if _store is None:
        _store = CaseVectorStore()
        if not _store.load():
            logger.warning("RAG 索引未构建，相似病例检索将不可用。请先运行 build_index.py")
    return _store
