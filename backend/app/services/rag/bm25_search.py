# -*- coding: utf-8 -*-
"""医学 BM25 检索引擎 — 基于 bm25s + jieba 医学词典

BM25 擅长精确术语匹配（药物名称、疾病编码、检查项目名称），
与向量检索互补，共同构成混合检索的基础。

使用 bm25s 替代手写 Okapi BM25，获得更好的性能和精度。
"""

import logging
import os
import re
from typing import Dict, List, Optional

import bm25s
import jieba

logger = logging.getLogger(__name__)

# ── BM25 超参数（医学文档调优）──
K1 = 1.2   # 词频饱和参数（医学文档较长，降低至 1.2）
B = 0.8    # 长度惩罚参数（提高至 0.8 惩罚长文档）


# ── 加载医学自定义词典（模块级别，只执行一次）──
_medical_dict_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'data', 'medical_dict.txt'
)
if os.path.exists(_medical_dict_path):
    jieba.load_userdict(_medical_dict_path)
    logger.info(f"jieba 医学词典已加载: {_medical_dict_path}")
else:
    logger.warning(f"医学词典文件不存在: {_medical_dict_path}，将使用 jieba 默认分词")


# ── 医学文本停用词表（精简版）──
MEDICAL_STOPWORDS = {
    "的", "了", "是", "在", "有", "和", "与", "及", "或", "等",
    "为", "被", "把", "将", "从", "到", "对", "以", "可", "也",
    "就", "都", "而", "且", "但", "则", "要", "能", "会", "应",
    "该", "其", "这", "那", "之", "于", "中", "上", "下", "不",
    "无", "未", "已", "所", "如", "若", "因", "由", "时", "后",
    "前", "间", "内", "外", "者", "用", "需", "可以", "应该",
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "and", "or", "of", "in", "on", "at", "to", "for", "with",
}


def tokenize_medical_text(text: str) -> List[str]:
    """医学文本分词 — jieba 分词 + 英文术语保护 + bigram 兜底

    策略：
    1. 保留完整的英文术语和数字（如 NSCLC、EGFR、PD-L1、20mg）
    2. 中文使用 jieba 分词（加载医学自定义词典提升复合术语召回）
    3. 保留 bigram 作为 fallback（对 jieba 可能切错的长词兜底）
    4. 去除停用词和标点符号

    Args:
        text: 输入文本

    Returns:
        token 列表
    """
    if not text:
        return []

    text = text.lower().strip()
    tokens = []

    # 1. 提取英文术语和数字（保留连字符，如 PD-L1）
    english_pattern = re.compile(r'[a-z][a-z0-9\-]*[a-z0-9]|[a-z]|\d+\.?\d*')
    english_tokens = english_pattern.findall(text)
    tokens.extend(english_tokens)

    # 2. jieba 中文分词（先移除英文/数字部分，避免干扰）
    chinese_text = re.sub(r'[a-zA-Z0-9\.\-\_]+', ' ', text)
    words = jieba.cut(chinese_text)

    for word in words:
        word = word.strip()
        if not word:
            continue
        # 过滤停用词
        if word in MEDICAL_STOPWORDS:
            continue
        # 过滤单字（除非是数字）
        if len(word) == 1 and not word.isdigit():
            continue
        tokens.append(word)

    # 3. 保留 bigram 作为补充（对 jieba 可能切错的长词提供 fallback）
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    for i in range(len(chinese_chars) - 1):
        bigram = chinese_chars[i] + chinese_chars[i + 1]
        if bigram not in MEDICAL_STOPWORDS:
            tokens.append(bigram)

    return tokens


class BM25Index:
    """基于 bm25s 引擎的文本检索索引（向后兼容接口）

    内部使用 bm25s 进行索引和检索，对外保持与原 Okapi BM25 实现相同的 API：
    - build(documents, text_field) 构建索引
    - search(query, top_k) 返回 List[Dict]（含 bm25_score 字段）
    """

    def __init__(self):
        self.documents: List[Dict] = []        # 原始文档列表
        self.doc_tokens: List[List[str]] = []  # 分词后的文档
        self.doc_count: int = 0                # 文档总数
        self._bm25: Optional[bm25s.BM25] = None  # bm25s 索引实例
        self.initialized: bool = False

    def build(self, documents: List[Dict], text_field: str = "text"):
        """构建 BM25 索引

        Args:
            documents: 文档列表，每个文档为 dict
            text_field: 用于检索的文本字段名
        """
        self.documents = documents
        self.doc_count = len(documents)

        if self.doc_count == 0:
            self.initialized = False
            return

        # 分词
        self.doc_tokens = [
            tokenize_medical_text(doc.get(text_field, ""))
            for doc in documents
        ]

        # 使用 bm25s 构建索引
        self._bm25 = bm25s.BM25(method="lucene", k1=K1, b=B)
        self._bm25.index(self.doc_tokens, show_progress=False)

        self.initialized = True
        logger.info(
            f"BM25 索引构建完成（bm25s 引擎）：{self.doc_count} 个文档，"
            f"平均词元数 {sum(len(t) for t in self.doc_tokens) / max(self.doc_count, 1):.0f}"
        )

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """执行 BM25 检索

        Args:
            query: 查询文本
            top_k: 返回条数

        Returns:
            检索结果列表，每个结果增加 "bm25_score" 字段
        """
        if not self.initialized or not self._bm25 or not query or not query.strip():
            return []

        query_tokens = tokenize_medical_text(query)
        if not query_tokens:
            return []

        # bm25s 检索（retrieve 接受 List[List[str]]，返回 2D numpy 数组）
        results, scores = self._bm25.retrieve(
            [query_tokens], k=min(top_k, self.doc_count), show_progress=False
        )

        # 转换为与原接口兼容的格式
        final_results = []
        for idx, score in zip(results[0], scores[0]):
            idx = int(idx)
            score = float(score)
            if score <= 0:
                continue
            doc_copy = dict(self.documents[idx])
            doc_copy["bm25_score"] = round(score, 4)
            # 与向量检索结果字段对齐，保证 RRF 跨路去重生效
            doc_copy["doc_id"] = doc_copy.get("id", "")
            final_results.append(doc_copy)
            if len(final_results) >= top_k:
                break

        return final_results


# ── 全局 BM25 索引单例 ──
_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """获取全局 BM25 索引单例（懒加载，从 ChromaDB 加载文档）"""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
        _try_load_documents()
    return _bm25_index


def _try_load_documents():
    """尝试从当前活跃版本的 ChromaDB collection 加载文档构建 BM25 索引"""
    try:
        from app.services.rag.medical_store import _get_collection_name, get_medical_store

        store = get_medical_store()
        if store.client is None:
            store._init_client()

        collection_name = _get_collection_name()
        try:
            collection = store.client.get_collection(collection_name)
        except Exception:
            logger.warning(f"BM25 索引: collection '{collection_name}' 不存在")
            return

        if collection.count() == 0:
            logger.warning(f"BM25 索引: collection '{collection_name}' 为空")
            return

        # 从 collection 获取所有文档
        count = collection.count()
        all_docs = []
        batch_size = 1000
        for offset in range(0, count, batch_size):
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            if result["ids"]:
                for i, doc_id in enumerate(result["ids"]):
                    doc_text = result["documents"][i] if result["documents"] else ""
                    metadata = result["metadatas"][i] if result["metadatas"] else {}
                    all_docs.append({
                        "id": doc_id,
                        "text": doc_text,
                        "source": metadata.get("source", "未知"),
                        "page": metadata.get("page", 0),
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
                    })

        if all_docs:
            _bm25_index.build(all_docs, text_field="text")
            logger.info(f"BM25 索引已从 collection '{collection_name}' 加载 {len(all_docs)} 个文档")
        else:
            logger.warning(f"Collection '{collection_name}' 中无文档，BM25 索引为空")

    except Exception as e:
        logger.warning(f"BM25 索引构建失败: {e}")


def rebuild_bm25_index():
    """强制重建 BM25 索引（在索引版本切换或数据更新后调用）"""
    global _bm25_index
    _bm25_index = BM25Index()
    _try_load_documents()
