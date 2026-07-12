# -*- coding: utf-8 -*-
"""BM25 关键词检索模块 — 基于 Okapi BM25 算法的医学文本关键词匹配

BM25 擅长精确术语匹配（药物名称、疾病编码、检查项目名称），
与向量检索互补，共同构成混合检索的基础。
"""

import logging
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import jieba

logger = logging.getLogger(__name__)

# ── BM25 超参数 ──
K1 = 1.5    # 词频饱和参数
B = 0.75    # 文档长度归一化参数


class BM25Index:
    """基于 Okapi BM25 算法的文本检索索引"""

    def __init__(self):
        self.documents: List[Dict] = []        # 原始文档列表
        self.doc_tokens: List[List[str]] = []  # 分词后的文档
        self.doc_lengths: List[int] = []       # 每个文档的 token 数
        self.avg_doc_length: float = 0.0       # 平均文档长度
        self.doc_count: int = 0                # 文档总数
        self.df: Dict[str, int] = {}           # 文档频率（包含某词的文档数）
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
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0

        # 计算文档频率 (DF)
        self.df = {}
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] = self.df.get(token, 0) + 1

        self.initialized = True
        logger.info(f"BM25 索引构建完成：{self.doc_count} 个文档，词汇量 {len(self.df)}")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """执行 BM25 检索

        Args:
            query: 查询文本
            top_k: 返回条数

        Returns:
            检索结果列表，每个结果增加 "bm25_score" 字段
        """
        if not self.initialized or not query or not query.strip():
            return []

        query_tokens = tokenize_medical_text(query)
        if not query_tokens:
            return []

        # 计算每个文档的 BM25 分数
        scores = []
        for doc_idx in range(self.doc_count):
            score = self._score_document(query_tokens, doc_idx)
            if score > 0:
                scores.append((doc_idx, score))

        # 按分数降序排列
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回 top_k 结果
        results = []
        for doc_idx, score in scores[:top_k]:
            doc_copy = dict(self.documents[doc_idx])
            doc_copy["bm25_score"] = round(score, 4)
            results.append(doc_copy)

        return results

    def _score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """计算单个文档的 BM25 得分"""
        doc_tokens = self.doc_tokens[doc_idx]
        doc_len = self.doc_lengths[doc_idx]

        if doc_len == 0:
            return 0.0

        # 文档中的词频统计
        tf_in_doc = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in self.df:
                continue

            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            df = self.df[term]
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

            # TF 归一化
            tf = tf_in_doc.get(term, 0)
            tf_norm = (tf * (K1 + 1)) / (
                tf + K1 * (1 - B + B * doc_len / self.avg_doc_length)
            )

            score += idf * tf_norm

        return score


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
    """医学文本分词 — jieba 分词 + 英文术语提取 + bigram 兜底

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
        from app.services.rag.medical_store import get_medical_store, _get_collection_name

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
