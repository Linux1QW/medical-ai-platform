# -*- coding: utf-8 -*-
"""BGE-M3 双表示编码器 — Dense + Learned Sparse

基于 FlagEmbedding 的 BGE-M3 模型，同时输出：
- Dense embedding（1024 维，用于语义检索）
- Learned Sparse embedding（类似 SPLADE 的词表级稀疏表示，用于精确匹配）

模型约 2GB，默认延迟加载；通过 BGE_M3_ENABLED 配置开关控制启用。
"""

import logging
from typing import Optional

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class DualEncoder:
    """
    BGE-M3 双表示编码器

    同时输出 dense embedding 和 learned sparse embedding，
    为后续三路融合检索（BM25 + Dense + Sparse）提供基础。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_fp16: Optional[bool] = None,
    ):
        self._model = None
        self._model_path = model_path or settings.BGE_M3_MODEL_PATH
        self._use_fp16 = use_fp16 if use_fp16 is not None else settings.BGE_M3_USE_FP16
        self._loaded = False

    def _load_model(self) -> None:
        """延迟加载模型（首次调用时触发）

        模型约 2GB，避免模块导入时即占用大量内存。
        FP16 量化可减半 GPU 显存占用（需 CUDA 支持）。
        """
        if self._loaded:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]

            self._model = BGEM3FlagModel(
                self._model_path,
                use_fp16=self._use_fp16,
            )
            self._loaded = True
            logger.info(
                f"BGE-M3 模型加载完成: path={self._model_path}, fp16={self._use_fp16}"
            )
        except ImportError as e:
            logger.error(
                f"FlagEmbedding 未安装，请执行 pip install FlagEmbedding>=1.2.0: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"BGE-M3 模型加载失败: {e}")
            raise

    def encode_query(self, query: str) -> dict:
        """编码查询文本

        Args:
            query: 查询文本（可包含 BGE_M3_QUERY_INSTRUCTION 前缀）

        Returns:
            {
                "dense": np.ndarray,   # shape (1024,)
                "sparse": dict,        # {token_id: weight}
            }
        """
        self._load_model()

        # 添加查询指令前缀（提升检索效果）
        instruction = settings.BGE_M3_QUERY_INSTRUCTION
        prefixed_query = f"{instruction}{query}" if instruction else query

        output = self._model.encode(
            prefixed_query,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense = (
            output["dense_vecs"]
            if isinstance(output["dense_vecs"], np.ndarray)
            else np.array(output["dense_vecs"])
        )
        return {
            "dense": dense,
            "sparse": output["lexical_weights"],
        }

    def encode_corpus(self, texts: list) -> dict:
        """批量编码语料（不带查询指令前缀）

        Args:
            texts: 文本列表

        Returns:
            {
                "dense": np.ndarray,    # shape (N, 1024)
                "sparse": list[dict],   # N 个 {token_id: weight}
            }
        """
        self._load_model()

        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense = (
            output["dense_vecs"]
            if isinstance(output["dense_vecs"], np.ndarray)
            else np.array(output["dense_vecs"])
        )
        return {
            "dense": dense,
            "sparse": output["lexical_weights"],
        }

    @property
    def is_available(self) -> bool:
        """模型是否已加载并可用"""
        return self._loaded


# ── 模块级单例 ─────────────────────────────────────────────────────────────────
# 延迟初始化：仅在 BGE_M3_ENABLED=True 且首次调用时加载模型

_dual_encoder: Optional[DualEncoder] = None


def get_dual_encoder() -> Optional[DualEncoder]:
    """获取全局 DualEncoder 实例

    当 BGE_M3_ENABLED=False 时返回 None，调用方应降级为现有检索方案。
    """
    global _dual_encoder

    if not settings.BGE_M3_ENABLED:
        return None

    if _dual_encoder is None:
        try:
            _dual_encoder = DualEncoder()
        except Exception as e:
            logger.warning(f"BGE-M3 DualEncoder 初始化失败，降级为现有检索方案: {e}")
            return None

    return _dual_encoder
