# -*- coding: utf-8 -*-
"""Langfuse 链路追踪客户端

提供 LLM 调用、RAG 检索、Agent 步骤的 trace 记录能力。
LANGFUSE_ENABLED=False 时所有操作静默跳过，不影响业务逻辑。
"""

import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── 懒初始化单例 ─────────────────────────────────────────────────────────────

_tracer: Optional["LangfuseTracer"] = None


def get_tracer() -> "LangfuseTracer":
    """获取全局 LangfuseTracer 单例（懒初始化）"""
    global _tracer
    if _tracer is None:
        _tracer = LangfuseTracer()
    return _tracer


class LangfuseTracer:
    """Langfuse 链路追踪客户端

    初始化时根据 settings.LANGFUSE_ENABLED 决定是否真正连接 Langfuse。
    未启用时所有 trace 方法为空操作（zero overhead）。
    """

    def __init__(self):
        self._client = None
        if settings.LANGFUSE_ENABLED and settings.LANGFUSE_PUBLIC_KEY:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST,
                )
                logger.info(
                    "Langfuse 链路追踪已启用",
                    extra={"host": settings.LANGFUSE_HOST},
                )
            except Exception as e:
                logger.warning(f"Langfuse 初始化失败，追踪功能已禁用: {e}")
                self._client = None
        else:
            logger.debug("Langfuse 链路追踪未启用（LANGFUSE_ENABLED=False）")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def trace_llm_call(
        self,
        trace_name: str,
        model: str,
        prompt: str,
        completion: str,
        tokens: int,
        latency_ms: float,
    ) -> None:
        """记录 LLM 调用 trace

        Args:
            trace_name: trace 名称（如 "qwen_chat"）
            model: 模型名称
            prompt: 输入 prompt（截取前 500 字符）
            completion: 模型输出（截取前 500 字符）
            tokens: token 用量
            latency_ms: 调用延迟（毫秒）
        """
        if not self._client:
            return
        try:
            trace = self._client.trace(name=trace_name)
            trace.span(
                name="llm_call",
                input={"prompt": prompt[:500]},
                output={"completion": completion[:500]},
                metadata={
                    "model": model,
                    "tokens": tokens,
                    "latency_ms": round(latency_ms, 2),
                },
            )
            self._client.flush()
        except Exception as e:
            logger.debug(f"Langfuse trace_llm_call 异常（静默）: {e}")

    def trace_rag_retrieval(
        self,
        trace_name: str,
        query: str,
        results: List[Dict[str, Any]],
        latency_ms: float,
    ) -> None:
        """记录 RAG 检索 trace

        Args:
            trace_name: trace 名称（如 "rag_retrieval"）
            query: 检索查询文本
            results: 检索结果列表
            latency_ms: 检索延迟（毫秒）
        """
        if not self._client:
            return
        try:
            trace = self._client.trace(name=trace_name)
            trace.span(
                name="rag_retrieval",
                input={"query": query[:500]},
                output={
                    "result_count": len(results),
                    "top_scores": [
                        r.get("score", 0) for r in results[:5]
                    ],
                },
                metadata={"latency_ms": round(latency_ms, 2)},
            )
            self._client.flush()
        except Exception as e:
            logger.debug(f"Langfuse trace_rag_retrieval 异常（静默）: {e}")

    def trace_agent_step(
        self,
        trace_name: str,
        agent_name: str,
        step_data: Dict[str, Any],
    ) -> None:
        """记录 Agent 步骤 trace

        Args:
            trace_name: trace 名称
            agent_name: Agent 名称（如 "knowledge_agent"）
            step_data: 步骤数据（如输入/输出/决策）
        """
        if not self._client:
            return
        try:
            trace = self._client.trace(name=trace_name)
            trace.span(
                name=f"agent_step:{agent_name}",
                input=step_data.get("input", {}),
                output=step_data.get("output", {}),
                metadata={
                    "agent_name": agent_name,
                    "step_type": step_data.get("step_type", "unknown"),
                },
            )
            self._client.flush()
        except Exception as e:
            logger.debug(f"Langfuse trace_agent_step 异常（静默）: {e}")
