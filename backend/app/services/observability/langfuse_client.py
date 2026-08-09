# -*- coding: utf-8 -*-
"""Langfuse 链路追踪客户端（Task 7 隐私安全版）

提供 LLM 调用、RAG 检索、Agent 步骤的 trace 记录能力。
LANGFUSE_ENABLED=False 时所有操作静默跳过，不影响业务逻辑。

隐私策略：
- capture=false：prompt/completion/query/content 只输出 hmac_sha256 + char_count
- capture=true：先 sanitize_for_observability() 脱敏，再按最大长度截断
- 外部 trace metadata 固定带 trace_id、run_id、HMAC 后的 consultation_ref
- 原始 consultation_id 不发送到外部 Langfuse
"""

import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.services.observability.trace_context import (
    get_current_trace_context,
    sanitize_for_observability,
    summarize_sensitive_text,
)

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
    """Langfuse 链路追踪客户端（隐私安全版）

    初始化时根据 settings.LANGFUSE_ENABLED 决定是否真正连接 Langfuse。
    未启用时所有 trace 方法为空操作（zero overhead）。

    隐私处理集中在本类内部，调用方即使误传原文也不能绕过。
    """

    def __init__(self):
        self._client = None
        self._capture = settings.OBSERVABILITY_CAPTURE_CONTENT
        self._max_chars = settings.OBSERVABILITY_CONTENT_MAX_CHARS
        self._hmac_key = self._get_hmac_key()
        self._flushed = False

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

    def _get_hmac_key(self) -> bytes:
        """获取 HMAC 密钥字节"""
        if settings.OBSERVABILITY_HMAC_KEY:
            return settings.OBSERVABILITY_HMAC_KEY.get_secret_value().encode("utf-8")
        # 开发/测试环境未配置时使用固定默认值（仅用于测试）
        return b"default-dev-key-not-for-production-use!!"

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _process_text_for_trace(self, text: str) -> Any:
        """处理单个文本：根据 capture 模式返回脱敏/摘要结果"""
        if not text:
            return text
        if self._capture:
            # capture=true: 先脱敏，再截断
            sanitized = sanitize_for_observability(text)
            if len(sanitized) > self._max_chars:
                sanitized = sanitized[:self._max_chars] + "...[truncated]"
            return sanitized
        else:
            # capture=false: 只返回 HMAC + char_count
            return summarize_sensitive_text(text, self._hmac_key)

    def _process_data_for_trace(self, data: Any) -> Any:
        """处理任意数据：递归处理 dict/list，字符串走 _process_text_for_trace"""
        if data is None:
            return None
        if isinstance(data, str):
            return self._process_text_for_trace(data)
        if isinstance(data, dict):
            return {k: self._process_data_for_trace(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._process_data_for_trace(item) for item in data]
        # int / float / bool 等原样返回
        return data

    def _get_trace_metadata(self) -> dict:
        """从当前 TraceContext 提取 metadata（consultation_id 做 HMAC 处理）"""
        ctx = get_current_trace_context()
        if not ctx:
            return {}
        metadata = {
            "trace_id": ctx.trace_id,
            "run_id": ctx.run_id,
        }
        if ctx.consultation_id:
            # consultation_id 做 HMAC 处理，不原文上传
            metadata["consultation_ref"] = summarize_sensitive_text(
                str(ctx.consultation_id), self._hmac_key
            )["hmac_sha256"][:16]
        if ctx.agent_name:
            metadata["agent_name"] = ctx.agent_name
        if ctx.tool_name:
            metadata["tool_name"] = ctx.tool_name
        return metadata

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
            prompt: 输入 prompt（隐私处理）
            completion: 模型输出（隐私处理）
            tokens: token 用量
            latency_ms: 调用延迟（毫秒）
        """
        if not self._client:
            return
        try:
            trace = self._client.trace(name=trace_name)
            # 隐私处理：prompt 和 completion 都经过 _process_data_for_trace
            processed_input = {"prompt": self._process_data_for_trace(prompt)}
            processed_output = {"completion": self._process_data_for_trace(completion)}

            metadata = {
                "model": model,
                "tokens": tokens,
                "latency_ms": round(latency_ms, 2),
                **self._get_trace_metadata(),
            }
            trace.span(
                name="llm_call",
                input=processed_input,
                output=processed_output,
                metadata=metadata,
            )
            # 注意：不再每次 span 后调用 flush()
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
            query: 检索查询文本（隐私处理）
            results: 检索结果列表
            latency_ms: 检索延迟（毫秒）
        """
        if not self._client:
            return
        try:
            trace = self._client.trace(name=trace_name)
            # 隐私处理：query 经过 _process_data_for_trace
            processed_input = {"query": self._process_data_for_trace(query)}
            processed_output = {
                "result_count": len(results),
                "top_scores": [
                    r.get("score", 0) for r in results[:5]
                ],
            }
            metadata = {
                "latency_ms": round(latency_ms, 2),
                **self._get_trace_metadata(),
            }
            trace.span(
                name="rag_retrieval",
                input=processed_input,
                output=processed_output,
                metadata=metadata,
            )
            # 注意：不再每次 span 后调用 flush()
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
            step_data: 步骤数据（如输入/输出/决策，隐私处理）
        """
        if not self._client:
            return
        try:
            trace = self._client.trace(name=trace_name)
            # 隐私处理：step_data 中的 input/output 都经过 _process_data_for_trace
            processed_input = self._process_data_for_trace(step_data.get("input", {}))
            processed_output = self._process_data_for_trace(step_data.get("output", {}))

            metadata = {
                "agent_name": agent_name,
                "step_type": step_data.get("step_type", "unknown"),
                **self._get_trace_metadata(),
            }
            trace.span(
                name=f"agent_step:{agent_name}",
                input=processed_input,
                output=processed_output,
                metadata=metadata,
            )
            # 注意：不再每次 span 后调用 flush()
        except Exception as e:
            logger.debug(f"Langfuse trace_agent_step 异常（静默）: {e}")

    def flush(self) -> None:
        """幂等 flush：应用 shutdown 时调用，多次调用只实际 flush 一次"""
        if self._flushed or not self._client:
            return
        try:
            self._client.flush()
            self._flushed = True
            logger.debug("Langfuse tracer flushed")
        except Exception as e:
            logger.debug(f"Langfuse flush 异常（静默）: {e}")

    def close(self) -> None:
        """关闭 tracer：调用 flush 并清理引用"""
        self.flush()
        self._client = None
