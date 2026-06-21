import asyncio
import logging
import time
import traceback
from typing import List, Dict, Optional

import httpx
from openai import AsyncOpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── HTTP 客户端 ──
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(120.0),
    limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
)

client = AsyncOpenAI(
    api_key=settings.QWEN_API_KEY,
    base_url=settings.QWEN_API_BASE_URL,
    http_client=http_client,
)


# ── 全局并发控制 ──────────────────────────────────────────────────────────────
#
# 核心机制：asyncio.Semaphore
# - 所有 call_qwen_chat 调用必须先 acquire 信号量
# - 信号量值为 0 时，后续调用排队等待（asyncio 协程级别挂起，不阻塞线程）
# - 对调用方（Agent / RAG）完全透明，无需修改任何上层代码
#
# 示例（LLM_MAX_CONCURRENT=3）：
#   评估1-AgentA acquire ✓  ──► 执行中
#   评估1-AgentB acquire ✓  ──► 执行中
#   评估1-AgentC acquire ✓  ──► 执行中
#   评估2-AgentA acquire ✗  ──► 排队等待...
#   评估1-AgentA release    ──► 评估2-AgentA acquire ✓ 继续执行

_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    """获取全局信号量（懒初始化，确保在事件循环中创建）"""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.LLM_MAX_CONCURRENT)
        logger.info(
            f"LLM 并发信号量已初始化: max_concurrent={settings.LLM_MAX_CONCURRENT}"
        )
    return _semaphore


# ── 运行时监控指标 ──
class _LLMMetrics:
    """轻量级 LLM 调用统计（仅用于监控/日志，不参与业务逻辑）"""

    __slots__ = (
        "total_calls", "total_failures", "total_retries",
        "total_wait_time", "total_exec_time",
    )

    def __init__(self):
        self.total_calls: int = 0
        self.total_failures: int = 0
        self.total_retries: int = 0
        self.total_wait_time: float = 0.0   # 排队等待总耗时（秒）
        self.total_exec_time: float = 0.0    # API 执行总耗时（秒）

    def snapshot(self) -> dict:
        calls = max(self.total_calls, 1)
        return {
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_retries": self.total_retries,
            "avg_wait_ms": round(self.total_wait_time / calls * 1000, 1),
            "avg_exec_ms": round(self.total_exec_time / calls * 1000, 1),
            "active_concurrent": settings.LLM_MAX_CONCURRENT - _get_semaphore()._value,
        }


metrics = _LLMMetrics()


def get_llm_metrics() -> dict:
    """获取 LLM 调用统计快照（供 /health 或管理接口使用）"""
    return metrics.snapshot()


# ── 自定义异常 ──
class LLMConcurrencyTimeoutError(Exception):
    """等待 LLM 并发信号量超时"""
    pass


# ── 核心调用函数 ──────────────────────────────────────────────────────────────

async def call_qwen_chat(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> str:
    """调用阿里云百炼平台 Qwen 模型

    具备三层保护机制：
    1. **全局并发限流** — asyncio.Semaphore 控制同时在飞的 LLM 请求数
    2. **指数退避重试** — 对网络超时、429 限流、5xx 服务端错误自动重试
    3. **等待超时保护** — 排队超过 LLM_SEMAPHORE_TIMEOUT 秒则快速失败

    Args:
        messages: OpenAI 格式的消息列表
        model: 模型名称（默认读取 settings.QWEN_MODEL）
        temperature: 采样温度
        max_tokens: 最大输出 token 数

    Returns:
        模型生成的文本内容

    Raises:
        LLMConcurrencyTimeoutError: 排队等待信号量超时
        RateLimitError: API 限流（重试耗尽后）
        APITimeoutError: API 超时（重试耗尽后）
    """
    sem = _get_semaphore()

    # ── Step 1: 获取信号量（带超时保护）──
    wait_start = time.monotonic()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=settings.LLM_SEMAPHORE_TIMEOUT)
    except asyncio.TimeoutError:
        wait_elapsed = time.monotonic() - wait_start
        logger.error(
            f"LLM 并发排队超时: 等待 {wait_elapsed:.1f}s 后仍未获取信号量 "
            f"(active={settings.LLM_MAX_CONCURRENT - sem._value}/{settings.LLM_MAX_CONCURRENT})"
        )
        metrics.total_failures += 1
        raise LLMConcurrencyTimeoutError(
            f"LLM 服务繁忙，排队等待超过 {settings.LLM_SEMAPHORE_TIMEOUT}s"
        )

    wait_elapsed = time.monotonic() - wait_start
    metrics.total_wait_time += wait_elapsed
    metrics.total_calls += 1

    if wait_elapsed > 1.0:
        logger.info(
            f"LLM 排队等待 {wait_elapsed:.1f}s 后开始执行 "
            f"(active={settings.LLM_MAX_CONCURRENT - sem._value + 1}/{settings.LLM_MAX_CONCURRENT})"
        )

    # ── Step 2: 在信号量保护下执行 API 调用 ──
    try:
        result = await _execute_with_retry(messages, model, temperature, max_tokens)
        return result
    finally:
        sem.release()


async def _execute_with_retry(
    messages: List[Dict[str, str]],
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    """带指数退避重试的 LLM API 调用（内部函数）"""
    max_retries = 3
    retry_delay = 1.0

    for attempt in range(max_retries + 1):
        exec_start = time.monotonic()
        try:
            response = await client.chat.completions.create(
                model=model or settings.QWEN_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            exec_elapsed = time.monotonic() - exec_start
            metrics.total_exec_time += exec_elapsed
            return response.choices[0].message.content

        except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as e:
            exec_elapsed = time.monotonic() - exec_start
            metrics.total_exec_time += exec_elapsed

            status_code = getattr(e, 'status_code', None)
            is_retryable = (
                isinstance(e, (APITimeoutError, APIConnectionError, RateLimitError))
                or (isinstance(e, APIError) and status_code in (500, 502, 503, 504))
            )

            if is_retryable and attempt < max_retries:
                metrics.total_retries += 1
                # 对 429 RateLimitError 使用更长的退避时间
                if isinstance(e, RateLimitError):
                    retry_delay = max(retry_delay, 5.0)

                logger.warning(
                    f"LLM调用失败(尝试 {attempt + 1}/{max_retries + 1}): "
                    f"{type(e).__name__}: {str(e)[:120]}，"
                    f"等待 {retry_delay}s 后重试"
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                metrics.total_failures += 1
                logger.error(
                    f"LLM调用最终失败: {type(e).__name__}: {str(e)}\n"
                    f"{traceback.format_exc()}"
                )
                raise

        except Exception as e:
            exec_elapsed = time.monotonic() - exec_start
            metrics.total_exec_time += exec_elapsed
            metrics.total_failures += 1
            logger.error(f"LLM调用发生未知异常: {str(e)}\n{traceback.format_exc()}")
            raise
