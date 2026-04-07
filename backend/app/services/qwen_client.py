import asyncio
import logging
import traceback
from typing import List, Dict

import httpx
from openai import AsyncOpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError

from app.core.config import settings

# 使用系统代理（让 httpx 自动读取环境变量中的代理配置）
http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

client = AsyncOpenAI(
    api_key=settings.QWEN_API_KEY,
    base_url=settings.QWEN_API_BASE_URL,
    http_client=http_client,
)


async def call_qwen_chat(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> str:
    """调用阿里云百炼平台 Qwen 模型，带指数退避重试机制"""
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model or settings.QWEN_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as e:
            # 判断是否为可重试错误：网络超时、连接错误、429频率限制、5xx服务端错误
            status_code = getattr(e, 'status_code', None)
            is_retryable = isinstance(e, (APITimeoutError, APIConnectionError, RateLimitError)) or \
                           (isinstance(e, APIError) and status_code in [500, 502, 503, 504])
            
            if is_retryable and attempt < max_retries:
                logging.warning(f"LLM调用失败(尝试 {attempt + 1}/{max_retries + 1}): {str(e)}，等待 {retry_delay}s 后重试")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                logging.error(f"LLM调用最终失败或不可重试: {str(e)}\n{traceback.format_exc()}")
                # 此处若有token计费逻辑，可在此处回滚
                raise e
        except Exception as e:
            logging.error(f"LLM调用发生未知异常: {str(e)}\n{traceback.format_exc()}")
            raise e
