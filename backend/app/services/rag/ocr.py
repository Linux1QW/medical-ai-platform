# -*- coding: utf-8 -*-
"""Qwen-VL 云端 OCR — 扫描版/图片型 PDF 页的文字识别兜底

复用 DashScope 的 OpenAI 兼容端点多模态能力（qwen-vl-ocr），无需额外 pip 依赖。
设计要点：
- 仅在 ENABLE_OCR 开启且页面文本过少时被调用（触发判断在抽取阶段完成）
- 未配置 QWEN_API_KEY 或调用失败时优雅降级（返回空串），绝不阻断索引构建
- 通过信号量限制并发，避免触发 API 限流
"""

import asyncio
import base64
import logging
from typing import List, Optional

import httpx
from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# OCR 提示词：要求纯文本输出，表格用 Markdown 还原，避免模型添加解释性文字
_OCR_PROMPT = (
    "请识别这张图片中的所有文字，按自然阅读顺序输出为纯文本；"
    "若包含表格，请用 Markdown 表格还原其行列结构。只输出识别到的内容，不要添加任何解释或评论。"
)

# 懒加载客户端与信号量（避免模块导入期产生副作用 / 绑定错误的事件循环）
_ocr_client: Optional[AsyncOpenAI] = None
_ocr_semaphore: Optional[asyncio.Semaphore] = None


def _get_client() -> Optional[AsyncOpenAI]:
    """获取 OCR 客户端；未配置 API Key 时返回 None（触发降级）"""
    global _ocr_client
    if not settings.QWEN_API_KEY:
        return None
    if _ocr_client is None:
        _ocr_client = AsyncOpenAI(
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_API_BASE_URL,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(settings.OCR_TIMEOUT_SECONDS)
            ),
        )
    return _ocr_client


def _get_semaphore() -> asyncio.Semaphore:
    """获取并发信号量（懒加载，绑定到首次调用时的事件循环）"""
    global _ocr_semaphore
    if _ocr_semaphore is None:
        _ocr_semaphore = asyncio.Semaphore(max(1, settings.OCR_MAX_CONCURRENCY))
    return _ocr_semaphore


async def ocr_image_bytes(png_bytes: bytes) -> str:
    """对单张 PNG 图片执行 OCR，返回识别文本。

    失败、未启用或未配置 API Key 时返回空串（调用方据此保留原文本）。
    """
    if not png_bytes:
        return ""

    client = _get_client()
    if client is None:
        logger.debug("OCR 跳过：未配置 QWEN_API_KEY")
        return ""

    b64 = base64.b64encode(png_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"
    max_retries = 2

    async with _get_semaphore():
        for attempt in range(max_retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=settings.OCR_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": _OCR_PROMPT},
                        ],
                    }],
                    temperature=0.0,
                )
                content = resp.choices[0].message.content or ""
                return content.strip()
            except Exception as e:
                if attempt < max_retries:
                    backoff = 2 ** (attempt + 1)  # 2s, 4s
                    logger.warning(
                        f"OCR 调用失败，第 {attempt + 1} 次重试，等待 {backoff}s: {e}"
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"OCR 调用失败，已达最大重试次数，降级跳过: {e}")
                    return ""
    return ""


async def apply_ocr_to_pages(pages: List[dict]) -> None:
    """对携带 `_ocr_image` 标记的低文本页并发执行 OCR，就地回填 text。

    - 仅处理 extract_text_from_pdf 标记的低文本页；无标记页直接跳过（零开销）
    - 仅当 OCR 文本更长时才覆盖，避免识别噪声破坏已有的有效文本层
    - 处理完毕移除 `_ocr_image` 字节，及时释放内存
    - 全流程降级安全：单页失败不影响其它页

    Args:
        pages: extract_text_from_pdf 返回的页面列表（就地修改）
    """
    targets = [p for p in pages if p.get("_ocr_image")]
    if not targets:
        return

    logger.info(f"OCR 兜底：{len(targets)} 个低文本页待识别")

    async def _run(page: dict) -> None:
        img = page.pop("_ocr_image", None)  # 取出并移除，释放内存
        if not img:
            return
        ocr_text = await ocr_image_bytes(img)
        if ocr_text and len(ocr_text) > len(page.get("text", "")):
            page["text"] = ocr_text

    await asyncio.gather(*[_run(p) for p in targets])

    filled = sum(1 for p in targets if p.get("text", "").strip())
    logger.info(f"OCR 兜底完成：{filled}/{len(targets)} 页成功回填文本")
