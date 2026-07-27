# -*- coding: utf-8 -*-
"""评估上下文预算 — 长对话早期摘要压缩与增量缓存（Harness）

评估链路的 conversation_text 原为全量拼接，长问诊会撑爆各 agent 的 LLM 上下文。
对话全文超过字符预算时，早期消息压缩为结构化摘要（复用虚拟患者侧的
_summarize_early_messages 四段摘要），仅保留近期消息全文。

摘要按早期消息内容哈希缓存到 Redis（db=2，复用 llm_cache 客户端）：
同一问诊重试/重评时早期段内容不变即可命中缓存，实现增量缓存——
每多一轮对话只需为新滑出窗口的早期段重新摘要一次。

全链路 best-effort：LLM 摘要失败或缓存不可用时降级返回全量文本，
不做盲目截断（评估准确性优先），不影响评估主流程。
"""

import hashlib
import logging

logger = logging.getLogger(__name__)

SUMMARY_CACHE_PREFIX = "eval_ctx_summary"
SUMMARY_CACHE_TTL = 604800  # 7 天，与 review_gate 快照一致


def _render_messages(messages) -> str:
    """按评估链路既有格式拼接消息（医生/患者 角色前缀）"""
    return "\n".join(
        f"{'医生' if m.role == 'doctor' else '患者'}: {m.content}" for m in messages
    )


def _summary_cache_key(messages) -> str:
    """缓存键：早期消息内容哈希（内容不变即命中，实现增量缓存）"""
    digest = hashlib.sha256(
        "\n".join(f"{m.role}:{m.content}" for m in messages).encode("utf-8")
    ).hexdigest()[:16]
    return f"{SUMMARY_CACHE_PREFIX}:{digest}"


async def _get_cached_summary(key: str) -> str | None:
    """读摘要缓存（best-effort，异常静默返回 None）"""
    try:
        from app.services.llm_cache import _get_redis

        r = await _get_redis()
        if r is None:
            return None
        return await r.get(key)
    except Exception as e:
        logger.debug(f"评估摘要缓存读取失败（忽略）: {e}")
        return None


async def _set_cached_summary(key: str, summary: str) -> None:
    """写摘要缓存（best-effort，异常静默）"""
    try:
        from app.services.llm_cache import _get_redis

        r = await _get_redis()
        if r is not None:
            await r.setex(key, SUMMARY_CACHE_TTL, summary)
    except Exception as e:
        logger.debug(f"评估摘要缓存写入失败（忽略）: {e}")


async def build_eval_conversation_text(messages, patient_profile: str = "") -> str:
    """构建评估用对话文本：短对话全量返回，长对话早期摘要 + 近期全文

    Args:
        messages: 按 sequence 排序的 ConsultationMessage 列表
        patient_profile: 患者基本情况（供摘要 LLM 参考，截断使用）

    Returns:
        对话文本；超预算时为「早期摘要 + 近期完整对话」合成文本。
    """
    from app.core.config import settings

    full_text = _render_messages(messages)
    if not settings.EVAL_CONTEXT_COMPRESS_ENABLED:
        return full_text

    threshold = settings.EVAL_CONTEXT_COMPRESS_THRESHOLD_CHARS
    keep = settings.EVAL_CONTEXT_RECENT_KEEP_MESSAGES
    if threshold <= 0 or keep <= 0:
        return full_text
    if len(full_text) <= threshold or len(messages) <= keep:
        return full_text

    early, recent = list(messages[:-keep]), list(messages[-keep:])

    cache_key = _summary_cache_key(early)
    summary = await _get_cached_summary(cache_key)
    if summary is None:
        # 复用虚拟患者侧滑动窗口摘要（四段结构化摘要，失败返回空串降级）
        from app.services.consultation_service import _summarize_early_messages

        summary = await _summarize_early_messages(early, patient_profile)
        if summary:
            await _set_cached_summary(cache_key, summary)

    if not summary:
        # 摘要失败：降级返回全量文本，不做盲目截断
        return full_text

    logger.info(
        f"评估上下文已压缩: 全文 {len(full_text)} 字符 → "
        f"早期 {len(early)} 条摘要 + 近期 {len(recent)} 条全文"
    )
    return (
        f"【早期对话摘要】（原 {len(early)} 条消息已压缩为要点）\n{summary}\n\n"
        f"【近期完整对话】\n{_render_messages(recent)}"
    )
