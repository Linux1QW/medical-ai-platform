# -*- coding: utf-8 -*-
"""评估上下文预算测试 — 长对话早期摘要压缩与增量缓存"""

from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.services import context_budget
from app.services.context_budget import (
    _render_messages,
    _summary_cache_key,
    build_eval_conversation_text,
)


def _msg(role: str, content: str):
    return SimpleNamespace(role=role, content=content)


def _make_messages(n: int, content: str = "这是一条足够长的问诊消息内容用于撑大文本体积"):
    return [
        _msg("doctor" if i % 2 == 0 else "patient", f"{content}-{i}")
        for i in range(n)
    ]


class _FakeRedis:
    """内存版 Redis 桩（get/setex）"""

    def __init__(self, store: dict | None = None):
        self.store = store or {}
        self.setex_calls: list[tuple] = []

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.setex_calls.append((key, ttl, value))
        self.store[key] = value


@pytest.fixture
def compress_settings(monkeypatch):
    """启用压缩并调小阈值，便于用少量消息触发"""
    monkeypatch.setattr(settings, "EVAL_CONTEXT_COMPRESS_ENABLED", True)
    monkeypatch.setattr(settings, "EVAL_CONTEXT_COMPRESS_THRESHOLD_CHARS", 100)
    monkeypatch.setattr(settings, "EVAL_CONTEXT_RECENT_KEEP_MESSAGES", 2)


@pytest.fixture
def no_redis(monkeypatch):
    """缓存不可用（_get_redis 返回 None），走纯 LLM 摘要路径"""
    import app.services.llm_cache as llm_cache

    async def _none():
        return None

    monkeypatch.setattr(llm_cache, "_get_redis", _none)


@pytest.mark.asyncio
async def test_short_conversation_returned_verbatim(compress_settings):
    """未超阈值：原样全量拼接，不触发摘要"""
    messages = [_msg("doctor", "您好"), _msg("patient", "肚子疼")]

    text = await build_eval_conversation_text(messages)

    assert text == "医生: 您好\n患者: 肚子疼"


@pytest.mark.asyncio
async def test_disabled_returns_full_text(monkeypatch):
    """开关关闭：长对话也原样返回"""
    monkeypatch.setattr(settings, "EVAL_CONTEXT_COMPRESS_ENABLED", False)
    messages = _make_messages(30)

    text = await build_eval_conversation_text(messages)

    assert text == _render_messages(messages)


@pytest.mark.asyncio
async def test_long_conversation_compressed(compress_settings, no_redis, monkeypatch):
    """超阈值：早期消息摘要化 + 近期全文保留"""
    import app.services.consultation_service as cs

    async def fake_summarize(early, profile):
        return "【已披露症状】腹痛三天"

    monkeypatch.setattr(cs, "_summarize_early_messages", fake_summarize)
    messages = _make_messages(6)

    text = await build_eval_conversation_text(messages, patient_profile="腹痛")

    assert "【早期对话摘要】（原 4 条消息已压缩为要点）" in text
    assert "【已披露症状】腹痛三天" in text
    # 近期 2 条完整保留，早期消息原文不再出现
    assert _render_messages(messages[-2:]) in text
    assert messages[0].content not in text


@pytest.mark.asyncio
async def test_summary_failure_falls_back_to_full_text(compress_settings, no_redis, monkeypatch):
    """LLM 摘要失败（空串）：降级返回全量文本，不做盲目截断"""
    import app.services.consultation_service as cs

    async def fake_summarize(early, profile):
        return ""

    monkeypatch.setattr(cs, "_summarize_early_messages", fake_summarize)
    messages = _make_messages(6)

    text = await build_eval_conversation_text(messages)

    assert text == _render_messages(messages)


@pytest.mark.asyncio
async def test_cache_hit_skips_llm(compress_settings, monkeypatch):
    """缓存命中：直接复用摘要，不再调用 LLM（增量缓存）"""
    import app.services.consultation_service as cs
    import app.services.llm_cache as llm_cache

    messages = _make_messages(6)
    cache_key = _summary_cache_key(messages[:-2])
    fake_redis = _FakeRedis({cache_key: "【已披露症状】缓存摘要"})

    async def _redis():
        return fake_redis

    monkeypatch.setattr(llm_cache, "_get_redis", _redis)

    async def boom(early, profile):
        raise AssertionError("缓存命中时不应调用 LLM 摘要")

    monkeypatch.setattr(cs, "_summarize_early_messages", boom)

    text = await build_eval_conversation_text(messages)

    assert "缓存摘要" in text


@pytest.mark.asyncio
async def test_cache_miss_writes_summary(compress_settings, monkeypatch):
    """缓存未命中：摘要生成后写入缓存（setex 带 TTL）"""
    import app.services.consultation_service as cs
    import app.services.llm_cache as llm_cache

    fake_redis = _FakeRedis()

    async def _redis():
        return fake_redis

    monkeypatch.setattr(llm_cache, "_get_redis", _redis)

    async def fake_summarize(early, profile):
        return "【已披露症状】新生成摘要"

    monkeypatch.setattr(cs, "_summarize_early_messages", fake_summarize)
    messages = _make_messages(6)

    await build_eval_conversation_text(messages)

    assert len(fake_redis.setex_calls) == 1
    key, ttl, value = fake_redis.setex_calls[0]
    assert key == _summary_cache_key(messages[:-2])
    assert ttl == context_budget.SUMMARY_CACHE_TTL
    assert value == "【已披露症状】新生成摘要"
