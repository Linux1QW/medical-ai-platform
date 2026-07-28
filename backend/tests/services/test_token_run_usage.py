# -*- coding: utf-8 -*-
"""run 级 Token 成本归因测试 — contextvars 透传 + Redis hash 记账"""

import pytest

from app.core.run_context import current_agent_name, current_run_id
from app.services import token_tracker as tracker_module
from app.services.token_tracker import token_tracker


class _FakeRedis:
    """内存字典模拟 Redis hash（hincrby/expire/hgetall）"""

    def __init__(self):
        self.store: dict[str, dict[str, int]] = {}

    async def hincrby(self, key, field, amount):
        bucket = self.store.setdefault(key, {})
        bucket[field] = bucket.get(field, 0) + amount
        return bucket[field]

    async def expire(self, key, seconds):
        return True

    async def hgetall(self, key):
        return dict(self.store.get(key, {}))


@pytest.fixture
def fake_redis(monkeypatch):
    redis = _FakeRedis()

    async def _get(*args, **kwargs):
        return redis

    monkeypatch.setattr(tracker_module, "_get_redis", _get)
    return redis


@pytest.fixture
def run_ctx():
    """设置 run/agent 上下文，测试结束后还原"""
    run_token = current_run_id.set("run-1")
    agent_token = current_agent_name.set("knowledge")
    yield
    current_run_id.reset(run_token)
    current_agent_name.reset(agent_token)


@pytest.mark.asyncio
async def test_record_usage_attributes_to_run_and_agent(fake_redis, run_ctx):
    await token_tracker.record_usage("qwen-max", prompt_tokens=100, completion_tokens=50)

    run_bucket = fake_redis.store["token_usage:run:run-1"]
    assert run_bucket["prompt_tokens"] == 100
    assert run_bucket["completion_tokens"] == 50
    assert run_bucket["total_tokens"] == 150
    assert run_bucket["agent:knowledge:total_tokens"] == 150


@pytest.mark.asyncio
async def test_record_usage_accumulates_across_calls(fake_redis, run_ctx):
    await token_tracker.record_usage("qwen-max", 10, 5)
    await token_tracker.record_usage("qwen-max", 20, 10)

    run_bucket = fake_redis.store["token_usage:run:run-1"]
    assert run_bucket["total_tokens"] == 45


@pytest.mark.asyncio
async def test_record_usage_without_run_context_skips_run_key(fake_redis):
    """非评估链路（run_id 为 None）不产生 run 键，日/模型统计不受影响"""
    assert current_run_id.get() is None

    await token_tracker.record_usage("qwen-max", 10, 5)

    assert not any(k.startswith("token_usage:run:") for k in fake_redis.store)
    daily_keys = [k for k in fake_redis.store if k.startswith("token_usage:daily:")]
    assert len(daily_keys) == 1


@pytest.mark.asyncio
async def test_get_run_usage_parses_by_agent(fake_redis):
    fake_redis.store["token_usage:run:run-9"] = {
        "prompt_tokens": 300,
        "completion_tokens": 100,
        "total_tokens": 400,
        "agent:knowledge:total_tokens": 250,
        "agent:inquiry:total_tokens": 150,
    }

    usage = await token_tracker.get_run_usage("run-9")

    assert usage["prompt_tokens"] == 300
    assert usage["completion_tokens"] == 100
    assert usage["total_tokens"] == 400
    assert usage["by_agent"] == {"knowledge": 250, "inquiry": 150}
    assert usage["estimated_cost"] >= 0


@pytest.mark.asyncio
async def test_get_run_usage_returns_zero_when_missing(fake_redis):
    usage = await token_tracker.get_run_usage("no-such-run")

    assert usage["total_tokens"] == 0
    assert usage["by_agent"] == {}
    assert usage["estimated_cost"] == 0.0
