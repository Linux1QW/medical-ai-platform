# -*- coding: utf-8 -*-
"""LLM-as-Judge judge_turn 解析/降级/钳制/overall 计算测试"""
import pytest

from evaluation import patient_judge
from evaluation.patient_judge import JudgeScore, judge_turn

_PROFILE = {"personality": "配合型", "diagnosis": "慢性胃炎", "chief_complaint": "腹痛"}


def _patch_llm(monkeypatch, return_text):
    async def fake_call(messages, model=None, temperature=0.7, max_tokens=2000):
        return return_text
    monkeypatch.setattr(patient_judge, "call_qwen_chat", fake_call)


@pytest.mark.asyncio
async def test_judge_turn_parses_four_dims(monkeypatch):
    _patch_llm(monkeypatch, (
        '{"role_consistency": 4, "medical_plausibility": 4, '
        '"naturalness": 5, "disclosure_timing": 3, "reason": "回答自然"}'
    ))
    score = await judge_turn("哪里不舒服？", "肚子疼三天了", _PROFILE, history=[])
    assert isinstance(score, JudgeScore)
    assert score.degraded is False
    assert score.role_consistency == 4
    assert score.disclosure_timing == 3
    assert score.overall == pytest.approx(4.0)  # (4+4+5+3)/4
    assert score.reason == "回答自然"


@pytest.mark.asyncio
async def test_judge_turn_degrades_on_parse_failure(monkeypatch):
    _patch_llm(monkeypatch, "抱歉我无法评分，这不是 JSON")
    score = await judge_turn("哪里不舒服？", "肚子疼", _PROFILE, history=[])
    assert score.degraded is True
    assert score.overall is None
    assert score.role_consistency is None


@pytest.mark.asyncio
async def test_judge_turn_clamps_out_of_range(monkeypatch):
    _patch_llm(monkeypatch, (
        '{"role_consistency": 9, "medical_plausibility": 0, '
        '"naturalness": 3, "disclosure_timing": 4, "reason": "越界"}'
    ))
    score = await judge_turn("问", "答", _PROFILE, history=[])
    assert score.degraded is False
    assert score.role_consistency == 5  # 9 钳制到 5
    assert score.medical_plausibility == 1  # 0 钳制到 1
    assert score.overall == pytest.approx((5 + 1 + 3 + 4) / 4)


@pytest.mark.asyncio
async def test_judge_turn_degrades_on_llm_exception(monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("LLM 挂了")
    monkeypatch.setattr(patient_judge, "call_qwen_chat", boom)
    score = await judge_turn("问", "答", _PROFILE, history=[])
    assert score.degraded is True


@pytest.mark.asyncio
async def test_judge_turn_partial_dims_degrade(monkeypatch):
    # 缺 disclosure_timing 维度 -> 降级但保留已解析维度
    _patch_llm(monkeypatch, (
        '{"role_consistency": 4, "medical_plausibility": 4, "naturalness": 5, "reason": "缺一维"}'
    ))
    score = await judge_turn("问", "答", _PROFILE, history=[])
    assert score.degraded is True
    assert score.role_consistency == 4
    assert score.disclosure_timing is None
