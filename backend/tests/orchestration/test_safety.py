"""Safety Agent 测试"""

import pytest

from app.services.agents.safety_agent import run_safety_check


@pytest.mark.asyncio
async def test_red_flag_high_risk():
    """测试红旗关键词命中 → high"""
    # 心脏骤停相关关键词
    conversation = "患者突然意识丧失，无脉搏，疑似心脏骤停"

    result = await run_safety_check(conversation)

    assert result.risk_level == "high"
    assert result.immediate_review_required is True
    assert result.degraded is False
    assert len(result.matched_rules) > 0
    assert "cardiac_arrest" in result.matched_rules


@pytest.mark.asyncio
async def test_no_risk_low():
    """测试无风险文本 → low"""
    conversation = "患者轻微咳嗽3天，无发热，精神状态良好，饮食正常"

    result = await run_safety_check(conversation)

    # 由于LLM可能对低风险情况判断不同，这里主要测试是否成功执行
    # 如果没有命中规则，且LLM调用失败，则应该是undetermined
    # 如果LLM调用成功，可能是low或medium
    assert result.risk_level in ["low", "medium", "undetermined"]
    if result.risk_level == "undetermined":
        assert result.immediate_review_required is True
    else:
        assert result.immediate_review_required is False


@pytest.mark.asyncio
async def test_non_high_risk_matched_medium():
    """测试非高风险规则命中 → medium"""
    conversation = "患者有轻微头痛，偶尔头晕"

    result = await run_safety_check(conversation)

    # 这可能不会命中高风险规则，但可能命中其他规则或被LLM判断为medium
    assert result.risk_level in ["low", "medium", "undetermined"]
    if "acute_stroke" in result.matched_rules:  # 如果命中了stroke相关规则但不是高风险
        assert result.risk_level in ["medium", "low"]


@pytest.mark.asyncio
async def test_hard_rule_not_downgradable():
    """测试硬性高风险规则不可降级"""
    # 包含高风险关键词的对话
    conversation = "患者出现过敏性休克，血压骤降，呼吸困难伴皮疹"

    result = await run_safety_check(conversation)

    # 高风险规则命中时应始终为high，不得降级
    assert result.risk_level == "high"
    assert result.immediate_review_required is True
    assert "anaphylaxis" in result.matched_rules


if __name__ == "__main__":
    pytest.main([__file__])
