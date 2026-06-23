"""Safety Agent — 急危重风险门控检查

执行顺序：
1. 先运行确定性红旗规则
2. 命中硬性高风险规则时直接标记 high，LLM 不得降级
3. 未命中硬规则时，允许一次低温度 LLM 语义补充
4. LLM 失败时：若规则命中则沿用规则结果；否则返回 undetermined + immediate_review_required=True
"""

import json
import logging

from app.orchestration.state import SafetyResult
from app.services.qwen_client import call_qwen_chat

logger = logging.getLogger(__name__)

# ── 确定性红旗规则 ──────────────────────────────────────────────────────────

RED_FLAG_KEYWORDS = {
    "cardiac_arrest": ["心脏骤停", "心跳停止", "心搏骤停", "意识丧失", "无脉搏"],
    "severe_hemorrhage": ["大出血", "喷射性出血", "失血性休克", "大量呕血", "咯血"],
    "acute_stroke": ["突发偏瘫", "意识障碍", "言语不清", "口角歪斜", "突发剧烈头痛"],
    "anaphylaxis": ["过敏性休克", "喉头水肿", "呼吸困难伴皮疹", "血压骤降"],
    "acute_mi": ["持续性胸痛", "压榨性胸痛伴大汗", "放射至左臂", "濒死感"],
    "respiratory_failure": ["呼吸衰竭", "血氧饱和度低于85", "端坐呼吸", "三凹征"],
    "sepsis": ["感染性休克", "高热伴寒战", "多器官功能衰竭", "乳酸升高"],
    "acute_abdomen": ["板状腹", "反跳痛", "急性腹膜炎", "肠穿孔"],
}

# 高风险直接标记规则（命中即 high，不可降级）
HIGH_RISK_RULES = {
    "cardiac_arrest", "severe_hemorrhage", "anaphylaxis",
    "respiratory_failure", "sepsis",
}


async def run_safety_check(conversation_text: str) -> SafetyResult:
    """执行 Safety 检查

    Args:
        conversation_text: 问诊对话文本

    Returns:
        SafetyResult
    """
    # 1. 确定性红旗规则扫描
    matched = []
    for rule_id, keywords in RED_FLAG_KEYWORDS.items():
        for kw in keywords:
            if kw in conversation_text:
                matched.append(rule_id)
                break

    # 2. 命中硬性高风险规则 → 直接 high
    high_risk_hits = [m for m in matched if m in HIGH_RISK_RULES]
    if high_risk_hits:
        return SafetyResult(
            risk_level="high",
            matched_rules=matched,
            reasoning_summary=f"命中高风险红旗规则: {', '.join(high_risk_hits)}",
            immediate_review_required=True,
            degraded=False,
        )

    # 3. 有非高风险红旗命中 → medium，仍尝试 LLM 补充
    if matched:
        # 尝试 LLM 补充确认
        llm_result = await _llm_safety_check(conversation_text, matched)
        if llm_result is not None:
            return llm_result
        # LLM 失败但有规则命中 → 沿用规则结果
        return SafetyResult(
            risk_level="medium",
            matched_rules=matched,
            reasoning_summary=f"红旗规则命中: {', '.join(matched)}，LLM补充失败，降级为medium",
            immediate_review_required=False,
            degraded=True,
        )

    # 4. 无规则命中 → LLM 语义补充
    llm_result = await _llm_safety_check(conversation_text, [])
    if llm_result is not None:
        return llm_result

    # 5. LLM 失败 + 无规则命中 → undetermined, fail closed
    return SafetyResult(
        risk_level="undetermined",
        matched_rules=[],
        reasoning_summary="确定性规则未命中，LLM语义检查失败，安全关闭为undetermined",
        immediate_review_required=True,
        degraded=True,
    )


async def _llm_safety_check(
    conversation_text: str, matched_rules: list[str]
) -> SafetyResult | None:
    """LLM 语义补充检查

    Returns:
        SafetyResult 或 None（LLM 调用失败时）
    """
    prompt = f"""你是一个医疗安全评估系统。请分析以下医患对话，判断是否存在急危重风险。

已有红旗规则命中: {matched_rules if matched_rules else '无'}

对话内容:
{conversation_text[:3000]}

请分析是否存在以下情况：
1. 患者描述了需要紧急处理的危重症状
2. 医生遗漏了明显的危险信号
3. 存在可能的医疗安全隐患

请以JSON格式返回：
"risk_level": "low/medium/high", "reasoning": "简要分析", "matched_concerns": ["关注点1", ...]

注意：
- risk_level 只能是 low、medium、high 之一
- 如果对话内容正常、无明显风险，返回 low
- 如果有潜在风险但不确定，返回 medium
- 如果明确存在急危重情况，返回 high"""

    try:
        messages = [
            {"role": "system", "content": "你是医疗安全评估专家，负责识别急危重风险。"},
            {"role": "user", "content": prompt},
        ]
        response = await call_qwen_chat(
            messages=messages,
            temperature=0.1,
            max_tokens=500,
        )

        # 提取 JSON
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        data = json.loads(text)
        risk_level = data.get("risk_level", "low")
        if risk_level not in ("low", "medium", "high"):
            risk_level = "medium"

        # LLM 不得将已命中的硬性高风险降级
        if matched_rules and any(m in HIGH_RISK_RULES for m in matched_rules):
            risk_level = "high"

        concerns = data.get("matched_concerns", [])
        reasoning = data.get("reasoning", "")

        return SafetyResult(
            risk_level=risk_level,
            matched_rules=matched_rules + [f"llm:{c}" for c in concerns],
            reasoning_summary=reasoning,
            immediate_review_required=(risk_level == "high"),
            degraded=False,
        )

    except Exception as e:
        logger.warning(f"Safety LLM check failed: {e}")
        return None
