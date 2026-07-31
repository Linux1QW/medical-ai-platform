# -*- coding: utf-8 -*-
"""患者模拟回复质量评审器（同模型 LLM-as-Judge，低温 rubric）。

对每轮患者回复按四维度 1-5 分制打分，与账本硬指标（披露率/矛盾数）双轨并列，
避免 self-judge 偏差被单独采信。任何异常/解析失败降级为 degraded（不阻断回放）。
"""
import logging
from typing import Optional

from pydantic import BaseModel

from app.services.qwen_client import call_qwen_chat
from app.utils.json_parser import extract_json_dict_from_text

logger = logging.getLogger(__name__)

DIMENSIONS = ["role_consistency", "medical_plausibility", "naturalness", "disclosure_timing"]

JUDGE_RUBRIC = """你是医疗问诊对话质量评审专家。请对「模拟患者」的单轮回复按以下四个维度各打 1-5 分（整数）：

1. role_consistency（角色一致性）：是否始终以患者身份作答，未泄露 AI 身份、未说出病历档案里没有的信息、未使用医生视角的专业措辞。5=完全一致，1=严重跳戏。
2. medical_plausibility（医学合理性）：症状描述、病史、用药等是否符合常识与该诊断的临床表现，无自相矛盾。5=完全合理，1=明显违背常识。
3. naturalness（口语自然度）：是否像真实患者的口语表达，而非机械罗列或书面语。5=非常自然，1=生硬机械。
4. disclosure_timing（披露时机）：信息披露节奏是否符合该人格设定（如配合型主动、对抗型谨慎、焦虑型啰嗦），既不一次倒完也不过度隐瞒。5=恰当，1=明显失当。

只输出如下 JSON，不要多余文字：
{"role_consistency": <1-5>, "medical_plausibility": <1-5>, "naturalness": <1-5>, "disclosure_timing": <1-5>, "reason": "<一句话中文理由>"}"""


class JudgeScore(BaseModel):
    """单轮回复的四维评分。degraded 为 True 时各维度为 None。"""

    role_consistency: Optional[int] = None
    medical_plausibility: Optional[int] = None
    naturalness: Optional[int] = None
    disclosure_timing: Optional[int] = None
    overall: Optional[float] = None
    reason: str = ""
    degraded: bool = False


def _clamp(value) -> Optional[int]:
    """把分数钳制到 [1, 5]；非数值返回 None。"""
    try:
        return max(1, min(5, int(round(float(value)))))
    except (TypeError, ValueError, OverflowError):
        return None


def build_judge_prompt(doctor: str, reply: str, patient_profile: dict, history=None) -> list[dict]:
    """构造评审 messages（system rubric + user 上下文）。"""
    profile_text = (
        f"人格类型：{patient_profile.get('personality', '未知')}；"
        f"预期诊断：{patient_profile.get('diagnosis', '未知')}；"
        f"主诉：{patient_profile.get('chief_complaint', '未知')}"
    )
    hist_lines = []
    for h in (history or [])[-6:]:
        role = h.get("role") if isinstance(h, dict) else getattr(h, "role", "")
        content = h.get("content") if isinstance(h, dict) else getattr(h, "content", "")
        hist_lines.append(f"{role}: {content}")
    hist_text = "\n".join(hist_lines) or "（无历史）"
    user = (
        f"【患者档案】{profile_text}\n"
        f"【近期对话】\n{hist_text}\n"
        f"【本轮医生提问】{doctor}\n"
        f"【本轮患者回复】{reply}\n\n"
        "请按 rubric 打分并只输出 JSON。"
    )
    return [
        {"role": "system", "content": JUDGE_RUBRIC},
        {"role": "user", "content": user},
    ]


def _parse_scores(data: dict) -> JudgeScore:
    """解析四维分：钳制到 [1,5]，缺任一维度则降级但保留已解析维度。"""
    parsed = {dim: _clamp(data.get(dim)) for dim in DIMENSIONS}
    reason = str(data.get("reason", "") or "")
    present = [v for v in parsed.values() if v is not None]
    degraded = len(present) < len(DIMENSIONS)
    overall = round(sum(present) / len(present), 3) if (present and not degraded) else None
    return JudgeScore(overall=overall, reason=reason, degraded=degraded, **parsed)


async def judge_turn(doctor: str, reply: str, patient_profile: dict, history=None) -> JudgeScore:
    """对单轮患者回复打分。任何异常降级为 JudgeScore(degraded=True)。"""
    try:
        messages = build_judge_prompt(doctor, reply, patient_profile, history)
        raw = await call_qwen_chat(messages, temperature=0.0, max_tokens=300)
        data = extract_json_dict_from_text(raw)
        return _parse_scores(data)
    except Exception:
        logger.warning("Judge 评分降级（解析/调用失败）", exc_info=True)
        return JudgeScore(degraded=True)
