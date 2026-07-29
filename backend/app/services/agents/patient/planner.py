# -*- coding: utf-8 -*-
"""问诊阶段规划 — 基于关键词规则的医生问题分类与阶段跟踪

阶段状态机只做记录与策略查询，不强制对话走向（约束患者行为而非医生行为）。
纯规则实现，零 LLM 成本、完全确定可测。
"""

STAGES = [
    "greeting", "chief_complaint", "hpi", "past_history",
    "personal_family_history", "physical_exam",
    "assessment_communication", "closing",
]

_STAGE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "greeting": ("请坐", "早上好", "下午好"),
    "chief_complaint": ("哪里不舒服", "哪儿不舒服", "哪不舒服", "怎么了", "什么问题", "为什么来", "看什么"),
    "hpi": ("多久", "多长时间", "什么时候开始", "加重", "缓解", "诱因", "什么样的疼",
            "怎么个疼", "一天几次", "什么性质", "什么部位", "伴随", "还有别的"),
    "past_history": ("以前", "既往", "得过", "手术", "过敏", "住过院", "老毛病", "病史",
                     "吃什么药", "用过什么药", "慢性病"),
    "personal_family_history": ("抽烟", "吸烟", "喝酒", "饮酒", "家里人", "家族", "父母",
                                "职业", "做什么工作", "结婚", "月经"),
    "physical_exam": ("量一下", "测一下", "体温", "血压", "心率", "听诊", "按一下",
                      "压痛", "查体", "张嘴", "看一下舌头"),
    "assessment_communication": ("诊断", "考虑是", "可能是", "建议你", "开点药", "做个检查",
                                 "化验", "拍个片", "初步判断"),
    "closing": ("再见", "注意休息", "按时吃药", "复诊", "有问题再来", "先这样"),
}


def classify_stage(doctor_message: str, current_stage: str) -> str:
    """按关键词命中数将医生消息归入问诊阶段；无命中保持当前阶段"""
    text = (doctor_message or "").strip()
    if not text:
        return current_stage
    best_stage, best_hits = current_stage, 0
    for stage in STAGES:
        hits = sum(1 for kw in _STAGE_KEYWORDS[stage] if kw in text)
        if hits > best_hits:
            best_stage, best_hits = stage, hits
    return best_stage
