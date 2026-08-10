"""72-case coach benchmark dataset.

Each case contains a visible_context, dialogue_prefix, expected_intent,
expected_stage, forbidden_hidden_facts, and target_slots.

Cases span 8 specialties × 9 interview intents = 72 cases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CoachCase:
    """A single benchmark case for the coach evaluation."""

    case_id: str
    specialty: str
    difficulty: str
    personality: str
    visible_context: dict[str, Any]
    dialogue_prefix: list[dict[str, str]]
    expected_intent: str
    expected_stage: str
    forbidden_hidden_facts: list[str] = field(default_factory=list)
    target_slots: list[str] = field(default_factory=list)


# ── Specialty definitions ────────────────────────────────────────────

SPECIALTIES = [
    "心血管内科",
    "呼吸与危重症医学科",
    "消化内科",
    "内分泌科",
    "神经内科",
    "肾内科",
    "风湿免疫科",
    "急诊医学",
]

DIFFICULTIES = ["easy", "medium", "hard"]

PERSONALITIES = ["配合型", "焦虑型", "回避型", "老年迟钝型"]

# ── Intent → stage mapping for expected values ───────────────────────

INTENT_STAGE_MAP: dict[str, str] = {
    "rapport": "rapport",
    "chief_complaint": "chief_complaint",
    "hpi_onset": "history_present_illness",
    "hpi_progression": "history_present_illness",
    "hpi_severity": "history_present_illness",
    "hpi_timing": "history_present_illness",
    "hpi_associated": "history_present_illness",
    "past_history": "past_medical_history",
    "medication": "medication_allergy",
    "allergy": "medication_allergy",
    "family_social": "family_social_history",
    "examination": "physical_examination",
    "assessment_communication": "assessment_communication",
    "treatment_communication": "assessment_communication",
    "closing": "closing",
    "off_topic": "off_topic",
    "unsafe": "off_topic",
}

# ── Dialogue templates per intent ────────────────────────────────────

_DIALOGUE_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "rapport": [
        {"role": "patient", "content": "医生你好。"},
        {"role": "doctor", "content": "您好，请坐。我是今天的接诊医生，请问您怎么称呼？"},
    ],
    "chief_complaint": [
        {"role": "patient", "content": "医生你好。"},
        {"role": "doctor", "content": "您好，请坐。请问您今天哪里不舒服？"},
    ],
    "hpi_onset": [
        {"role": "patient", "content": "我胸口疼了三天了。"},
        {"role": "doctor", "content": "好的，请问这个疼痛是什么时候开始的？具体哪个时间点？"},
    ],
    "hpi_progression": [
        {"role": "patient", "content": "我胸口疼了三天了，一开始只是闷闷的。"},
        {"role": "doctor", "content": "那这几天疼痛有没有加重或者减轻的变化？"},
    ],
    "hpi_severity": [
        {"role": "patient", "content": "我头痛得厉害。"},
        {"role": "doctor", "content": "疼痛程度如果0到10分，您觉得大概几分？严重到什么程度？"},
    ],
    "hpi_timing": [
        {"role": "patient", "content": "我肚子疼有一周了。"},
        {"role": "doctor", "content": "是一直持续还是间断发作？每次大概多长时间？"},
    ],
    "hpi_associated": [
        {"role": "patient", "content": "我发烧咳嗽快一周了。"},
        {"role": "doctor", "content": "除了发烧咳嗽，还有没有其他伴随症状？比如乏力、肌肉酸痛？"},
    ],
    "past_history": [
        {"role": "patient", "content": "我最近血压偏高。"},
        {"role": "doctor", "content": "您以前得过什么病吗？有没有住院或手术的历史？"},
    ],
    "medication": [
        {"role": "patient", "content": "我血糖控制不好。"},
        {"role": "doctor", "content": "您目前在吃什么药？用药情况怎么样？"},
    ],
    "allergy": [
        {"role": "patient", "content": "我皮肤起了疹子。"},
        {"role": "doctor", "content": "您有没有药物过敏史？对什么药物过敏？"},
    ],
    "family_social": [
        {"role": "patient", "content": "我最近总觉得疲劳。"},
        {"role": "doctor", "content": "您家人有没有类似的情况？有没有遗传病史？吸烟喝酒吗？"},
    ],
    "examination": [
        {"role": "patient", "content": "我头晕好几天了。"},
        {"role": "doctor", "content": "我先给您量一下血压，做个基本检查。之前化验结果有吗？"},
    ],
    "assessment_communication": [
        {"role": "patient", "content": "医生，我的检查结果出来了吗？"},
        {"role": "doctor", "content": "结果出来了，我来跟您解释一下病情和检查发现。"},
    ],
    "treatment_communication": [
        {"role": "patient", "content": "医生，我这个病要怎么治？"},
        {"role": "doctor", "content": "根据您的情况，我们来讨论一下治疗方案。"},
    ],
    "closing": [
        {"role": "patient", "content": "好的医生，我了解了。"},
        {"role": "doctor", "content": "那您还有其他问题吗？没有的话我们今天就到这里。"},
    ],
}

# ── Hidden facts per specialty (things the doctor must NOT see) ──────

_HIDDEN_FACTS: dict[str, list[str]] = {
    "心血管内科": [
        "expected_diagnosis:急性心肌梗死",
        "gold_standard_treatment:立即PCI",
        "hidden_risk:猝死风险极高",
    ],
    "呼吸与危重症医学科": [
        "expected_diagnosis:肺栓塞",
        "gold_standard_treatment:溶栓治疗",
        "hidden_risk:呼吸衰竭",
    ],
    "消化内科": [
        "expected_diagnosis:消化道出血",
        "gold_standard_treatment:内镜止血",
        "hidden_risk:失血性休克",
    ],
    "内分泌科": [
        "expected_diagnosis:糖尿病酮症酸中毒",
        "gold_standard_treatment:胰岛素泵",
        "hidden_risk:昏迷风险",
    ],
    "神经内科": [
        "expected_diagnosis:脑卒中",
        "gold_standard_treatment:rt-PA溶栓",
        "hidden_risk:脑疝形成",
    ],
    "肾内科": [
        "expected_diagnosis:急性肾衰竭",
        "gold_standard_treatment:紧急透析",
        "hidden_risk:高钾血症致死",
    ],
    "风湿免疫科": [
        "expected_diagnosis:系统性红斑狼疮",
        "gold_standard_treatment:激素冲击",
        "hidden_risk:多脏器衰竭",
    ],
    "急诊医学": [
        "expected_diagnosis:脓毒症休克",
        "gold_standard_treatment:液体复苏+血管活性药",
        "hidden_risk:多器官功能衰竭",
    ],
}

# ── Target slots per intent ──────────────────────────────────────────

_TARGET_SLOTS: dict[str, list[str]] = {
    "rapport": ["建立信任", "自我介绍"],
    "chief_complaint": ["主诉采集", "症状定位"],
    "hpi_onset": ["发病时间", "诱因"],
    "hpi_progression": ["变化趋势", "加重因素"],
    "hpi_severity": ["严重程度评分", "功能影响"],
    "hpi_timing": ["持续性/间断性", "发作频率"],
    "hpi_associated": ["伴随症状", "鉴别症状"],
    "past_history": ["既往疾病", "手术史"],
    "medication": ["当前用药", "用药依从性"],
    "allergy": ["药物过敏", "过敏表现"],
    "family_social": ["家族史", "生活习惯"],
    "examination": ["体格检查", "辅助检查"],
    "assessment_communication": ["结果解释", "病情沟通"],
    "treatment_communication": ["方案说明", "风险告知"],
    "closing": ["总结回顾", "随访安排"],
}

# ── Chief complaints per specialty ───────────────────────────────────

_CHIEF_COMPLAINTS: dict[str, str] = {
    "心血管内科": "胸闷胸痛3天",
    "呼吸与危重症医学科": "咳嗽气促1周",
    "消化内科": "腹痛伴黑便5天",
    "内分泌科": "多饮多尿消瘦2月",
    "神经内科": "头痛头晕伴肢体麻木1周",
    "肾内科": "浮肿尿少2周",
    "风湿免疫科": "关节痛伴皮疹3月",
    "急诊医学": "高热寒战1天",
}


def _build_case(
    specialty: str,
    intent: str,
    case_index: int,
) -> CoachCase:
    """Build a single benchmark case."""
    difficulty = DIFFICULTIES[case_index % len(DIFFICULTIES)]
    personality = PERSONALITIES[case_index % len(PERSONALITIES)]
    stage = INTENT_STAGE_MAP.get(intent, "off_topic")
    dialogue = _DIALOGUE_TEMPLATES.get(intent, _DIALOGUE_TEMPLATES["rapport"])
    hidden = _HIDDEN_FACTS.get(specialty, [])
    slots = _TARGET_SLOTS.get(intent, [])
    cc = _CHIEF_COMPLAINTS.get(specialty, "问诊")

    return CoachCase(
        case_id=f"C{case_index + 1:03d}",
        specialty=specialty,
        difficulty=difficulty,
        personality=personality,
        visible_context={
            "patient_age": 30 + (case_index * 7) % 50,
            "patient_gender": "男" if case_index % 2 == 0 else "女",
            "chief_complaint": cc,
        },
        dialogue_prefix=list(dialogue),  # copy
        expected_intent=intent,
        expected_stage=stage,
        forbidden_hidden_facts=list(hidden),
        target_slots=list(slots),
    )


# ── All 72 intents to cover (8 specialties × 9 intents each) ────────
# We pick 9 representative intents per specialty to reach 72 cases.

_COVERED_INTENTS: list[str] = [
    "rapport",
    "chief_complaint",
    "hpi_onset",
    "hpi_progression",
    "hpi_severity",
    "hpi_timing",
    "hpi_associated",
    "past_history",
    "medication",
]


def _generate_all_cases() -> list[CoachCase]:
    """Generate 72 benchmark cases: 8 specialties × 9 intents."""
    cases: list[CoachCase] = []
    idx = 0
    for specialty in SPECIALTIES:
        for intent in _COVERED_INTENTS:
            cases.append(_build_case(specialty, intent, idx))
            idx += 1
    return cases


_ALL_CASES: list[CoachCase] | None = None


def load_cases() -> list[CoachCase]:
    """Load all 72 benchmark cases."""
    global _ALL_CASES
    if _ALL_CASES is None:
        _ALL_CASES = _generate_all_cases()
    return list(_ALL_CASES)


def validate_dataset(cases: list[CoachCase]) -> list[str]:
    """Validate dataset integrity. Returns list of error messages."""
    errors: list[str] = []
    if len(cases) != 72:
        errors.append(f"Expected 72 cases, got {len(cases)}")

    seen_ids: set[str] = set()
    for case in cases:
        if case.case_id in seen_ids:
            errors.append(f"Duplicate case_id: {case.case_id}")
        seen_ids.add(case.case_id)

        if not case.dialogue_prefix:
            errors.append(f"{case.case_id}: empty dialogue_prefix")

        if case.expected_intent not in INTENT_STAGE_MAP:
            errors.append(f"{case.case_id}: unknown intent '{case.expected_intent}'")

        for msg in case.dialogue_prefix:
            if "role" not in msg or "content" not in msg:
                errors.append(f"{case.case_id}: malformed message in dialogue_prefix")

    return errors
