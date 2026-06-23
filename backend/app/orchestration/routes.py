"""场景分类与动态路由"""

from app.orchestration.state import (
    EvaluationContext, SubmissionFlags, RoutePlan
)
from app.models.consultation import Consultation


# ── 路由矩阵 ──────────────────────────────────────────────────────────────

_ROUTE_MATRIX = {
    "initial": {
        "required": ["inquiry", "humanistic"],
        "conditional": ["diagnosis", "treatment", "knowledge"],
    },
    "follow_up": {
        "required": ["inquiry", "humanistic"],
        "conditional": ["diagnosis", "treatment", "knowledge"],
    },
    "communication": {
        "required": ["inquiry", "humanistic"],
        "conditional": [],
    },
    "emergency": {
        "required": ["inquiry", "humanistic"],
        "conditional": ["diagnosis", "treatment", "knowledge"],
    },
}


def build_submission_flags(consultation: Consultation) -> SubmissionFlags:
    """从 Consultation 模型构建提交标志

    判断规则：字段非空且去除空白后不为空字符串即为已提交。
    不使用占位字符串判断。
    """
    has_diagnosis = bool(
        consultation.diagnosis and consultation.diagnosis.strip()
    )
    has_treatment = bool(
        consultation.treatment_plan and consultation.treatment_plan.strip()
    )
    return SubmissionFlags(
        has_diagnosis=has_diagnosis,
        has_treatment=has_treatment,
    )


def build_route_plan(
    consultation_type: str,
    flags: SubmissionFlags,
) -> RoutePlan:
    """根据场景和提交状态构建路由计划"""
    if consultation_type not in _ROUTE_MATRIX:
        consultation_type = "initial"

    matrix = _ROUTE_MATRIX[consultation_type]
    selected = list(matrix["required"])
    skipped = []
    skip_reasons = {}

    for agent_name in matrix["conditional"]:
        if agent_name == "diagnosis" and not flags.has_diagnosis:
            skipped.append(agent_name)
            skip_reasons[agent_name] = "未提交诊断结果"
        elif agent_name == "treatment" and not flags.has_treatment:
            skipped.append(agent_name)
            skip_reasons[agent_name] = "未提交治疗方案"
        else:
            selected.append(agent_name)

    return RoutePlan(
        consultation_type=consultation_type,
        selected_agents=selected,
        skipped_agents=skipped,
        skip_reasons=skip_reasons,
    )


def get_consultation_type(consultation: Consultation) -> str:
    """从 Consultation 获取问诊类型，兼容旧数据"""
    ct = getattr(consultation, "consultation_type", None)
    if ct and ct.strip():
        return ct
    return "initial"
