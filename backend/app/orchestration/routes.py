"""场景分类与动态路由"""

from app.orchestration.state import (
    EvaluationContext, SubmissionFlags, RoutePlan, EvaluationPlan, PlanStep
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


# ── 评估计划构建（Plan-Execute 模式） ────────────────────────────────────────

# Agent 描述矩阵：用于生成有意义的步骤描述
_AGENT_DESCRIPTIONS = {
    "inquiry": "病史采集与问诊技巧评估",
    "diagnosis": "诊断结果准确性评估",
    "treatment": "治疗方案合理性评估",
    "knowledge": "医学知识一致性核对",
    "humanistic": "人文关怀与沟通能力评估",
}


def build_evaluation_plan(
    consultation_type: str,
    flags: SubmissionFlags,
) -> EvaluationPlan:
    """根据场景和提交状态构建完整评估计划

    每个 agent 被映射为一个 PlanStep，包含 step_id、依赖关系和描述信息。
    此计划由 plan_evaluation 节点调用，替代旧版 build_route_plan。
    """
    if consultation_type not in _ROUTE_MATRIX:
        consultation_type = "initial"

    matrix = _ROUTE_MATRIX[consultation_type]
    steps: list[PlanStep] = []
    skipped: list[str] = []
    skip_reasons: dict[str, str] = {}

    # 1. 必填 agent → 无依赖的 PlanStep
    for agent_name in matrix["required"]:
        steps.append(PlanStep(
            step_id=f"step_{agent_name}",
            step_type="agent_evaluation",
            agent_name=agent_name,
            description=_AGENT_DESCRIPTIONS.get(agent_name, f"{agent_name}评估"),
            depends_on=[],  # 必填步骤无前置依赖
        ))

    # 2. 条件 agent → 根据 submission_flags 决定执行或跳过
    for agent_name in matrix["conditional"]:
        if agent_name == "diagnosis" and not flags.has_diagnosis:
            skipped.append(agent_name)
            skip_reasons[agent_name] = "未提交诊断结果"
        elif agent_name == "treatment" and not flags.has_treatment:
            skipped.append(agent_name)
            skip_reasons[agent_name] = "未提交治疗方案"
        else:
            # 条件步骤依赖必填步骤完成
            required_ids = [f"step_{a}" for a in matrix["required"]]
            steps.append(PlanStep(
                step_id=f"step_{agent_name}",
                step_type="agent_evaluation",
                agent_name=agent_name,
                description=_AGENT_DESCRIPTIONS.get(agent_name, f"{agent_name}评估"),
                depends_on=required_ids,
            ))

    return EvaluationPlan(
        consultation_type=consultation_type,
        steps=steps,
        skipped_agents=skipped,
        skip_reasons=skip_reasons,
        plan_version="v1",
    )
