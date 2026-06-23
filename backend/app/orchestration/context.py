"""评估上下文构建工具"""

from app.orchestration.state import EvaluationContext, SubmissionFlags


def build_context(
    conversation_text: str,
    patient_age: int | None = None,
    patient_gender: str | None = None,
    chief_complaint: str = "",
    medical_history: str = "",
    symptoms: list[str] | None = None,
    doctor_diagnosis: str | None = None,
    treatment_plan: str | None = None,
) -> EvaluationContext:
    """构建去标识化的评估上下文"""
    return EvaluationContext(
        conversation_text=conversation_text,
        patient_age=patient_age,
        patient_gender=patient_gender,
        chief_complaint=chief_complaint,
        medical_history=medical_history,
        symptoms=symptoms or [],
        doctor_diagnosis=doctor_diagnosis,
        treatment_plan=treatment_plan,
    )


def build_submission_flags(
    doctor_diagnosis: str | None,
    treatment_plan: str | None,
) -> SubmissionFlags:
    """基于诊断和治疗方案判断提交状态"""
    has_diagnosis = bool(
        doctor_diagnosis and doctor_diagnosis.strip()
        and not doctor_diagnosis.startswith("（")
    )
    has_treatment = bool(
        treatment_plan and treatment_plan.strip()
        and not treatment_plan.startswith("（")
    )
    return SubmissionFlags(has_diagnosis=has_diagnosis, has_treatment=has_treatment)
