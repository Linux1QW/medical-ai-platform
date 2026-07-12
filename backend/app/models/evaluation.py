from datetime import datetime

from sqlalchemy import Column, Integer, Float, String, Boolean, Text, JSON, DateTime, ForeignKey

from app.models.base import Base


class Evaluation(Base):
    """问诊评估报告 — 五维度评估"""

    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    consultation_id = Column(
        Integer, ForeignKey("consultations.id"), nullable=False, unique=True, index=True
    )

    # 维度1: 病史采集（问诊分析智能体）
    inquiry_score = Column(Float, default=0, comment="病史采集评分")
    inquiry_analysis = Column(Text, default="", comment="病史采集分析详情")

    # 维度2: 医学知识（医学知识核对智能体）
    knowledge_score = Column(Float, nullable=True, default=None, comment="医学知识评分")
    knowledge_analysis = Column(Text, default="", comment="知识核对详情")

    # 维度3: 沟通交流（人文关怀评估智能体）
    humanistic_score = Column(Float, default=0, comment="沟通交流评分")
    humanistic_analysis = Column(Text, default="", comment="沟通交流评估详情")

    # 维度4: 诊断结果（诊断评估智能体）
    diagnosis_score = Column(Float, default=0, comment="诊断结果评分")
    diagnosis_analysis = Column(Text, default="", comment="诊断结果评估详情")

    # 维度5: 治疗方案（治疗方案评估智能体）
    treatment_score = Column(Float, default=0, comment="治疗方案评分")
    treatment_analysis = Column(Text, default="", comment="治疗方案评估详情")

    # 综合评分
    total_score = Column(Float, nullable=True, default=None, comment="综合评分")
    overall_summary = Column(Text, default="", comment="综合评估摘要")

    # 建议指导
    improvement_suggestions = Column(Text, default="", comment="改进建议")

    # RAG 审计字段
    citation_data = Column(JSON, nullable=True, comment="引用数据")
    retrieval_status = Column(String(20), nullable=False, default='not_run', comment="检索状态")
    evidence_stance = Column(String(20), nullable=False, default='undetermined', comment="证据立场")
    human_review_needed = Column(Boolean, nullable=False, default=False, comment="是否需要人工复核")
    review_reason = Column(Text, nullable=True, comment="复核原因")
    rag_trace_data = Column(JSON, nullable=True, comment="RAG 追踪数据")
    evaluation_status = Column(String(20), nullable=False, default='completed', comment="评估状态")

    # LangGraph 审计字段
    run_id = Column(String(36), nullable=True, comment="关联的评估运行ID")
    safety_data = Column(JSON, nullable=True, comment="Safety检查结果")
    applicable_dimensions = Column(JSON, nullable=True, comment="适用维度列表")
    scoring_policy_version = Column(String(50), nullable=True, comment="评分策略版本")
    graph_version = Column(String(50), nullable=True, comment="编排图版本")
    review_completed_by = Column(String(50), nullable=True, comment="复核完成人ID")
    review_completed_at = Column(DateTime, nullable=True, comment="复核完成时间")

    created_at = Column(DateTime, default=datetime.utcnow)
