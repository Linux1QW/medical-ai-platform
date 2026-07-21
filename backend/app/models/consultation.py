from datetime import datetime

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, String, Text

from app.models.base import Base


class Consultation(Base):
    """问诊会话"""

    __tablename__ = "consultations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey("virtual_patients.id"), nullable=False)
    status = Column(
        Enum("in_progress", "completed", "evaluated", name="consultation_status"),
        default="in_progress",
    )
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    summary = Column(Text, default="")
    diagnosis = Column(Text, default="", comment="医生提交的诊断结果")
    treatment_plan = Column(Text, default="", comment="医生提交的治疗方案")
    consultation_type = Column(
        String(20), nullable=False, default="initial",
        comment="问诊类型: initial/follow_up/communication"
    )
    max_rounds = Column(Integer, default=20, comment="最大允许问诊轮次")
    created_at = Column(DateTime, default=datetime.utcnow)


class ConsultationMessage(Base):
    """问诊对话消息"""

    __tablename__ = "consultation_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    consultation_id = Column(Integer, ForeignKey("consultations.id"), nullable=False, index=True)
    role = Column(Enum("doctor", "patient", name="message_role"), nullable=False)
    content = Column(Text, nullable=False)
    sequence = Column(Integer, nullable=False, comment="消息序号")
    created_at = Column(DateTime, default=datetime.utcnow)
