from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Consultation(Base):
    """问诊会话"""

    __tablename__ = "consultations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doctor_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    patient_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("virtual_patients.id"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        Enum("in_progress", "completed", "evaluated", name="consultation_status"),
        default="in_progress",
        nullable=True,
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=True)
    diagnosis: Mapped[str] = mapped_column(
        Text, default="", nullable=True, comment="医生提交的诊断结果"
    )
    treatment_plan: Mapped[str] = mapped_column(
        Text, default="", nullable=True, comment="医生提交的治疗方案"
    )
    consultation_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="initial",
        comment="问诊类型: initial/follow_up/communication"
    )
    max_rounds: Mapped[int] = mapped_column(
        Integer, default=20, nullable=True, comment="最大允许问诊轮次"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=True)


class ConsultationMessage(Base):
    """问诊对话消息"""

    __tablename__ = "consultation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    consultation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("consultations.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(Enum("doctor", "patient", name="message_role"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, comment="消息序号")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=True)
