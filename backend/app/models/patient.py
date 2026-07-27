from datetime import datetime

from sqlalchemy import DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class VirtualPatient(Base):
    __tablename__ = "virtual_patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(Enum("male", "female", name="gender_type"), nullable=False)
    personality_type: Mapped[str] = mapped_column(
        Enum("配合型", "焦虑型", "沉默型", "对抗型", name="personality_type"),
        nullable=False,
        comment="人格类型：配合型/焦虑型/沉默型/对抗型",
    )
    chief_complaint: Mapped[str] = mapped_column(String(200), nullable=False, comment="主诉")
    medical_history: Mapped[str] = mapped_column(Text, nullable=False, comment="病史")
    symptoms: Mapped[str] = mapped_column(Text, nullable=False, comment="症状描述（JSON）")
    expected_diagnosis: Mapped[str] = mapped_column(
        String(200), default="", nullable=True, comment="预期诊断"
    )
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False, comment="虚拟患者的系统提示词")
    difficulty_level: Mapped[int] = mapped_column(
        Integer, default=1, nullable=True, comment="难度等级 1-5"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=True
    )
