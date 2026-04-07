from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime, Enum

from app.models.base import Base


class VirtualPatient(Base):
    __tablename__ = "virtual_patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(Enum("male", "female", name="gender_type"), nullable=False)
    personality_type = Column(
        Enum("配合型", "焦虑型", "沉默型", "对抗型", name="personality_type"),
        nullable=False,
        comment="人格类型：配合型/焦虑型/沉默型/对抗型",
    )
    chief_complaint = Column(String(200), nullable=False, comment="主诉")
    medical_history = Column(Text, nullable=False, comment="病史")
    symptoms = Column(Text, nullable=False, comment="症状描述（JSON）")
    expected_diagnosis = Column(String(200), default="", comment="预期诊断")
    system_prompt = Column(Text, nullable=False, comment="虚拟患者的系统提示词")
    difficulty_level = Column(Integer, default=1, comment="难度等级 1-5")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
