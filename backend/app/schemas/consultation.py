from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel


class ConsultationCreate(BaseModel):
    patient_id: int


class MessageCreate(BaseModel):
    content: str


class DiagnosisSubmit(BaseModel):
    diagnosis: str
    treatment_plan: str


class MessageOut(BaseModel):
    id: int
    consultation_id: int
    role: str
    content: str
    sequence: int
    created_at: datetime

    class Config:
        from_attributes = True


class ConsultationOut(BaseModel):
    id: int
    doctor_id: int
    patient_id: int
    patient_name: Optional[str] = None
    personality_type: Optional[str] = None
    doctor_username: Optional[str] = None
    status: str
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    total_score: Optional[float] = None
    duration_minutes: Optional[int] = None
    summary: Optional[str] = ""
    diagnosis: Optional[str] = ""
    treatment_plan: Optional[str] = ""
    consultation_type: str = "initial"
    max_rounds: int = 20
    created_at: datetime

    class Config:
        from_attributes = True


class ConsultationDetail(ConsultationOut):
    messages: List[MessageOut] = []
