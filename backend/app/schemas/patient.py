from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    personality_type: str
    chief_complaint: str
    medical_history: str
    symptoms: str
    expected_diagnosis: Optional[str] = ""
    system_prompt: str
    difficulty_level: Optional[int] = 1


class PatientUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    personality_type: Optional[str] = None
    chief_complaint: Optional[str] = None
    medical_history: Optional[str] = None
    symptoms: Optional[str] = None
    expected_diagnosis: Optional[str] = None
    system_prompt: Optional[str] = None
    difficulty_level: Optional[int] = None


class PatientOut(BaseModel):
    id: int
    name: str
    age: int
    gender: str
    personality_type: str
    chief_complaint: str
    medical_history: str
    symptoms: str
    expected_diagnosis: str
    system_prompt: str = ""
    difficulty_level: int
    created_at: datetime

    class Config:
        from_attributes = True
