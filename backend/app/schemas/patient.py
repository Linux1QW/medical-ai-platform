from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PatientCreate(BaseModel):
    name: str = Field(..., max_length=50)
    age: int = Field(..., ge=0, le=200)
    gender: str = Field(..., max_length=10)
    personality_type: str = Field(..., max_length=20)
    chief_complaint: str = Field(..., max_length=500)
    medical_history: str = Field(..., max_length=10000)
    symptoms: str = Field(..., max_length=10000)
    expected_diagnosis: str = Field(default="", max_length=200)
    system_prompt: str = Field(..., max_length=10000)
    difficulty_level: int = Field(default=1, ge=1, le=5)


class PatientUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=50)
    age: int | None = Field(default=None, ge=0, le=200)
    gender: str | None = Field(default=None, max_length=10)
    personality_type: str | None = Field(default=None, max_length=20)
    chief_complaint: str | None = Field(default=None, max_length=500)
    medical_history: str | None = Field(default=None, max_length=10000)
    symptoms: str | None = Field(default=None, max_length=10000)
    expected_diagnosis: str | None = Field(default=None, max_length=200)
    system_prompt: str | None = Field(default=None, max_length=10000)
    difficulty_level: int | None = Field(default=None, ge=1, le=5)


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
