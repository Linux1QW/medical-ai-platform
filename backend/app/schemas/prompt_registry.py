"""Prompt registry API schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RegisterBundleRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=20)
    system_prompt: str = Field(min_length=1)
    safety_policy: str = Field(min_length=1)


class BundleResponse(BaseModel):
    bundle_id: str
    name: str
    version: str
    content_hash: str


class CreateExperimentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    baseline_bundle_id: str
    treatment_bundle_id: str


class ExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    baseline_bundle_id: str
    treatment_bundle_id: str
    stage: str
    rollout_percentage: int
    auto_rollback_triggered: bool


class AssignmentRequest(BaseModel):
    doctor_id: int


class AssignmentResponse(BaseModel):
    experiment_id: str
    doctor_id: int
    assigned_bundle_id: str
