"""Prompt registry API schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RegisterBundleRequest(BaseModel):
    """Bundle registration request.

    Source commit and author are recorded for audit trail.
    """
    name: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=50)
    system_prompt: str = Field(min_length=1)
    source_commit: str = Field(min_length=7, max_length=40)
    node_prompts: dict | None = None
    output_schemas: dict | None = None
    llm_config: dict | None = None


class BundleResponse(BaseModel):
    bundle_id: int
    name: str
    version: str
    status: str
    content_hash: str | None = None
    source_commit: str | None = None
    author: str | None = None


class CreateExperimentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    baseline_bundle_name: str
    treatment_bundle_name: str
    rollout_pct: int = Field(default=0, ge=0, le=100)


class ExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    baseline_bundle_name: str
    treatment_bundle_name: str
    rollout_pct: int
    stage: str = "planned"


class AssignmentRequest(BaseModel):
    """Assignment request.

    Subject identity from token or explicit subject_id.
    """
    subject_id: str


class AssignmentResponse(BaseModel):
    experiment_id: str
    subject_id: str
    assigned_variant: str
