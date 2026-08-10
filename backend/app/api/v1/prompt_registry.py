"""Prompt registry API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.agent_runtime.prompt_registry import PromptRegistry
from app.schemas.prompt_registry import (
    AssignmentRequest,
    AssignmentResponse,
    BundleResponse,
    CreateExperimentRequest,
    ExperimentResponse,
    RegisterBundleRequest,
)

router = APIRouter(prefix="/prompt-registry", tags=["prompt-registry"])
_registry = PromptRegistry()


def get_registry() -> PromptRegistry:
    return _registry


@router.post("/bundles", response_model=BundleResponse)
async def register_bundle(request: RegisterBundleRequest) -> BundleResponse:
    bundle = _registry.register_bundle(name=request.name, version=request.version,
                                       system_prompt=request.system_prompt, safety_policy=request.safety_policy)
    return BundleResponse(bundle_id=bundle.bundle_id, name=bundle.name, version=bundle.version, content_hash=bundle.content_hash)


@router.get("/bundles/{bundle_id}", response_model=BundleResponse)
async def get_bundle(bundle_id: str) -> BundleResponse:
    bundle = _registry.get_bundle(bundle_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail=f"Bundle {bundle_id} not found")
    return BundleResponse(bundle_id=bundle.bundle_id, name=bundle.name, version=bundle.version, content_hash=bundle.content_hash)


@router.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(request: CreateExperimentRequest) -> ExperimentResponse:
    try:
        exp = _registry.create_experiment(name=request.name, baseline_bundle_id=request.baseline_bundle_id,
                                          treatment_bundle_id=request.treatment_bundle_id)
        return ExperimentResponse(experiment_id=exp.experiment_id, name=exp.name,
                                  baseline_bundle_id=exp.baseline_bundle_id, treatment_bundle_id=exp.treatment_bundle_id,
                                  stage=exp.stage, rollout_percentage=exp.rollout_percentage,
                                  auto_rollback_triggered=exp.auto_rollback_triggered)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/experiments/{experiment_id}/advance", response_model=ExperimentResponse)
async def advance_experiment(experiment_id: str) -> ExperimentResponse:
    exp = _registry.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    exp.advance_stage()
    return ExperimentResponse(experiment_id=exp.experiment_id, name=exp.name,
                              baseline_bundle_id=exp.baseline_bundle_id, treatment_bundle_id=exp.treatment_bundle_id,
                              stage=exp.stage, rollout_percentage=exp.rollout_percentage,
                              auto_rollback_triggered=exp.auto_rollback_triggered)


@router.post("/experiments/{experiment_id}/assign", response_model=AssignmentResponse)
async def assign_bundle(experiment_id: str, request: AssignmentRequest) -> AssignmentResponse:
    try:
        bundle_id = _registry.assign_bundle(experiment_id, request.doctor_id)
        return AssignmentResponse(experiment_id=experiment_id, doctor_id=request.doctor_id, assigned_bundle_id=bundle_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
