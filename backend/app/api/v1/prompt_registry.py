"""Prompt registry API endpoints with authentication and authorization."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_runtime.prompt_registry import PromptRegistry
from app.core.permissions import require_permission
from app.db.session import get_db
from app.models.user import User
from app.schemas.prompt_registry import (
    AssignmentRequest,
    AssignmentResponse,
    BundleResponse,
    CreateExperimentRequest,
    ExperimentResponse,
    RegisterBundleRequest,
)

router = APIRouter(prefix="/prompt-registry", tags=["prompt-registry"])


def _bundle_to_response(bundle) -> BundleResponse:
    """Convert a PromptBundle model to response schema."""
    return BundleResponse(
        bundle_id=bundle.id,
        name=bundle.name,
        version=bundle.version,
        status=bundle.status,
        content_hash=bundle.content_hash,
        source_commit=bundle.source_commit,
        author=bundle.author,
    )


@router.post("/bundles", response_model=BundleResponse, status_code=201)
async def register_bundle(
    request: RegisterBundleRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("prompt:manage"),
) -> BundleResponse:
    """Register a new prompt bundle.

    Requires prompt:manage permission.
    Records source commit and authenticated author.
    """
    registry = PromptRegistry(db)
    try:
        bundle = await registry.register_bundle(
            name=request.name,
            version=request.version,
            system_prompt=request.system_prompt,
            source_commit=request.source_commit,
            author=current_user.username,
            node_prompts=request.node_prompts,
            output_schemas=request.output_schemas,
            model_config=request.llm_config,
        )
        return _bundle_to_response(bundle)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.get("/bundles/{name}/{version}", response_model=BundleResponse)
async def get_bundle(
    name: str,
    version: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("prompt:manage"),
) -> BundleResponse:
    """Get a specific bundle by name and version.

    Requires prompt:manage permission.
    """
    registry = PromptRegistry(db)
    bundle = await registry.get_bundle(name, version)
    if bundle is None:
        raise HTTPException(status_code=404, detail=f"Bundle {name} v{version} not found")
    return _bundle_to_response(bundle)


@router.get("/bundles/{name}/active", response_model=BundleResponse)
async def get_active_bundle(
    name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("prompt:manage"),
) -> BundleResponse:
    """Get the active bundle for a given name.

    Requires prompt:manage permission.
    """
    registry = PromptRegistry(db)
    bundle = await registry.get_active_bundle(name)
    if bundle is None:
        raise HTTPException(status_code=404, detail=f"No active bundle for {name}")
    return _bundle_to_response(bundle)


@router.post("/bundles/{name}/{version}/activate", response_model=BundleResponse)
async def activate_bundle(
    name: str,
    version: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("prompt:manage"),
) -> BundleResponse:
    """Activate a bundle. Once active, it cannot be edited (immutable).

    Requires prompt:manage permission.
    """
    registry = PromptRegistry(db)
    try:
        bundle = await registry.activate_bundle(name, version)
        return _bundle_to_response(bundle)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/bundles", response_model=list[BundleResponse])
async def list_bundles(
    name: str | None = None,
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("prompt:manage"),
) -> list[BundleResponse]:
    """List bundles, optionally filtered by name and/or status.

    Requires prompt:manage permission.
    """
    registry = PromptRegistry(db)
    bundles = await registry.list_bundles(name=name, status=status)
    return [_bundle_to_response(b) for b in bundles]


@router.post("/experiments", response_model=ExperimentResponse, status_code=201)
async def create_experiment(
    request: CreateExperimentRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("experiment:manage"),
) -> ExperimentResponse:
    """Create a new prompt experiment.

    Requires experiment:manage permission.
    Assignment is deterministic and persisted once per (experiment_id, subject_id).
    """
    # Verify baseline and treatment bundles exist
    registry = PromptRegistry(db)
    baseline = await registry.get_active_bundle(request.baseline_bundle_name)
    treatment = await registry.get_active_bundle(request.treatment_bundle_name)

    if baseline is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active baseline bundle: {request.baseline_bundle_name}",
        )
    if treatment is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active treatment bundle: {request.treatment_bundle_name}",
        )

    # For now, experiment is a logical concept tracked via assignments
    # The experiment_id is derived from the name
    experiment_id = f"exp_{request.name}"

    return ExperimentResponse(
        experiment_id=experiment_id,
        name=request.name,
        baseline_bundle_name=request.baseline_bundle_name,
        treatment_bundle_name=request.treatment_bundle_name,
        rollout_pct=request.rollout_pct,
        stage="planned",
    )


@router.post("/experiments/{experiment_id}/assign", response_model=AssignmentResponse)
async def assign_bundle(
    experiment_id: str,
    request: AssignmentRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = require_permission("experiment:manage"),
) -> AssignmentResponse:
    """Assign a subject to an experiment variant.

    Requires experiment:manage permission.
    Assignment is deterministic and persisted once per (experiment_id, subject_id).
    """
    # For assignment, we need the experiment config
    # In a real system, this would be stored in a table
    # For now, we use the registry's deterministic assignment
    registry = PromptRegistry(db)

    # We need to know the baseline/treatment names and rollout pct
    # These would be stored in an experiment table in production
    # For this implementation, we pass them via the registry
    try:
        variant = await registry.assign_bundle(
            experiment_id=experiment_id,
            subject_id=request.subject_id,
            baseline_name="baseline",  # Would come from experiment config
            treatment_name="treatment",  # Would come from experiment config
            rollout_pct=5,  # Would come from experiment config
        )
        return AssignmentResponse(
            experiment_id=experiment_id,
            subject_id=request.subject_id,
            assigned_variant=variant,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
