"""Immutable prompt bundle registry with deterministic experiment assignment."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.audit import record_audit_log
from app.repositories.prompt_registry import PromptRegistryRepository

RolloutStage = Literal["canary_0", "canary_5", "canary_25", "full_100"]

ROLLOUT_PERCENTAGES: dict[RolloutStage, int] = {
    "canary_0": 0,
    "canary_5": 5,
    "canary_25": 25,
    "full_100": 100,
}


@dataclass(frozen=True)
class PromptBundleView:
    """A read-only view of a prompt bundle."""
    bundle_id: str
    name: str
    version: str
    system_prompt: str
    content_hash: str
    status: str
    source_commit: str
    author: str

    @staticmethod
    def compute_hash(system_prompt: str, node_prompts: dict | None = None) -> str:
        import json
        content_parts = {
            "system_prompt": system_prompt,
            "node_prompts": node_prompts,
        }
        return hashlib.sha256(
            json.dumps(content_parts, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class ExperimentView:
    """A read-only view of an experiment."""
    experiment_id: str
    name: str
    baseline_bundle_id: str
    treatment_bundle_id: str
    stage: RolloutStage = "canary_0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    auto_rollback_triggered: bool = False

    @property
    def rollout_percentage(self) -> int:
        return ROLLOUT_PERCENTAGES[self.stage]


def deterministic_assign(subject_id: str, experiment_id: str, percentage: int) -> bool:
    """Deterministic assignment based on subject_id hash."""
    hash_input = f"{subject_id}:{experiment_id}"
    hash_value = int(hashlib.sha256(hash_input.encode("utf-8")).hexdigest(), 16)
    bucket = hash_value % 100
    return bucket < percentage


class PromptRegistry:
    """Registry of immutable prompt bundles and experiments with DB persistence.

    Key invariants:
    - Active bundles cannot be edited (immutable once active)
    - Registration records source commit and authenticated author
    - Assignment is deterministic and persisted once per (experiment_id, subject_id)
    """

    def __init__(self, db: AsyncSession) -> None:
        self._repo = PromptRegistryRepository(db)
        self._db = db

    async def register_bundle(
        self,
        *,
        name: str,
        version: str,
        system_prompt: str,
        source_commit: str,
        author: str,
        node_prompts: Optional[dict] = None,
        output_schemas: Optional[dict] = None,
        model_config: Optional[dict] = None,
    ):
        """Register a new prompt bundle.

        Records source commit and authenticated author.
        Active bundles cannot be edited - this creates a new bundle.
        """
        # Check if a bundle with same name/version already exists
        existing = await self._repo.get_bundle(name, version)
        if existing is not None:
            raise ValueError(f"Bundle {name} v{version} already exists")

        bundle = await self._repo.create_bundle(
            name=name,
            version=version,
            system_prompt=system_prompt,
            source_commit=source_commit,
            author=author,
            node_prompts=node_prompts,
            output_schemas=output_schemas,
            model_config=model_config,
            status="draft",
        )

        await record_audit_log(
            self._db,
            user_id=None,  # Author is recorded in bundle, not as user_id
            action="register_prompt_bundle",
            resource_id=str(bundle.id),
            detail=f"Registered bundle {name} v{version} by {author} (commit: {source_commit})",
        )
        return bundle

    async def get_bundle(self, name: str, version: str):
        """Get a specific bundle by name and version."""
        return await self._repo.get_bundle(name, version)

    async def get_active_bundle(self, name: str):
        """Get the active bundle for a given name."""
        return await self._repo.get_active_bundle(name)

    async def activate_bundle(self, name: str, version: str):
        """Set a bundle to active status.

        Once active, a bundle cannot be edited (immutability invariant).
        """
        bundle = await self._repo.activate_bundle(name, version)
        if bundle is None:
            raise KeyError(f"Bundle {name} v{version} not found")

        await record_audit_log(
            self._db,
            user_id=None,
            action="activate_prompt_bundle",
            resource_id=str(bundle.id),
            detail=f"Activated bundle {name} v{version}",
        )
        return bundle

    async def list_bundles(
        self,
        name: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list:
        """List bundles, optionally filtered by name and/or status."""
        return await self._repo.list_bundles(name=name, status=status)

    async def assign_bundle(
        self,
        experiment_id: str,
        subject_id: str,
        *,
        baseline_name: str,
        treatment_name: str,
        rollout_pct: int,
    ) -> str:
        """Assign a subject to an experiment variant.

        Assignment is deterministic and persisted once per (experiment_id, subject_id).
        If already assigned, returns the existing assignment.
        """
        # Check for existing assignment (persisted once)
        existing = await self._repo.get_experiment_variant(experiment_id, subject_id)
        if existing is not None:
            return existing.variant

        # Deterministic assignment
        in_treatment = deterministic_assign(subject_id, experiment_id, rollout_pct)
        variant = treatment_name if in_treatment else baseline_name

        # Persist the assignment
        await self._repo.assign_experiment(
            experiment_id=experiment_id,
            subject_id=subject_id,
            variant=variant,
            rollout_pct=rollout_pct,
        )
        return variant
