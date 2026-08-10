"""Repository for immutable prompt bundle storage and experiment lookup."""

import hashlib
import json
from typing import Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment_assignment import ExperimentAssignment
from app.models.prompt_bundle import PromptBundle


class PromptRegistryRepository:
    """Handles immutable prompt bundle storage and experiment assignment lookup."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Prompt bundles
    # ------------------------------------------------------------------

    async def create_bundle(
        self,
        name: str,
        version: str,
        system_prompt: str,
        source_commit: str,
        author: str,
        node_prompts: Optional[dict] = None,
        output_schemas: Optional[dict] = None,
        model_config: Optional[dict] = None,
        skill_manifest_checksum: Optional[str] = None,
        approval_metadata: Optional[dict] = None,
        status: str = "draft",
    ) -> PromptBundle:
        """Create a new prompt bundle with computed content hash."""
        # Compute content hash for integrity verification
        content_parts = {
            "system_prompt": system_prompt,
            "node_prompts": node_prompts,
            "output_schemas": output_schemas,
            "model_config": model_config,
        }
        content_hash = hashlib.sha256(
            json.dumps(content_parts, sort_keys=True).encode()
        ).hexdigest()

        bundle = PromptBundle(
            name=name,
            version=version,
            status=status,
            system_prompt=system_prompt,
            node_prompts=node_prompts,
            output_schemas=output_schemas,
            model_config=model_config,
            content_hash=content_hash,
            skill_manifest_checksum=skill_manifest_checksum,
            source_commit=source_commit,
            author=author,
            approval_metadata=approval_metadata,
        )
        self.db.add(bundle)
        await self.db.flush()
        return bundle

    async def get_bundle(
        self,
        name: str,
        version: str,
    ) -> Optional[PromptBundle]:
        """Get a specific bundle by name and version."""
        stmt = select(PromptBundle).where(
            and_(
                PromptBundle.name == name,
                PromptBundle.version == version,
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active_bundle(self, name: str) -> Optional[PromptBundle]:
        """Get the active bundle for a given name."""
        stmt = select(PromptBundle).where(
            and_(
                PromptBundle.name == name,
                PromptBundle.status == "active",
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def activate_bundle(self, name: str, version: str) -> Optional[PromptBundle]:
        """Set a bundle to active status, archiving other active bundles of same name."""
        bundle = await self.get_bundle(name, version)
        if bundle is None:
            return None

        # Archive any currently active bundles with the same name
        stmt = select(PromptBundle).where(
            and_(
                PromptBundle.name == name,
                PromptBundle.status == "active",
            )
        )
        result = await self.db.execute(stmt)
        for existing in result.scalars().all():
            existing.status = "archived"

        bundle.status = "active"
        await self.db.flush()
        return bundle

    async def list_bundles(
        self,
        name: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[PromptBundle]:
        """List bundles, optionally filtered by name and/or status."""
        conditions = []
        if name is not None:
            conditions.append(PromptBundle.name == name)
        if status is not None:
            conditions.append(PromptBundle.status == status)

        stmt = select(PromptBundle)
        if conditions:
            stmt = stmt.where(and_(*conditions))
        stmt = stmt.order_by(PromptBundle.created_at.desc())

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Experiment assignment lookup
    # ------------------------------------------------------------------

    async def get_experiment_variant(
        self,
        experiment_id: str,
        subject_id: str,
    ) -> Optional[ExperimentAssignment]:
        """Look up experiment assignment for a subject."""
        stmt = select(ExperimentAssignment).where(
            and_(
                ExperimentAssignment.experiment_id == experiment_id,
                ExperimentAssignment.subject_id == subject_id,
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def assign_experiment(
        self,
        experiment_id: str,
        subject_id: str,
        variant: str,
        weight: int = 0,
        rollout_pct: int = 0,
    ) -> ExperimentAssignment:
        """Assign a subject to an experiment variant."""
        assignment = ExperimentAssignment(
            experiment_id=experiment_id,
            subject_id=subject_id,
            variant=variant,
            weight=weight,
            rollout_pct=rollout_pct,
        )
        self.db.add(assignment)
        await self.db.flush()
        return assignment

    async def get_bundle_for_experiment(
        self,
        experiment_id: str,
        subject_id: str,
    ) -> Optional[PromptBundle]:
        """Look up the prompt bundle variant for an experiment subject.

        Returns the active bundle matching the variant name from the assignment.
        """
        assignment = await self.get_experiment_variant(experiment_id, subject_id)
        if assignment is None:
            return None
        return await self.get_active_bundle(assignment.variant)
