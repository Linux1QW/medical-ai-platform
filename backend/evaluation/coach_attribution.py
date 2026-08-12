"""Feedback attribution for the coach data flywheel.

Only admin-reviewed + deidentified decisions can become training candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class AttributionCandidate:
    """A decision that is a candidate for the training flywheel."""
    decision_id: str
    session_id: str
    feedback: Literal["accepted", "rejected", "ignored"]
    admin_reviewed: bool = False
    deidentified: bool = False
    eligible: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


class AttributionService:
    """Manages feedback attribution for the data flywheel.

    Rules:
    - Only admin-reviewed decisions can become candidates
    - Only deidentified decisions can become candidates
    - Both conditions must be true for eligibility
    """

    def __init__(self) -> None:
        self._candidates: dict[str, AttributionCandidate] = {}
        self._counter = 0

    def create_candidate(
        self,
        *,
        decision_id: str,
        session_id: str,
        feedback: str,
    ) -> AttributionCandidate:
        """Create a new attribution candidate."""
        self._counter += 1
        candidate = AttributionCandidate(
            decision_id=decision_id,
            session_id=session_id,
            feedback=feedback,  # type: ignore[arg-type]
        )
        self._candidates[decision_id] = candidate
        return candidate

    def mark_admin_reviewed(self, decision_id: str) -> AttributionCandidate:
        """Mark a candidate as admin-reviewed."""
        candidate = self._candidates.get(decision_id)
        if candidate is None:
            raise KeyError(f"Candidate {decision_id} not found")
        candidate.admin_reviewed = True
        candidate.eligible = candidate.admin_reviewed and candidate.deidentified
        return candidate

    def mark_deidentified(self, decision_id: str) -> AttributionCandidate:
        """Mark a candidate as deidentified."""
        candidate = self._candidates.get(decision_id)
        if candidate is None:
            raise KeyError(f"Candidate {decision_id} not found")
        candidate.deidentified = True
        candidate.eligible = candidate.admin_reviewed and candidate.deidentified
        return candidate

    def get_eligible(self) -> list[AttributionCandidate]:
        """Get all eligible candidates (admin-reviewed AND deidentified)."""
        return [c for c in self._candidates.values() if c.eligible]

    def get_candidate(self, decision_id: str) -> AttributionCandidate | None:
        """Get a candidate by ID."""
        return self._candidates.get(decision_id)
