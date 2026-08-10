"""Trainee profile memory service with approval lifecycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

MemoryStatus = Literal["candidate", "approved", "rejected", "expired"]

VALID_DIMENSIONS = [
    "rapport", "information_gathering", "clinical_reasoning",
    "communication", "safety_awareness", "professionalism",
]


@dataclass
class ProfileMemoryEntry:
    """In-memory profile memory entry."""
    memory_id: str
    doctor_id: int
    status: MemoryStatus
    skill_dimension: str
    summary: str
    evidence_refs: list[str] = field(default_factory=list)
    reviewer_id: int | None = None
    review_comment: str | None = None
    reviewed_at: datetime | None = None
    expires_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    consent_given: bool = False  # Default OFF


class ProfileMemoryService:
    """Manages trainee profile memories.

    Lifecycle: candidate → approved/rejected/expired
    - Only approved memories are visible to the coach
    - Consent defaults to False (opt-in)
    - Max 5 approved memories loaded per context compilation
    - 6 skill dimensions
    """

    def __init__(self) -> None:
        self._memories: dict[str, ProfileMemoryEntry] = {}
        self._consent: dict[int, bool] = {}  # doctor_id → consent
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"mem_{self._counter:06d}"

    def create_candidate(
        self,
        *,
        doctor_id: int,
        skill_dimension: str,
        summary: str,
        evidence_refs: list[str] | None = None,
    ) -> ProfileMemoryEntry:
        """Create a new candidate memory entry."""
        if skill_dimension not in VALID_DIMENSIONS:
            raise ValueError(
                f"Invalid skill dimension: {skill_dimension}. "
                f"Must be one of {VALID_DIMENSIONS}"
            )

        entry = ProfileMemoryEntry(
            memory_id=self._next_id(),
            doctor_id=doctor_id,
            status="candidate",
            skill_dimension=skill_dimension,
            summary=summary[:500],
            evidence_refs=evidence_refs or [],
        )
        self._memories[entry.memory_id] = entry
        return entry

    def approve(
        self,
        memory_id: str,
        *,
        reviewer_id: int,
        review_comment: str | None = None,
        ttl_days: int = 90,
    ) -> ProfileMemoryEntry:
        """Approve a candidate memory."""
        entry = self._memories.get(memory_id)
        if entry is None:
            raise KeyError(f"Memory {memory_id} not found")
        if entry.status != "candidate":
            raise ValueError(
                f"Memory {memory_id} is {entry.status}, not candidate"
            )

        entry.status = "approved"
        entry.reviewer_id = reviewer_id
        entry.review_comment = review_comment
        entry.reviewed_at = datetime.utcnow()
        entry.expires_at = datetime.utcnow() + timedelta(days=ttl_days)
        return entry

    def reject(
        self,
        memory_id: str,
        *,
        reviewer_id: int,
        review_comment: str | None = None,
    ) -> ProfileMemoryEntry:
        """Reject a candidate memory."""
        entry = self._memories.get(memory_id)
        if entry is None:
            raise KeyError(f"Memory {memory_id} not found")
        if entry.status != "candidate":
            raise ValueError(
                f"Memory {memory_id} is {entry.status}, not candidate"
            )

        entry.status = "rejected"
        entry.reviewer_id = reviewer_id
        entry.review_comment = review_comment
        entry.reviewed_at = datetime.utcnow()
        return entry

    def expire_stale(self) -> int:
        """Expire all approved memories past their TTL. Returns count expired."""
        now = datetime.utcnow()
        count = 0
        for entry in self._memories.values():
            if entry.status == "approved" and entry.expires_at and entry.expires_at < now:
                entry.status = "expired"
                count += 1
        return count

    def get_approved_memories(
        self,
        doctor_id: int,
        *,
        max_count: int = 5,
    ) -> list[ProfileMemoryEntry]:
        """Get approved memories for a doctor. Max 5 by default.

        Only returns memories if consent is given.
        """
        if not self._consent.get(doctor_id, False):
            return []

        approved = [
            m for m in self._memories.values()
            if m.doctor_id == doctor_id and m.status == "approved"
        ]
        # Sort by most recently reviewed
        approved.sort(key=lambda m: m.reviewed_at or m.created_at, reverse=True)
        return approved[:max_count]

    def set_consent(self, doctor_id: int, consent: bool) -> None:
        """Set consent for a doctor. Default is False."""
        self._consent[doctor_id] = consent

    def get_consent(self, doctor_id: int) -> bool:
        """Get consent status for a doctor."""
        return self._consent.get(doctor_id, False)

    def list_memories(
        self,
        doctor_id: int,
        status: MemoryStatus | None = None,
    ) -> list[ProfileMemoryEntry]:
        """List memories for a doctor, optionally filtered by status."""
        results = [m for m in self._memories.values() if m.doctor_id == doctor_id]
        if status:
            results = [m for m in results if m.status == status]
        return results
