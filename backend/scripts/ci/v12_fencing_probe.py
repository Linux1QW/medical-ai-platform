"""Exercise EvaluationRun ownership fencing against the real E2E database."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import delete, update

from app.core.database import AsyncSessionLocal
from app.models.evaluation_run import EvaluationRun
from app.services.evaluation_run_service import (
    RunClaimDisposition,
    RunLeaseLost,
    claim_run,
    create_queued_run,
    mark_run_terminal,
)


async def run_probe() -> dict[str, object]:
    run_id = str(uuid4())
    owner_a = "fault-probe-owner-a"
    owner_b = "fault-probe-owner-b"
    now = datetime.utcnow()

    try:
        async with AsyncSessionLocal() as session:
            await create_queued_run(session, run_id=run_id, consultation_id=2)
            await session.commit()

        async with AsyncSessionLocal() as owner_a_session:
            first = await claim_run(
                owner_a_session,
                run_id=run_id,
                celery_task_id="fault-task-a",
                execution_owner=owner_a,
                now=now,
                lease_seconds=60,
            )
            assert first.disposition is RunClaimDisposition.STARTED
            await owner_a_session.commit()

        async with AsyncSessionLocal() as owner_b_session:
            active_other = await claim_run(
                owner_b_session,
                run_id=run_id,
                celery_task_id="fault-task-b",
                execution_owner=owner_b,
                now=now + timedelta(seconds=1),
                lease_seconds=60,
            )
            assert active_other.disposition is RunClaimDisposition.ACTIVE_OTHER_TASK
            await owner_b_session.commit()

        async with AsyncSessionLocal() as control_session:
            await control_session.execute(
                update(EvaluationRun)
                .where(EvaluationRun.id == run_id)
                .values(lease_expires_at=now - timedelta(seconds=1))
            )
            await control_session.commit()

        async with AsyncSessionLocal() as owner_b_session:
            reclaimed = await claim_run(
                owner_b_session,
                run_id=run_id,
                celery_task_id="fault-task-b",
                execution_owner=owner_b,
                now=now + timedelta(seconds=2),
                lease_seconds=60,
            )
            assert reclaimed.disposition is RunClaimDisposition.RECLAIMED_STALE
            await owner_b_session.commit()

        stale_owner_rejected = False
        async with AsyncSessionLocal() as owner_a_session:
            try:
                await mark_run_terminal(
                    owner_a_session,
                    run_id=run_id,
                    status="failed",
                    execution_owner=owner_a,
                )
            except RunLeaseLost:
                stale_owner_rejected = True
                await owner_a_session.rollback()
            assert stale_owner_rejected

        async with AsyncSessionLocal() as owner_b_session:
            await mark_run_terminal(
                owner_b_session,
                run_id=run_id,
                status="failed",
                execution_owner=owner_b,
                error_code="FAULT_PROBE_CLEANUP",
            )
            await owner_b_session.commit()

        return {
            "run_id": run_id,
            "first_claim": first.disposition.value,
            "active_other": active_other.disposition.value,
            "reclaimed": reclaimed.disposition.value,
            "stale_owner_rejected": stale_owner_rejected,
        }
    finally:
        async with AsyncSessionLocal() as cleanup_session:
            await cleanup_session.execute(
                delete(EvaluationRun).where(EvaluationRun.id == run_id)
            )
            await cleanup_session.commit()


def main() -> int:
    evidence = asyncio.run(run_probe())
    print(json.dumps(evidence, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
