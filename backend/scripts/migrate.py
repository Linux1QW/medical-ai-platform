# -*- coding: utf-8 -*-
"""Version-agnostic migration entry point (V1.0 → V1.2 head).

Replaces ``migrate_v11.py`` as the single Compose migrate command.

Idempotent paths covered:
1. Empty DB (no revision)        → upgrade head
2. V1.1 head (3c4d5e6f7a8b)      → upgrade head (V1.2 migrations)
3. V1.2 first版 (4d5e6f7a8b9c)   → upgrade head (remediation)
4. Coach tables with existing rows → backfill then upgrade head
5. downgrade -1 then re-upgrade  → upgrade head
6. Already at head (5e6f7a8b9c0d) → no-op upgrade head (idempotent)

Exit code 0 only when final revision == 5e6f7a8b9c0d.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# ── Revision constants ─────────────────────────────────────────────────────────
# V1.1 pre-head: last revision before the V1.1 backfill FK (needs legacy backfill)
_V11_PRE_HEAD = "2b3c4d5e6f7a"
# V1.1 head: last V1.1 revision
_V11_HEAD = "3c4d5e6f7a8b"
# Target final head
_TARGET_HEAD = "6f7a8b9c0d1e"


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _backend_cwd() -> str:
    """Return the backend/ directory (where alembic.ini lives)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    """Run a subprocess command; never uses shell=True."""
    logger.info("Running: %s", " ".join(args))
    return subprocess.run(args, capture_output=True, text=True, cwd=cwd, shell=False)


def _get_current_revision(cwd: str) -> Optional[str]:
    """Read the current Alembic revision from the database.

    Returns None when no revision is recorded (empty / uninitialised DB).
    """
    result = _run(["alembic", "current"], cwd=cwd)
    if result.returncode != 0:
        # alembic current returns non-zero when alembic_version table is missing
        logger.info("alembic current failed (likely empty DB); treating as no revision.")
        return None
    output = result.stdout.strip()
    if not output or "None" in output:
        return None
    # Format: "<revision> (head)" or just "<revision>"
    return output.split()[0]


def _run_backfill(cwd: str, operator_id: Optional[str]) -> int:
    """Execute the V1.1 legacy evaluation_runs backfill.

    Returns 0 on success, non-zero on failure.
    """
    import asyncio

    try:
        from scripts.backfill_legacy_evaluation_runs import main_async
    except ImportError:
        logger.warning("backfill_legacy_evaluation_runs not importable; skipping backfill.")
        return 0

    exit_code = asyncio.run(
        main_async(
            operator_id=operator_id or "migration-script",
            batch_size=100,
            dry_run=False,
        )
    )
    return exit_code


# ── Main orchestration ──────────────────────────────────────────────────────────

def run_migration(operator_id: Optional[str] = None) -> int:
    """Run the full migration pipeline.

    Always ends with ``alembic upgrade head`` and verifies the final revision.
    Returns 0 on success, non-zero on failure.
    """
    cwd = _backend_cwd()
    operator_id = operator_id or os.environ.get("MIGRATION_OPERATOR_ID")

    # Step 1: read current revision
    current_rev = _get_current_revision(cwd)
    logger.info("Current revision: %s", current_rev)

    # Step 2: if at or before V1.1 pre-head, run backfill before proceeding.
    # The backfill is only needed when upgrading FROM pre-head (outbox table just created,
    # legacy evaluations need run_id populated before the FK is added in V1.1 head).
    if current_rev == _V11_PRE_HEAD:
        logger.info("At V1.1 pre-head; running legacy evaluation backfill…")
        bf_exit = _run_backfill(cwd, operator_id)
        if bf_exit != 0:
            logger.error("Legacy backfill failed (exit=%d). Aborting.", bf_exit)
            return 1

    # Step 3: ALWAYS run alembic upgrade head.
    # This covers every path: empty DB, V1.1 head, V1.2 first revision, already at head.
    result = _run(["alembic", "upgrade", "head"], cwd=cwd)
    if result.returncode != 0:
        logger.error("alembic upgrade head failed:\n%s", result.stderr)
        return 1

    # Step 4: verify final revision
    final_rev = _get_current_revision(cwd)
    logger.info("Final revision: %s", final_rev)

    if final_rev != _TARGET_HEAD:
        logger.error(
            "Migration verification failed: expected %s but got %s",
            _TARGET_HEAD,
            final_rev,
        )
        return 1

    logger.info("Migration complete — at %s", _TARGET_HEAD)
    return 0


def main() -> None:
    """CLI entry point."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Version-agnostic migration entry point")
    parser.add_argument(
        "--operator-id",
        default=os.environ.get("MIGRATION_OPERATOR_ID"),
        help="Operator ID for legacy backfill (or set MIGRATION_OPERATOR_ID env var)",
    )
    args = parser.parse_args()

    sys.exit(run_migration(operator_id=args.operator_id))


if __name__ == "__main__":
    main()
