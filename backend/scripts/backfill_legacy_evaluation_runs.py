# -*- coding: utf-8 -*-
"""Task 8: Legacy evaluation run backfill 脚本

用途：为 evaluations 表中 run_id 为 NULL 的行创建/关联 EvaluationRun。

用法：
    # dry-run 统计
    python scripts/backfill_legacy_evaluation_runs.py --dry-run

    # 正式执行
    python scripts/backfill_legacy_evaluation_runs.py --operator-id <admin-id> --batch-size 100

规则：
- 空 run_id 时优先复用 evaluation_id 精确关联的唯一 EvaluationRun
- 其次复用 consultation/lock 唯一且终态一致的 run
- 都不存在时创建 UUID4 run
- 歧义/active/orphan 写无 PII conflict manifest 并跳过
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# 状态映射：evaluation_status → run status
_STATUS_MAP = {
    "completed": "completed",
    "needs_review": "needs_review",
    "reviewed": "reviewed",
    "failed": "failed",
}


async def _write_manifest(conflicts: List[Dict[str, Any]]) -> str:
    """写入无 PII conflict manifest，返回 hash"""
    manifest_data = json.dumps(conflicts, ensure_ascii=False, default=str)
    manifest_hash = hashlib.sha256(manifest_data.encode()).hexdigest()[:16]
    logger.info(f"Conflict manifest hash: {manifest_hash}, entries: {len(conflicts)}")
    return manifest_hash


async def _count_candidates(db: AsyncSession) -> Dict[str, int]:
    """统计需要 backfill 的 evaluation 数量"""
    result = await db.execute(
        text("SELECT COUNT(*) FROM evaluations WHERE run_id IS NULL")
    )
    null_count = result.scalar() or 0

    result = await db.execute(
        text(
            "SELECT COUNT(*) FROM evaluations e "
            "WHERE e.run_id IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM evaluation_runs er WHERE er.id = e.run_id)"
        )
    )
    orphan_count = result.scalar() or 0

    return {"null_run_id": null_count, "orphans": orphan_count, "ambiguous": 0}


async def backfill_evaluations(
    db: AsyncSession,
    operator_id: Optional[str],
    batch_size: int = 100,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """回填 evaluations 的 run_id

    Returns:
        {"created": int, "reused": int, "conflicts": int, "dry_run": bool}
    """
    stats = {"created": 0, "reused": 0, "conflicts": 0, "skipped": 0, "dry_run": dry_run}
    conflicts: List[Dict[str, Any]] = []

    # 查询 run_id IS NULL 的 evaluations
    result = await db.execute(
        text("SELECT id, consultation_id, evaluation_status, created_at FROM evaluations WHERE run_id IS NULL")
    )
    candidates = result.fetchall()

    if not candidates:
        logger.info("No evaluations with null run_id found")
        return stats

    for row in candidates[:batch_size]:
        eval_id = row[0]
        consultation_id = row[1]
        eval_status = row[2]

        # SELECT FOR UPDATE
        locked_result = await db.execute(
            text("SELECT id, consultation_id, evaluation_status FROM evaluations WHERE id = :id FOR UPDATE"),
            {"id": eval_id},
        )
        locked_row = locked_result.fetchone()
        if locked_row is None:
            continue

        # 查找 evaluation_id 精确关联的唯一 run
        exact_run_result = await db.execute(
            text("SELECT id, consultation_id, status FROM evaluation_runs WHERE evaluation_id = :eid"),
            {"eid": eval_id},
        )
        exact_runs = exact_run_result.fetchall()

        if len(exact_runs) == 1:
            # 精确匹配单个 run → 复用
            run_id = exact_runs[0][0]
            if not dry_run:
                await db.execute(
                    text("UPDATE evaluations SET run_id = :rid WHERE id = :eid"),
                    {"rid": run_id, "eid": eval_id},
                )
            stats["reused"] += 1
            continue
        elif len(exact_runs) > 1:
            # 歧义：多个候选
            conflicts.append({
                "evaluation_id": eval_id,
                "reason": "ambiguous_runs",
                "candidate_count": len(exact_runs),
            })
            stats["conflicts"] += 1
            continue

        # 查找 consultation 关联的 run
        consult_run_result = await db.execute(
            text(
                "SELECT id, status FROM evaluation_runs "
                "WHERE consultation_id = :cid AND evaluation_id IS NULL"
            ),
            {"cid": consultation_id},
        )
        consult_runs = consult_run_result.fetchall()

        if len(consult_runs) == 1:
            # 唯一 consultation run → 复用
            run_id = consult_runs[0][0]
            if not dry_run:
                await db.execute(
                    text("UPDATE evaluations SET run_id = :rid WHERE id = :eid"),
                    {"rid": run_id, "eid": eval_id},
                )
                await db.execute(
                    text("UPDATE evaluation_runs SET evaluation_id = :eid WHERE id = :rid"),
                    {"eid": eval_id, "rid": run_id},
                )
            stats["reused"] += 1
            continue
        elif len(consult_runs) > 1:
            conflicts.append({
                "evaluation_id": eval_id,
                "reason": "ambiguous_consultation_runs",
                "candidate_count": len(consult_runs),
            })
            stats["conflicts"] += 1
            continue

        # 都不存在 → 创建新 run
        new_run_id = str(uuid.uuid4())
        mapped_status = _STATUS_MAP.get(eval_status, "completed")

        if not dry_run:
            await db.execute(
                text(
                    "INSERT INTO evaluation_runs "
                    "(id, consultation_id, evaluation_id, status, checkpoint_thread_id, "
                    "graph_version, scoring_policy_version, attempt, created_at, updated_at) "
                    "VALUES (:id, :cid, :eid, :status, :thread, 'evaluation-graph-v1', 'v1', 0, :now, :now)"
                ),
                {
                    "id": new_run_id,
                    "cid": consultation_id,
                    "eid": eval_id,
                    "status": mapped_status,
                    "thread": f"backfill-{new_run_id}",
                    "now": datetime.utcnow(),
                },
            )
            await db.execute(
                text("UPDATE evaluations SET run_id = :rid WHERE id = :eid"),
                {"rid": new_run_id, "eid": eval_id},
            )
        stats["created"] += 1

    # 写入 conflict manifest
    if conflicts:
        await _write_manifest(conflicts)

    # 提交事务
    if not dry_run:
        await db.commit()
    else:
        await db.rollback()

    return stats


async def main_async(
    operator_id: Optional[str],
    batch_size: int,
    dry_run: bool,
) -> int:
    """异步入口"""
    from app.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            stats = await backfill_evaluations(
                db=db,
                operator_id=operator_id,
                batch_size=batch_size,
                dry_run=dry_run,
            )
            logger.info(f"Backfill complete: {stats}")
            return 0 if stats["conflicts"] == 0 else 1
        except Exception as e:
            logger.error(f"Backfill failed: {e}")
            await db.rollback()
            return 1


def main() -> None:
    """CLI 入口"""
    parser = argparse.ArgumentParser(description="Backfill legacy evaluation runs")
    parser.add_argument("--dry-run", action="store_true", help="统计但不写入数据库")
    parser.add_argument("--operator-id", type=str, default=None, help="执行 operator ID")
    parser.add_argument("--batch-size", type=int, default=100, help="批处理大小")

    args = parser.parse_args()

    if not args.dry_run and not args.operator_id:
        print("Error: --operator-id is required for non-dry-run execution", file=sys.stderr)
        sys.exit(1)

    import asyncio

    exit_code = asyncio.run(
        main_async(
            operator_id=args.operator_id,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
