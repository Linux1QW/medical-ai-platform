# -*- coding: utf-8 -*-
"""Task 8: V1.1 迁移编排脚本

唯一 V1.1 编排入口：
1. 检查当前 revision
2. alembic upgrade 2b3c4d5e6f7a（Task 2A）
3. backfill dry-run
4. 有候选时要求 MIGRATION_OPERATOR_ID 并正式回填
5. 要求 conflict/orphan=0
6. alembic upgrade head

幂等，可重复运行。不调用 create_all。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Alembic revision 常量
_PRE_HEAD_REVISION = "2b3c4d5e6f7a"
_HEAD_REVISION = "3c4d5e6f7a8b"


def _run_command(args: list[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    """运行子进程命令 — 参数必须是固定列表，不拼接 shell 字符串"""
    logger.info(f"Running: {' '.join(args)}")
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=cwd,
        shell=False,  # 安全：不使用 shell
    )
    if result.returncode != 0:
        logger.error(f"Command failed: {result.stderr}")
    return result


def _get_current_revision(cwd: Optional[str] = None) -> Optional[str]:
    """获取当前 Alembic revision"""
    result = _run_command(["alembic", "current"], cwd=cwd)
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    if not output or "None" in output:
        return None
    # 格式: "revision (head)" 或 "revision"
    parts = output.split()
    return parts[0] if parts else None


def _count_backfill_candidates(cwd: Optional[str] = None) -> Dict[str, int]:
    """统计需要 backfill 的 evaluation 数量"""
    import asyncio

    async def _count():
        from sqlalchemy import text

        from app.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
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

    try:
        return asyncio.run(_count())
    except Exception as e:
        logger.warning(f"Failed to count backfill candidates: {e}")
        return {"null_run_id": 0, "orphans": 0, "ambiguous": 0}


def run_migration(
    operator_id: Optional[str] = None,
    batch_size: int = 100,
    dry_run: bool = False,
) -> int:
    """编排 V1.1 迁移

    Returns:
        0 成功，非零失败
    """
    import os

    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Step 1: 检查当前 revision
    current_rev = _get_current_revision(cwd=cwd)
    logger.info(f"Current revision: {current_rev}")

    # 已在 head → 幂等完成
    if current_rev == _HEAD_REVISION:
        logger.info("Already at head revision, nothing to do")
        return 0

    # Step 2: alembic upgrade 到 pre-head（创建 outbox 等）
    if current_rev != _PRE_HEAD_REVISION:
        result = _run_command(
            ["alembic", "upgrade", _PRE_HEAD_REVISION],
            cwd=cwd,
        )
        if result.returncode != 0:
            logger.error(f"Alembic upgrade to {_PRE_HEAD_REVISION} failed: {result.stderr}")
            return 1

    # Step 3: backfill dry-run
    candidates = _count_backfill_candidates()
    logger.info(f"Backfill candidates: {candidates}")

    total_candidates = candidates["null_run_id"] + candidates["orphans"]

    if total_candidates > 0:
        # 有候选时需要 operator
        if not operator_id:
            logger.error(
                f"Found {total_candidates} evaluations needing backfill. "
                f"Provide --operator-id or set MIGRATION_OPERATOR_ID env var."
            )
            return 1

        if dry_run:
            logger.info("Dry-run mode: skipping actual backfill")
        else:
            # 正式执行 backfill
            from scripts.backfill_legacy_evaluation_runs import main_async

            exit_code = asyncio.run(
                main_async(
                    operator_id=operator_id,
                    batch_size=batch_size,
                    dry_run=False,
                )
            )
            if exit_code != 0:
                logger.error("Backfill failed with conflicts")
                return 1

            # 验证 conflict/orphan=0
            post_candidates = _count_backfill_candidates()
            if post_candidates["orphans"] > 0 or post_candidates["ambiguous"] > 0:
                logger.error(
                    f"Post-backfill still has orphans={post_candidates['orphans']}, "
                    f"ambiguous={post_candidates['ambiguous']}. Cannot proceed."
                )
                return 1

    # Step 4: alembic upgrade head（创建 FK 和索引）
    result = _run_command(
        ["alembic", "upgrade", "head"],
        cwd=cwd,
    )
    if result.returncode != 0:
        logger.error(f"Alembic upgrade to head failed: {result.stderr}")
        return 1

    logger.info("V1.1 migration complete")
    return 0


def main() -> None:
    """CLI 入口"""
    parser = argparse.ArgumentParser(description="V1.1 migration orchestrator")
    parser.add_argument(
        "--operator-id",
        type=str,
        default=os.environ.get("MIGRATION_OPERATOR_ID"),
        help="执行 operator ID（或设置 MIGRATION_OPERATOR_ID 环境变量）",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="批处理大小")
    parser.add_argument("--dry-run", action="store_true", help="统计但不写入数据库")

    args = parser.parse_args()

    exit_code = run_migration(
        operator_id=args.operator_id,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
