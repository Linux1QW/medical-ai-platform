# -*- coding: utf-8 -*-
"""Task 13 — 复核反馈导出 CLI

从数据库读取 ReviewRecord + Evaluation，生成去标识的 candidate manifest JSONL。

用法：
  python -m scripts.export_review_feedback --output candidates.jsonl
  python -m scripts.export_review_feedback --output candidates.jsonl --since 2026-08-01 --until 2026-08-10
  python -m scripts.export_review_feedback --output private.jsonl --include-content --acknowledge-sensitive-data

环境变量：
  FEEDBACK_EXPORT_HMAC_KEY  必填，用于 HMAC-SHA256 哈希 consultation_id
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 私有目录（含完整问诊内容，gitignore）
PRIVATE_FEEDBACK_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "private_feedback"


async def fetch_review_records(
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int | None = None,
) -> list[tuple[Any, Any]]:
    """从数据库获取 ReviewRecord + 关联 Evaluation

    Returns:
        list of (ReviewRecord, Evaluation) 元组
    """
    try:
        from sqlalchemy import String, cast, select

        from app.db.session import AsyncSessionLocal
        from app.models.evaluation import Evaluation
        from app.models.review_record import ReviewRecord

        async with AsyncSessionLocal() as session:
            query = (
                select(ReviewRecord, Evaluation)
                .join(Evaluation, ReviewRecord.evaluation_id == cast(Evaluation.id, String))
                .order_by(ReviewRecord.created_at.desc())
            )

            if since:
                query = query.where(ReviewRecord.created_at >= since)
            if until:
                query = query.where(ReviewRecord.created_at <= until)
            if limit:
                query = query.limit(limit)

            result = await session.execute(query)
            rows = result.all()
            return [(row.ReviewRecord, row.Evaluation) for row in rows]

    except Exception as e:
        logger.error(f"数据库查询失败: {e}")
        raise


def _atomic_write_jsonl(records: list[dict], output_path: Path) -> None:
    """原子写 JSONL（临时文件 + replace）"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        suffix=".jsonl.tmp",
        dir=str(output_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        # 原子替换
        if sys.platform == "win32":
            # Windows 不支持 atomic replace，先删除再移动
            if output_path.exists():
                output_path.unlink()
        Path(tmp_path).replace(output_path)
    except Exception:
        # 清理临时文件
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass
        raise


async def async_main(args: argparse.Namespace) -> int:
    """异步主逻辑"""
    from evaluation.feedback_cases import (
        ExportConfig,
        build_candidate,
    )

    # 检查 HMAC key
    hmac_key = os.environ.get("FEEDBACK_EXPORT_HMAC_KEY", "")
    if not hmac_key:
        print("ERROR: 环境变量 FEEDBACK_EXPORT_HMAC_KEY 未设置", file=sys.stderr)
        return 2

    # 解析时间范围
    since = datetime.fromisoformat(args.since) if args.since else None
    until = datetime.fromisoformat(args.until) if args.until else None

    # 导出配置
    config = ExportConfig(
        hmac_key=hmac_key,
        include_content=args.include_content,
        acknowledge_sensitive_data=args.acknowledge_sensitive_data,
        output_path=args.output,
        since=since,
        until=until,
        limit=args.limit,
    )

    # 敏感模式双确认检查
    if args.include_content and not args.acknowledge_sensitive_data:
        print(
            "ERROR: 导出含受限内容需同时提供 --include-content --acknowledge-sensitive-data",
            file=sys.stderr,
        )
        return 3

    # 确定输出路径
    output_path = Path(args.output)
    if config.include_content:
        # 受限内容写入私有目录
        output_path = PRIVATE_FEEDBACK_DIR / output_path.name

    # 获取数据
    try:
        pairs = await fetch_review_records(since=since, until=until, limit=args.limit)
    except Exception as e:
        print(f"ERROR: 数据库查询失败: {e}", file=sys.stderr)
        return 4

    if not pairs:
        print("无复核记录可导出")
        # 空结果写空文件
        _atomic_write_jsonl([], output_path)
        return 0

    # 构造候选样本
    candidates = []
    for review, evaluation in pairs:
        try:
            candidate = build_candidate(review, evaluation, hmac_key=hmac_key)
            data = candidate.model_dump(mode="json")

            # 默认模式移除敏感字段
            if not config.include_content:
                data.pop("conversation_text", None)

            candidates.append(data)
        except Exception as e:
            logger.warning(f"跳过 review {review.id}: {e}")

    # 应用 limit（fetch 层可能已限制，这里兜底）
    if args.limit:
        candidates = candidates[: args.limit]

    # 原子写
    try:
        _atomic_write_jsonl(candidates, output_path)
    except OSError as e:
        print(f"ERROR: 写入失败: {e}", file=sys.stderr)
        return 5

    print(f"导出 {len(candidates)} 条候选到: {output_path}")
    return 0


def main() -> int:
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="导出复核反馈候选样本 (Task 13)",
        prog="export_review_feedback",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="起始时间 (ISO 8601, 如 2026-08-01)",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="截止时间 (ISO 8601, 如 2026-08-10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制导出数量",
    )
    parser.add_argument(
        "--include-content",
        action="store_true",
        default=False,
        help="包含受限内容（完整问诊对话）",
    )
    parser.add_argument(
        "--acknowledge-sensitive-data",
        action="store_true",
        default=False,
        help="确认了解数据敏感性（与 --include-content 配合使用）",
    )

    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
