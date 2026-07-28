# -*- coding: utf-8 -*-
"""管理接口 — 缓存清理、数据留存、运维操作与运行监控"""

from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_admin
from app.db.session import get_db
from app.models.evaluation_node_result import EvaluationNodeResult
from app.models.evaluation_run import EvaluationRun
from app.models.user import User
from app.services.llm_cache import LLMResponseCache
from app.services.rag.retrieval_cache import clear_retrieval_cache, get_retrieval_cache_stats

router = APIRouter()


@router.post("/cache/retrieval/clear")
async def clear_cache(current_user: User = Depends(get_current_admin)):
    """清除检索缓存（仅管理员）"""
    deleted = await clear_retrieval_cache()
    return {"message": "检索缓存已清除", "deleted": deleted}


@router.get("/cache/retrieval/stats")
async def cache_stats(current_user: User = Depends(get_current_admin)):
    """获取检索缓存统计信息（仅管理员）"""
    return await get_retrieval_cache_stats()


@router.get("/cache-stats")
async def cache_stats_all(current_user: User = Depends(get_current_admin)):
    """获取所有缓存详细统计信息（LLM + 检索，仅管理员）"""
    llm_stats = await LLMResponseCache.get_stats()
    retrieval_stats = await get_retrieval_cache_stats()
    return {
        "llm_cache": llm_stats,
        "retrieval_cache": retrieval_stats,
    }


@router.post("/cleanup")
async def trigger_cleanup(current_user: User = Depends(get_current_admin)):
    """手动触发数据清理（审计日志 + 评估运行记录）"""
    from app.tasks.data_cleanup import cleanup_expired_records

    result = cleanup_expired_records.delay()
    return {
        "message": "清理任务已提交",
        "task_id": result.id,
    }


# ── 运行监控 ────────────────────────────────────────────────────────────


@router.get("/monitoring/tool-runtime")
async def tool_runtime_snapshot(current_user: User = Depends(get_current_admin)):
    """工具 Harness 运行时快照（熔断器/预算/健康状态，仅管理员）"""
    from app.services.tools.runtime import get_tool_runtime_snapshot

    return get_tool_runtime_snapshot()


@router.get("/monitoring/runs/{run_id}/trace")
async def get_run_trace(
    run_id: str,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """run 级 trace 树：以 run_id 为根，聚合各 agent 节点及其工具调用明细与 Token 成本（仅管理员）"""
    from app.services.token_tracker import token_tracker

    run_result = await db.execute(
        select(EvaluationRun).where(EvaluationRun.id == run_id)
    )
    run = run_result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail={"error_code": "RUN_NOT_FOUND"})

    nodes_result = await db.execute(
        select(EvaluationNodeResult)
        .where(EvaluationNodeResult.run_id == run_id)
        .order_by(EvaluationNodeResult.started_at, EvaluationNodeResult.id)
    )
    nodes = nodes_result.scalars().all()

    return {
        "run": {
            "run_id": run.id,
            "consultation_id": run.consultation_id,
            "evaluation_id": run.evaluation_id,
            "status": run.status,
            "graph_version": run.graph_version,
            "selected_agents": run.selected_agents,
            "error_type": run.error_type,
            "error_message": run.error_message,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        },
        "nodes": [
            {
                "id": n.id,
                "node_name": n.node_name,
                "attempt": n.attempt,
                "status": n.status,
                "duration_ms": n.duration_ms,
                "result_summary": n.result_summary,
                "error_type": n.error_type,
                "started_at": n.started_at.isoformat() if n.started_at else None,
                "finished_at": n.finished_at.isoformat() if n.finished_at else None,
            }
            for n in nodes
        ],
        "node_count": len(nodes),
        # run 级 Token 成本归因（含按 agent 细分；Redis 无记录时为零值）
        "usage": await token_tracker.get_run_usage(run_id),
    }


@router.get("/monitoring/failures/summary")
async def failures_summary(
    days: int = 7,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """失败归因聚合：按标准化失败原因码统计近 N 天评估 run 失败分布（仅管理员）

    原因码体系见 app.core.failure_reasons（timeout/connection_error/cancelled 等），
    error_type 非空即计入失败分布，覆盖失败与被取消的 run。
    """
    days = max(1, min(days, 90))
    since = datetime.utcnow() - timedelta(days=days)

    total_result = await db.execute(
        select(func.count())
        .select_from(EvaluationRun)
        .where(EvaluationRun.started_at >= since)
    )
    total_runs = total_result.scalar_one() or 0

    grouped_result = await db.execute(
        select(EvaluationRun.error_type, func.count())
        .where(EvaluationRun.started_at >= since, EvaluationRun.error_type.isnot(None))
        .group_by(EvaluationRun.error_type)
    )
    by_reason = {row[0]: row[1] for row in grouped_result.all()}

    recent_result = await db.execute(
        select(EvaluationRun)
        .where(EvaluationRun.started_at >= since, EvaluationRun.error_type.isnot(None))
        .order_by(EvaluationRun.started_at.desc())
        .limit(10)
    )
    recent = recent_result.scalars().all()

    failed_runs = sum(by_reason.values())
    return {
        "days": days,
        "total_runs": total_runs,
        "failed_runs": failed_runs,
        "failure_rate": round(failed_runs / total_runs, 4) if total_runs else 0.0,
        "by_reason": by_reason,
        "recent_failures": [
            {
                "run_id": r.id,
                "consultation_id": r.consultation_id,
                "status": r.status,
                "error_type": r.error_type,
                "error_message": (r.error_message or "")[:200] or None,
                "attempt": r.attempt,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            }
            for r in recent
        ],
    }


@router.get("/monitoring/usage/summary")
async def usage_summary(
    days: int = 7,
    current_user: User = Depends(get_current_admin),
):
    """Token 成本聚合：全部 run 的用量汇总/by_agent 排行 + 近 N 天日趋势（仅管理员）

    run 级数据来自 Redis token_usage:run:*（7 天 TTL），Redis 不可用时降级返回零值。
    """
    from app.services.token_tracker import token_tracker

    days = max(1, min(days, 30))
    daily = [
        await token_tracker.get_daily_usage((date.today() - timedelta(days=i)).isoformat())
        for i in range(days)
    ]
    return {
        "days": days,
        "runs": await token_tracker.get_runs_usage_summary(),
        "daily": daily,
    }
