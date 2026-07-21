# -*- coding: utf-8 -*-
"""管理接口 — 缓存清理、数据留存与运维操作"""

from fastapi import APIRouter, Depends

from app.core.deps import get_current_admin
from app.models.user import User
from app.services.llm_cache import LLMResponseCache
from app.services.rag.retrieval_cache import clear_retrieval_cache, get_retrieval_cache_stats

router = APIRouter()


@router.post("/cache/retrieval/clear")
async def clear_cache():
    """清除检索缓存"""
    deleted = await clear_retrieval_cache()
    return {"message": "检索缓存已清除", "deleted": deleted}


@router.get("/cache/retrieval/stats")
async def cache_stats():
    """获取检索缓存统计信息"""
    return await get_retrieval_cache_stats()


@router.get("/cache-stats")
async def cache_stats_all():
    """获取所有缓存详细统计信息（LLM + 检索）"""
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
