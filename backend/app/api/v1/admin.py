# -*- coding: utf-8 -*-
"""管理接口 — 缓存清理与运维操作"""

from fastapi import APIRouter

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
