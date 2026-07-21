# -*- coding: utf-8 -*-
"""医学知识库管理 API — 支持增量索引、来源查询与删除"""

import asyncio
import logging
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.deps import get_current_admin
from app.models.user import User
from app.services.rag.build_medical_index import (
    PDF_DIR,
    get_indexed_sources,
    index_single_pdf,
)
from app.services.rag.embeddings import clear_embed_cache, get_embed_cache_stats
from app.services.rag.medical_store import get_medical_store

logger = logging.getLogger(__name__)
router = APIRouter()


# ── 响应 Schema ────────────────────────────────────────────────────────────────

class KBStatsResponse(BaseModel):
    total_chunks: int
    total_sources: int
    sources: List[dict]
    embed_cache: dict


class AddPDFRequest(BaseModel):
    filename: str           # PDF 文件名（相对于 data/ 目录），如 "内科学第10版.pdf"
    force_replace: bool = False  # True 则先删旧索引再重建


class AddPDFResponse(BaseModel):
    source: str
    status: str             # "added" | "skipped" | "replaced"
    chunks: int
    message: str


class DeleteSourceResponse(BaseModel):
    source: str
    deleted_chunks: int
    message: str


# ── 后台任务 ───────────────────────────────────────────────────────────────────

_rebuild_lock = asyncio.Lock()
_rebuild_status: dict = {"running": False, "progress": "", "error": ""}


async def _run_rebuild_task():
    """后台全量重建任务（加锁防止并发）"""
    async with _rebuild_lock:
        _rebuild_status.update({"running": True, "progress": "开始全量重建...", "error": ""})
        target_version = settings.ACTIVE_INDEX_VERSION
        try:
            from app.services.rag.build_medical_index import build_medical_index, switch_index_version
            clear_embed_cache()
            await build_medical_index(target_version=target_version)
            store = get_medical_store()
            # 重建完成后自动切换到新版本
            await switch_index_version(target_version)
            _rebuild_status["progress"] = f"重建完成，已自动切换到 {target_version}，共 {store.count()} 条文档"
            logger.info(f"全量重建完成，已自动切换到 {target_version}")
        except Exception as e:
            logger.error(f"知识库全量重建失败: {e}")
            _rebuild_status["error"] = str(e)
        finally:
            _rebuild_status["running"] = False


# ── 路由 ───────────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=KBStatsResponse, summary="获取知识库统计信息")
async def get_kb_stats(_: User = Depends(get_current_admin)):
    """返回知识库文档总量、已索引来源列表及 Embedding 缓存状态。"""
    store = get_medical_store()
    total_chunks = store.count()
    sources = await get_indexed_sources()
    return KBStatsResponse(
        total_chunks=total_chunks,
        total_sources=len(sources),
        sources=sources,
        embed_cache=get_embed_cache_stats(),
    )


@router.post("/add-pdf", response_model=AddPDFResponse, summary="增量添加单个 PDF")
async def add_pdf(
    body: AddPDFRequest,
    _: User = Depends(get_current_admin),
):
    """对 data/ 目录下的单个 PDF 执行增量索引。

    - 若该文件已有索引且 force_replace=False，则跳过并返回 status="skipped"
    - 若 force_replace=True，则先删除旧索引再重建
    """
    pdf_path = PDF_DIR / body.filename
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"文件不存在: {body.filename}（PDF 目录: {PDF_DIR}）",
        )
    if not body.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 .pdf 文件")

    try:
        result = await index_single_pdf(pdf_path, force_replace=body.force_replace)
    except Exception as e:
        logger.error(f"增量索引失败 [{body.filename}]: {e}")
        raise HTTPException(status_code=500, detail=f"索引失败: {e}")

    # 增量操作完成后重建 BM25 索引
    if result["status"] in ("added", "replaced"):
        try:
            from app.services.rag.bm25_search import rebuild_bm25_index
            await asyncio.to_thread(rebuild_bm25_index)
            logger.info("增量索引完成，BM25 索引已重建")
        except Exception as e:
            logger.warning(f"BM25 索引重建失败（非致命）: {e}")

    status_msg = {
        "added": f"成功新增 {result['chunks']} 个文本块",
        "skipped": f"已有索引（{result['chunks']} 块），跳过",
        "replaced": f"已替换旧索引，新增 {result['chunks']} 个文本块",
    }.get(result["status"], "完成")

    return AddPDFResponse(
        source=result["source"],
        status=result["status"],
        chunks=result["chunks"],
        message=status_msg,
    )


@router.delete(
    "/sources/{source_name:path}",
    response_model=DeleteSourceResponse,
    summary="删除指定来源的全部索引",
)
async def delete_source(
    source_name: str,
    _: User = Depends(get_current_admin),
):
    """删除指定来源文件的全部文档块索引。source_name 为文件名，如 '内科学第10版.pdf'。"""
    store = get_medical_store()
    deleted = store.delete_by_source(source_name)
    if deleted == 0:
        raise HTTPException(
            status_code=404, detail=f"未找到来源 '{source_name}' 的任何索引"
        )

    # 删除操作完成后重建 BM25 索引
    try:
        from app.services.rag.bm25_search import rebuild_bm25_index
        await asyncio.to_thread(rebuild_bm25_index)
        logger.info("删除操作完成，BM25 索引已重建")
    except Exception as e:
        logger.warning(f"BM25 索引重建失败（非致命）: {e}")

    return DeleteSourceResponse(
        source=source_name,
        deleted_chunks=deleted,
        message=f"已删除 {deleted} 个文本块",
    )


@router.post("/rebuild", summary="触发全量重建（后台异步执行）")
async def rebuild_index(
    background_tasks: BackgroundTasks,
    _: User = Depends(get_current_admin),
):
    """触发后台全量重建任务。重建期间可通过 GET /stats 查看进度。"""
    if _rebuild_status["running"]:
        raise HTTPException(status_code=409, detail="重建任务正在运行中，请稍后再试")
    background_tasks.add_task(_run_rebuild_task)
    return {"message": "全量重建任务已启动，可通过 GET /knowledge-base/stats 查看进度"}


@router.get("/rebuild/status", summary="查询重建任务状态")
async def get_rebuild_status(_: User = Depends(get_current_admin)):
    """返回后台重建任务的当前状态。"""
    return _rebuild_status


@router.post("/cache/clear", summary="清空 Embedding 缓存")
async def clear_cache(_: User = Depends(get_current_admin)):
    """清空内存中的 Embedding LRU 缓存（重建索引后可调用以释放内存）。"""
    clear_embed_cache()
    return {"message": "Embedding 缓存已清空"}
