# -*- coding: utf-8 -*-
"""医学知识库管理 API — 只负责校验请求并投递 Celery 索引任务。"""

import logging
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, List, Optional

try:
    from celery.result import AsyncResult
except ModuleNotFoundError:  # pragma: no cover - dependency-light test fallback
    from app.celery_app import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.celery_app import celery_app
from app.core.deps import get_current_admin
from app.models.user import User
from app.services.rag.build_medical_index import PDF_DIR, get_indexed_sources
from app.services.rag.embeddings import clear_embed_cache, get_embed_cache_stats
from app.services.rag.indexing.manifest import load_rag_index_manifest
from app.services.rag.indexing.versioning import get_active_index_generation
from app.services.rag.medical_store import get_medical_store

logger = logging.getLogger(__name__)
router = APIRouter()

SUPPORTED_DOCUMENT_EXTENSIONS = (".pdf", ".docx")


class KBStatsResponse(BaseModel):
    total_chunks: int
    total_sources: int
    sources: List[dict]
    embed_cache: dict


class AddPDFRequest(BaseModel):
    filename: str
    force_replace: bool = False


class IndexTaskResponse(BaseModel):
    task_id: str
    status: str
    operation: str
    source_id: Optional[str] = None


def _reject_unsafe_relative_path(value: str) -> None:
    """Reject absolute, drive-qualified, and traversal paths on either OS."""
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=400, detail="文件路径不能为空")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise HTTPException(status_code=400, detail="不接受绝对文件路径")
    if ".." in posix.parts or ".." in windows.parts:
        raise HTTPException(status_code=400, detail="文件路径不能包含 ..")


def _resolve_pdf_path(filename: str) -> Path:
    """Resolve a requested document and enforce the PDF_DIR boundary."""
    _reject_unsafe_relative_path(filename)
    requested = Path(filename)
    if requested.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 .pdf 或 .docx 文件")

    root = PDF_DIR.resolve()
    resolved = (root / requested).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="文件必须位于 PDF_DIR 内") from exc
    if not resolved.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"文件不存在: {filename}（PDF 目录: {root}）",
        )
    return resolved


def _source_id_for_path(pdf_path: Path) -> str:
    """Create a stable opaque source ID without trusting request path text."""
    root = PDF_DIR.resolve()
    relative = pdf_path.resolve().relative_to(root).as_posix()
    digest = sha256(relative.encode("utf-8")).hexdigest()[:24]
    return f"source-{digest}"


def _queued_response(result: Any, operation: str, source_id: str | None = None) -> dict:
    return {
        "task_id": str(result.id),
        "status": "queued",
        "operation": operation,
        "source_id": source_id,
    }


def _manifest_payload(generation: str | None) -> dict | None:
    if not generation:
        return None
    try:
        return load_rag_index_manifest(generation).model_dump(mode="json")
    except Exception:
        logger.warning("无法读取 RAG generation manifest: %s", generation)
        return None


@router.get("/stats", response_model=KBStatsResponse, summary="获取知识库统计信息")
async def get_kb_stats(_: User = Depends(get_current_admin)):
    store = get_medical_store()
    generation = await get_active_index_generation()
    if generation is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG active generation is unavailable",
        )
    collection = store.get_collection_for_generation(generation)
    sources = await get_indexed_sources(generation=generation)
    return KBStatsResponse(
        total_chunks=collection.count(),
        total_sources=len(sources),
        sources=sources,
        embed_cache=get_embed_cache_stats(),
    )


@router.post(
    "/add-pdf",
    response_model=IndexTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="异步添加或替换单个 PDF",
)
async def add_pdf(
    body: AddPDFRequest,
    _: User = Depends(get_current_admin),
):
    pdf_path = _resolve_pdf_path(body.filename)
    source_id = _source_id_for_path(pdf_path)
    from app.tasks import rag_index_task

    if body.force_replace:
        result = rag_index_task.replace_rag_index.delay(
            str(pdf_path), source_name=source_id
        )
    else:
        result = rag_index_task.add_rag_index.delay(
            str(pdf_path), source_name=source_id
        )
    return _queued_response(
        result,
        "replace" if body.force_replace else "add",
        source_id,
    )


@router.delete(
    "/sources/{source_name:path}",
    response_model=IndexTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="异步删除指定来源的全部索引",
)
async def delete_source(
    source_name: str,
    _: User = Depends(get_current_admin),
):
    _reject_unsafe_relative_path(source_name)
    from app.tasks import rag_index_task

    result = rag_index_task.delete_rag_index.delay(source_name)
    return _queued_response(result, "delete", source_name)


@router.post(
    "/rebuild",
    response_model=IndexTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="异步触发全量重建",
)
async def rebuild_index(_: User = Depends(get_current_admin)):
    from app.tasks import rag_index_task

    result = rag_index_task.rebuild_rag_index.delay()
    return _queued_response(result, "rebuild")


@router.get("/rebuild/status", summary="查询 Celery 重建任务状态")
async def get_rebuild_status(
    task_id: str | None = Query(default=None),
    _: User = Depends(get_current_admin),
):
    if task_id:
        result = AsyncResult(task_id, app=celery_app)
        info = result.info if isinstance(result.info, dict) else {}
        state = result.state if isinstance(result.state, str) else "PENDING"
        task_status = result.status if isinstance(result.status, str) else state
        response = {
            "task_id": task_id,
            "state": state,
            "status": task_status,
            **info,
        }
        if state == "SUCCESS":
            response["result"] = result.result
        elif state == "FAILURE":
            response["error"] = str(result.result)
        generation = response.get("generation")
        manifest = _manifest_payload(generation)
        if manifest is not None:
            response["manifest"] = manifest
        return response

    try:
        generation = await get_active_index_generation()
    except Exception:
        logger.warning("读取 active RAG generation 失败", exc_info=True)
        generation = None
    return {
        "task_id": None,
        "state": "IDLE",
        "status": "IDLE",
        "generation": generation,
        "manifest": _manifest_payload(generation),
    }


@router.post("/cache/clear", summary="清空 Embedding 缓存")
async def clear_cache(_: User = Depends(get_current_admin)):
    clear_embed_cache()
    return {"message": "Embedding 缓存已清空"}
