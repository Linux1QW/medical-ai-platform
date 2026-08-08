"""Celery tasks for immutable RAG index generations."""

import asyncio
import hashlib
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import redis

from app.celery_app import celery_app
from app.core.config import settings

logger = logging.getLogger(__name__)

RAG_INDEX_BUILD_LOCK = "rag:index-build-lock"
RAG_INDEX_SWITCHED_CHANNEL = "rag:index-switched"
INDEX_BUILD_PHASES = (
    "snapshot",
    "parse",
    "chunk",
    "embed",
    "chroma",
    "bm25",
    "sparse",
    "validate",
    "switch",
    "publish",
)
INDEX_BUILD_LOCK_TTL = 30 * 60
INDEX_SWITCH_PUBLISH_ATTEMPTS = 3
_RENEW_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return 0
"""
_RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


class RedisIndexBuildLock:
    """A Redis ownership lock with compare-and-set renewal/release."""

    def __init__(self, redis_client: Any, *, ttl: int = INDEX_BUILD_LOCK_TTL) -> None:
        self.redis = redis_client
        self.ttl = ttl

    def acquire(self, task_id: str) -> bool:
        return bool(
            self.redis.set(
                RAG_INDEX_BUILD_LOCK,
                task_id,
                nx=True,
                ex=self.ttl,
            )
        )

    def renew(self, task_id: str) -> bool:
        return bool(
            self.redis.eval(
                _RENEW_SCRIPT,
                1,
                RAG_INDEX_BUILD_LOCK,
                task_id,
                self.ttl,
            )
        )

    def is_owned_by(self, task_id: str) -> bool:
        return self.redis.get(RAG_INDEX_BUILD_LOCK) == task_id

    def release(self, task_id: str) -> bool:
        return bool(
            self.redis.eval(
                _RELEASE_SCRIPT,
                1,
                RAG_INDEX_BUILD_LOCK,
                task_id,
            )
        )


class LostIndexBuildLock(RuntimeError):
    """Raised when a task no longer owns the distributed build lock."""


def _publish_switch_event_with_retries(
    publish: Any,
    *,
    generation: str,
    previous: Optional[str],
    manifest_sha256: str,
    redis_client: Any,
) -> dict[str, Any]:
    """Best-effort notification after the active pointer has been committed.

    Once CAS succeeds, notification failures must not turn the task into a
    misleading FAILURE: request paths reconcile against the durable active
    pointer.  The result still exposes the warning so operators can replay or
    restart listeners if all bounded attempts fail.
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, INDEX_SWITCH_PUBLISH_ATTEMPTS + 1):
        try:
            publish(
                generation,
                previous,
                manifest_sha256,
                redis=redis_client,
            )
            return {"ok": True, "attempts": attempt}
        except Exception as exc:
            last_error = exc
            logger.warning(
                "RAG generation %s switch notification attempt %s/%s failed: %s",
                generation,
                attempt,
                INDEX_SWITCH_PUBLISH_ATTEMPTS,
                exc,
            )
    return {
        "ok": False,
        "attempts": INDEX_SWITCH_PUBLISH_ATTEMPTS,
        "error": str(last_error) if last_error is not None else "unknown error",
    }


def _get_redis() -> Any:
    return redis.Redis.from_url(
        settings.REDIS_CHECKPOINT_URL,
        decode_responses=True,
        socket_connect_timeout=3.0,
        socket_timeout=3.0,
    )


def _heartbeat(
    lock: RedisIndexBuildLock,
    task_id: str,
    stop_event: threading.Event,
    lost_lock_event: threading.Event,
) -> None:
    interval = max(1.0, lock.ttl / 3)
    while not stop_event.wait(interval):
        try:
            renewed = lock.renew(task_id)
        except Exception:
            logger.exception("RAG index lock renewal failed: task_id=%s", task_id)
            lost_lock_event.set()
            return
        if not renewed:
            logger.warning("RAG index lock renewal failed: task_id=%s", task_id)
            lost_lock_event.set()
            return


def _set_phase(task: Any, phase: str, **metadata: Any) -> None:
    task.update_state(
        state="PROGRESS",
        meta={"phase": phase, "status": "running", **metadata},
    )


def _manifest_sha256(manifest: Any, artifact_root: Any) -> str:
    from app.services.rag.indexing.manifest import manifest_path

    return hashlib.sha256(
        manifest_path(manifest.index_generation, artifact_root).read_bytes()
    ).hexdigest()


def _server_source_id(pdf_path: Path, root: Path) -> str:
    relative = pdf_path.resolve().relative_to(root.resolve()).as_posix()
    return f"source-{hashlib.sha256(relative.encode('utf-8')).hexdigest()[:24]}"


def _resolve_worker_pdf_path(pdf_path: str, pdf_dir: Path) -> Path:
    root = pdf_dir.resolve()
    resolved = Path(pdf_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("RAG source file must remain inside PDF_DIR")
    if not resolved.is_file():
        raise FileNotFoundError(f"RAG source file does not exist: {resolved}")
    return resolved


def _raise_if_lock_lost(
    lock: RedisIndexBuildLock,
    task_id: str,
    lost_lock_event: threading.Event,
    *,
    verify_owner: bool = False,
) -> None:
    if lost_lock_event.is_set() or (verify_owner and not lock.is_owned_by(task_id)):
        lost_lock_event.set()
        raise LostIndexBuildLock(
            f"RAG index task {task_id!r} lost the distributed build lock"
        )


async def _build_candidate(
    task: Any,
    *,
    operation: str,
    pdf_path: Optional[str] = None,
    force_replace: bool = False,
    source_name: Optional[str] = None,
    redis_client: Any,
    lock: RedisIndexBuildLock,
    task_id: str,
    lost_lock_event: threading.Event,
) -> dict[str, Any]:
    from app.services.rag.indexing import builder
    from app.services.rag.indexing.manifest import validate_candidate_manifest
    from app.services.rag.indexing.versioning import (
        _get_generation_redis,
        activate_candidate_generation,
        get_active_index_generation,
        publish_index_switched,
    )
    from app.services.rag.medical_store import get_medical_store

    resolved_path = (
        _resolve_worker_pdf_path(pdf_path, builder.PDF_DIR) if pdf_path else None
    )
    artifact_root = Path(settings.BM25_ARTIFACT_ROOT)
    _raise_if_lock_lost(lock, task_id, lost_lock_event)
    _set_phase(task, "snapshot", operation=operation)
    generation_redis = await _get_generation_redis()
    active_generation = await get_active_index_generation(redis=generation_redis)
    if not active_generation:
        active_generation = str(getattr(settings, "ACTIVE_INDEX_VERSION", "rag-v1"))

    def report_phase(phase: str) -> None:
        _raise_if_lock_lost(lock, task_id, lost_lock_event)
        _set_phase(task, phase, operation=operation)

    if operation == "rebuild":
        pdf_root = builder.PDF_DIR.resolve()
        document_paths = sorted(
            path
            for path in pdf_root.iterdir()
            if path.is_file() and path.suffix.lower() in builder.SUPPORTED_EXTENSIONS
        )
        manifest = await builder.build_full_index_candidate(
            document_paths=document_paths,
            source_names={
                path: _server_source_id(path, pdf_root) for path in document_paths
            },
            artifact_root=artifact_root,
            phase_callback=report_phase,
        )
    elif operation in {"add", "replace"}:
        if resolved_path is None:
            raise ValueError("incremental RAG operation requires a PDF path")
        manifest = await builder.build_incremental_index_candidate(
            resolved_path,
            active_generation=active_generation,
            force_replace=force_replace,
            source_name=source_name,
            artifact_root=artifact_root,
            phase_callback=report_phase,
        )
    elif operation == "delete":
        if not source_name:
            raise ValueError("delete operation requires a source name")
        active_documents = get_medical_store().export_generation_documents(
            active_generation
        )
        if not any(
            builder._record_source(document) == source_name
            for document in active_documents
        ):
            raise ValueError(f"source {source_name!r} was not found")
        records = builder.build_candidate_snapshot(
            active_documents,
            [],
            source_name=source_name,
            force_replace=True,
        )
        manifest = builder._publish_candidate_generation(
            records,
            artifact_root=artifact_root,
            phase_callback=report_phase,
        )
    else:
        raise ValueError(f"unsupported RAG index operation: {operation}")

    _raise_if_lock_lost(lock, task_id, lost_lock_event)
    _set_phase(
        task,
        "validate",
        operation=operation,
        generation=manifest.index_generation,
    )
    validate_candidate_manifest(manifest)
    digest = _manifest_sha256(manifest, artifact_root)
    _raise_if_lock_lost(lock, task_id, lost_lock_event, verify_owner=True)
    _set_phase(task, "switch", generation=manifest.index_generation)
    switched = await activate_candidate_generation(
        manifest,
        expected_generation=active_generation,
        redis=generation_redis,
        artifact_root=artifact_root,
    )
    try:
        _set_phase(task, "publish", generation=manifest.index_generation)
    except Exception:
        logger.warning(
            "RAG generation %s switched but publish phase state could not be recorded",
            manifest.index_generation,
            exc_info=True,
        )
    notification = _publish_switch_event_with_retries(
        publish_index_switched,
        generation=manifest.index_generation,
        previous=switched.get("previous", active_generation),
        manifest_sha256=digest,
        redis_client=redis_client,
    )
    return {
        "status": "completed" if notification["ok"] else "completed_with_warning",
        "operation": operation,
        "generation": manifest.index_generation,
        "previous": switched.get("previous", active_generation),
        "manifest": manifest.model_dump(mode="json"),
        "manifest_sha256": digest,
        "validation": {"ok": True, "chunk_count": manifest.chunk_count},
        "switch": switched,
        "notification": notification,
        "phase": "publish",
    }


def _run_index_task(
    task: Any,
    *,
    operation: str,
    pdf_path: Optional[str] = None,
    force_replace: bool = False,
    source_name: Optional[str] = None,
) -> dict[str, Any]:
    task_id = str(getattr(task.request, "id", "rag-index-task"))
    redis_client = _get_redis()
    lock = RedisIndexBuildLock(redis_client)
    if not lock.acquire(task_id):
        raise RuntimeError("another RAG index build is already running")
    stop_event = threading.Event()
    lost_lock_event = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=(lock, task_id, stop_event, lost_lock_event),
        name="rag-index-lock-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    try:
        return asyncio.run(
            _build_candidate(
                task,
                operation=operation,
                pdf_path=pdf_path,
                force_replace=force_replace,
                source_name=source_name,
                redis_client=redis_client,
                lock=lock,
                task_id=task_id,
                lost_lock_event=lost_lock_event,
            )
        )
    except Exception as exc:
        task.update_state(state="FAILURE", meta={"phase": "failed", "error": str(exc)})
        raise
    finally:
        stop_event.set()
        heartbeat.join(timeout=2.0)
        lock.release(task_id)


@celery_app.task(bind=True, name="rebuild_rag_index")
def rebuild_rag_index(self: Any) -> dict[str, Any]:
    return _run_index_task(self, operation="rebuild")


@celery_app.task(bind=True, name="add_rag_index")
def add_rag_index(
    self: Any,
    pdf_path: str,
    force_replace: bool = False,
    source_name: Optional[str] = None,
) -> dict[str, Any]:
    return _run_index_task(
        self,
        operation="replace" if force_replace else "add",
        pdf_path=pdf_path,
        force_replace=force_replace,
        source_name=source_name,
    )


@celery_app.task(bind=True, name="replace_rag_index")
def replace_rag_index(
    self: Any,
    pdf_path: str,
    source_name: Optional[str] = None,
) -> dict[str, Any]:
    return _run_index_task(
        self,
        operation="replace",
        pdf_path=pdf_path,
        force_replace=True,
        source_name=source_name,
    )


@celery_app.task(bind=True, name="delete_rag_index")
def delete_rag_index(self: Any, source_name: str) -> dict[str, Any]:
    return _run_index_task(self, operation="delete", source_name=source_name)
