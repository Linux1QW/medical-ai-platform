import logging
import time
import traceback
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from slowapi.errors import RateLimitExceeded
from sqlalchemy.exc import SQLAlchemyError

from contextlib import asynccontextmanager

from app.api.v1 import router as api_v1_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.session import engine
from app.models.base import Base
from app.orchestration.checkpointer import init_checkpointer, close_checkpointer, get_checkpointer
from app.orchestration.adapters import register_all as register_all_adapters
from app.services.qwen_client import get_llm_metrics
from app.services.llm_cache import LLMResponseCache, close_cache_redis
from app.services.rag.retrieval_cache import close_retrieval_cache_redis, get_retrieval_cache_stats
from app.services.token_tracker import token_tracker
from app.services.observability.metrics import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION,
    CACHE_HIT_RATE,
)
from prometheus_client import generate_latest
from app.services.jwt_blacklist import close_blacklist_redis
import app.models  # noqa: F401

# 初始化结构化日志
setup_logging()
logger = logging.getLogger(__name__)

from app.core.limiter import limiter

# ── 速率限制器 ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured.")

    # 安全检查
    settings.check_security()

    # 注册所有 Agent 适配器（LangGraph dispatch 依赖）
    register_all_adapters()
    logger.info("Agent adapters registered.")

    # 初始化 checkpointer（LANGGRAPH_ENABLED=false 时返回 None）
    # LANGGRAPH_ENABLED=true 但 Redis 失败时会抛出 RuntimeError，阻止服务启动
    await init_checkpointer(
        redis_url=settings.REDIS_CHECKPOINT_URL,
        ttl=settings.REDIS_CHECKPOINT_TTL,
    )

    yield

    # 关闭 checkpointer（None 时无操作）
    await close_checkpointer()
    # 关闭 LLM 缓存 Redis 连接
    await close_cache_redis()
    # 关闭检索缓存 Redis 连接
    await close_retrieval_cache_redis()
    # 关闭 JWT 黑名单 Redis 连接
    await close_blacklist_redis()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    lifespan=lifespan,
)

# ── 速率限制中间件 ─────────────────────────────────────────────────────────────
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error_code": "RATE_LIMIT_EXCEEDED",
            "message": "请求过于频繁，请稍后再试",
            "detail": f"速率限制：{exc.detail}",
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def _error_response(
    request: Request,
    status_code: int,
    error_code: str,
    message: str,
    error_type: str | None = None,
) -> JSONResponse:
    request_id = _get_request_id(request)
    content: dict = {
        "error_code": error_code,
        "message": message,
        "detail": message,
        "request_id": request_id,
    }
    if error_type:
        content["error_type"] = error_type
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers={"X-Request-ID": request_id},
    )


@app.middleware("http")
async def request_log_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.error(
            "Request failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
            exc_info=True,
        )
        raise
    duration_ms = int((time.perf_counter() - start) * 1000)
    duration_s = time.perf_counter() - start
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    # ── Prometheus HTTP 指标 ──
    _path = request.url.path
    _method = request.method
    _status = str(response.status_code)
    HTTP_REQUESTS_TOTAL.labels(method=_method, path=_path, status=_status).inc()
    HTTP_REQUEST_DURATION.labels(method=_method, path=_path).observe(duration_s)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        error_code = detail.get("error_code", f"HTTP_{exc.status_code}")
        message = detail.get("message", "请求失败")
        error_type = detail.get("error_type")
    else:
        error_code = f"HTTP_{exc.status_code}"
        message = str(detail)
        error_type = None
    return _error_response(request, exc.status_code, error_code, message, error_type=error_type)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        f"Validation failed request_id={_get_request_id(request)} method={request.method} path={request.url.path} errors={exc.errors()}"
    )
    return _error_response(request, 422, "VALIDATION_ERROR", "请求参数不合法")


@app.exception_handler(SQLAlchemyError)
async def db_exception_handler(request: Request, exc: SQLAlchemyError):
    logger.error(
        f"Database failed request_id={_get_request_id(request)} method={request.method} path={request.url.path}\n{traceback.format_exc()}"
    )
    return _error_response(request, 503, "DB_UNAVAILABLE", "数据库服务暂不可用")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled error request_id={_get_request_id(request)} method={request.method} path={request.url.path}\n{traceback.format_exc()}"
    )
    return _error_response(request, 500, "INTERNAL_SERVER_ERROR", "服务器内部错误，请稍后重试")


app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)


@app.get("/health")
async def health_check():
    checkpointer = get_checkpointer()

    # 根据 LANGGRAPH_ENABLED 和 checkpointer 状态返回健康信息
    if settings.LANGGRAPH_ENABLED:
        if checkpointer is not None:
            langgraph_status = "available"
        else:
            langgraph_status = "not_available"
    else:
        langgraph_status = "disabled"

    llm_cache_stats = await LLMResponseCache.get_stats()
    retrieval_cache_stats = await get_retrieval_cache_stats()

    # ── 更新 Prometheus 缓存命中率 Gauge ──
    try:
        CACHE_HIT_RATE.labels(cache="llm").set(llm_cache_stats.get("hit_rate", 0))
        CACHE_HIT_RATE.labels(cache="retrieval").set(retrieval_cache_stats.get("hit_rate", 0))
    except Exception:
        pass

    return {
        "status": "ok",
        "version": settings.VERSION,
        "llm": get_llm_metrics(),
        "llm_cache": llm_cache_stats,
        "retrieval_cache": retrieval_cache_stats,
        "token_usage": await token_tracker.get_summary(),
        "langgraph_enabled": settings.LANGGRAPH_ENABLED,
        "checkpointer": langgraph_status,
    }


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus 指标导出端点"""
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
