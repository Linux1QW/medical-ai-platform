import logging
import time
import traceback
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.exc import SQLAlchemyError

from contextlib import asynccontextmanager

from app.api.v1 import router as api_v1_router
from app.core.config import settings
from app.db.session import engine
from app.models.base import Base
from app.orchestration.checkpointer import init_checkpointer, close_checkpointer, get_checkpointer
from app.orchestration.adapters import register_all as register_all_adapters
from app.services.qwen_client import get_llm_metrics
from app.services.llm_cache import LLMResponseCache, close_cache_redis
import app.models  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
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


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    lifespan=lifespan,
)

# ── 速率限制中间件 ─────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def _error_response(request: Request, status_code: int, error_code: str, message: str) -> JSONResponse:
    request_id = _get_request_id(request)
    return JSONResponse(
        status_code=status_code,
        content={
            "error_code": error_code,
            "message": message,
            "detail": message,
            "request_id": request_id,
        },
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
            f"Request failed request_id={request_id} method={request.method} path={request.url.path}\n{traceback.format_exc()}"
        )
        raise
    duration_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        f"Request completed request_id={request_id} method={request.method} path={request.url.path} status={response.status_code} duration_ms={duration_ms}"
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        error_code = detail.get("error_code", f"HTTP_{exc.status_code}")
        message = detail.get("message", "请求失败")
    else:
        error_code = f"HTTP_{exc.status_code}"
        message = str(detail)
    return _error_response(request, exc.status_code, error_code, message)


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

    return {
        "status": "ok",
        "version": settings.VERSION,
        "llm": get_llm_metrics(),
        "llm_cache": await LLMResponseCache.get_stats(),
        "langgraph_enabled": settings.LANGGRAPH_ENABLED,
        "checkpointer": langgraph_status,
    }
