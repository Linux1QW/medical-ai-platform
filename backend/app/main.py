import logging
import time
import traceback
from contextlib import asynccontextmanager
from uuid import uuid4

# Patch starlette BaseHTTPMiddleware to prevent MemoryObjectReceiveStream leaks
# Must be imported before any @app.middleware("http") decorator runs.
import app._starlette_patch  # noqa: F401

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import generate_latest
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text as sa_text
from sqlalchemy.exc import SQLAlchemyError

from app.api.v1 import router as api_v1_router
from app.core.config import settings
from app.core.limiter import limiter
from app.core.logging import setup_logging
from app.db.session import engine
from app.orchestration.adapters import register_all as register_all_adapters
from app.orchestration.checkpointer import close_checkpointer, get_checkpointer, init_checkpointer
from app.services.jwt_blacklist import close_blacklist_redis
from app.services.llm_cache import _get_redis as _get_cache_redis
from app.services.llm_cache import close_cache_redis
from app.services.observability.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
)
from app.services.progress_bus import RedisProgressBus
from app.services.rag.retrieval_cache import close_retrieval_cache_redis

# 初始化结构化日志
setup_logging()
logger = logging.getLogger(__name__)

# ── 速率限制器 ────────────────────────────────────────────────────────────────


# Module-level reference to progress bus (set during lifespan, used by /health/ready)
_progress_bus = None


async def _check_progress_bus() -> bool:
    """检查 progress bus Redis 连通性"""
    return _progress_bus is not None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 生命周期顺序固定：check_security → adapters → checkpointer → progress listener → tool health
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

    # Validate Coach production dependencies when COACH_ENABLED=true
    if settings.COACH_ENABLED and not settings.TESTING:
        from app.services.coach_runtime_factory import validate_production_dependencies

        missing = validate_production_dependencies()
        if missing:
            raise RuntimeError(
                f"Coach enabled but missing dependencies: {', '.join(missing)}"
            )
        logger.info("Coach production dependencies validated.")

    # 初始化 Progress Bus（Redis Pub/Sub 跨进程进度广播）
    import redis.asyncio as aioredis

    from app.core.websocket import init_manager
    global _progress_bus
    progress_redis = aioredis.from_url(settings.PROGRESS_REDIS_URL, decode_responses=True)
    progress_bus = RedisProgressBus(progress_redis, ttl=settings.PROGRESS_EVENT_TTL_SECONDS)
    _progress_bus = progress_bus
    mgr = init_manager(bus=progress_bus)
    await mgr.start()

    # 启动工具健康探测（TOOL_HEALTH_CHECK_ENABLED=false 时无操作）
    from app.services.tools.runtime import start_tool_health_checks, stop_tool_health_checks
    await start_tool_health_checks()

    yield

    # ── Shutdown（反向关闭）──
    # Task 7: 统一 shutdown flush（Langfuse tracer 幂等 flush）
    from app.services.observability.langfuse_client import get_tracer
    try:
        get_tracer().flush()
    except Exception as e:
        logger.debug(f"Langfuse tracer flush on shutdown failed: {e}")

    # 停止工具健康探测
    await stop_tool_health_checks()

    # 关闭 Progress Bus
    from app.core.websocket import get_manager
    mgr = get_manager()
    await mgr.close()
    await progress_redis.aclose()
    _progress_bus = None

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
    context: dict | None = None,
    detail: str | None = None,
) -> JSONResponse:
    request_id = _get_request_id(request)
    content: dict = {
        "error_code": error_code,
        "message": message,
        "detail": detail if detail is not None else message,
        "request_id": request_id,
    }
    if error_type:
        content["error_type"] = error_type
    content["context"] = context
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


# Swagger UI / ReDoc 依赖内联脚本与 CDN 资源，严格 CSP 会将其破坏，故豁免
_CSP_EXEMPT_PATHS = ("/docs", "/redoc", "/openapi.json")


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """统一注入安全响应头（纯 API 后端，默认拒绝一切内嵌/脚本加载）"""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if not request.url.path.startswith(_CSP_EXEMPT_PATHS):
        response.headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
    if settings.ENVIRONMENT == "production":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        error_code = detail.get("error_code", f"HTTP_{exc.status_code}")
        message = detail.get("message", "请求失败")
        error_type = detail.get("error_type")
        context = detail.get("context")
    else:
        error_code = f"HTTP_{exc.status_code}"
        message = str(detail)
        error_type = None
        context = None
    return _error_response(
        request, exc.status_code, error_code, message,
        error_type=error_type, context=context,
    )


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


# ── V1.1 健康端点 ─────────────────────────────────────────────────────────────


@app.get("/health/live")
async def health_live():
    """Liveness probe：只证明进程存活，永远不查询外部依赖"""
    return {"status": "ok", "version": "1.1.0"}


@app.get("/health/ready")
async def health_ready():
    """Readiness probe：检查必需依赖（MySQL、状态 Redis、checkpointer、progress bus）

    - 任一必需依赖不可用 → 503
    - redis-cache 不可用 → 200 + degraded=["cache"]（缓存降级不影响核心功能）
    """
    checks = {}
    degraded = []

    # 1. MySQL（必需）
    try:
        async with engine.connect() as conn:
            await conn.execute(sa_text("SELECT 1"))
        checks["mysql"] = "ok"
    except Exception as e:
        checks["mysql"] = "unavailable"
        logger.warning(f"Readiness: MySQL unavailable: {e}")

    # 2. LangGraph checkpointer（必需；LANGGRAPH_ENABLED=false 时视为 ok）
    checkpointer = get_checkpointer()
    if settings.LANGGRAPH_ENABLED:
        if checkpointer is not None:
            checks["checkpointer"] = "ok"
        else:
            checks["checkpointer"] = "unavailable"
    else:
        checks["checkpointer"] = "disabled"

    # 3. Progress bus（必需）
    if await _check_progress_bus():
        checks["progress_bus"] = "ok"
    else:
        checks["progress_bus"] = "unavailable"

    # 4. redis-cache（非必需，降级不阻止服务）
    try:
        cache_redis = await _get_cache_redis()
        if cache_redis is not None and await cache_redis.ping():
            checks["cache"] = "ok"
        else:
            checks["cache"] = "degraded"
            degraded.append("cache")
    except Exception:
        checks["cache"] = "degraded"
        degraded.append("cache")

    # 必需依赖任一不可用 → 503
    required = [k for k in ("mysql", "checkpointer", "progress_bus") if checks.get(k) == "unavailable"]
    if required:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "checks": checks, "version": settings.VERSION},
        )

    body = {"status": "ok", "checks": checks, "version": settings.VERSION}
    if degraded:
        body["degraded"] = degraded
    return body


@app.get("/health")
async def health_check():
    """兼容旧版健康端点：返回 minimal ready 结果，不暴露敏感指标"""
    checkpointer = get_checkpointer()

    if settings.LANGGRAPH_ENABLED:
        langgraph_status = "available" if checkpointer is not None else "not_available"
    else:
        langgraph_status = "disabled"

    checks = {"mysql": "ok", "redis": "ok"}
    try:
        async with engine.connect() as conn:
            await conn.execute(sa_text("SELECT 1"))
    except Exception as e:
        checks["mysql"] = "unavailable"
        logger.warning(f"Health check: MySQL unavailable: {e}")
    try:
        redis_client = await _get_cache_redis()
        if redis_client is None or not await redis_client.ping():
            checks["redis"] = "unavailable"
    except Exception as e:
        checks["redis"] = "unavailable"
        logger.warning(f"Health check: Redis unavailable: {e}")

    healthy = all(v == "ok" for v in checks.values())
    body = {
        "status": "ok" if healthy else "degraded",
        "checks": checks,
        "version": settings.VERSION,
        "langgraph_enabled": settings.LANGGRAPH_ENABLED,
        "checkpointer": langgraph_status,
    }
    if not healthy:
        return JSONResponse(status_code=503, content=jsonable_encoder(body))
    return body


@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request):
    """Prometheus 指标导出端点（METRICS_TOKEN 非空时需 Bearer 鉴权）"""
    if settings.METRICS_TOKEN:
        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {settings.METRICS_TOKEN}":
            raise HTTPException(status_code=403, detail="Forbidden")
    elif settings.ENVIRONMENT == "production":
        # 生产环境未配置 METRICS_TOKEN 时默认关闭，避免内部指标外泄
        raise HTTPException(status_code=403, detail="Forbidden")
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
