import logging
import time
import traceback
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from contextlib import asynccontextmanager

from app.api.v1 import router as api_v1_router
from app.core.config import settings
from app.db.session import engine
from app.models.base import Base
from app.orchestration.checkpointer import init_checkpointer, close_checkpointer
from app.services.qwen_client import get_llm_metrics
import app.models  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured.")
    try:
        await init_checkpointer(
            redis_url=settings.REDIS_CHECKPOINT_URL,
            ttl=settings.REDIS_CHECKPOINT_TTL,
        )
        yield
    finally:
        await close_checkpointer()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    lifespan=lifespan,
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
    return {
        "status": "ok",
        "version": settings.VERSION,
        "llm": get_llm_metrics(),
    }
