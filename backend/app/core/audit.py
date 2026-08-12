"""审计日志记录模块"""
import logging
import re
from typing import Optional

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.audit_log import AuditLog

logger = logging.getLogger(__name__)

# 用于脱敏的 pattern：数据库连接串
_DB_CONN_PATTERN = re.compile(
    r"(mysql|postgresql|redis)://[^\s]+", re.IGNORECASE
)


def _sanitize_error_message(msg: str) -> str:
    """脱敏错误消息 — 移除数据库连接串和敏感信息"""
    sanitized = _DB_CONN_PATTERN.sub("<redacted>", msg)
    # 截断过长的消息
    if len(sanitized) > 200:
        sanitized = sanitized[:200] + "..."
    return sanitized


async def record_audit_log(
    db: Optional[AsyncSession],
    user_id: Optional[int],
    action: str,
    request: Optional[Request] = None,
    resource_id: Optional[str] = None,
    detail: Optional[str] = None,
    *,
    strict: bool = False,
) -> None:
    """记录审计日志

    Args:
        db: 数据库会话
        user_id: 操作用户ID（登录失败时可为 None）
        action: 操作类型（login/create_consultation/submit_diagnosis/trigger_evaluation/admin_action）
        request: FastAPI 请求对象（用于提取 IP 和 UA）
        resource_id: 关联资源ID
        detail: 操作详情（禁止记录密码等敏感信息）
        strict: 严格模式 — flush 失败时原样抛出供上层 rollback；
                默认 False — flush 失败只记录安全 warning（不含 PII）
    """
    if db is None:
        return
    if not settings.AUDIT_LOG_ENABLED:
        return

    ip_address = None
    user_agent = None
    if request:
        # 优先从 X-Forwarded-For 获取真实 IP
        ip_address = request.headers.get("X-Forwarded-For", request.client.host if request.client else None)
        user_agent = request.headers.get("User-Agent", "")[:500]

    log_entry = AuditLog(
        user_id=user_id,
        action=action,
        resource_id=str(resource_id) if resource_id else None,
        ip_address=ip_address,
        user_agent=user_agent,
        detail=detail,
    )
    db.add(log_entry)
    try:
        await db.flush()
    except Exception as e:
        if strict:
            # strict 模式：原样抛出供上层 rollback
            raise
        # 默认模式：记录安全 warning（不含 PII、不含数据库连接串）
        safe_msg = _sanitize_error_message(str(e))
        logger.warning(f"Audit log flush failed: {safe_msg}")
