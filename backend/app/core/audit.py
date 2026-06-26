"""审计日志记录模块"""
import logging
from typing import Optional

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.audit_log import AuditLog

logger = logging.getLogger(__name__)


async def record_audit_log(
    db: AsyncSession,
    user_id: Optional[int],
    action: str,
    request: Optional[Request] = None,
    resource_id: Optional[str] = None,
    detail: Optional[str] = None,
) -> None:
    """记录审计日志

    Args:
        db: 数据库会话
        user_id: 操作用户ID（登录失败时可为 None）
        action: 操作类型（login/create_consultation/submit_diagnosis/trigger_evaluation/admin_action）
        request: FastAPI 请求对象（用于提取 IP 和 UA）
        resource_id: 关联资源ID
        detail: 操作详情（禁止记录密码等敏感信息）
    """
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
        # 审计日志写入失败不应影响主流程，仅记录警告
        logger.warning(f"Audit log write failed: {e}")
