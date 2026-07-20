"""细粒度 RBAC 权限检查"""

from fastapi import Depends, HTTPException

from app.core.deps import get_current_user
from app.models.user import User

# 预定义角色权限映射
PERMISSIONS: dict[str, list[str]] = {
    "admin": [
        "evaluation:create", "evaluation:view", "evaluation:review",
        "consultation:create", "consultation:view",
        "patient:create", "patient:view", "patient:export",
        "user:manage", "system:manage", "model:manage",
    ],
    "doctor": [
        "evaluation:create", "evaluation:view",
        "consultation:create", "consultation:view",
        "patient:view",
    ],
}


def get_user_permissions(user: User) -> list[str]:
    """获取用户权限列表

    优先使用用户自定义 permissions 字段，
    若未设置则回退到角色默认权限。
    """
    if user.permissions:
        return user.permissions
    return PERMISSIONS.get(user.role, [])


def require_permission(permission: str):
    """权限检查依赖

    用法:
        current_user: User = require_permission("evaluation:create")
    """
    async def checker(
        user: User = Depends(get_current_user),
    ) -> User:
        user_perms = get_user_permissions(user)
        if permission not in user_perms:
            raise HTTPException(
                status_code=403,
                detail={
                    "error_code": "AUTH_PERMISSION_DENIED",
                    "message": f"权限不足，需要 {permission} 权限",
                },
            )
        return user

    return Depends(checker)
