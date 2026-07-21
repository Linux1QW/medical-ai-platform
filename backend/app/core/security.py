import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt_sha256", "bcrypt"], deprecated="auto")


def normalize_password(password: str) -> str:
    return f"sha256${hashlib.sha256(password.encode('utf-8')).hexdigest()}"


def hash_password(password: str) -> str:
    return pwd_context.hash(normalize_password(password))


def verify_password(plain_password: str, hashed_password: str) -> bool:
    normalized_password = normalize_password(plain_password)
    if pwd_context.verify(normalized_password, hashed_password):
        return True
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    jti = str(uuid.uuid4())
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "type": "access", "jti": jti})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """生成 refresh token（更长过期时间，默认 7 天）"""
    to_encode = data.copy()
    jti = str(uuid.uuid4())
    expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh", "jti": jti})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "access":
            return None
        return payload
    except JWTError:
        return None


def verify_refresh_token(token: str) -> Optional[dict]:
    """验证 refresh token，返回 payload 或 None"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "refresh":
            return None
        return payload
    except JWTError:
        return None


def get_token_jti(token: str) -> Optional[str]:
    """从 token 中提取 JTI（不验证过期，仅解码用于黑名单 TTL 计算）"""
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM],
            options={"verify_exp": False},
        )
        return payload.get("jti")
    except JWTError:
        return None


def get_token_remaining_ttl(token: str) -> Optional[int]:
    """获取 token 剩余有效期（秒），用于黑名单 TTL"""
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM],
            options={"verify_exp": False},
        )
        exp = payload.get("exp")
        if exp is None:
            return None
        remaining = int(exp - datetime.now(timezone.utc).timestamp())
        return max(remaining, 1)
    except JWTError:
        return None
