"""模型版本注册表 API"""

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.permissions import require_permission
from app.db.session import get_db
from app.models.model_version import ModelVersion

router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────

class ModelVersionCreate(BaseModel):
    name: str
    version: str
    config_json: Optional[dict] = None
    description: Optional[str] = None


class ModelVersionOut(BaseModel):
    id: str
    name: str
    version: str
    config_json: Optional[dict] = None
    status: str
    description: Optional[str] = None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


# ── API ──────────────────────────────────────────────────────────────────────

@router.get("/", response_model=List[ModelVersionOut])
async def list_model_versions(
    name: Optional[str] = Query(None, description="按模型名称筛选"),
    status: Optional[str] = Query(None, description="按状态筛选"),
    db: AsyncSession = Depends(get_db),
):
    """列出所有模型版本"""
    stmt = select(ModelVersion).order_by(ModelVersion.created_at.desc())
    if name:
        stmt = stmt.where(ModelVersion.name == name)
    if status:
        stmt = stmt.where(ModelVersion.status == status)
    result = await db.execute(stmt)
    versions = result.scalars().all()
    return [_version_to_dict(v) for v in versions]


@router.post("/", response_model=ModelVersionOut, status_code=201)
async def register_model_version(
    data: ModelVersionCreate,
    db: AsyncSession = Depends(get_db),
    _=require_permission("model:manage"),
):
    """注册新模型版本"""
    # 检查 name+version 唯一性
    existing = await db.execute(
        select(ModelVersion).where(
            ModelVersion.name == data.name,
            ModelVersion.version == data.version,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail={"error_code": "MODEL_VERSION_EXISTS", "message": "该模型版本已存在"},
        )

    mv = ModelVersion(
        id=str(uuid.uuid4()),
        name=data.name,
        version=data.version,
        config_json=data.config_json,
        description=data.description,
    )
    db.add(mv)
    await db.commit()
    await db.refresh(mv)
    return _version_to_dict(mv)


@router.get("/{name}/active", response_model=ModelVersionOut)
async def get_active_version(
    name: str,
    db: AsyncSession = Depends(get_db),
):
    """获取某模型的活跃版本"""
    result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.name == name,
            ModelVersion.status == "active",
        ).order_by(ModelVersion.created_at.desc()).limit(1)
    )
    mv = result.scalar_one_or_none()
    if not mv:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "MODEL_VERSION_NOT_FOUND", "message": f"模型 {name} 无活跃版本"},
        )
    return _version_to_dict(mv)


@router.put("/{version_id}/deprecate", response_model=ModelVersionOut)
async def deprecate_version(
    version_id: str,
    db: AsyncSession = Depends(get_db),
    _=require_permission("model:manage"),
):
    """标记模型版本为废弃"""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == version_id)
    )
    mv = result.scalar_one_or_none()
    if not mv:
        raise HTTPException(status_code=404, detail="模型版本不存在")

    mv.status = "deprecated"
    await db.commit()
    await db.refresh(mv)
    return _version_to_dict(mv)


@router.post("/{version_id}/rollback", response_model=ModelVersionOut)
async def rollback_version(
    version_id: str,
    db: AsyncSession = Depends(get_db),
    _=require_permission("model:manage"),
):
    """回滚到此版本（将其设为 active，同模型其他版本设为 inactive）"""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == version_id)
    )
    mv = result.scalar_one_or_none()
    if not mv:
        raise HTTPException(status_code=404, detail="模型版本不存在")

    if mv.status == "deprecated":
        raise HTTPException(
            status_code=400,
            detail={"error_code": "MODEL_VERSION_DEPRECATED", "message": "已废弃的版本不可回滚"},
        )

    # 将同模型的其他活跃版本设为 inactive
    others = await db.execute(
        select(ModelVersion).where(
            ModelVersion.name == mv.name,
            ModelVersion.id != version_id,
            ModelVersion.status == "active",
        )
    )
    for other in others.scalars().all():
        other.status = "inactive"

    mv.status = "active"
    await db.commit()
    await db.refresh(mv)
    return _version_to_dict(mv)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _version_to_dict(mv: ModelVersion) -> dict:
    return {
        "id": mv.id,
        "name": mv.name,
        "version": mv.version,
        "config_json": mv.config_json,
        "status": mv.status,
        "description": mv.description,
        "created_at": mv.created_at.isoformat() if mv.created_at else None,
        "updated_at": mv.updated_at.isoformat() if mv.updated_at else None,
    }
