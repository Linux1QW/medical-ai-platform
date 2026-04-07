from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.patients import router as patients_router
from app.api.v1.consultations import router as consultations_router
from app.api.v1.evaluations import router as evaluations_router
from app.api.v1.stats import router as stats_router

router = APIRouter()

router.include_router(auth_router, prefix="/auth", tags=["认证"])
router.include_router(patients_router, prefix="/patients", tags=["虚拟患者"])
router.include_router(consultations_router, prefix="/consultations", tags=["问诊交互"])
router.include_router(evaluations_router, prefix="/evaluations", tags=["评估"])
router.include_router(stats_router, prefix="/stats", tags=["数据统计"])
