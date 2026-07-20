from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.patients import router as patients_router
from app.api.v1.consultations import router as consultations_router
from app.api.v1.evaluations import router as evaluations_router
from app.api.v1.stats import router as stats_router
from app.api.v1.knowledge_base import router as knowledge_base_router
from app.api.v1.admin import router as admin_router
from app.api.v1.cases import router as cases_router
from app.api.v1.review import router as review_router
from app.api.v1.model_versions import router as model_versions_router
from app.api.v1.data_export import router as data_export_router

router = APIRouter()

router.include_router(auth_router, prefix="/auth", tags=["认证"])
router.include_router(patients_router, prefix="/patients", tags=["虚拟患者"])
router.include_router(consultations_router, prefix="/consultations", tags=["问诊交互"])
router.include_router(evaluations_router, prefix="/evaluations", tags=["评估"])
router.include_router(stats_router, prefix="/stats", tags=["数据统计"])
router.include_router(knowledge_base_router, prefix="/knowledge-base", tags=["知识库管理"])
router.include_router(admin_router, prefix="/admin", tags=["管理"])
router.include_router(cases_router, prefix="/cases", tags=["病例推荐"])
router.include_router(review_router)
router.include_router(model_versions_router, prefix="/model-versions", tags=["模型版本"])
router.include_router(data_export_router, prefix="/users", tags=["数据导出"])
