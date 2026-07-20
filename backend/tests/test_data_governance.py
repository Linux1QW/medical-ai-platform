# -*- coding: utf-8 -*-
"""数据治理模块测试

测试内容：
- 数据留存策略：配置默认值、清理任务逻辑
- 模型版本注册表：ORM 模型、API 端点
- 细粒度 RBAC：权限映射、权限检查
- 数据导出 API：端点可用性
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.config import Settings
from app.core.permissions import get_user_permissions, PERMISSIONS
from app.models.user import User
from app.models.model_version import ModelVersion


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db():
    """创建模拟数据库会话"""
    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.rollback = AsyncMock()
    db.flush = AsyncMock()
    return db


@pytest.fixture
def admin_user():
    """创建管理员用户"""
    return User(
        id=1,
        username="admin",
        email="admin@test.com",
        role="admin",
        hashed_password="hashed",
    )


@pytest.fixture
def doctor_user():
    """创建医生用户"""
    return User(
        id=2,
        username="doctor",
        email="doctor@test.com",
        role="doctor",
        hashed_password="hashed",
    )


@pytest.fixture
def custom_perm_user():
    """创建自定义权限用户"""
    return User(
        id=3,
        username="custom",
        email="custom@test.com",
        role="doctor",
        hashed_password="hashed",
        permissions=["evaluation:view", "patient:view"],
    )


# ── 测试数据留存策略配置 ──────────────────────────────────────────────────────

class TestRetentionConfig:
    """测试数据留存配置"""

    def test_audit_log_retention_default(self):
        """审计日志默认保留 90 天"""
        settings = Settings()
        assert settings.AUDIT_LOG_RETENTION_DAYS == 90

    def test_evaluation_run_retention_default(self):
        """评估运行记录默认保留 180 天"""
        settings = Settings()
        assert settings.EVALUATION_RUN_RETENTION_DAYS == 180

    def test_retention_config_customizable(self):
        """留存天数可通过环境变量配置"""
        settings = Settings(
            AUDIT_LOG_RETENTION_DAYS=30,
            EVALUATION_RUN_RETENTION_DAYS=60,
        )
        assert settings.AUDIT_LOG_RETENTION_DAYS == 30
        assert settings.EVALUATION_RUN_RETENTION_DAYS == 60


# ── 测试 Celery Beat 配置 ────────────────────────────────────────────────────

try:
    import celery  # noqa: F401
    celery_available = True
except ImportError:
    celery_available = False


@pytest.mark.skipif(not celery_available, reason="Celery 未安装")
class TestCeleryBeatConfig:
    """测试 Celery Beat 定时任务配置"""

    def test_beat_schedule_exists(self):
        """Beat 定时任务已配置"""
        from app.celery_app import celery_app
        assert hasattr(celery_app.conf, 'beat_schedule')
        assert "cleanup-expired-records" in celery_app.conf.beat_schedule

    def test_beat_schedule_task_name(self):
        """定时任务名称正确"""
        from app.celery_app import celery_app
        schedule = celery_app.conf.beat_schedule["cleanup-expired-records"]
        assert schedule["task"] == "cleanup_expired_records"

    def test_beat_schedule_interval(self):
        """定时任务间隔为每天（86400秒）"""
        from app.celery_app import celery_app
        schedule = celery_app.conf.beat_schedule["cleanup-expired-records"]
        assert schedule["schedule"] == 86400.0

    def test_cleanup_task_registered(self):
        """清理任务已注册到 Celery"""
        from app.celery_app import celery_app
        # 导入任务模块以触发注册
        import app.tasks.data_cleanup  # noqa: F401
        assert "cleanup_expired_records" in celery_app.tasks


# ── 测试细粒度 RBAC ──────────────────────────────────────────────────────────

class TestRBACPermissions:
    """测试细粒度权限系统"""

    def test_admin_has_all_permissions(self):
        """管理员拥有所有预定义权限"""
        admin = User(id=1, username="admin", role="admin", hashed_password="x")
        perms = get_user_permissions(admin)
        assert "evaluation:create" in perms
        assert "evaluation:view" in perms
        assert "evaluation:review" in perms
        assert "user:manage" in perms
        assert "system:manage" in perms
        assert "model:manage" in perms
        assert "patient:export" in perms

    def test_doctor_has_limited_permissions(self):
        """医生只有有限权限"""
        doctor = User(id=2, username="doctor", role="doctor", hashed_password="x")
        perms = get_user_permissions(doctor)
        assert "evaluation:create" in perms
        assert "evaluation:view" in perms
        assert "consultation:create" in perms
        assert "consultation:view" in perms
        assert "patient:view" in perms
        # 医生不应有管理权限
        assert "user:manage" not in perms
        assert "system:manage" not in perms
        assert "model:manage" not in perms
        assert "patient:export" not in perms

    def test_custom_permissions_override_role(self):
        """自定义 permissions 字段覆盖角色默认权限"""
        user = User(
            id=3, username="custom", role="doctor", hashed_password="x",
            permissions=["evaluation:view", "patient:view"],
        )
        perms = get_user_permissions(user)
        assert perms == ["evaluation:view", "patient:view"]
        # 即使角色是 doctor，自定义权限不包含的也不应有
        assert "evaluation:create" not in perms

    def test_empty_permissions_fallback_to_role(self):
        """空 permissions 列表回退到角色默认"""
        user = User(
            id=4, username="empty", role="admin", hashed_password="x",
            permissions=[],
        )
        # 空列表是 falsy，应回退到角色默认
        perms = get_user_permissions(user)
        assert "user:manage" in perms

    def test_none_permissions_fallback_to_role(self):
        """None permissions 回退到角色默认"""
        user = User(
            id=5, username="none", role="doctor", hashed_password="x",
            permissions=None,
        )
        perms = get_user_permissions(user)
        assert perms == PERMISSIONS.get("doctor", [])

    def test_unknown_role_returns_empty(self):
        """未知角色返回空权限列表"""
        user = User(
            id=6, username="unknown", role="nurse", hashed_password="x",
            permissions=None,
        )
        perms = get_user_permissions(user)
        assert perms == []

    def test_permissions_dict_structure(self):
        """权限映射结构正确"""
        assert "admin" in PERMISSIONS
        assert "doctor" in PERMISSIONS
        assert isinstance(PERMISSIONS["admin"], list)
        assert isinstance(PERMISSIONS["doctor"], list)
        # 所有权限格式为 "resource:action"
        for role_perms in PERMISSIONS.values():
            for perm in role_perms:
                assert ":" in perm


# ── 测试 ModelVersion ORM ────────────────────────────────────────────────────

class TestModelVersionORM:
    """测试模型版本 ORM 模型"""

    def test_model_version_table_name(self):
        """表名正确"""
        assert ModelVersion.__tablename__ == "model_versions"

    def test_model_version_has_required_columns(self):
        """模型包含所有必需字段"""
        columns = {c.name for c in ModelVersion.__table__.columns}
        assert "id" in columns
        assert "name" in columns
        assert "version" in columns
        assert "config_json" in columns
        assert "status" in columns
        assert "description" in columns
        assert "created_at" in columns
        assert "updated_at" in columns

    def test_model_version_default_status(self):
        """默认状态为 active（通过 Column default 定义）"""
        # SQLAlchemy Column default 在持久化时生效，检查列定义
        status_col = ModelVersion.__table__.columns["status"]
        assert status_col.default is not None
        assert status_col.default.arg == "active"


# ── 测试 User 模型 permissions 字段 ──────────────────────────────────────────

class TestUserPermissionsField:
    """测试 User 模型 permissions 字段"""

    def test_user_has_permissions_column(self):
        """User 模型包含 permissions 字段"""
        columns = {c.name for c in User.__table__.columns}
        assert "permissions" in columns

    def test_user_default_permissions_none(self):
        """默认 permissions 为 None"""
        user = User(username="test", email="test@test.com", hashed_password="x")
        assert user.permissions is None

    def test_user_can_set_permissions(self):
        """可以设置 permissions"""
        user = User(
            username="test", email="test@test.com", hashed_password="x",
            permissions=["evaluation:create", "evaluation:view"],
        )
        assert user.permissions == ["evaluation:create", "evaluation:view"]


# ── 测试数据清理任务 ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not celery_available, reason="Celery 未安装")
class TestCleanupTask:
    """测试数据清理任务"""

    def test_cleanup_task_exists(self):
        """清理任务函数存在"""
        from app.tasks.data_cleanup import cleanup_expired_records
        assert callable(cleanup_expired_records)

    def test_cleanup_task_name(self):
        """清理任务名称正确"""
        from app.tasks.data_cleanup import cleanup_expired_records
        assert cleanup_expired_records.name == "cleanup_expired_records"

    @pytest.mark.asyncio
    async def test_cleanup_logic(self):
        """清理逻辑正确删除过期数据"""
        from app.tasks.data_cleanup import _do_cleanup
        from app.models.audit_log import AuditLog
        from app.models.evaluation_run import EvaluationRun

        mock_result = MagicMock()
        mock_result.rowcount = 5

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_db)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("app.tasks.data_cleanup.AsyncSessionLocal", return_value=mock_session):
            result = await _do_cleanup()

        assert result["audit_logs_deleted"] == 5
        assert result["evaluation_runs_deleted"] == 5
        assert "audit_cutoff" in result
        assert "run_cutoff" in result
        # 验证 execute 被调用了两次（审计日志 + 评估运行记录）
        assert mock_db.execute.call_count == 2
        mock_db.commit.assert_called_once()


# ── 测试数据导出 API ──────────────────────────────────────────────────────────

class TestDataExportAPI:
    """测试数据导出 API"""

    def test_export_endpoint_exists(self):
        """数据导出端点存在"""
        from app.api.v1.data_export import export_my_data
        assert callable(export_my_data)

    @pytest.mark.asyncio
    async def test_export_returns_user_data(self):
        """导出返回用户数据"""
        from app.api.v1.data_export import export_my_data

        mock_user = User(
            id=1, username="doctor", email="doc@test.com",
            real_name="测试医生", role="doctor", department="内科",
            created_at=datetime(2025, 1, 1),
            hashed_password="x",
        )

        # Mock 空查询结果
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await export_my_data(db=mock_db, current_user=mock_user)

        assert "user" in result
        assert "consultations" in result
        assert result["user"]["username"] == "doctor"
        assert result["user"]["email"] == "doc@test.com"
        assert result["consultations"] == []
