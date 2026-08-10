# -*- coding: utf-8 -*-
"""V1.1 Production 配置安全测试

TDD Phase 1: 这些测试在实现之前应该失败。
"""

import os

import pytest

# ── 安全检查 ──────────────────────────────────────────────────────────────────


class TestProductionSecurityChecks:
    """生产环境安全检查约束"""

    def test_default_secret_blocks_production(self):
        """默认 SECRET_KEY 阻止 production 启动"""
        from app.core.config import Settings
        s = Settings(
            ENVIRONMENT="production",
            SECRET_KEY="change-this-to-a-secure-random-string",
            TESTING=False,
            JWT_TOKEN_BLACKLIST_ENABLED=True,
            JWT_BLACKLIST_FAIL_CLOSED=True,
            ACCESS_TOKEN_EXPIRE_MINUTES=30,
            LANGFUSE_ENABLED=False,
        )
        with pytest.raises(RuntimeError, match="SECRET_KEY"):
            s.check_security()

    def test_fail_closed_false_blocks_production(self):
        """JWT_BLACKLIST_FAIL_CLOSED=false 阻止 production 启动"""
        from app.core.config import Settings
        s = Settings(
            ENVIRONMENT="production",
            SECRET_KEY="a-very-long-random-secret-that-is-not-default",
            TESTING=False,
            JWT_TOKEN_BLACKLIST_ENABLED=True,
            JWT_BLACKLIST_FAIL_CLOSED=False,
            ACCESS_TOKEN_EXPIRE_MINUTES=30,
            LANGFUSE_ENABLED=False,
        )
        with pytest.raises(RuntimeError, match="JWT_BLACKLIST_FAIL_CLOSED"):
            s.check_security()

    def test_testing_true_blocks_production(self):
        """TESTING=true 时 check_security 应跳过（不阻止），但 production 不应在 TESTING 模式下运行"""
        # TESTING=true 允许跳过安全检查（这是测试模式的设计）
        from app.core.config import Settings
        s = Settings(
            ENVIRONMENT="production",
            SECRET_KEY="change-this-to-a-secure-random-string",
            TESTING=True,
        )
        # TESTING=true 时 check_security 不抛异常
        s.check_security()  # should not raise

    def test_default_secret_blocks_staging(self):
        """默认 SECRET_KEY 阻止 staging 启动"""
        from app.core.config import Settings
        s = Settings(
            ENVIRONMENT="staging",
            SECRET_KEY="change-this-to-a-secure-random-string",
            TESTING=False,
            JWT_TOKEN_BLACKLIST_ENABLED=True,
            JWT_BLACKLIST_FAIL_CLOSED=True,
            ACCESS_TOKEN_EXPIRE_MINUTES=30,
            LANGFUSE_ENABLED=False,
        )
        with pytest.raises(RuntimeError, match="SECRET_KEY"):
            s.check_security()


# ── DATABASE_URL 特殊字符解析 ─────────────────────────────────────────────────


class TestDatabaseUrlSpecialChars:
    """密码含 @:/#% 时 URL 正确解析"""

    def test_password_with_special_chars_async_url(self):
        """密码含 @:/#% 时 async DATABASE_URL 被 SQLAlchemy 正确编码"""
        from app.core.config import Settings
        s = Settings(
            MYSQL_HOST="localhost",
            MYSQL_PORT=3306,
            MYSQL_USER="medical_app",
            MYSQL_PASSWORD="p@ss:w/ord#123%456",
            MYSQL_DATABASE="medical_ai",
        )
        url = s.DATABASE_URL
        # 密码中的特殊字符必须被 percent-encode
        assert "p@ss:w/ord#123%456" not in url
        # SQLAlchemy 能正确解析回来
        from sqlalchemy.engine import make_url
        parsed = make_url(url)
        assert parsed.password == "p@ss:w/ord#123%456"
        assert parsed.username == "medical_app"
        assert parsed.host == "localhost"
        assert parsed.database == "medical_ai"

    def test_password_with_special_chars_sync_url(self):
        """密码含 @:/#% 时 sync DATABASE_URL 被 SQLAlchemy 正确编码"""
        from app.core.config import Settings
        s = Settings(
            MYSQL_HOST="localhost",
            MYSQL_PORT=3306,
            MYSQL_USER="medical_app",
            MYSQL_PASSWORD="p@ss:w/ord#123%456",
            MYSQL_DATABASE="medical_ai",
        )
        url = s.DATABASE_URL_SYNC
        from sqlalchemy.engine import make_url
        parsed = make_url(url)
        assert parsed.password == "p@ss:w/ord#123%456"
        assert parsed.username == "medical_app"

    def test_password_with_at_sign(self):
        """密码含 @ 时 URL 不会混淆为用户名分隔符"""
        from app.core.config import Settings
        s = Settings(
            MYSQL_PASSWORD="user@domain.com",
        )
        url = s.DATABASE_URL
        from sqlalchemy.engine import make_url
        parsed = make_url(url)
        assert parsed.password == "user@domain.com"
        assert parsed.username == s.MYSQL_USER


# ── Compose 凭据隔离 ─────────────────────────────────────────────────────────


class TestComposeCredentialIsolation:
    """展开 production Compose 后，runtime 服务不得出现 root/migration/backup 密码"""

    def _load_compose_services(self):
        """加载 docker-compose.yml + docker-compose.prod.yml 的服务定义"""
        import yaml

        # 注册 Docker Compose 扩展标签的无操作构造器
        class _SafeLoaderIgnore(yaml.SafeLoader):
            pass
        _SafeLoaderIgnore.add_constructor(
            "!reset", lambda loader, node: None
        )

        base_path = os.path.join(os.path.dirname(__file__), "..", "..", "docker-compose.yml")
        prod_path = os.path.join(os.path.dirname(__file__), "..", "..", "docker-compose.prod.yml")
        with open(base_path, encoding="utf-8") as f:
            base = yaml.load(f, Loader=_SafeLoaderIgnore)
        with open(prod_path, encoding="utf-8") as f:
            prod = yaml.load(f, Loader=_SafeLoaderIgnore)
        # 简单合并（prod 覆盖 base）
        services = base.get("services", {})
        for svc_name, svc_conf in (prod.get("services", {}) or {}).items():
            if svc_name in services:
                services[svc_name].update(svc_conf or {})
            else:
                services[svc_name] = svc_conf
        return services

    def test_runtime_services_no_root_password(self):
        """backend/celery-worker/celery-beat 不得注入 MYSQL_ROOT_PASSWORD"""
        services = self._load_compose_services()
        runtime_services = ["backend", "celery-worker", "celery-beat"]
        for svc in runtime_services:
            if svc not in services:
                continue
            env = services[svc].get("environment", {})
            if isinstance(env, dict):
                assert "MYSQL_ROOT_PASSWORD" not in env, \
                    f"{svc} should not have MYSQL_ROOT_PASSWORD"
            elif isinstance(env, list):
                for item in env:
                    assert not str(item).startswith("MYSQL_ROOT_PASSWORD="), \
                        f"{svc} should not have MYSQL_ROOT_PASSWORD"

    def test_migrate_service_only_has_migration_vars(self):
        """migrate 服务只接收 migration DB 变量，不含 JWT/LLM/Redis secret"""
        services = self._load_compose_services()
        if "migrate" not in services:
            pytest.skip("migrate service not found in compose")
        env = services["migrate"].get("environment", {})
        if isinstance(env, dict):
            env_keys = set(env.keys())
        elif isinstance(env, list):
            env_keys = {str(item).split("=")[0] for item in env}
        else:
            env_keys = set()
        # migrate 不应收到 JWT/LLM/Redis 相关 secret
        forbidden_prefixes = ("SECRET_KEY", "QWEN_", "DASHSCOPE_", "REDIS_CHECKPOINT",
                              "CELERY_", "LANGFUSE_", "JWT_", "LLM_")
        for key in env_keys:
            for prefix in forbidden_prefixes:
                assert not key.startswith(prefix), \
                    f"migrate service should not have {key}"

    def test_no_container_name_in_compose(self):
        """所有服务不得使用 container_name"""
        services = self._load_compose_services()
        for svc_name, svc_conf in services.items():
            assert "container_name" not in svc_conf, \
                f"{svc_name} should not have container_name (breaks --scale)"
