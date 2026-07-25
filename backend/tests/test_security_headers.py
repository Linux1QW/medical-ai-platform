# -*- coding: utf-8 -*-
"""安全响应头中间件回归测试"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


@pytest.fixture
def client():
    # 不进入上下文管理器，避免触发 lifespan（LangGraph checkpointer 等真实依赖）
    c = TestClient(app, raise_server_exceptions=False)
    yield c
    c.close()


class TestSecurityHeaders:
    def test_api_response_has_security_headers(self, client):
        resp = client.get("/api/v1/patients")  # 401 也应带安全头
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["Referrer-Policy"] == "no-referrer"
        assert resp.headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"

    def test_docs_exempt_from_csp(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert "Content-Security-Policy" not in resp.headers
        # 其余安全头仍然生效
        assert resp.headers["X-Content-Type-Options"] == "nosniff"

    def test_hsts_only_in_production(self, client, monkeypatch):
        monkeypatch.setattr(settings, "ENVIRONMENT", "development")
        assert "Strict-Transport-Security" not in client.get("/health").headers

        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        resp = client.get("/health")
        assert resp.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
