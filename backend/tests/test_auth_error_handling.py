import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.api.v1 import auth as auth_module
from app.db.session import get_db
from app.main import app


def override_get_db():
    yield None


class TestAuthErrorHandling(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self):
        self.client.close()
        app.dependency_overrides.clear()

    def test_login_invalid_credentials_returns_standard_error_code(self):
        async def fake_authenticate_user(db, username, password):
            return None

        with patch.object(auth_module, "authenticate_user", fake_authenticate_user):
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": "bad-pass"},
            )
        body = response.json()
        self.assertEqual(response.status_code, 401)
        self.assertEqual(body["error_code"], "AUTH_INVALID_CREDENTIALS")
        self.assertEqual(body["message"], "用户名或密码错误")
        self.assertTrue(body["request_id"])

    def test_login_runtime_error_returns_masked_500_and_request_id(self):
        async def fake_authenticate_user(db, username, password):
            raise RuntimeError("sensitive stack detail")

        with self.assertLogs("app.main", level="ERROR") as logs:
            with patch.object(auth_module, "authenticate_user", fake_authenticate_user):
                response = self.client.post(
                    "/api/v1/auth/login",
                    json={"username": "admin", "password": "admin123"},
                )
        joined_logs = "\n".join(logs.output)
        self.assertIn("request_id=", joined_logs)
        self.assertIn("Traceback", joined_logs)
        self.assertIn("/api/v1/auth/login", joined_logs)

        body = response.json()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(body["error_code"], "INTERNAL_SERVER_ERROR")
        self.assertEqual(body["message"], "服务器内部错误，请稍后重试")
        self.assertNotIn("sensitive stack detail", str(body))
        self.assertTrue(body["request_id"])

    def test_login_runtime_error_response_contains_request_id_header(self):
        async def fake_authenticate_user(db, username, password):
            raise RuntimeError("boom")

        with patch.object(auth_module, "authenticate_user", fake_authenticate_user):
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": "admin123"},
            )
        body = response.json()
        self.assertIn("X-Request-ID", response.headers)
        self.assertEqual(response.headers["X-Request-ID"], body["request_id"])
        self.assertTrue(body["request_id"])

    def test_login_db_error_returns_503(self):
        async def fake_authenticate_user(db, username, password):
            raise SQLAlchemyError("db down")

        with patch.object(auth_module, "authenticate_user", fake_authenticate_user):
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": "admin123"},
            )
        body = response.json()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(body["error_code"], "DB_UNAVAILABLE")
        self.assertEqual(body["message"], "数据库服务暂不可用")
        self.assertTrue(body["request_id"])

    def test_login_validation_error_returns_422_standard_code(self):
        response = self.client.post(
            "/api/v1/auth/login",
            json={"username": "admin"},
        )
        body = response.json()
        self.assertEqual(response.status_code, 422)
        self.assertEqual(body["error_code"], "VALIDATION_ERROR")
        self.assertEqual(body["message"], "请求参数不合法")
        self.assertTrue(body["request_id"])
