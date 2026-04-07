from datetime import datetime
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import json
import locale

from fastapi.testclient import TestClient
from passlib.context import CryptContext

from app.api.v1 import auth as auth_module
from app.core.security import (
    hash_password,
    verify_password,
)
from app.db.session import get_db
from app.main import app
from app.models.user import User


def override_get_db():
    yield None


class TestAuthPasswordLength(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        app.dependency_overrides.clear()

    def test_hash_and_verify_admin_password_success(self):
        hashed_password = hash_password("admin123")
        self.assertTrue(verify_password("admin123", hashed_password))

    def test_hash_and_verify_empty_password_success(self):
        hashed_password = hash_password("")
        self.assertTrue(verify_password("", hashed_password))

    def test_hash_and_verify_super_long_password_success(self):
        long_password = "中a" * 5000
        hashed_password = hash_password(long_password)
        self.assertTrue(verify_password(long_password, hashed_password))

    def test_verify_legacy_bcrypt_hash_success(self):
        legacy_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        legacy_hash = legacy_context.hash("admin123")
        self.assertTrue(verify_password("admin123", legacy_hash))

    def test_admin_can_login_with_admin123(self):
        admin_hashed_password = hash_password("admin123")
        admin_user = SimpleNamespace(
            id=1,
            username="admin",
            email="admin@medical.com",
            real_name="系统管理员",
            role="admin",
            department="系统管理",
            avatar="",
            created_at=datetime.utcnow(),
        )

        async def fake_authenticate_user(db, username, password):
            if username != "admin":
                return None
            if verify_password(password, admin_hashed_password):
                return admin_user
            return None

        with patch.object(auth_module, "authenticate_user", fake_authenticate_user):
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": "admin123"},
            )
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(response_json["user"]["username"], "admin")
        self.assertEqual(response_json["user"]["role"], "admin")
        self.assertTrue(response_json["access_token"])

    def test_login_accepts_overlong_password_without_length_error(self):
        very_long_password = "a" * 10000

        async def fake_authenticate_user(db, username, password):
            return None

        with patch.object(auth_module, "authenticate_user", fake_authenticate_user):
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": very_long_password},
            )
        self.assertEqual(response.status_code, 401)
        self.assertIn("用户名或密码错误", str(response.json()))

    def test_login_accepts_empty_password_without_length_error(self):
        async def fake_authenticate_user(db, username, password):
            return None

        with patch.object(auth_module, "authenticate_user", fake_authenticate_user):
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": ""},
            )
        self.assertEqual(response.status_code, 401)
        self.assertIn("用户名或密码错误", str(response.json()))

    def test_password_diagnostics_memory_and_network_bytes(self):
        payload = {"username": "admin", "password": "admin123"}
        raw_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        memory_password_bytes = len(payload["password"].encode("utf-8"))
        network_password_bytes = len(json.loads(raw_body.decode("utf-8"))["password"].encode("utf-8"))
        print("MEMORY_PASSWORD_BYTES", memory_password_bytes)
        print("NETWORK_PASSWORD_BYTES", network_password_bytes)
        self.assertEqual(memory_password_bytes, 8)
        self.assertEqual(network_password_bytes, 8)

    def test_password_diagnostics_database_schema_and_hash_bytes(self):
        username_length_limit = User.__table__.c.username.type.length
        admin_hashed_password = hash_password("admin123")
        hash_bytes = len(admin_hashed_password.encode("utf-8"))
        hashed_password_type = User.__table__.c.hashed_password.type.__class__.__name__
        print("DB_USERNAME_LENGTH_LIMIT", username_length_limit)
        print("DB_HASHED_PASSWORD_TYPE", hashed_password_type)
        print("DB_HASHED_PASSWORD_BYTES", hash_bytes)
        self.assertEqual(username_length_limit, 50)
        self.assertEqual(hashed_password_type, "Text")
        self.assertGreater(hash_bytes, 0)

    def test_admin123_not_affected_by_locale_charset(self):
        original_getpreferredencoding = locale.getpreferredencoding
        try:
            locale.getpreferredencoding = lambda do_setlocale=True: "gbk"
            validate_ok = verify_password("admin123", hash_password("admin123"))
            self.assertTrue(validate_ok)
            locale.getpreferredencoding = lambda do_setlocale=True: "cp1252"
            validate_ok = verify_password("admin123", hash_password("admin123"))
            self.assertTrue(validate_ok)
        finally:
            locale.getpreferredencoding = original_getpreferredencoding

    def test_login_handles_boundary_password_without_length_error(self):
        boundary_password = "a" * 72
        async def fake_authenticate_user(db, username, password):
            return None

        with patch.object(auth_module, "authenticate_user", fake_authenticate_user):
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": boundary_password},
            )
        self.assertEqual(response.status_code, 401)
        self.assertIn("用户名或密码错误", str(response.json()))
