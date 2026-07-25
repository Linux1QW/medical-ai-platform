# -*- coding: utf-8 -*-
"""WebSocket 首条消息鉴权回归测试（#18）"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import app.api.v1.evaluations as eval_module
from app.core.security import create_access_token
from app.main import app

WS_URL = "/api/v1/evaluations/ws/1"


@pytest.fixture
def client():
    # 不进入上下文管理器，避免触发 lifespan（LangGraph checkpointer 等真实依赖）
    c = TestClient(app, raise_server_exceptions=False)
    yield c
    c.close()


def _assert_closed_1008(ws):
    msg = ws.receive()
    assert msg["type"] == "websocket.close"
    assert msg["code"] == 1008


class TestWebSocketFirstMessageAuth:
    def test_invalid_json_first_message_rejected(self, client):
        with client.websocket_connect(WS_URL) as ws:
            ws.send_text("not-a-json")
            _assert_closed_1008(ws)

    def test_missing_token_rejected(self, client):
        with client.websocket_connect(WS_URL) as ws:
            ws.send_text(json.dumps({"type": "auth"}))
            _assert_closed_1008(ws)

    def test_invalid_token_rejected(self, client):
        with client.websocket_connect(WS_URL) as ws:
            ws.send_text(json.dumps({"type": "auth", "token": "bad.token.here"}))
            _assert_closed_1008(ws)

    def test_auth_timeout_closes_connection(self, client, monkeypatch):
        monkeypatch.setattr(eval_module, "WS_AUTH_TIMEOUT", 0.1)
        with client.websocket_connect(WS_URL) as ws:
            # 不发送任何消息，等待服务端超时关闭
            _assert_closed_1008(ws)

    def test_valid_token_receives_auth_ok(self, client, monkeypatch):
        # 绕过真实数据库：mock 用户查询与访问控制
        fake_user = MagicMock(id=1)
        monkeypatch.setattr(eval_module, "get_user_by_id", AsyncMock(return_value=fake_user))
        monkeypatch.setattr(eval_module, "require_consultation_access", AsyncMock(return_value=None))

        token = create_access_token({"sub": "1"})
        with client.websocket_connect(WS_URL) as ws:
            ws.send_text(json.dumps({"type": "auth", "token": token}))
            assert ws.receive_json() == {"type": "auth_ok"}

    def test_token_no_longer_accepted_via_query_string(self, client, monkeypatch):
        """token 出现在 query string 中不再生效（必须走首条消息）"""
        monkeypatch.setattr(eval_module, "WS_AUTH_TIMEOUT", 0.1)
        token = create_access_token({"sub": "1"})
        with client.websocket_connect(f"{WS_URL}?token={token}") as ws:
            # 仅带 query token 而不发首条消息 → 超时关闭
            _assert_closed_1008(ws)
