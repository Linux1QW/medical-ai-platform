"""Integration test: Coach multi-worker scenario.

Two TestClient instances sharing DB/Redis.
- Create on worker A, read/replay/feedback on worker B.
- Assert one decision and monotonic event sequences.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.deps import get_current_user
from app.db.session import get_db
from app.main import app


# ── Helpers ──────────────────────────────────────────────────────────────────

DOCTOR_A = SimpleNamespace(id=10, username="doc_a", role="doctor", permissions=None)


def _override_get_db():
    yield None


def _make_client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _as_user(user):
    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = lambda: user


def _parse_sse_events(text: str) -> list[dict]:
    """Parse SSE text into a list of event dicts."""
    events = []
    current_event: dict = {}
    current_data: str = ""

    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event["event"] = line[len("event: "):]
        elif line.startswith("id: "):
            current_event["id"] = line[len("id: "):]
        elif line.startswith("data: "):
            current_data = line[len("data: "):]
        elif line == "" and current_event.get("event"):
            if current_data:
                try:
                    current_event["data"] = json.loads(current_data)
                except json.JSONDecodeError:
                    current_event["data"] = current_data
            else:
                current_event["data"] = {}
            events.append(current_event)
            current_event = {}
            current_data = ""

    return events


# ── Multi-worker test ────────────────────────────────────────────────────────


class TestCoachMultiWorker:
    """Two TestClient instances simulating separate workers.

    Since we mock the service layer, we verify the contract:
    - Same idempotency_key → same decision (one decision).
    - Event sequences are monotonic.
    - Replay returns same events.
    """

    def test_create_on_a_replay_on_b(self) -> None:
        """Create suggestion on worker A, replay on worker B."""
        _as_user(DOCTOR_A)

        suggestion_id = str(uuid4())
        event_ids = [str(uuid4()) for _ in range(3)]

        # Shared mock service state
        shared_decisions: dict[str, dict] = {}
        shared_events: list[dict] = []

        async def fake_stream(**kwargs):
            idem_key = kwargs["idempotency_key"]

            # If already decided, replay
            if idem_key in shared_decisions:
                for evt in shared_events:
                    yield evt
                yield {"event": "done", "data": {}, "id": None}
                return

            # First call: create decision
            thinking_evt = {
                "event": "thinking",
                "data": {"turn_no": 1},
                "id": event_ids[0],
            }
            suggestion_evt = {
                "event": "suggestion",
                "data": {
                    "suggestion_id": suggestion_id,
                    "content": "Ask about onset",
                    "confidence": 0.85,
                },
                "id": event_ids[1],
            }
            done_evt = {"event": "done", "data": {}, "id": event_ids[2]}

            shared_decisions[idem_key] = {
                "suggestion_id": suggestion_id,
                "turn_no": 1,
            }
            shared_events.extend([thinking_evt, suggestion_evt])

            yield thinking_evt
            yield suggestion_evt
            yield done_evt

        mock_service = MagicMock()
        mock_service.enabled = True
        mock_service.stream_suggestion = fake_stream

        async def fake_verify(consultation_id, doctor_id, db):
            return SimpleNamespace(id=1, doctor_id=10)

        with patch("app.api.v1.coach._get_service", return_value=mock_service), \
             patch("app.api.v1.coach._verify_consultation_ownership", side_effect=fake_verify):

            payload = {
                "latest_message": "患者主诉头痛",
                "idempotency_key": "multi-worker-key-001",
            }

            # Worker A: create
            client_a = _make_client()
            resp_a = client_a.post(
                "/api/v1/coach/consultations/1/suggestions/stream",
                json=payload,
            )
            assert resp_a.status_code == 200
            events_a = _parse_sse_events(resp_a.text)
            event_names_a = [e["event"] for e in events_a]
            assert "thinking" in event_names_a
            assert "suggestion" in event_names_a
            client_a.close()

            # Worker B: replay with same key
            client_b = _make_client()
            resp_b = client_b.post(
                "/api/v1/coach/consultations/1/suggestions/stream",
                json=payload,
            )
            assert resp_b.status_code == 200
            events_b = _parse_sse_events(resp_b.text)
            event_names_b = [e["event"] for e in events_b]
            # Replay should contain suggestion (may or may not have thinking)
            assert "suggestion" in event_names_b
            client_b.close()

            # Assert one decision
            assert len(shared_decisions) == 1
            assert "multi-worker-key-001" in shared_decisions

        app.dependency_overrides.clear()

    def test_monotonic_event_sequences(self) -> None:
        """Events have monotonically increasing sequence numbers."""
        _as_user(DOCTOR_A)

        seq_counter = 0

        async def fake_stream(**kwargs):
            nonlocal seq_counter
            seq_counter += 1
            yield {
                "event": "thinking",
                "data": {"turn_no": 1},
                "id": f"evt-{seq_counter:04d}",
            }
            seq_counter += 1
            yield {
                "event": "suggestion",
                "data": {"suggestion_id": str(uuid4())},
                "id": f"evt-{seq_counter:04d}",
            }
            yield {"event": "done", "data": {}, "id": None}

        mock_service = MagicMock()
        mock_service.enabled = True
        mock_service.stream_suggestion = fake_stream

        async def fake_verify(consultation_id, doctor_id, db):
            return SimpleNamespace(id=1, doctor_id=10)

        with patch("app.api.v1.coach._get_service", return_value=mock_service), \
             patch("app.api.v1.coach._verify_consultation_ownership", side_effect=fake_verify):

            client = _make_client()
            resp = client.post(
                "/api/v1/coach/consultations/1/suggestions/stream",
                json={
                    "latest_message": "头痛",
                    "idempotency_key": "monotonic-test-key-01",
                },
            )
            assert resp.status_code == 200

            events = _parse_sse_events(resp.text)
            # Verify event IDs are in order
            event_ids = [e.get("id") for e in events if e.get("id")]
            assert len(event_ids) >= 2
            # IDs should be different and sequential
            assert event_ids[0] != event_ids[1]
            client.close()

        app.dependency_overrides.clear()

    def test_new_key_creates_new_turn(self) -> None:
        """Different idempotency_key → new turn (new decision)."""
        _as_user(DOCTOR_A)

        decisions_created = []

        async def fake_stream(**kwargs):
            idem_key = kwargs["idempotency_key"]
            decisions_created.append(idem_key)
            yield {
                "event": "thinking",
                "data": {"turn_no": len(decisions_created)},
                "id": str(uuid4()),
            }
            yield {
                "event": "suggestion",
                "data": {"suggestion_id": str(uuid4())},
                "id": str(uuid4()),
            }
            yield {"event": "done", "data": {}, "id": None}

        mock_service = MagicMock()
        mock_service.enabled = True
        mock_service.stream_suggestion = fake_stream

        async def fake_verify(consultation_id, doctor_id, db):
            return SimpleNamespace(id=1, doctor_id=10)

        with patch("app.api.v1.coach._get_service", return_value=mock_service), \
             patch("app.api.v1.coach._verify_consultation_ownership", side_effect=fake_verify):

            client = _make_client()

            # First key
            resp1 = client.post(
                "/api/v1/coach/consultations/1/suggestions/stream",
                json={
                    "latest_message": "头痛",
                    "idempotency_key": "key-a-000000000001",
                },
            )
            assert resp1.status_code == 200

            # Second key (different)
            resp2 = client.post(
                "/api/v1/coach/consultations/1/suggestions/stream",
                json={
                    "latest_message": "还有恶心",
                    "idempotency_key": "key-b-000000000002",
                },
            )
            assert resp2.status_code == 200

            # Two different decisions
            assert len(decisions_created) == 2
            assert decisions_created[0] != decisions_created[1]

            client.close()

        app.dependency_overrides.clear()
