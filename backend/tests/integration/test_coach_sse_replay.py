"""Integration test: Coach SSE replay with Last-Event-ID.

Tests:
  - SSE replay returns events after the given Last-Event-ID
  - Disconnect/reconnect recovery: client reconnects with Last-Event-ID,
    server replays missed events and continues
  - Idempotency: same key → same decision across reconnects
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

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
    events: list[dict] = []
    current_event: dict = {}
    current_data_lines: list[str] = []

    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event["event"] = line[len("event: "):]
        elif line.startswith("id: "):
            current_event["id"] = line[len("id: "):]
        elif line.startswith("data: "):
            current_data_lines.append(line[len("data: "):])
        elif line == "":
            if current_event.get("event"):
                raw_data = "\n".join(current_data_lines)
                if raw_data:
                    try:
                        current_event["data"] = json.loads(raw_data)
                    except json.JSONDecodeError:
                        current_event["data"] = raw_data
                else:
                    current_event["data"] = {}
                events.append(current_event)
            current_event = {}
            current_data_lines = []

    return events


# ── Shared mock state ────────────────────────────────────────────────────────

def _build_mock_service(
    event_sequence: list[dict],
    replay_from_id: str | None = None,
) -> MagicMock:
    """Build a mock CoachService that yields a fixed event sequence.

    If replay_from_id is set, only yield events whose id comes after that id.
    """
    mock_service = MagicMock()
    mock_service.enabled = True

    async def fake_stream(**kwargs):
        last_event_id = kwargs.get("last_event_id")

        if last_event_id:
            # Replay mode: yield events after the given id
            found = False
            for evt in event_sequence:
                if evt.get("id") == last_event_id:
                    found = True
                    continue
                if found:
                    yield evt
            if not found:
                # If id not found, yield all events (full replay)
                for evt in event_sequence:
                    yield evt
        else:
            for evt in event_sequence:
                yield evt

    mock_service.stream_suggestion = fake_stream
    return mock_service


def _patch_coach(mock_service: MagicMock, doctor_id: int = 10):
    """Patch coach dependencies."""
    async def fake_verify(consultation_id, doctor_id_param, db):
        return SimpleNamespace(id=1, doctor_id=doctor_id)

    return patch("app.api.v1.coach._get_service", return_value=mock_service), \
           patch("app.api.v1.coach._verify_consultation_ownership", side_effect=fake_verify)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestCoachSSEReplay:
    """Test SSE replay with Last-Event-ID header."""

    def test_replay_returns_events_after_last_event_id(self) -> None:
        """When Last-Event-ID is provided, server replays events after that id."""
        _as_user(DOCTOR_A)

        event_ids = [f"evt-{uuid4().hex[:8]}" for _ in range(5)]
        full_sequence = [
            {"event": "thinking", "data": {"turn_no": 1}, "id": event_ids[0]},
            {"event": "suggestion", "data": {"suggestion_id": str(uuid4()), "content": "Ask about onset"}, "id": event_ids[1]},
            {"event": "evidence", "data": {"source": "guideline-A"}, "id": event_ids[2]},
            {"event": "suggestion", "data": {"suggestion_id": str(uuid4()), "content": "Consider CBC"}, "id": event_ids[3]},
            {"event": "done", "data": {}, "id": event_ids[4]},
        ]

        mock_service = _build_mock_service(full_sequence)
        p1, p2 = _patch_coach(mock_service)

        with p1, p2:
            client = _make_client()

            # First: full stream to get all events
            resp_full = client.post(
                "/api/v1/coach/consultations/1/suggestions/stream",
                json={"latest_message": "头痛", "idempotency_key": "replay-test-key-001"},
            )
            assert resp_full.status_code == 200
            full_events = _parse_sse_events(resp_full.text)
            assert len(full_events) == 5
            client.close()

        app.dependency_overrides.clear()

        # Now replay from event_ids[1] (the first suggestion)
        mock_service_replay = _build_mock_service(full_sequence)
        p1, p2 = _patch_coach(mock_service_replay)

        with p1, p2:
            _as_user(DOCTOR_A)
            client = _make_client()
            resp_replay = client.post(
                "/api/v1/coach/consultations/1/suggestions/stream",
                json={"latest_message": "头痛", "idempotency_key": "replay-test-key-001"},
                headers={"Last-Event-ID": event_ids[1]},
            )
            assert resp_replay.status_code == 200
            replay_events = _parse_sse_events(resp_replay.text)

            # Should only contain events after event_ids[1]
            replay_event_ids = [e.get("id") for e in replay_events if e.get("id")]
            assert event_ids[1] not in replay_event_ids
            assert event_ids[2] in replay_event_ids
            assert event_ids[3] in replay_event_ids
            assert event_ids[4] in replay_event_ids
            client.close()

        app.dependency_overrides.clear()

    def test_replay_with_unknown_id_returns_all_events(self) -> None:
        """When Last-Event-ID is unknown, server replays all events."""
        _as_user(DOCTOR_A)

        event_ids = [f"evt-{uuid4().hex[:8]}" for _ in range(3)]
        sequence = [
            {"event": "thinking", "data": {"turn_no": 1}, "id": event_ids[0]},
            {"event": "suggestion", "data": {"suggestion_id": str(uuid4())}, "id": event_ids[1]},
            {"event": "done", "data": {}, "id": event_ids[2]},
        ]

        mock_service = _build_mock_service(sequence)
        p1, p2 = _patch_coach(mock_service)

        with p1, p2:
            client = _make_client()
            resp = client.post(
                "/api/v1/coach/consultations/1/suggestions/stream",
                json={"latest_message": "头痛", "idempotency_key": "unknown-id-test-001"},
                headers={"Last-Event-ID": "nonexistent-event-id"},
            )
            assert resp.status_code == 200
            events = _parse_sse_events(resp.text)
            # All events should be returned
            assert len(events) == 3
            client.close()

        app.dependency_overrides.clear()


class TestCoachDisconnectReconnect:
    """Test disconnect/reconnect recovery."""

    def test_reconnect_recovers_missed_events(self) -> None:
        """Client disconnects mid-stream, reconnects with Last-Event-ID,
        and receives remaining events."""
        _as_user(DOCTOR_A)

        event_ids = [f"evt-{uuid4().hex[:8]}" for _ in range(4)]
        full_sequence = [
            {"event": "thinking", "data": {"turn_no": 1}, "id": event_ids[0]},
            {"event": "suggestion", "data": {"suggestion_id": str(uuid4()), "content": "First suggestion"}, "id": event_ids[1]},
            {"event": "evidence", "data": {"source": "guideline-B"}, "id": event_ids[2]},
            {"event": "done", "data": {}, "id": event_ids[3]},
        ]

        # Simulate: client received first 2 events, then disconnected
        _received_before_disconnect = full_sequence[:2]
        last_received_id = event_ids[1]

        # Reconnect: should receive events after last_received_id
        mock_service = _build_mock_service(full_sequence)
        p1, p2 = _patch_coach(mock_service)

        with p1, p2:
            client = _make_client()

            # Simulate reconnect with Last-Event-ID
            resp_reconnect = client.post(
                "/api/v1/coach/consultations/1/suggestions/stream",
                json={"latest_message": "头痛加剧", "idempotency_key": "reconnect-key-001"},
                headers={"Last-Event-ID": last_received_id},
            )
            assert resp_reconnect.status_code == 200
            recovered_events = _parse_sse_events(resp_reconnect.text)

            # Should have events after the disconnect point
            recovered_ids = [e.get("id") for e in recovered_events if e.get("id")]
            assert event_ids[0] not in recovered_ids
            assert event_ids[1] not in recovered_ids
            assert event_ids[2] in recovered_ids
            assert event_ids[3] in recovered_ids

            # Verify done event is present (stream completed)
            event_names = [e["event"] for e in recovered_events]
            assert "done" in event_names

            client.close()

        app.dependency_overrides.clear()

    def test_idempotency_preserved_across_reconnects(self) -> None:
        """Same idempotency_key across disconnect/reconnect yields same decision."""
        _as_user(DOCTOR_A)

        suggestion_id = str(uuid4())
        event_ids = [f"evt-{uuid4().hex[:8]}" for _ in range(3)]
        sequence = [
            {"event": "thinking", "data": {"turn_no": 1}, "id": event_ids[0]},
            {"event": "suggestion", "data": {"suggestion_id": suggestion_id, "content": "Same suggestion"}, "id": event_ids[1]},
            {"event": "done", "data": {}, "id": event_ids[2]},
        ]

        idem_key = f"reconnect-idem-{uuid4().hex[:16]}"

        # First connection (partial — simulates disconnect after thinking)
        mock_service_1 = _build_mock_service(sequence)
        p1, p2 = _patch_coach(mock_service_1)

        with p1, p2:
            client = _make_client()
            resp1 = client.post(
                "/api/v1/coach/consultations/1/suggestions/stream",
                json={"latest_message": "头痛", "idempotency_key": idem_key},
            )
            assert resp1.status_code == 200
            events1 = _parse_sse_events(resp1.text)
            event_names1 = [e["event"] for e in events1]
            assert "suggestion" in event_names1
            client.close()

        app.dependency_overrides.clear()

        # Reconnect with same idempotency_key and Last-Event-ID
        mock_service_2 = _build_mock_service(sequence)
        p1, p2 = _patch_coach(mock_service_2)

        with p1, p2:
            _as_user(DOCTOR_A)
            client = _make_client()
            resp2 = client.post(
                "/api/v1/coach/consultations/1/suggestions/stream",
                json={"latest_message": "头痛", "idempotency_key": idem_key},
                headers={"Last-Event-ID": event_ids[0]},
            )
            assert resp2.status_code == 200
            events2 = _parse_sse_events(resp2.text)

            # Replayed suggestion should have same suggestion_id
            suggestion_events = [e for e in events2 if e["event"] == "suggestion"]
            assert len(suggestion_events) >= 1
            replayed_suggestion_id = suggestion_events[0].get("data", {}).get("suggestion_id")
            assert replayed_suggestion_id == suggestion_id

            client.close()

        app.dependency_overrides.clear()
