"""Tests for Coach API endpoints."""
from __future__ import annotations

import json
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.coach import _coach_service, router


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_service():
    """Reset coach service before each test."""
    _coach_service._sessions.clear()
    _coach_service.set_enabled(True)
    yield
    _coach_service._sessions.clear()
    _coach_service.set_enabled(False)


def test_get_state_idle(client: TestClient) -> None:
    """GET state for unknown consultation → idle."""
    resp = client.get("/coach/consultations/999/state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["consultation_id"] == 999
    assert data["status"] == "idle"
    assert data["turn_no"] == 0


def test_get_state_disabled(client: TestClient) -> None:
    """When coach disabled → status='disabled'."""
    _coach_service.set_enabled(False)
    resp = client.get("/coach/consultations/999/state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "disabled"


def test_stream_suggestion_produces_sse(client: TestClient) -> None:
    """POST stream → SSE events (thinking, suggestion, done)."""
    payload = {
        "consultation_id": 1,
        "doctor_id": 1,
        "latest_message": "你好",
        "idempotency_key": "test-key-001",
    }
    resp = client.post("/coach/consultations/1/suggestions/stream", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse SSE events
    events = _parse_sse_events(resp.text)
    event_names = [e["event"] for e in events]

    assert "thinking" in event_names
    assert "suggestion" in event_names
    assert "done" in event_names


def test_stream_suggestion_idempotent(client: TestClient) -> None:
    """Same idempotency_key → cached result (no thinking event on 2nd call)."""
    payload = {
        "consultation_id": 2,
        "doctor_id": 1,
        "latest_message": "你好",
        "idempotency_key": "idem-key-002",
    }

    # First call
    resp1 = client.post("/coach/consultations/2/suggestions/stream", json=payload)
    events1 = _parse_sse_events(resp1.text)
    event_names1 = [e["event"] for e in events1]
    assert "thinking" in event_names1

    # Second call with same key → cached
    resp2 = client.post("/coach/consultations/2/suggestions/stream", json=payload)
    events2 = _parse_sse_events(resp2.text)
    event_names2 = [e["event"] for e in events2]
    # Should NOT have thinking event (cached)
    assert "thinking" not in event_names2
    assert "suggestion" in event_names2


def test_stream_suggestion_disabled(client: TestClient) -> None:
    """Coach disabled → error event."""
    _coach_service.set_enabled(False)
    payload = {
        "consultation_id": 3,
        "doctor_id": 1,
        "latest_message": "你好",
        "idempotency_key": "test-key-003",
    }
    resp = client.post("/coach/consultations/3/suggestions/stream", json=payload)
    assert resp.status_code == 200

    events = _parse_sse_events(resp.text)
    event_names = [e["event"] for e in events]
    assert "error" in event_names

    # Check error message
    error_event = next(e for e in events if e["event"] == "error")
    assert "disabled" in error_event["data"]["message"].lower()


def test_submit_feedback_accepted(client: TestClient) -> None:
    """POST feedback → recorded=True after a suggestion is produced."""
    # First, produce a suggestion to get a valid suggestion_id
    payload = {
        "consultation_id": 4,
        "doctor_id": 1,
        "latest_message": "你好",
        "idempotency_key": "test-key-004",
    }
    resp = client.post("/coach/consultations/4/suggestions/stream", json=payload)
    events = _parse_sse_events(resp.text)
    suggestion_event = next(e for e in events if e["event"] == "suggestion")
    suggestion_id = suggestion_event["data"]["suggestion_id"]

    # Submit feedback
    feedback_resp = client.post(
        f"/coach/suggestions/{suggestion_id}/feedback",
        json={"feedback": "accepted", "reason": "Good suggestion"},
    )
    assert feedback_resp.status_code == 200
    data = feedback_resp.json()
    assert data["recorded"] is True
    assert data["feedback"] == "accepted"


def test_submit_feedback_unknown_suggestion(client: TestClient) -> None:
    """Unknown suggestion_id → recorded=False."""
    unknown_id = uuid4()
    feedback_resp = client.post(
        f"/coach/suggestions/{unknown_id}/feedback",
        json={"feedback": "rejected"},
    )
    assert feedback_resp.status_code == 200
    data = feedback_resp.json()
    assert data["recorded"] is False


def test_stream_sse_format(client: TestClient) -> None:
    """Verify SSE format (event:, data:, id: lines)."""
    payload = {
        "consultation_id": 5,
        "doctor_id": 1,
        "latest_message": "你好",
        "idempotency_key": "test-key-005",
    }
    resp = client.post("/coach/consultations/5/suggestions/stream", json=payload)
    assert resp.status_code == 200

    # Check raw SSE format
    raw = resp.text
    # Each event block should have "event: " and "data: " lines
    assert "event: thinking" in raw
    assert "data: " in raw
    assert "event: done" in raw

    # Verify JSON-parseable data
    events = _parse_sse_events(raw)
    for event in events:
        assert "event" in event
        assert "data" in event


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
            # End of event block
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
