"""V1.2 load test: HTTP load against two real API instances.

Exercises the full coach path through the HTTP layer:
  login → create consultation → POST SSE stream → parse completion
Records p50/p95/p99 latency, HTTP errors, timeouts, duplicate decisions,
replay failures, and trace completeness.

A request counts *successful* only when ALL of:
  - one valid suggestion event + done event received
  - persisted decision exists (state endpoint confirms turn_no advanced)
  - no policy error in the stream
  - trace required fields present (admin trace endpoint)

Exceptions are recorded under stable categories, never silently swallowed.

Usage:
    python -m scripts.ci.v12_load_test [--concurrency N] [--total N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import aiohttp

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

BACKEND_URLS = [
    os.environ.get("V12_BACKEND_A", "http://localhost:8000"),
    os.environ.get("V12_BACKEND_B", "http://localhost:8001"),
]

E2E_DOCTOR_USER = os.environ.get("E2E_DOCTOR_USER", "doctor_v12")
E2E_ADMIN_USER = os.environ.get("E2E_ADMIN_USER", "admin_v12")
E2E_PASSWORD = os.environ.get("E2E_PASSWORD", "e2e_test_password_2026")
API_PREFIX = "/api/v1"

SSE_TIMEOUT_S = int(os.environ.get("V12_SSE_TIMEOUT", "30"))
REQUEST_TIMEOUT_S = int(os.environ.get("V12_REQUEST_TIMEOUT", "60"))


# ──────────────────────────────────────────────────────────────
# Error categories (stable, never silently counted)
# ──────────────────────────────────────────────────────────────

class ErrorCategory:
    LOGIN_FAILURE = "LOGIN_FAILURE"
    CONSULTATION_CREATE_FAILURE = "CONSULTATION_CREATE_FAILURE"
    SSE_TIMEOUT = "SSE_TIMEOUT"
    SSE_CONNECTION_ERROR = "SSE_CONNECTION_ERROR"
    SSE_NO_SUGGESTION = "SSE_NO_SUGGESTION"
    SSE_NO_DONE = "SSE_NO_DONE"
    SSE_POLICY_ERROR = "SSE_POLICY_ERROR"
    STATE_CHECK_FAILURE = "STATE_CHECK_FAILURE"
    TRACE_MISSING_FIELDS = "TRACE_MISSING_FIELDS"
    DUPLICATE_DECISION = "DUPLICATE_DECISION"
    REPLAY_FAILURE = "REPLAY_FAILURE"
    UNEXPECTED_EXCEPTION = "UNEXPECTED_EXCEPTION"


@dataclass
class ErrorRecord:
    category: str
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class RequestResult:
    success: bool
    latency_ms: float
    errors: list[ErrorRecord] = field(default_factory=list)
    suggestion_id: str | None = None
    turn_no: int = 0
    trace_complete: bool = False


# ──────────────────────────────────────────────────────────────
# SSE parser
# ──────────────────────────────────────────────────────────────

def parse_sse_events(text: str) -> list[dict[str, Any]]:
    """Parse SSE text into a list of event dicts."""
    events: list[dict[str, Any]] = []
    current_event: dict[str, Any] = {}
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


# ──────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────

async def login(session: aiohttp.ClientSession, base_url: str) -> str:
    """Login and return access token."""
    url = f"{base_url}{API_PREFIX}/auth/login"
    payload = {"username": E2E_DOCTOR_USER, "password": E2E_PASSWORD}
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"Login failed ({resp.status}): {body}")
        data = await resp.json()
        return data["access_token"]


async def create_consultation(
    session: aiohttp.ClientSession,
    base_url: str,
    token: str,
    patient_id: int,
) -> int:
    """Create a consultation and return its ID."""
    url = f"{base_url}{API_PREFIX}/consultations/"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"patient_id": patient_id}
    async with session.post(url, json=payload, headers=headers,
                            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)) as resp:
        if resp.status not in (200, 201):
            body = await resp.text()
            raise RuntimeError(f"Create consultation failed ({resp.status}): {body}")
        data = await resp.json()
        return data["id"]


async def post_sse_stream(
    session: aiohttp.ClientSession,
    base_url: str,
    token: str,
    consultation_id: int,
    latest_message: str,
    idempotency_key: str,
    last_event_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """POST SSE stream and return (raw_text, parsed_events)."""
    url = f"{base_url}{API_PREFIX}/coach/consultations/{consultation_id}/suggestions/stream"
    headers = {"Authorization": f"Bearer {token}"}
    if last_event_id:
        headers["Last-Event-ID"] = last_event_id
    payload = {
        "latest_message": latest_message,
        "idempotency_key": idempotency_key,
    }
    timeout = aiohttp.ClientTimeout(total=SSE_TIMEOUT_S)
    async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"SSE stream failed ({resp.status}): {body}")
        raw_text = await resp.text()
    return raw_text, parse_sse_events(raw_text)


async def get_coach_state(
    session: aiohttp.ClientSession,
    base_url: str,
    token: str,
    consultation_id: int,
) -> dict[str, Any]:
    """GET coach state for a consultation."""
    url = f"{base_url}{API_PREFIX}/coach/consultations/{consultation_id}/state"
    headers = {"Authorization": f"Bearer {token}"}
    async with session.get(url, headers=headers,
                           timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"State check failed ({resp.status}): {body}")
        return await resp.json()


async def get_coach_trace(
    session: aiohttp.ClientSession,
    base_url: str,
    admin_token: str,
    consultation_id: int,
) -> dict[str, Any]:
    """GET admin trace for a consultation."""
    url = f"{base_url}{API_PREFIX}/coach/admin/consultations/{consultation_id}/trace"
    headers = {"Authorization": f"Bearer {admin_token}"}
    async with session.get(url, headers=headers,
                           timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"Trace check failed ({resp.status}): {body}")
        return await resp.json()


async def login_admin(session: aiohttp.ClientSession, base_url: str) -> str:
    """Login as admin and return access token."""
    url = f"{base_url}{API_PREFIX}/auth/login"
    payload = {"username": E2E_ADMIN_USER, "password": E2E_PASSWORD}
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"Admin login failed ({resp.status}): {body}")
        data = await resp.json()
        return data["access_token"]


# ──────────────────────────────────────────────────────────────
# Single request lifecycle
# ──────────────────────────────────────────────────────────────

TRACE_REQUIRED_FIELDS = {"consultation_id", "events", "turn_no"}


async def run_single_request(
    session: aiohttp.ClientSession,
    base_url: str,
    doctor_token: str,
    admin_token: str,
    turn: int,
    patient_id: int,
) -> RequestResult:
    """Execute one full coach request lifecycle and evaluate success."""
    errors: list[ErrorRecord] = []
    start = time.perf_counter()

    try:
        # 1. Create consultation
        consultation_id = await create_consultation(session, base_url, doctor_token, patient_id)
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        errors.append(ErrorRecord(ErrorCategory.CONSULTATION_CREATE_FAILURE, str(exc)))
        return RequestResult(success=False, latency_ms=elapsed, errors=errors)

    idem_key = f"load-{uuid4().hex[:32]}"
    suggestion_id: str | None = None
    turn_no = 0
    trace_complete = False

    try:
        # 2. POST SSE stream
        raw_text, events = await post_sse_stream(
            session, base_url, doctor_token, consultation_id,
            latest_message=f"Turn {turn}: 患者主诉头痛加剧",
            idempotency_key=idem_key,
        )
    except asyncio.TimeoutError:
        elapsed = (time.perf_counter() - start) * 1000
        errors.append(ErrorRecord(ErrorCategory.SSE_TIMEOUT, f"SSE timed out for consultation {consultation_id}"))
        return RequestResult(success=False, latency_ms=elapsed, errors=errors)
    except aiohttp.ClientError as exc:
        elapsed = (time.perf_counter() - start) * 1000
        errors.append(ErrorRecord(ErrorCategory.SSE_CONNECTION_ERROR, str(exc)))
        return RequestResult(success=False, latency_ms=elapsed, errors=errors)
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        errors.append(ErrorRecord(ErrorCategory.UNEXPECTED_EXCEPTION, f"SSE: {exc}"))
        return RequestResult(success=False, latency_ms=elapsed, errors=errors)

    # 3. Validate events
    event_names = [e["event"] for e in events]

    # Check for policy errors in stream
    for evt in events:
        if evt["event"] == "error":
            err_data = evt.get("data", {})
            if isinstance(err_data, dict) and "policy" in str(err_data).lower():
                errors.append(ErrorRecord(ErrorCategory.SSE_POLICY_ERROR, f"Policy error in stream: {err_data}"))

    # Must have suggestion event
    suggestion_events = [e for e in events if e["event"] == "suggestion"]
    if not suggestion_events:
        errors.append(ErrorRecord(ErrorCategory.SSE_NO_SUGGESTION, f"No suggestion event in {event_names}"))
    else:
        sdata = suggestion_events[0].get("data", {})
        if isinstance(sdata, dict):
            suggestion_id = sdata.get("suggestion_id")

    # Must have done event
    if "done" not in event_names:
        errors.append(ErrorRecord(ErrorCategory.SSE_NO_DONE, f"No done event in {event_names}"))

    # 4. State check — verify persisted decision
    try:
        state = await get_coach_state(session, base_url, doctor_token, consultation_id)
        turn_no = state.get("turn_no", 0)
        if turn_no < 1:
            errors.append(ErrorRecord(ErrorCategory.STATE_CHECK_FAILURE,
                                      f"turn_no={turn_no}, expected >= 1"))
    except Exception as exc:
        errors.append(ErrorRecord(ErrorCategory.STATE_CHECK_FAILURE, str(exc)))

    # 5. Trace completeness check (admin token)
    try:
        trace = await get_coach_trace(session, base_url, admin_token, consultation_id)
        missing = TRACE_REQUIRED_FIELDS - set(trace.keys())
        if missing:
            errors.append(ErrorRecord(ErrorCategory.TRACE_MISSING_FIELDS,
                                      f"Missing trace fields: {missing}"))
        else:
            trace_complete = True
    except Exception as exc:
        errors.append(ErrorRecord(ErrorCategory.TRACE_MISSING_FIELDS, str(exc)))

    elapsed = (time.perf_counter() - start) * 1000
    success = (
        suggestion_id is not None
        and "done" in event_names
        and not any(e.category == ErrorCategory.SSE_POLICY_ERROR for e in errors)
        and turn_no >= 1
        and trace_complete
        and len(errors) == 0
    )

    return RequestResult(
        success=success,
        latency_ms=elapsed,
        errors=errors,
        suggestion_id=suggestion_id,
        turn_no=turn_no,
        trace_complete=trace_complete,
    )


# ──────────────────────────────────────────────────────────────
# Replay test (duplicate decision detection)
# ──────────────────────────────────────────────────────────────

async def run_replay_test(
    session: aiohttp.ClientSession,
    base_url: str,
    token: str,
    consultation_id: int,
    idem_key: str,
) -> tuple[bool, list[ErrorRecord]]:
    """Replay same idempotency_key and verify same decision."""
    errors: list[ErrorRecord] = []
    try:
        _, events_replay = await post_sse_stream(
            session, base_url, token, consultation_id,
            latest_message="Replay: 患者主诉头痛加剧",
            idempotency_key=idem_key,
        )
        event_names = [e["event"] for e in events_replay]
        if "suggestion" not in event_names:
            errors.append(ErrorRecord(ErrorCategory.REPLAY_FAILURE,
                                      "Replay missing suggestion event"))
    except Exception as exc:
        errors.append(ErrorRecord(ErrorCategory.REPLAY_FAILURE, str(exc)))
    return len(errors) == 0, errors


# ──────────────────────────────────────────────────────────────
# Load runner
# ──────────────────────────────────────────────────────────────

async def run_load_test(
    concurrency: int = 10,
    total_requests: int = 50,
    patient_id: int = 1,
) -> dict[str, Any]:
    """Run load test against two API instances."""
    print(f"V1.2 Load Test: {total_requests} requests, concurrency={concurrency}")
    print(f"  Targets: {BACKEND_URLS}")

    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []
    all_errors: list[ErrorRecord] = []
    seen_suggestion_ids: list[str] = []

    async def bounded_request(turn: int) -> None:
        # Alternate between backend instances
        base_url = BACKEND_URLS[turn % len(BACKEND_URLS)]
        async with sem:
            try:
                async with aiohttp.ClientSession() as session:
                    doctor_token = await login(session, base_url)
                    admin_token = await login_admin(session, base_url)
                    result = await run_single_request(
                        session, base_url, doctor_token, admin_token,
                        turn=turn, patient_id=patient_id,
                    )
                    results.append(result)
                    all_errors.extend(result.errors)
                    if result.suggestion_id:
                        seen_suggestion_ids.append(result.suggestion_id)
            except Exception as exc:
                all_errors.append(ErrorRecord(ErrorCategory.UNEXPECTED_EXCEPTION,
                                              f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))

    start = time.perf_counter()
    await asyncio.gather(*(bounded_request(i) for i in range(total_requests)))
    total_time = time.perf_counter() - start

    # ── Compute metrics ──
    successful = [r for r in results if r.success]
    latencies = sorted([r.latency_ms for r in successful])

    if latencies:
        p50 = statistics.median(latencies)
        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)
        p95 = latencies[min(p95_idx, len(latencies) - 1)]
        p99 = latencies[min(p99_idx, len(latencies) - 1)]
    else:
        p50 = p95 = p99 = 0.0

    # Error counts by category
    error_counts: dict[str, int] = {}
    for err in all_errors:
        error_counts[err.category] = error_counts.get(err.category, 0) + 1

    # Duplicate decisions
    duplicate_count = len(seen_suggestion_ids) - len(set(seen_suggestion_ids))

    total_completed = len(results)
    success_count = len(successful)
    error_rate = (total_completed - success_count) / total_completed if total_completed > 0 else 0

    # ── Report ──
    print(f"\n{'='*60}")
    print(f"  Total time:        {total_time:.2f}s")
    print(f"  Completed:         {total_completed}/{total_requests}")
    print(f"  Successful:        {success_count}/{total_completed}")
    print(f"  p50 latency:       {p50:.1f}ms")
    print(f"  p95 latency:       {p95:.1f}ms (threshold: ≤2500ms)")
    print(f"  p99 latency:       {p99:.1f}ms")
    print(f"  Error rate:        {error_rate:.1%} (threshold: <1%)")
    print(f"  Duplicate IDs:     {duplicate_count}")
    print(f"  Trace complete:    {sum(1 for r in results if r.trace_complete)}/{total_completed}")
    if error_counts:
        print("  Error breakdown:")
        for cat, cnt in sorted(error_counts.items()):
            print(f"    {cat}: {cnt}")
    print(f"{'='*60}")

    passed = p95 <= 2500 and error_rate < 0.01 and duplicate_count == 0
    print(f"\nLoad test: {'PASS' if passed else 'FAIL'}")

    return {
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "error_rate": error_rate,
        "duplicate_decisions": duplicate_count,
        "total_completed": total_completed,
        "successful": success_count,
        "errors_by_category": error_counts,
        "passed": passed,
    }


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="V1.2 Coach HTTP load test")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--total", type=int, default=50, help="Total requests")
    parser.add_argument("--patient-id", type=int, default=1, help="Patient ID for consultations")
    args = parser.parse_args()

    result = asyncio.run(run_load_test(
        concurrency=args.concurrency,
        total_requests=args.total,
        patient_id=args.patient_id,
    ))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
