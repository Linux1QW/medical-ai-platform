"""V1.2 load test: verify coach latency under concurrent load."""
from __future__ import annotations

import asyncio
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.agent_runtime.contracts import CoachContextView, VisiblePatientProfile, VisibleMessage
from app.agent_runtime.graph import CoachGraph, CoachGraphState


async def run_single_request(turn: int) -> float:
    """Run a single coach request and return latency in ms."""
    context_view = CoachContextView(
        consultation_id=1,
        doctor_id=1,
        visible_patient=VisiblePatientProfile(age=45, gender="男", chief_complaint="头痛"),
        messages=[VisibleMessage(sequence=1, role="doctor", content="你好，我最近头痛")],
    )
    state = CoachGraphState(
        context_view=context_view,
        latest_message="你好，我最近头痛",
        turn_no=turn,
    )
    graph = CoachGraph()
    start = time.perf_counter()
    await graph.run(state)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms


async def run_load_test(concurrency: int = 10, total_requests: int = 50) -> dict:
    """Run load test with given concurrency."""
    print(f"Load test: {total_requests} requests, concurrency={concurrency}")
    
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0
    
    async def bounded_request(turn: int) -> None:
        nonlocal errors
        async with sem:
            try:
                latency = await run_single_request(turn)
                latencies.append(latency)
            except Exception:
                errors += 1
    
    start = time.perf_counter()
    await asyncio.gather(*(bounded_request(i) for i in range(total_requests)))
    total_time = time.perf_counter() - start
    
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0
    
    error_rate = errors / total_requests if total_requests > 0 else 0
    
    print(f"  Total time: {total_time:.2f}s")
    print(f"  p50 latency: {p50:.1f}ms")
    print(f"  p95 latency: {p95:.1f}ms (threshold: ≤2500ms)")
    print(f"  p99 latency: {p99:.1f}ms")
    print(f"  Error rate: {error_rate:.1%} (threshold: <1%)")
    
    passed = p95 <= 2500 and error_rate < 0.01
    print(f"\nLoad test: {'PASS' if passed else 'FAIL'}")
    return {"p95": p95, "error_rate": error_rate, "passed": passed}


if __name__ == "__main__":
    result = asyncio.run(run_load_test())
    sys.exit(0 if result["passed"] else 1)
