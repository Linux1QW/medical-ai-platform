# -*- coding: utf-8 -*-
"""V1.1 轻量负载冒烟测试

对 live / ready / status 端点做受控并发请求，输出 JSON 统计。
不触发真实 LLM 调用。

用法：
    python scripts/v11_load_smoke.py [--base-url http://localhost:8000] [--concurrency 20] [--requests 100]

退出码：
    0 — 所有阈值达标
    1 — 阈值未达标或请求失败
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from typing import Any

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx", file=sys.stderr)
    sys.exit(2)


# ── Thresholds ────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "p95_latency_seconds": 0.5,
    "error_rate": 0.01,
}

ENDPOINTS = [
    "/health/live",
    "/health/ready",
]


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _do_request(
    client: httpx.AsyncClient,
    url: str,
    results: list[dict[str, Any]],
) -> None:
    """单次请求，记录延迟和状态"""
    t0 = time.monotonic()
    try:
        resp = await client.get(url, timeout=10.0)
        elapsed = time.monotonic() - t0
        results.append({
            "url": url,
            "status": resp.status_code,
            "latency": elapsed,
            "error": None,
        })
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.append({
            "url": url,
            "status": 0,
            "latency": elapsed,
            "error": str(e),
        })


async def _run_load_test(
    base_url: str,
    concurrency: int,
    total_requests: int,
) -> list[dict[str, Any]]:
    """受控并发负载测试"""
    results: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(client: httpx.AsyncClient, url: str) -> None:
        async with sem:
            await _do_request(client, url, results)

    async with httpx.AsyncClient(base_url=base_url) as client:
        tasks = []
        for i in range(total_requests):
            url = ENDPOINTS[i % len(ENDPOINTS)]
            tasks.append(_bounded(client, url))
        await asyncio.gather(*tasks)

    return results


def _compute_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """计算 p50/p95/p99/error_rate"""
    latencies = sorted(r["latency"] for r in results)
    errors = sum(1 for r in results if r["error"] or r["status"] >= 500)
    total = len(results)

    if not latencies:
        return {
            "total": 0,
            "errors": 0,
            "error_rate": 1.0,
            "p50_latency_seconds": 0,
            "p95_latency_seconds": 0,
            "p99_latency_seconds": 0,
            "mean_latency_seconds": 0,
            "max_latency_seconds": 0,
        }

    def _percentile(data: list[float], pct: float) -> float:
        if not data:
            return 0.0
        k = (len(data) - 1) * (pct / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(data):
            return data[-1]
        return data[f] + (k - f) * (data[c] - data[f])

    return {
        "total": total,
        "errors": errors,
        "error_rate": errors / total if total else 1.0,
        "p50_latency_seconds": round(_percentile(latencies, 50), 4),
        "p95_latency_seconds": round(_percentile(latencies, 95), 4),
        "p99_latency_seconds": round(_percentile(latencies, 99), 4),
        "mean_latency_seconds": round(statistics.mean(latencies), 4),
        "max_latency_seconds": round(max(latencies), 4),
    }


def _check_thresholds(stats: dict[str, Any]) -> bool:
    """检查是否达标"""
    ok = True
    for key, threshold in THRESHOLDS.items():
        value = stats.get(key, 0)
        if isinstance(value, (int, float)) and value > threshold:
            print(
                f"FAIL: {key}={value} exceeds threshold={threshold}",
                file=sys.stderr,
            )
            ok = False
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> int:
    parser = argparse.ArgumentParser(description="V1.1 load smoke test")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the backend",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Max concurrent requests",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Total number of requests",
    )
    args = parser.parse_args()

    print(
        f"Running load smoke test: {args.requests} requests, "
        f"concurrency={args.concurrency}, base_url={args.base_url}"
    )

    results = await _run_load_test(args.base_url, args.concurrency, args.requests)
    stats = _compute_stats(results)

    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if not _check_thresholds(stats):
        print("SMOKE TEST FAILED", file=sys.stderr)
        return 1

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
