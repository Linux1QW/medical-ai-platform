# -*- coding: utf-8 -*-
"""Prometheus 指标定义

所有可观测指标集中定义于此，供各模块引用并更新。
指标命名遵循 Prometheus 规范：snake_case + 单位后缀。
"""

from prometheus_client import Counter, Histogram, Gauge

# ── HTTP 指标 ────────────────────────────────────────────────────────────────

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "path"],
)

# ── LLM 指标 ─────────────────────────────────────────────────────────────────

LLM_CALLS_TOTAL = Counter(
    "llm_calls_total",
    "Total LLM calls",
    ["model", "status"],
)

LLM_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration",
    ["model"],
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total LLM tokens",
    ["model", "type"],
)

# ── RAG 指标 ─────────────────────────────────────────────────────────────────

RAG_RETRIEVAL_DURATION = Histogram(
    "rag_retrieval_duration_seconds",
    "RAG retrieval duration",
)

# ── 评估指标 ─────────────────────────────────────────────────────────────────

EVALUATION_RUNS_TOTAL = Counter(
    "evaluation_runs_total",
    "Total evaluation runs",
    ["status"],
)

# ── 缓存指标 ─────────────────────────────────────────────────────────────────

CACHE_HIT_RATE = Gauge(
    "cache_hit_rate",
    "Cache hit rate",
    ["cache"],
)
