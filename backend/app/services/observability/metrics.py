# -*- coding: utf-8 -*-
"""Prometheus 指标定义

所有可观测指标集中定义于此，供各模块引用并更新。
指标命名遵循 Prometheus 规范：snake_case + 单位后缀。
"""

from collections.abc import Mapping

from prometheus_client import Counter, Gauge, Histogram

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

RAG_INDEX_GENERATION = Gauge(
    "rag_index_generation",
    "Currently observed RAG index generation",
    ["generation"],
)
BM25_LOAD_SECONDS = Histogram(
    "bm25_load_seconds",
    "BM25 index load duration",
)
BM25_QUERY_SECONDS = Histogram(
    "bm25_query_seconds",
    "BM25 query duration",
)
BM25_CANDIDATES = Histogram(
    "bm25_candidates",
    "Number of BM25 candidates returned",
)
BM25_TOP_SCORE = Histogram(
    "bm25_top_score",
    "Top BM25 score",
)
LEXICAL_EXPANSION_COUNT = Counter(
    "lexical_expansion_count",
    "Number of lexical query expansions",
)
FILTER_FALLBACK = Counter(
    "filter_fallback",
    "Number of metadata filter fallbacks",
)
CACHE_HIT = Counter(
    "cache_hit",
    "Retrieval cache outcomes",
    ["result"],
)
RETRIEVAL_LEVEL = Counter(
    "retrieval_level",
    "Retrieval level selected",
    ["level"],
)
RAG_CHANNEL_CANDIDATES = Histogram(
    "rag_channel_candidates",
    "Candidates returned by each retrieval channel",
    ["channel"],
)


def record_rag_observability(trace: Mapping[str, object]) -> dict:
    """Record bounded RAG metrics and return a normalized trace payload.

    Duration values are observed only when the retrieval path measured them;
    missing timings stay missing instead of being reported as fake zeros.
    """
    normalized = dict(trace)
    normalized.setdefault("cache_hit", False)
    normalized.setdefault("channel_candidates", {})

    generation = normalized.get("index_generation")
    if generation is not None:
        RAG_INDEX_GENERATION.labels(generation=str(generation)).set(1)

    numeric_metrics = (
        ("bm25_load_seconds", BM25_LOAD_SECONDS),
        ("bm25_query_seconds", BM25_QUERY_SECONDS),
        ("bm25_candidates", BM25_CANDIDATES),
        ("bm25_top_score", BM25_TOP_SCORE),
    )
    for field, metric in numeric_metrics:
        value = normalized.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metric.observe(float(value))

    expansion_count = normalized.get("lexical_expansion_count")
    if isinstance(expansion_count, (int, float)) and expansion_count > 0:
        LEXICAL_EXPANSION_COUNT.inc(float(expansion_count))
    if normalized.get("filter_fallback"):
        FILTER_FALLBACK.inc()
    CACHE_HIT.labels(result="hit" if normalized["cache_hit"] else "miss").inc()

    retrieval_level = normalized.get("retrieval_level")
    if retrieval_level:
        RETRIEVAL_LEVEL.labels(level=str(retrieval_level)).inc()
    channels = normalized.get("channel_candidates")
    if isinstance(channels, Mapping):
        for channel, count in channels.items():
            if isinstance(count, (int, float)) and not isinstance(count, bool):
                RAG_CHANNEL_CANDIDATES.labels(channel=str(channel)).observe(float(count))
    return normalized

# ── 工具调用指标 ──────────────────────────────────────────────────────────────

TOOL_CALLS_TOTAL = Counter(
    "tool_calls_total",
    "Total tool calls",
    ["tool", "agent", "status"],
)

TOOL_CALL_DURATION = Histogram(
    "tool_call_duration_seconds",
    "Tool call duration",
    ["tool"],
)

# ── 评估指标 ─────────────────────────────────────────────────────────────────

EVALUATION_RUNS_TOTAL = Counter(
    "evaluation_runs_total",
    "Total evaluation runs",
    ["status", "error_code"],
)

EVALUATION_RUN_DURATION = Histogram(
    "evaluation_run_duration_seconds",
    "Evaluation run duration",
    ["status"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
)

EVALUATION_QUEUE_WAIT = Histogram(
    "evaluation_queue_wait_seconds",
    "Time a run spends in queue before being claimed",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)

EVALUATION_ACTIVE_RUNS = Gauge(
    "evaluation_active_runs",
    "Currently active evaluation runs",
    ["status"],
)

EVALUATION_RETRIES_TOTAL = Counter(
    "evaluation_retries_total",
    "Total evaluation run retries",
    ["error_code"],
)

EVALUATION_CANCELLATIONS_TOTAL = Counter(
    "evaluation_cancellations_total",
    "Total evaluation cancellations",
    ["phase"],
)

EVALUATION_PROGRESS_PUBLISH_TOTAL = Counter(
    "evaluation_progress_publish_total",
    "Total progress publish operations",
    ["result"],
)

EVALUATION_PROGRESS_DELIVERY = Histogram(
    "evaluation_progress_delivery_seconds",
    "Progress delivery latency (Redis publish → WS arrival)",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
)

EVALUATION_STALE_RUNS_TOTAL = Counter(
    "evaluation_stale_runs_total",
    "Total stale runs detected",
    ["reason"],
)

EVALUATION_RUN_LEASE_LOST_TOTAL = Counter(
    "evaluation_run_lease_lost_total",
    "Total run lease lost events",
    ["phase"],
)

EVALUATION_OUTBOX_EVENTS_TOTAL = Counter(
    "evaluation_outbox_events_total",
    "Total outbox events processed",
    ["result"],
)

EVALUATION_OUTBOX_PENDING = Gauge(
    "evaluation_outbox_pending",
    "Current number of pending outbox events",
)

EVALUATION_OUTBOX_OLDEST_SECONDS = Gauge(
    "evaluation_outbox_oldest_seconds",
    "Age of the oldest pending outbox event in seconds",
)

EVALUATION_DISPATCH_DURATION = Histogram(
    "evaluation_dispatch_duration_seconds",
    "Dispatch tick duration",
    ["result"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

EVALUATION_DISPATCH_DEAD_LETTER_TOTAL = Counter(
    "evaluation_dispatch_dead_letter_total",
    "Total dispatches moved to dead letter",
    ["reason"],
)

EVALUATION_DISPATCH_BREAKER_OPEN = Gauge(
    "evaluation_dispatch_breaker_open",
    "Whether the dispatch circuit breaker is open (1=open, 0=closed)",
)

# ── 基础设施指标 ─────────────────────────────────────────────────────────────

REDIS_DEPENDENCY_STATUS = Gauge(
    "redis_dependency_status",
    "Redis dependency health (1=healthy, 0=degraded)",
    ["role"],
)

BACKUP_LAST_SUCCESS_TIMESTAMP = Gauge(
    "backup_last_success_timestamp_seconds",
    "Timestamp of last successful backup (Unix epoch seconds)",
)

REVIEW_QUEUE_DEPTH = Gauge(
    "review_queue_depth",
    "Current number of items in review queue",
)

REVIEW_COMPLETION_SECONDS = Histogram(
    "review_completion_seconds",
    "Review completion duration",
    buckets=(1, 5, 10, 30, 60, 300, 600, 1800, 3600),
)

# ── 缓存指标 ─────────────────────────────────────────────────────────────────

CACHE_HIT_RATE = Gauge(
    "cache_hit_rate",
    "Cache hit rate",
    ["cache"],
)

# ── Coach 安全门控指标 ────────────────────────────────────────────────────────

COACH_SAFETY_CHECKS_TOTAL = Counter(
    "coach_safety_checks_total",
    "Total coach safety checks performed",
    ["check_category", "result"],
)

COACH_SAFETY_GATE_DECISIONS = Counter(
    "coach_safety_gate_decisions_total",
    "Total coach safety gate decisions",
    ["decision", "risk_level"],
)

COACH_POLICY_BLOCKS_TOTAL = Counter(
    "coach_policy_blocks_total",
    "Total coach policy blocks",
    ["reason"],
)

COACH_EVENTS_TOTAL = Counter(
    "coach_events_total",
    "Total coach lifecycle events",
    ["event_type", "status"],
)

COACH_SUGGESTION_DURATION = Histogram(
    "coach_suggestion_duration_seconds",
    "Coach suggestion generation duration",
    ["agent_name"],
)
