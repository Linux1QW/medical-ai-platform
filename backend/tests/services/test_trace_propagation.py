# -*- coding: utf-8 -*-
"""Task 11 — 全链路 Trace 和观测测试

覆盖：
- TraceContext 创建与默认值
- serialize / restore 往返一致性
- Celery 重试：trace_id 不变、attempt 增加
- 脱敏函数：姓名、电话、身份证、地址
- trace 缺失时自动生成新 ID
- 上下文绑定和恢复
"""


from app.services.observability.trace_context import (
    TraceContext,
    bind_trace_context,
    get_current_trace_context,
    new_trace_id,
    restore_trace_context,
    sanitize_for_observability,
    serialize_trace_context,
)

# ── TraceContext 基础 ─────────────────────────────────────────────────────


class TestTraceContext:
    def test_default_values(self):
        ctx = TraceContext()
        assert ctx.trace_id is not None
        assert len(ctx.trace_id) > 0
        assert ctx.run_id is None
        assert ctx.consultation_id is None
        assert ctx.attempt == 1

    def test_custom_values(self):
        ctx = TraceContext(
            trace_id="trace-001",
            run_id="run-001",
            consultation_id="consult-001",
            celery_task_id="celery-001",
            graph_thread_id="thread-001",
            attempt=2,
        )
        assert ctx.trace_id == "trace-001"
        assert ctx.run_id == "run-001"
        assert ctx.attempt == 2

    def test_trace_id_auto_generated(self):
        ctx1 = TraceContext()
        ctx2 = TraceContext()
        assert ctx1.trace_id != ctx2.trace_id


# ── serialize / restore 往返 ──────────────────────────────────────────────


class TestTraceSerialization:
    def test_round_trip(self):
        ctx = TraceContext(
            trace_id="trace-abc",
            run_id="run-xyz",
            consultation_id="consult-123",
            celery_task_id="celery-456",
            graph_thread_id="thread-789",
            attempt=3,
        )
        payload = serialize_trace_context(ctx)
        restored = restore_trace_context(payload)
        assert restored.trace_id == ctx.trace_id
        assert restored.run_id == ctx.run_id
        assert restored.consultation_id == ctx.consultation_id
        assert restored.celery_task_id == ctx.celery_task_id
        assert restored.graph_thread_id == ctx.graph_thread_id
        assert restored.attempt == ctx.attempt

    def test_serialize_returns_dict(self):
        ctx = TraceContext(trace_id="t1", run_id="r1")
        payload = serialize_trace_context(ctx)
        assert isinstance(payload, dict)
        assert "trace_id" in payload
        assert "run_id" in payload

    def test_restore_with_missing_fields(self):
        payload = {"trace_id": "t1"}
        ctx = restore_trace_context(payload)
        assert ctx.trace_id == "t1"
        assert ctx.run_id is None
        assert ctx.attempt == 1

    def test_restore_with_empty_dict(self):
        ctx = restore_trace_context({})
        assert ctx.trace_id is not None  # 自动生成


# ── Celery 重试语义 ──────────────────────────────────────────────────────


class TestCeleryRetrySemantics:
    def test_retry_increments_attempt(self):
        ctx = TraceContext(trace_id="trace-retry", run_id="run-retry", attempt=1)
        # 模拟重试：复用 trace_id 和 run_id，attempt +1
        retry_ctx = TraceContext(
            trace_id=ctx.trace_id,
            run_id=ctx.run_id,
            attempt=ctx.attempt + 1,
        )
        assert retry_ctx.trace_id == "trace-retry"
        assert retry_ctx.run_id == "run-retry"
        assert retry_ctx.attempt == 2

    def test_retry_preserves_trace_id(self):
        """Celery 重试应保持 trace_id 不变"""
        original = TraceContext(trace_id="same-trace", run_id="same-run", attempt=1)
        payload = serialize_trace_context(original)
        # 模拟重试时修改 attempt
        payload["attempt"] = 2
        restored = restore_trace_context(payload)
        assert restored.trace_id == "same-trace"
        assert restored.attempt == 2


# ── 脱敏函数 ─────────────────────────────────────────────────────────────


class TestSanitization:
    def test_sanitize_phone_number(self):
        text = "患者电话13812345678"
        result = sanitize_for_observability(text)
        assert "13812345678" not in result
        assert "138****5678" in result or "***" in result

    def test_sanitize_id_card(self):
        text = "身份证号110101199001011234"
        result = sanitize_for_observability(text)
        assert "110101199001011234" not in result

    def test_sanitize_name(self):
        text = "患者姓名张三丰"
        result = sanitize_for_observability(text)
        assert "张三丰" not in result or "***" in result

    def test_sanitize_preserves_clinical_content(self):
        text = "主诉：胸闷三天，伴有气短"
        result = sanitize_for_observability(text)
        assert "胸闷" in result
        assert "气短" in result

    def test_sanitize_dict(self):
        data = {
            "prompt": "患者张三，电话13800001111，主诉头痛",
            "response": "建议检查",
        }
        result = sanitize_for_observability(data)
        assert isinstance(result, dict)
        assert "13800001111" not in result["prompt"]
        assert "建议检查" in result["response"]

    def test_sanitize_list(self):
        data = ["患者李明", "电话13900002222"]
        result = sanitize_for_observability(data)
        assert isinstance(result, list)

    def test_sanitize_empty_string(self):
        assert sanitize_for_observability("") == ""

    def test_sanitize_none(self):
        assert sanitize_for_observability(None) is None


# ── 上下文绑定 ────────────────────────────────────────────────────────────


class TestContextBinding:
    def test_bind_and_get(self):
        ctx = TraceContext(trace_id="bind-test", run_id="run-bind")
        bind_trace_context(ctx)
        current = get_current_trace_context()
        assert current is not None
        assert current.trace_id == "bind-test"

    def test_bind_returns_token(self):
        ctx = TraceContext(trace_id="token-test")
        token = bind_trace_context(ctx)
        assert token is not None

    def test_new_trace_id_format(self):
        tid = new_trace_id()
        assert isinstance(tid, str)
        assert len(tid) > 10
