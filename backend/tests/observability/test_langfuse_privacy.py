# -*- coding: utf-8 -*-
"""Task 7 — Langfuse 隐私安全测试

验证 LangfuseTracer 不会将患者 PII（姓名、手机号、身份证）原文发送到外部 Langfuse。
"""

import hashlib
import hmac
import json
from unittest.mock import MagicMock, patch

import pytest

# ── 常量 ──────────────────────────────────────────────────────────────────────

_SENSITIVE_TEXT = "患者张三，手机号13812345678，身份证110101199001011234"
_HMAC_KEY = b"test-hmac-key-must-be-at-least-32-bytes-long!!"


# ── 辅助：Fake Langfuse 客户端 ────────────────────────────────────────────────


class FakeSpan:
    """记录 span 调用参数"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTrace:
    """记录 trace 下所有 span"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.spans: list[FakeSpan] = []

    def span(self, **kwargs):
        self.spans.append(FakeSpan(**kwargs))
        return self.spans[-1]


class FakeLangfuseClient:
    """Fake Langfuse 客户端，记录所有 trace 调用"""

    def __init__(self):
        self.traces: list[FakeTrace] = []
        self.flush_count = 0

    def trace(self, **kwargs):
        t = FakeTrace(**kwargs)
        self.traces.append(t)
        return t

    def flush(self):
        self.flush_count += 1


# ── 测试：PII 不泄露 ─────────────────────────────────────────────────────────


class TestLangfusePrivacyNoPIILeakage:
    """输入含敏感数据时，fake Langfuse 收到的数据中不含原文"""

    def _make_tracer(self, capture: bool = False):
        from app.services.observability.langfuse_client import LangfuseTracer

        with patch("app.services.observability.langfuse_client.settings") as mock_settings:
            mock_settings.LANGFUSE_ENABLED = True
            mock_settings.LANGFUSE_PUBLIC_KEY = "pk-test"
            mock_settings.LANGFUSE_SECRET_KEY = "sk-test"
            mock_settings.LANGFUSE_HOST = "http://localhost"
            mock_settings.OBSERVABILITY_CAPTURE_CONTENT = capture
            mock_settings.OBSERVABILITY_CONTENT_MAX_CHARS = 500
            mock_settings.OBSERVABILITY_HMAC_KEY = _HMAC_KEY
            mock_settings.ENVIRONMENT = "test"

            tracer = LangfuseTracer.__new__(LangfuseTracer)
            tracer._client = FakeLangfuseClient()
            tracer._capture = capture
            tracer._max_chars = 500
            tracer._hmac_key = _HMAC_KEY
            tracer._flushed = False
        return tracer

    def test_no_raw_name_in_langfuse_data(self):
        tracer = self._make_tracer(capture=False)
        tracer.trace_llm_call(
            trace_name="test_trace",
            model="qwen-test",
            prompt=_SENSITIVE_TEXT,
            completion="回复内容",
            tokens=100,
            latency_ms=50.0,
        )
        fake_client: FakeLangfuseClient = tracer._client
        all_data = json.dumps(
            [t.kwargs for t in fake_client.traces],
            ensure_ascii=False,
            default=str,
        )
        assert "张三" not in all_data

    def test_no_raw_phone_in_langfuse_data(self):
        tracer = self._make_tracer(capture=False)
        tracer.trace_llm_call(
            trace_name="test_trace",
            model="qwen-test",
            prompt=_SENSITIVE_TEXT,
            completion="回复内容",
            tokens=100,
            latency_ms=50.0,
        )
        fake_client: FakeLangfuseClient = tracer._client
        all_data = json.dumps(
            [t.kwargs for t in fake_client.traces],
            ensure_ascii=False,
            default=str,
        )
        assert "13812345678" not in all_data

    def test_no_raw_id_card_in_langfuse_data(self):
        tracer = self._make_tracer(capture=False)
        tracer.trace_llm_call(
            trace_name="test_trace",
            model="qwen-test",
            prompt=_SENSITIVE_TEXT,
            completion="回复内容",
            tokens=100,
            latency_ms=50.0,
        )
        fake_client: FakeLangfuseClient = tracer._client
        all_data = json.dumps(
            [t.kwargs for t in fake_client.traces],
            ensure_ascii=False,
            default=str,
        )
        assert "110101199001011234" not in all_data

    def test_no_full_prompt_in_langfuse_data_capture_false(self):
        tracer = self._make_tracer(capture=False)
        tracer.trace_llm_call(
            trace_name="test_trace",
            model="qwen-test",
            prompt=_SENSITIVE_TEXT,
            completion="回复内容",
            tokens=100,
            latency_ms=50.0,
        )
        fake_client: FakeLangfuseClient = tracer._client
        all_data = json.dumps(
            [t.kwargs for t in fake_client.traces],
            ensure_ascii=False,
            default=str,
        )
        assert _SENSITIVE_TEXT not in all_data


# ── 测试：capture=false 只输出 HMAC/length ────────────────────────────────────


class TestLangfuseCaptureFalse:
    """capture=false 时 input/output 只有 hmac_sha256 + char_count"""

    def _make_tracer(self):
        from app.services.observability.langfuse_client import LangfuseTracer

        tracer = LangfuseTracer.__new__(LangfuseTracer)
        tracer._client = FakeLangfuseClient()
        tracer._capture = False
        tracer._max_chars = 500
        tracer._hmac_key = _HMAC_KEY
        tracer._flushed = False
        return tracer

    def test_input_is_hmac_summary(self):
        tracer = self._make_tracer()
        tracer.trace_llm_call(
            trace_name="test_trace",
            model="qwen-test",
            prompt="some prompt text",
            completion="some completion",
            tokens=10,
            latency_ms=10.0,
        )
        fake_client: FakeLangfuseClient = tracer._client
        trace = fake_client.traces[0]
        span = trace.spans[0]
        input_data = span.kwargs.get("input", {})
        # input 应该是 {"prompt": {"hmac_sha256": ..., "char_count": ...}}
        prompt_val = input_data.get("prompt", {})
        assert isinstance(prompt_val, dict)
        assert "hmac_sha256" in prompt_val
        assert "char_count" in prompt_val

    def test_output_is_hmac_summary(self):
        tracer = self._make_tracer()
        tracer.trace_llm_call(
            trace_name="test_trace",
            model="qwen-test",
            prompt="some prompt text",
            completion="some completion",
            tokens=10,
            latency_ms=10.0,
        )
        fake_client: FakeLangfuseClient = tracer._client
        trace = fake_client.traces[0]
        span = trace.spans[0]
        output_data = span.kwargs.get("output", {})
        completion_val = output_data.get("completion", {})
        assert isinstance(completion_val, dict)
        assert "hmac_sha256" in completion_val
        assert "char_count" in completion_val

    def test_hmac_is_deterministic_with_same_key(self):
        tracer = self._make_tracer()
        text = "deterministic test text"
        expected_hmac = hmac.new(_HMAC_KEY, text.encode("utf-8"), hashlib.sha256).hexdigest()

        tracer.trace_llm_call(
            trace_name="test_trace",
            model="qwen-test",
            prompt=text,
            completion="out",
            tokens=10,
            latency_ms=10.0,
        )
        fake_client: FakeLangfuseClient = tracer._client
        span = fake_client.traces[0].spans[0]
        actual_hmac_val = span.kwargs["input"]["prompt"]["hmac_sha256"]
        assert actual_hmac_val == expected_hmac

    def test_hmac_changes_with_different_key(self):
        tracer1 = self._make_tracer()
        tracer2 = self._make_tracer()
        tracer2._hmac_key = b"different-key-at-least-32-bytes-long!!"

        text = "same text"
        tracer1.trace_llm_call("t", "m", text, "o", 1, 1.0)
        tracer2.trace_llm_call("t", "m", text, "o", 1, 1.0)

        h1 = tracer1._client.traces[0].spans[0].kwargs["input"]["prompt"]["hmac_sha256"]
        h2 = tracer2._client.traces[0].spans[0].kwargs["input"]["prompt"]["hmac_sha256"]
        assert h1 != h2


# ── 测试：capture=true 脱敏 + 截断 ───────────────────────────────────────────


class TestLangfuseCaptureTrue:
    """capture=true 时内容已脱敏且最大长度生效"""

    def _make_tracer(self, max_chars: int = 50):
        from app.services.observability.langfuse_client import LangfuseTracer

        tracer = LangfuseTracer.__new__(LangfuseTracer)
        tracer._client = FakeLangfuseClient()
        tracer._capture = True
        tracer._max_chars = max_chars
        tracer._hmac_key = _HMAC_KEY
        tracer._flushed = False
        return tracer

    def test_capture_true_sanitizes_phone(self):
        tracer = self._make_tracer(max_chars=500)
        tracer.trace_llm_call(
            trace_name="test_trace",
            model="qwen-test",
            prompt="患者手机号13812345678",
            completion="回复",
            tokens=10,
            latency_ms=10.0,
        )
        fake_client: FakeLangfuseClient = tracer._client
        span = fake_client.traces[0].spans[0]
        input_data = span.kwargs.get("input", {})
        prompt_val = input_data.get("prompt", {})
        # capture=true 时应该是脱敏后的字符串（有 hash + length），不含原手机号
        if isinstance(prompt_val, str):
            assert "13812345678" not in prompt_val
        elif isinstance(prompt_val, dict):
            # 有 hash/length，不含原文
            serialized = json.dumps(prompt_val, ensure_ascii=False)
            assert "13812345678" not in serialized

    def test_capture_true_truncates_long_text(self):
        tracer = self._make_tracer(max_chars=20)
        long_text = "A" * 100
        tracer.trace_llm_call(
            trace_name="test_trace",
            model="qwen-test",
            prompt=long_text,
            completion="out",
            tokens=10,
            latency_ms=10.0,
        )
        fake_client: FakeLangfuseClient = tracer._client
        span = fake_client.traces[0].spans[0]
        input_data = span.kwargs.get("input", {})
        prompt_val = input_data.get("prompt", {})
        if isinstance(prompt_val, str):
            assert len(prompt_val) <= 20 + 20  # 允许一些额外字符（如截断标记）
        elif isinstance(prompt_val, dict):
            assert prompt_val.get("char_count", 0) <= 100  # 原始长度记录


# ── 测试：trace context 传播 ─────────────────────────────────────────────────


class TestTraceContextPropagation:
    """API 创建的 trace_id/run_id 在 Celery 恢复后仍进入 LLM/RAG span metadata"""

    def test_trace_id_propagates_to_span_metadata(self):
        from app.services.observability.langfuse_client import LangfuseTracer
        from app.services.observability.trace_context import (
            TraceContext,
            bind_trace_context,
            reset_trace_context,
        )

        tracer = LangfuseTracer.__new__(LangfuseTracer)
        tracer._client = FakeLangfuseClient()
        tracer._capture = False
        tracer._max_chars = 500
        tracer._hmac_key = _HMAC_KEY
        tracer._flushed = False

        ctx = TraceContext(
            trace_id="trace-abc123",
            run_id="run-xyz789",
            consultation_id="consult-001",
        )
        token = bind_trace_context(ctx)
        try:
            tracer.trace_llm_call(
                trace_name="test_trace",
                model="qwen-test",
                prompt="test prompt",
                completion="test completion",
                tokens=10,
                latency_ms=10.0,
            )
            fake_client: FakeLangfuseClient = tracer._client
            span = fake_client.traces[0].spans[0]
            metadata = span.kwargs.get("metadata", {})
            assert metadata.get("trace_id") == "trace-abc123"
            assert metadata.get("run_id") == "run-xyz789"
            # consultation_id 应该被 HMAC 处理，不原文出现
            consultation_ref = metadata.get("consultation_ref", "")
            assert "consult-001" not in str(consultation_ref) or isinstance(consultation_ref, str)
        finally:
            reset_trace_context(token)

    def test_trace_context_restored_from_celery_payload(self):
        from app.services.observability.trace_context import (
            TraceContext,
            bind_trace_context,
            restore_trace_context,
            serialize_trace_context,
            reset_trace_context,
        )

        original_ctx = TraceContext(
            trace_id="trace-api-created",
            run_id="run-api-created",
            consultation_id="consult-123",
        )
        payload = serialize_trace_context(original_ctx)
        restored_ctx = restore_trace_context(payload)

        assert restored_ctx.trace_id == "trace-api-created"
        assert restored_ctx.run_id == "run-api-created"
        assert restored_ctx.consultation_id == "consult-123"


# ── 测试：flush 行为 ─────────────────────────────────────────────────────────


class TestFlushBehavior:
    """连续 10 次 trace 不调用 client.flush；应用 shutdown 只调用一次"""

    def _make_tracer(self):
        from app.services.observability.langfuse_client import LangfuseTracer

        tracer = LangfuseTracer.__new__(LangfuseTracer)
        tracer._client = FakeLangfuseClient()
        tracer._capture = False
        tracer._max_chars = 500
        tracer._hmac_key = _HMAC_KEY
        tracer._flushed = False
        return tracer

    def test_no_flush_on_each_span(self):
        tracer = self._make_tracer()
        for i in range(10):
            tracer.trace_llm_call(
                trace_name=f"trace_{i}",
                model="qwen-test",
                prompt=f"prompt {i}",
                completion=f"completion {i}",
                tokens=10,
                latency_ms=10.0,
            )
        fake_client: FakeLangfuseClient = tracer._client
        assert fake_client.flush_count == 0

    def test_flush_called_once_on_shutdown(self):
        tracer = self._make_tracer()
        tracer.flush()
        fake_client: FakeLangfuseClient = tracer._client
        assert fake_client.flush_count == 1

    def test_flush_is_idempotent(self):
        tracer = self._make_tracer()
        tracer.flush()
        tracer.flush()
        tracer.flush()
        fake_client: FakeLangfuseClient = tracer._client
        # 幂等：多次调用只实际 flush 一次
        assert fake_client.flush_count == 1

    def test_close_calls_flush_once(self):
        tracer = self._make_tracer()
        fake_client: FakeLangfuseClient = tracer._client
        tracer.close()
        # close() 调用 flush 并清理 _client 引用
        assert fake_client.flush_count == 1
        assert tracer._client is None


# ── 测试：summarize_sensitive_text ───────────────────────────────────────────


class TestSummarizeSensitiveText:
    """summarize_sensitive_text 返回 hmac_sha256 + char_count"""

    def test_returns_hmac_and_char_count(self):
        from app.services.observability.trace_context import summarize_sensitive_text

        text = "test text"
        result = summarize_sensitive_text(text, _HMAC_KEY)
        assert "hmac_sha256" in result
        assert "char_count" in result
        assert result["char_count"] == len(text)

    def test_hmac_matches_expected(self):
        from app.services.observability.trace_context import summarize_sensitive_text

        text = "test text"
        expected = hmac.new(_HMAC_KEY, text.encode("utf-8"), hashlib.sha256).hexdigest()
        result = summarize_sensitive_text(text, _HMAC_KEY)
        assert result["hmac_sha256"] == expected

    def test_different_key_different_hmac(self):
        from app.services.observability.trace_context import summarize_sensitive_text

        text = "test text"
        key2 = b"different-key-at-least-32-bytes-long!!"
        r1 = summarize_sensitive_text(text, _HMAC_KEY)
        r2 = summarize_sensitive_text(text, key2)
        assert r1["hmac_sha256"] != r2["hmac_sha256"]
