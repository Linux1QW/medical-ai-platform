"""Tests for the read-only MCP demo server.

Includes:
- Direct API tests (list_tools, call_tool)
- Subprocess JSON-RPC protocol tests (initialize, tools/list, tools/call)
- Malformed args and unknown tool handling
- PII safety check
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from app.mcp_demo.server import DEMO_FIXTURES, MCPDemoServer, handle_jsonrpc_message


@pytest.fixture()
def server() -> MCPDemoServer:
    return MCPDemoServer()


# ---------------------------------------------------------------------------
# 1. list_tools
# ---------------------------------------------------------------------------

def test_list_tools_returns_two_tools(server: MCPDemoServer) -> None:
    tools = server.list_tools()
    assert len(tools) == 2
    names = {t["name"] for t in tools}
    assert names == {"search_teaching_rubric", "search_medical_kb"}


# ---------------------------------------------------------------------------
# 2. all tools readonly
# ---------------------------------------------------------------------------

def test_list_tools_all_readonly(server: MCPDemoServer) -> None:
    for tool in server.list_tools():
        assert tool.get("readOnly") is True, f"{tool['name']} must be readOnly"


# ---------------------------------------------------------------------------
# 3. teaching rubric — filter by stage
# ---------------------------------------------------------------------------

def test_search_teaching_rubric_by_stage(server: MCPDemoServer) -> None:
    result = server.call_tool(
        "search_teaching_rubric",
        {"query": "", "stage": "chief_complaint"},
    )
    data = result["data"]
    assert len(data) >= 1
    for item in data:
        assert item["stage"] == "chief_complaint"


# ---------------------------------------------------------------------------
# 4. teaching rubric — keyword search
# ---------------------------------------------------------------------------

def test_search_teaching_rubric_by_keyword(server: MCPDemoServer) -> None:
    result = server.call_tool(
        "search_teaching_rubric",
        {"query": "onset exploration"},
    )
    data = result["data"]
    assert len(data) >= 1
    # The OPQRST rubric should match best
    assert data[0]["id"] == "rubric_002"


# ---------------------------------------------------------------------------
# 5. medical KB — filter by topic
# ---------------------------------------------------------------------------

def test_search_medical_kb_by_topic(server: MCPDemoServer) -> None:
    result = server.call_tool(
        "search_medical_kb",
        {"query": "", "topic": "hypertension"},
    )
    data = result["data"]
    assert len(data) == 1
    assert data[0]["topic"] == "hypertension"


# ---------------------------------------------------------------------------
# 6. medical KB — keyword search
# ---------------------------------------------------------------------------

def test_search_medical_kb_by_keyword(server: MCPDemoServer) -> None:
    result = server.call_tool(
        "search_medical_kb",
        {"query": "diabetes screening"},
    )
    data = result["data"]
    assert len(data) >= 1
    assert data[0]["id"] == "kb_002"


# ---------------------------------------------------------------------------
# 7. unknown tool
# ---------------------------------------------------------------------------

def test_unknown_tool_returns_error(server: MCPDemoServer) -> None:
    result = server.call_tool("nonexistent_tool", {"query": "test"})
    assert "error" in result
    assert "Unknown tool" in result["error"]
    assert result["data"] is None


# ---------------------------------------------------------------------------
# 8. top_k capped at 5
# ---------------------------------------------------------------------------

def test_top_k_capped_at_5(server: MCPDemoServer) -> None:
    # Request top_k=100 — should still return at most 5
    result = server.call_tool(
        "search_teaching_rubric",
        {"query": "", "top_k": 100},
    )
    assert len(result["data"]) <= 5


# ---------------------------------------------------------------------------
# 9. no patient PII in fixtures
# ---------------------------------------------------------------------------

# Patterns that suggest real patient PII
_PII_PATTERNS = [
    re.compile(r"\b\d{17}[\dXx]\b"),            # Chinese ID card
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),        # US SSN
    re.compile(r"\b1[3-9]\d{9}\b"),              # Chinese phone
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
    re.compile(r"(\bname|姓名|身份证|电话|手机|地址|address\b)", re.IGNORECASE),
]


def test_no_patient_data_in_fixtures() -> None:
    raw = json.dumps(DEMO_FIXTURES, ensure_ascii=False)
    for pattern in _PII_PATTERNS:
        match = pattern.search(raw)
        assert match is None, f"Potential PII found in fixtures: {match.group()!r}"


# ---------------------------------------------------------------------------
# 10. Malformed arguments
# ---------------------------------------------------------------------------


def test_missing_required_argument(server: MCPDemoServer) -> None:
    """Missing 'query' (required) should return an error."""
    result = server.call_tool("search_medical_kb", {})
    assert "error" in result
    assert "Missing required" in result["error"] or "query" in result["error"]


def test_wrong_type_argument(server: MCPDemoServer) -> None:
    """query must be string, not integer."""
    result = server.call_tool("search_medical_kb", {"query": 123})
    assert "error" in result
    assert "must be a string" in result["error"]


def test_invalid_enum_value(server: MCPDemoServer) -> None:
    """stage must be one of the allowed enum values."""
    result = server.call_tool(
        "search_teaching_rubric",
        {"query": "test", "stage": "invalid_stage"},
    )
    assert "error" in result
    assert "must be one of" in result["error"]


# ---------------------------------------------------------------------------
# 11. JSON-RPC protocol (subprocess)
# ---------------------------------------------------------------------------

# Backend directory for subprocess cwd: resolve from this file's location
# tests/mcp_demo/test_mcp_server.py → parent.parent.parent = backend/
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent.parent)


def _run_jsonrpc(messages: list[dict]) -> list[dict]:
    """Send JSON-RPC messages to the MCP demo server subprocess and collect responses."""
    input_data = "\n".join(json.dumps(m) for m in messages) + "\n"
    proc = subprocess.run(
        [sys.executable, "-m", "app.mcp_demo"],
        input=input_data,
        capture_output=True,
        text=True,
        timeout=10,
        cwd=_BACKEND_DIR,
    )
    responses = []
    for line in proc.stdout.strip().split("\n"):
        if line.strip():
            responses.append(json.loads(line))
    return responses


class TestSubprocessProtocol:
    """Test the JSON-RPC protocol via subprocess."""

    def test_initialize(self) -> None:
        msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        responses = _run_jsonrpc([msg])
        assert len(responses) >= 1
        resp = responses[0]
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "result" in resp
        assert resp["result"]["protocolVersion"] == "2024-11-05"

    def test_tools_list(self) -> None:
        msgs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ]
        responses = _run_jsonrpc(msgs)
        # Find the tools/list response
        list_resp = next(r for r in responses if r.get("id") == 2)
        tools = list_resp["result"]["tools"]
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert "search_teaching_rubric" in names
        assert "search_medical_kb" in names

    def test_tools_call(self) -> None:
        msgs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "search_medical_kb", "arguments": {"query": "diabetes"}},
            },
        ]
        responses = _run_jsonrpc(msgs)
        call_resp = next(r for r in responses if r.get("id") == 2)
        assert "result" in call_resp
        content = call_resp["result"]["content"]
        assert len(content) >= 1
        data = json.loads(content[0]["text"])
        assert "data" in data

    def test_unknown_method(self) -> None:
        msg = {"jsonrpc": "2.0", "id": 1, "method": "unknown/method", "params": {}}
        responses = _run_jsonrpc([msg])
        resp = responses[0]
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_malformed_json(self) -> None:
        """Sending invalid JSON should return a parse error."""
        proc = subprocess.run(
            [sys.executable, "-m", "app.mcp_demo"],
            input="not valid json\n",
            capture_output=True,
            text=True,
            timeout=10,
            cwd=_BACKEND_DIR,
        )
        responses = []
        for line in proc.stdout.strip().split("\n"):
            if line.strip():
                responses.append(json.loads(line))
        assert len(responses) >= 1
        assert responses[0]["error"]["code"] == -32700
