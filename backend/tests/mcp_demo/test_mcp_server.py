"""Tests for the read-only MCP demo server."""
from __future__ import annotations

import json
import re

import pytest

from app.mcp_demo.server import DEMO_FIXTURES, MCPDemoServer


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
