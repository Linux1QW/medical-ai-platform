"""Read-only MCP demo server with deidentified clinical fixtures.

Implements a JSON-RPC 2.0 stdio transport compatible with the MCP protocol.
Exposes only static, deidentified clinical data.

Usage:
    python -m app.mcp_demo

This server:
- Reads JSON-RPC requests from stdin (one per line)
- Writes JSON-RPC responses to stdout
- Handles: initialize, tools/list, tools/call
- Does NOT bind any network port
- Does NOT connect to production databases
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# ── Deidentified demo fixtures ─────────────────────────────────────────────────

DEMO_FIXTURES: dict[str, list[dict[str, Any]]] = {
    "teaching_rubrics": [
        {
            "id": "rubric_001",
            "stage": "chief_complaint",
            "criteria": "Open-ended question about primary concern",
            "example": "您今天主要是因为什么问题来看诊？",
            "score_range": [1, 5],
        },
        {
            "id": "rubric_002",
            "stage": "history_present_illness",
            "criteria": "Systematic onset exploration (OPQRST)",
            "example": "这个疼痛是什么时候开始的？是突然还是逐渐的？",
            "score_range": [1, 5],
        },
        {
            "id": "rubric_003",
            "stage": "past_medical_history",
            "criteria": "Comprehensive past history including surgeries",
            "example": "您以前有没有住过院或者做过手术？",
            "score_range": [1, 5],
        },
    ],
    "medical_kb": [
        {
            "id": "kb_001",
            "topic": "hypertension",
            "summary": "Primary hypertension: common risk factors include age, family history, obesity, high salt intake",
            "references": ["JNC-8", "ESC/ESH 2023"],
        },
        {
            "id": "kb_002",
            "topic": "diabetes_t2",
            "summary": "Type 2 diabetes: screening recommended for BMI > 25 with risk factors",
            "references": ["ADA 2024", "CDS Guidelines"],
        },
    ],
}

# ── Tool schemas ───────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_teaching_rubric",
        "description": "Search teaching rubric for clinical interview guidelines",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "maxLength": 200},
                "stage": {
                    "type": "string",
                    "enum": [
                        "rapport",
                        "chief_complaint",
                        "history_present_illness",
                        "past_medical_history",
                        "medication_allergy",
                        "closing",
                    ],
                },
                "top_k": {"type": "integer", "default": 3, "maximum": 5},
            },
            "required": ["query"],
        },
        "readOnly": True,
    },
    {
        "name": "search_medical_kb",
        "description": "Search medical knowledge base for clinical information",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "maxLength": 300},
                "topic": {"type": "string"},
                "top_k": {"type": "integer", "default": 3, "maximum": 5},
            },
            "required": ["query"],
        },
        "readOnly": True,
    },
]


# ── Argument validation ────────────────────────────────────────────────────────


def _validate_arguments(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Validate arguments against tool schema. Returns error message or None."""
    schema = next((t for t in TOOL_SCHEMAS if t["name"] == tool_name), None)
    if schema is None:
        return f"Unknown tool: {tool_name}"

    input_schema = schema["inputSchema"]
    required = input_schema.get("required", [])
    properties = input_schema.get("properties", {})

    # Check required fields
    for field_name in required:
        if field_name not in arguments:
            return f"Missing required argument: '{field_name}'"

    # Check types and constraints
    for key, value in arguments.items():
        if key not in properties:
            continue  # Extra args are ignored
        prop = properties[key]
        expected_type = prop.get("type")

        if expected_type == "string" and not isinstance(value, str):
            return f"Argument '{key}' must be a string"
        if expected_type == "integer" and not isinstance(value, int):
            return f"Argument '{key}' must be an integer"

        # maxLength
        max_length = prop.get("maxLength")
        if max_length and isinstance(value, str) and len(value) > max_length:
            return f"Argument '{key}' exceeds maxLength ({max_length})"

        # maximum — for top_k, clamp silently (backward compat); others error
        maximum = prop.get("maximum")
        if maximum and isinstance(value, int) and value > maximum:
            if key == "top_k":
                arguments[key] = maximum  # clamp in-place
            else:
                return f"Argument '{key}' exceeds maximum ({maximum})"

        # enum
        enum_values = prop.get("enum")
        if enum_values and value not in enum_values:
            return f"Argument '{key}' must be one of {enum_values}"

    return None


# ── MCPDemoServer (sync API for direct use) ────────────────────────────────────


class MCPDemoServer:
    """Read-only MCP demo server.

    Provides two tools:
    - search_teaching_rubric: Search teaching rubric entries
    - search_medical_kb: Search medical knowledge base entries

    All data is static, deidentified, and read-only.
    Communication is stdio-only (no network binding).
    """

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {
            "search_teaching_rubric": self._search_teaching_rubric,
            "search_medical_kb": self._search_medical_kb,
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """List available MCP tools."""
        return TOOL_SCHEMAS

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool. All outputs are read-only and deidentified."""
        # Validate arguments
        error = _validate_arguments(tool_name, arguments)
        if error:
            return {"error": error, "data": None}

        handler = self._tools.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}", "data": None}
        result = handler(arguments)
        return dict(result) if isinstance(result, dict) else {}

    def _search_teaching_rubric(self, args: dict[str, Any]) -> dict[str, Any]:
        """Search teaching rubric fixtures."""
        query = args.get("query", "").lower()
        stage_filter = args.get("stage")
        top_k = min(args.get("top_k", 3), 5)

        results: list[dict[str, Any]] = []
        for rubric in DEMO_FIXTURES["teaching_rubrics"]:
            if stage_filter and rubric["stage"] != stage_filter:
                continue
            # Simple keyword matching for demo
            score = sum(
                1 for word in query.split()
                if word in rubric["criteria"].lower() or word in rubric["example"].lower()
            )
            if score > 0 or not query:
                results.append({**rubric, "_score": score})

        results.sort(key=lambda r: r["_score"], reverse=True)
        return {"data": results[:top_k], "total": len(results)}

    def _search_medical_kb(self, args: dict[str, Any]) -> dict[str, Any]:
        """Search medical KB fixtures."""
        query = args.get("query", "").lower()
        topic_filter = args.get("topic")
        top_k = min(args.get("top_k", 3), 5)

        results: list[dict[str, Any]] = []
        for entry in DEMO_FIXTURES["medical_kb"]:
            if topic_filter and entry["topic"] != topic_filter:
                continue
            score = sum(
                1 for word in query.split()
                if word in entry["summary"].lower() or word in entry["topic"].lower()
            )
            if score > 0 or not query:
                results.append({**entry, "_score": score})

        results.sort(key=lambda r: r["_score"], reverse=True)
        return {"data": results[:top_k], "total": len(results)}


# ── JSON-RPC stdio handler ─────────────────────────────────────────────────────


def _make_response(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _make_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_jsonrpc_message(message: dict[str, Any], server: MCPDemoServer) -> dict[str, Any]:
    """Handle a single JSON-RPC 2.0 message and return a response."""
    jsonrpc = message.get("jsonrpc")
    if jsonrpc != "2.0":
        return _make_error(message.get("id"), -32600, "Invalid JSON-RPC version")

    request_id = message.get("id")
    method = message.get("method", "")
    params = message.get("params", {})

    if method == "initialize":
        return _make_response(request_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "medical-mcp-demo",
                "version": "1.0.0",
            },
        })

    if method == "notifications/initialized":
        # Notification, no response needed
        return {}

    if method == "tools/list":
        return _make_response(request_id, {"tools": server.list_tools()})

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if not isinstance(arguments, dict):
            return _make_error(request_id, -32602, "Arguments must be an object")

        result = server.call_tool(tool_name, arguments)

        if "error" in result and result.get("data") is None:
            return _make_response(request_id, {
                "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                "isError": True,
            })

        return _make_response(request_id, {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
        })

    return _make_error(request_id, -32601, f"Method not found: {method}")


def run_stdio_server() -> None:
    """Run the MCP demo server reading from stdin and writing to stdout."""
    server = MCPDemoServer()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = _make_error(None, -32700, "Parse error")
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue

        response = handle_jsonrpc_message(message, server)
        if response:  # Skip empty responses (notifications)
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
