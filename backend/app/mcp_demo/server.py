"""Read-only MCP demo server with deidentified clinical fixtures.

This server is for development/demo purposes only.
It serves static, deidentified clinical data via the MCP protocol.
"""
from __future__ import annotations

from typing import Any

# Deidentified demo fixtures
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
        return [
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

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool. All outputs are read-only and deidentified."""
        handler = self._tools.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}", "data": None}
        return handler(arguments)

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
