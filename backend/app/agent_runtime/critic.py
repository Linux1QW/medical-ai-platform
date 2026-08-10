"""Critic agent: validates suggestions for safety and correctness."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.agent_runtime.contracts import CoachSuggestion


@dataclass
class CriticFinding:
    """A single finding from the critic."""

    severity: str  # "error", "warning", "info"
    category: str  # "hidden_leak", "diagnostic", "repeated", "unsupported_citation"
    message: str


@dataclass
class CriticResult:
    """Result of critic evaluation."""

    passed: bool
    findings: list[CriticFinding] = field(default_factory=list)
    adjusted_confidence: float | None = None


HIDDEN_FACT_PATTERNS = [
    "expected_diagnosis",
    "gold_standard",
    "hidden_fact",
    "unrevealed",
    "evaluation_only",
    "secret",
]

DIAGNOSTIC_PATTERNS = [
    r"你应该诊断",
    r"这是.*病",
    r"确诊为",
    r"处方",
    r"建议.*用药",
]


class CriticAgent:
    """Critic that validates coach suggestions.

    Checks:
    1. Hidden-fact leakage
    2. Diagnostic phrasing (coach must not diagnose)
    3. Repeated questions
    4. Unsupported citations
    """

    def evaluate(
        self,
        suggestion: CoachSuggestion,
        *,
        evidence_results: list[dict[str, Any]] | None = None,
    ) -> CriticResult:
        """Evaluate a suggestion for safety and correctness."""
        findings: list[CriticFinding] = []

        text_to_check = (
            f"{suggestion.suggested_question} {suggestion.rationale_summary}".lower()
        )

        # Check 1: Hidden-fact leakage
        for pattern in HIDDEN_FACT_PATTERNS:
            if pattern in text_to_check:
                findings.append(
                    CriticFinding(
                        severity="error",
                        category="hidden_leak",
                        message=f"Hidden fact pattern detected: {pattern}",
                    )
                )

        # Check 2: Diagnostic phrasing
        for pattern in DIAGNOSTIC_PATTERNS:
            if re.search(pattern, suggestion.suggested_question):
                findings.append(
                    CriticFinding(
                        severity="error",
                        category="diagnostic",
                        message="Diagnostic phrasing detected",
                    )
                )

        # Check 3: Unsupported citations
        if suggestion.citation_ids and not evidence_results:
            findings.append(
                CriticFinding(
                    severity="warning",
                    category="unsupported_citation",
                    message="Citations present but no evidence results to support them",
                )
            )

        has_errors = any(f.severity == "error" for f in findings)

        return CriticResult(
            passed=not has_errors,
            findings=findings,
        )
