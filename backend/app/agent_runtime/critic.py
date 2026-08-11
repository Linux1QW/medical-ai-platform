"""Critic agent: validates suggestions for safety and correctness.

Expanded safety checks include:
- Direct and paraphrased diagnosis/medication instructions
- Emergency claims
- Hidden label markers
- Unsupported citations
- Prompt injection detection
- Forbidden tools
- Schema mismatch
- Risk calculated from findings, never hard-coded low
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.agent_runtime.contracts import CoachSuggestion
from app.agent_runtime.telemetry import COACH_POLICY_BLOCKED


@dataclass
class CriticFinding:
    """A single finding from the critic."""

    severity: str  # "error", "warning", "info"
    category: str  # "hidden_leak", "diagnostic", "medication", "emergency", etc.
    message: str


@dataclass
class CriticResult:
    """Result of critic evaluation."""

    passed: bool
    findings: list[CriticFinding] = field(default_factory=list)
    risk_level: str = "low"  # Calculated from findings, never hard-coded
    error_code: str | None = None


# Hidden fact patterns that must never leak
HIDDEN_FACT_PATTERNS = [
    "expected_diagnosis",
    "gold_standard",
    "hidden_fact",
    "unrevealed",
    "evaluation_only",
    "secret",
    "标准答案",
    "隐藏信息",
    "内部评分",
]

# Direct diagnosis patterns
DIAGNOSTIC_PATTERNS = [
    r"你应该诊断",
    r"这是.*病",
    r"确诊为",
    r"你患.*病",
    r"诊断.*是",
    r"可以确诊",
    r"明确诊断",
]

# Paraphrased diagnosis patterns
PARAPHRASED_DIAGNOSTIC_PATTERNS = [
    r"你的情况.*(很|可能|应该)是",
    r"根据症状(判断|考虑|推测)",
    r"你(得|患)了",
    r"应该是.*引起的",
    r"典型(的|的).*(症状|表现|病例)",
]

# Medication instruction patterns
MEDICATION_PATTERNS = [
    r"处方",
    r"建议.*用药",
    r"应该.*吃.*药",
    r"需要.*服用",
    r"推荐.*药物",
    r"剂量.*mg",
    r"每日.*次",
    r"用药方案",
]

# Emergency claim patterns
EMERGENCY_PATTERNS = [
    r"立即.*急诊",
    r"马上.*手术",
    r"必须.*住院",
    r"危及生命",
    r"紧急情况",
    r"立刻.*抢救",
    r"需要.*急救",
]

# Hidden label marker patterns (system prompts leaking)
HIDDEN_LABEL_PATTERNS = [
    r"\[hidden\]",
    r"\[internal\]",
    r"\[system\]",
    r"\[confidential\]",
    r"<!--.*hidden.*-->",
    r"\[HIDDEN\]",
    r"\[INTERNAL\]",
]

# Prompt injection patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"disregard (previous|all|above)",
    r"you are now",
    r"new role.*:",
    r"override.*system.*prompt",
    r"forget.*instructions",
    r"act as if",
    r"pretend (you are|to be)",
    r"忽略.*(之前|以上|所有).*(指令|指示)",
    r"你现在是",
    r"覆盖.*系统.*提示",
]

# Forbidden tools that coach must not invoke
FORBIDDEN_TOOLS = [
    "prescribe_medication",
    "order_lab_test",
    "generate_prescription",
    "modify_patient_record",
    "delete_consultation",
    "access_other_patient",
    "bypass_safety_check",
]

# Schema required fields for coach suggestions
REQUIRED_SCHEMA_FIELDS = [
    "suggested_question",
    "rationale_summary",
    "intent",
    "confidence",
]


class CriticAgent:
    """Critic that validates coach suggestions.

    Checks:
    1. Hidden-fact leakage
    2. Direct diagnostic phrasing
    3. Paraphrased diagnostic phrasing
    4. Medication instructions
    5. Emergency claims
    6. Hidden label markers
    7. Unsupported citations
    8. Prompt injection
    9. Forbidden tools
    10. Schema mismatch
    """

    def evaluate(
        self,
        suggestion: CoachSuggestion,
        *,
        evidence_results: list[dict[str, Any]] | None = None,
        available_tools: list[str] | None = None,
    ) -> CriticResult:
        """Evaluate a suggestion for safety and correctness."""
        findings: list[CriticFinding] = []

        text_to_check = (
            f"{suggestion.suggested_question} {suggestion.rationale_summary}".lower()
        )

        # Extract checks into separate methods to reduce complexity
        findings.extend(self._check_hidden_fact_leakage(text_to_check))
        findings.extend(self._check_diagnostic_phrasing(suggestion.suggested_question))
        findings.extend(self._check_paraphrased_diagnostic(suggestion.suggested_question))
        findings.extend(self._check_medication_instructions(suggestion.suggested_question))
        findings.extend(self._check_emergency_claims(text_to_check))
        findings.extend(self._check_hidden_labels(text_to_check))
        findings.extend(self._check_unsupported_citations(suggestion, evidence_results))
        findings.extend(self._check_prompt_injection(text_to_check))
        findings.extend(self._check_forbidden_tools(suggestion, available_tools))
        findings.extend(self._check_schema_mismatch(suggestion))

        # Calculate risk from findings, never hard-coded
        risk_level = self._calculate_risk(findings)
        has_errors = any(f.severity == "error" for f in findings)

        return CriticResult(
            passed=not has_errors,
            findings=findings,
            risk_level=risk_level,
            error_code=COACH_POLICY_BLOCKED if has_errors else None,
        )

    def _check_hidden_fact_leakage(self, text_to_check: str) -> list[CriticFinding]:
        findings = []
        for pattern in HIDDEN_FACT_PATTERNS:
            if pattern.lower() in text_to_check:
                findings.append(
                    CriticFinding(
                        severity="error",
                        category="hidden_leak",
                        message=f"Hidden fact pattern detected: {pattern}",
                    )
                )
        return findings

    def _check_diagnostic_phrasing(self, question: str) -> list[CriticFinding]:
        findings = []
        for pattern in DIAGNOSTIC_PATTERNS:
            if re.search(pattern, question, re.IGNORECASE):
                findings.append(
                    CriticFinding(
                        severity="error",
                        category="diagnostic",
                        message="Direct diagnostic phrasing detected",
                    )
                )
                break
        return findings

    def _check_paraphrased_diagnostic(self, question: str) -> list[CriticFinding]:
        findings = []
        for pattern in PARAPHRASED_DIAGNOSTIC_PATTERNS:
            if re.search(pattern, question, re.IGNORECASE):
                findings.append(
                    CriticFinding(
                        severity="error",
                        category="paraphrased_diagnostic",
                        message="Paraphrased diagnostic phrasing detected",
                    )
                )
                break
        return findings

    def _check_medication_instructions(self, question: str) -> list[CriticFinding]:
        findings = []
        for pattern in MEDICATION_PATTERNS:
            if re.search(pattern, question, re.IGNORECASE):
                findings.append(
                    CriticFinding(
                        severity="error",
                        category="medication_instruction",
                        message="Medication instruction detected",
                    )
                )
                break
        return findings

    def _check_emergency_claims(self, text_to_check: str) -> list[CriticFinding]:
        findings = []
        for pattern in EMERGENCY_PATTERNS:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                findings.append(
                    CriticFinding(
                        severity="error",
                        category="emergency_claim",
                        message="Emergency claim detected",
                    )
                )
                break
        return findings

    def _check_hidden_labels(self, text_to_check: str) -> list[CriticFinding]:
        findings = []
        for pattern in HIDDEN_LABEL_PATTERNS:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                findings.append(
                    CriticFinding(
                        severity="error",
                        category="hidden_label",
                        message="Hidden label marker detected",
                    )
                )
                break
        return findings

    def _check_unsupported_citations(
        self,
        suggestion: CoachSuggestion,
        evidence_results: list[dict[str, Any]] | None,
    ) -> list[CriticFinding]:
        if suggestion.citation_ids and not evidence_results:
            return [
                CriticFinding(
                    severity="warning",
                    category="unsupported_citation",
                    message="Citations present but no evidence results to support them",
                )
            ]
        return []

    def _check_prompt_injection(self, text_to_check: str) -> list[CriticFinding]:
        findings = []
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                findings.append(
                    CriticFinding(
                        severity="error",
                        category="prompt_injection",
                        message="Prompt injection pattern detected",
                    )
                )
                break
        return findings

    def _check_forbidden_tools(
        self,
        suggestion: CoachSuggestion,
        available_tools: list[str] | None,
    ) -> list[CriticFinding]:
        findings = []
        if available_tools:
            for tool in suggestion.targeted_slots:
                if tool.lower() in [t.lower() for t in FORBIDDEN_TOOLS]:
                    findings.append(
                        CriticFinding(
                            severity="error",
                            category="forbidden_tool",
                            message=f"Forbidden tool referenced: {tool}",
                        )
                    )
        return findings

    def _check_schema_mismatch(self, suggestion: CoachSuggestion) -> list[CriticFinding]:
        findings = []
        suggestion_dict = suggestion.model_dump()
        for required_field in REQUIRED_SCHEMA_FIELDS:
            if required_field not in suggestion_dict or suggestion_dict[required_field] is None:
                findings.append(
                    CriticFinding(
                        severity="error",
                        category="schema_mismatch",
                        message=f"Required field missing: {required_field}",
                    )
                )
        return findings

    def _calculate_risk(self, findings: list[CriticFinding]) -> str:
        """Calculate risk level from findings.

        Risk is derived from the number and severity of findings,
        never hard-coded to 'low'.
        """
        if not findings:
            return "low"

        error_count = sum(1 for f in findings if f.severity == "error")
        warning_count = sum(1 for f in findings if f.severity == "warning")

        # High risk: any error-level finding in critical categories
        critical_categories = {
            "hidden_leak", "prompt_injection", "emergency_claim",
            "medication_instruction", "forbidden_tool",
        }
        has_critical = any(
            f.severity == "error" and f.category in critical_categories
            for f in findings
        )

        if has_critical or error_count >= 3:
            return "high"
        if error_count >= 1 or warning_count >= 2:
            return "medium"
        return "low"
