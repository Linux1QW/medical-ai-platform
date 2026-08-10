"""Versioned skill manifest and registry with startup validation."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

ALLOWED_CONTEXT_VIEWS = {"coach", "evaluation", "patient", "system"}

# JSON-schema-like parameter type whitelist
ALLOWED_PARAM_TYPES = {"string", "integer", "number", "boolean", "array", "object"}

# Hard limits
MAX_TIMEOUT_SECONDS = 60.0
MAX_BUDGET_TOKENS = 32000
MAX_RESULT_ITEMS = 100


# ── Validation errors ─────────────────────────────────────────────────────────


class SkillValidationError(ValueError):
    """Raised when a skill manifest fails validation."""


# ── SkillManifest ─────────────────────────────────────────────────────────────


@dataclass
class SkillManifest:
    """A versioned skill definition."""

    name: str
    version: str
    description: str = ""
    allowed_agents: list[str] = field(default_factory=list)
    allowed_context_views: list[str] = field(default_factory=list)
    read_only: bool = True
    timeout_seconds: float = 10.0
    budget_tokens: int = 1000
    budget_results: int = 10
    parameters: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> SkillManifest:
        """Load a skill manifest from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise SkillValidationError(f"YAML file '{path}' does not contain a mapping")
        manifest = cls(
            name=data.get("name", ""),
            version=str(data.get("version", "")),
            description=data.get("description", ""),
            allowed_agents=data.get("allowed_agents", []),
            allowed_context_views=data.get("allowed_context_views", []),
            read_only=data.get("read_only", True),
            timeout_seconds=float(data.get("timeout_seconds", 10.0)),
            budget_tokens=int(data.get("budget_tokens", 1000)),
            budget_results=int(data.get("budget_results", 10)),
            parameters=data.get("parameters", {}),
            source_path=path,
        )
        return manifest


# ── Validation ─────────────────────────────────────────────────────────────────


def validate_manifest(manifest: SkillManifest) -> list[str]:
    """Validate a skill manifest and return a list of error messages.

    Returns an empty list if the manifest is valid.
    """
    errors: list[str] = []

    # 1. Name must be non-empty and alphanumeric + underscores
    if not manifest.name or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", manifest.name):
        errors.append(
            f"Invalid skill name '{manifest.name}': must be non-empty alphanumeric/underscore"
        )

    # 2. Semantic version
    if not SEMVER_RE.match(manifest.version):
        errors.append(
            f"Invalid version '{manifest.version}' for skill '{manifest.name}': "
            "must be semantic version (MAJOR.MINOR.PATCH)"
        )

    # 3. Allowed agents must be list of non-empty strings
    if not isinstance(manifest.allowed_agents, list):
        errors.append(f"'allowed_agents' for '{manifest.name}' must be a list")
    else:
        for agent in manifest.allowed_agents:
            if not isinstance(agent, str) or not agent.strip():
                errors.append(
                    f"Invalid agent '{agent}' in allowed_agents for '{manifest.name}'"
                )

    # 4. Allowed context views must be in whitelist
    if not isinstance(manifest.allowed_context_views, list):
        errors.append(f"'allowed_context_views' for '{manifest.name}' must be a list")
    else:
        for view in manifest.allowed_context_views:
            if view not in ALLOWED_CONTEXT_VIEWS:
                errors.append(
                    f"Unknown context_view '{view}' for skill '{manifest.name}': "
                    f"allowed values are {sorted(ALLOWED_CONTEXT_VIEWS)}"
                )

    # 5. read_only must be bool
    if not isinstance(manifest.read_only, bool):
        errors.append(f"'read_only' for '{manifest.name}' must be a boolean")

    # 6. timeout_seconds within bounds
    if not (0.1 <= manifest.timeout_seconds <= MAX_TIMEOUT_SECONDS):
        errors.append(
            f"timeout_seconds={manifest.timeout_seconds} for '{manifest.name}' "
            f"out of range [0.1, {MAX_TIMEOUT_SECONDS}]"
        )

    # 7. budget_tokens within bounds
    if not (1 <= manifest.budget_tokens <= MAX_BUDGET_TOKENS):
        errors.append(
            f"budget_tokens={manifest.budget_tokens} for '{manifest.name}' "
            f"out of range [1, {MAX_BUDGET_TOKENS}]"
        )

    # 8. budget_results within bounds
    if not (1 <= manifest.budget_results <= MAX_RESULT_ITEMS):
        errors.append(
            f"budget_results={manifest.budget_results} for '{manifest.name}' "
            f"out of range [1, {MAX_RESULT_ITEMS}]"
        )

    # 9. Parameters schema validation
    if not isinstance(manifest.parameters, dict):
        errors.append(f"'parameters' for '{manifest.name}' must be a dict")
    else:
        for param_name, param_spec in manifest.parameters.items():
            if not isinstance(param_spec, dict):
                errors.append(
                    f"Parameter '{param_name}' in '{manifest.name}' must be a mapping"
                )
                continue
            ptype = param_spec.get("type")
            if ptype and ptype not in ALLOWED_PARAM_TYPES:
                errors.append(
                    f"Parameter '{param_name}' in '{manifest.name}' has invalid type '{ptype}'"
                )

    return errors


# ── SkillRegistry ─────────────────────────────────────────────────────────────


class SkillRegistry:
    """Registry of available skills with startup validation."""

    def __init__(self, *, strict: bool = False) -> None:
        """
        Args:
            strict: If True (staging/production), fail on duplicate names or
                    validation errors. If False (dev), log warnings only.
        """
        self._skills: dict[str, SkillManifest] = {}
        self._strict = strict
        self._validation_errors: list[str] = []

    def register(self, manifest: SkillManifest) -> None:
        """Register a skill manifest after validation."""
        # Validate
        errors = validate_manifest(manifest)
        if errors:
            self._validation_errors.extend(errors)
            msg = f"Skill '{manifest.name}' validation failed: {'; '.join(errors)}"
            if self._strict:
                raise SkillValidationError(msg)
            logger.warning(msg)

        key = f"{manifest.name}@{manifest.version}"

        # Check duplicate
        if key in self._skills:
            msg = f"Duplicate skill registration: '{key}'"
            if self._strict:
                raise SkillValidationError(msg)
            logger.warning(msg)

        self._skills[key] = manifest

    def register_from_yaml(self, path: str) -> None:
        """Load and register a skill from YAML."""
        manifest = SkillManifest.from_yaml(path)
        self.register(manifest)

    def get(self, name: str, version: str | None = None) -> SkillManifest | None:
        """Get a skill by name, optionally pinned to a version."""
        if version:
            return self._skills.get(f"{name}@{version}")
        # Return latest version
        candidates = [s for s in self._skills.values() if s.name == name]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.version)

    def list_skills(self) -> list[SkillManifest]:
        """List all registered skills."""
        return list(self._skills.values())

    def load_directory(self, directory: str) -> int:
        """Load all YAML skills from a directory. Returns count loaded.

        In strict mode, raises SkillValidationError if any manifest fails validation.
        """
        count = 0
        for fname in sorted(os.listdir(directory)):
            if fname.endswith((".yaml", ".yml")):
                self.register_from_yaml(os.path.join(directory, fname))
                count += 1
        return count

    @property
    def validation_errors(self) -> list[str]:
        """Return accumulated validation errors (non-strict mode)."""
        return list(self._validation_errors)

    def assert_valid(self) -> None:
        """Raise if any validation errors were accumulated (call after loading)."""
        if self._validation_errors:
            raise SkillValidationError(
                f"Skill registry has {len(self._validation_errors)} validation error(s): "
                + "; ".join(self._validation_errors[:5])
            )
