"""Versioned skill manifest and registry."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


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
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> SkillManifest:
        """Load a skill manifest from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            allowed_agents=data.get("allowed_agents", []),
            allowed_context_views=data.get("allowed_context_views", []),
            read_only=data.get("read_only", True),
            timeout_seconds=data.get("timeout_seconds", 10.0),
            budget_tokens=data.get("budget_tokens", 1000),
            parameters=data.get("parameters", {}),
        )


class SkillRegistry:
    """Registry of available skills."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillManifest] = {}

    def register(self, manifest: SkillManifest) -> None:
        """Register a skill manifest."""
        key = f"{manifest.name}@{manifest.version}"
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
        """Load all YAML skills from a directory. Returns count loaded."""
        import os

        count = 0
        for fname in sorted(os.listdir(directory)):
            if fname.endswith((".yaml", ".yml")):
                self.register_from_yaml(os.path.join(directory, fname))
                count += 1
        return count
