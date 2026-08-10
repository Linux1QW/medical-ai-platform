"""Immutable prompt bundle registry with deterministic experiment assignment."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

RolloutStage = Literal["canary_0", "canary_5", "canary_25", "full_100"]

ROLLOUT_PERCENTAGES: dict[RolloutStage, int] = {
    "canary_0": 0,
    "canary_5": 5,
    "canary_25": 25,
    "full_100": 100,
}


@dataclass(frozen=True)
class PromptBundle:
    """An immutable prompt bundle."""
    bundle_id: str
    name: str
    version: str
    system_prompt: str
    safety_policy: str
    content_hash: str

    @staticmethod
    def compute_hash(system_prompt: str, safety_policy: str) -> str:
        content = f"{system_prompt}||{safety_policy}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class Experiment:
    """A prompt experiment with staged rollout."""
    experiment_id: str
    name: str
    baseline_bundle_id: str
    treatment_bundle_id: str
    stage: RolloutStage = "canary_0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    auto_rollback_triggered: bool = False

    @property
    def rollout_percentage(self) -> int:
        return ROLLOUT_PERCENTAGES[self.stage]

    def advance_stage(self) -> RolloutStage:
        stages: list[RolloutStage] = ["canary_0", "canary_5", "canary_25", "full_100"]
        current_idx = stages.index(self.stage)
        if current_idx < len(stages) - 1:
            self.stage = stages[current_idx + 1]
        return self.stage

    def should_rollback(self, *, hidden_leak: bool, unsafe_suggestion: bool, error_rate: float) -> bool:
        if hidden_leak or unsafe_suggestion:
            self.auto_rollback_triggered = True
            return True
        if error_rate > 0.05:
            self.auto_rollback_triggered = True
            return True
        return False


def deterministic_assign(doctor_id: int, experiment_id: str, percentage: int) -> bool:
    """Deterministic assignment based on doctor_id hash."""
    hash_input = f"{doctor_id}:{experiment_id}"
    hash_value = int(hashlib.sha256(hash_input.encode("utf-8")).hexdigest(), 16)
    bucket = hash_value % 100
    return bucket < percentage


class PromptRegistry:
    """Registry of immutable prompt bundles and experiments."""

    def __init__(self) -> None:
        self._bundles: dict[str, PromptBundle] = {}
        self._experiments: dict[str, Experiment] = {}
        self._counter = 0

    def register_bundle(self, *, name: str, version: str, system_prompt: str, safety_policy: str) -> PromptBundle:
        self._counter += 1
        bundle_id = f"bundle_{self._counter:06d}"
        content_hash = PromptBundle.compute_hash(system_prompt, safety_policy)
        bundle = PromptBundle(bundle_id=bundle_id, name=name, version=version,
                              system_prompt=system_prompt, safety_policy=safety_policy, content_hash=content_hash)
        self._bundles[bundle_id] = bundle
        return bundle

    def get_bundle(self, bundle_id: str) -> PromptBundle | None:
        return self._bundles.get(bundle_id)

    def create_experiment(self, *, name: str, baseline_bundle_id: str, treatment_bundle_id: str) -> Experiment:
        if baseline_bundle_id not in self._bundles:
            raise KeyError(f"Baseline bundle {baseline_bundle_id} not found")
        if treatment_bundle_id not in self._bundles:
            raise KeyError(f"Treatment bundle {treatment_bundle_id} not found")
        self._counter += 1
        experiment_id = f"exp_{self._counter:06d}"
        experiment = Experiment(experiment_id=experiment_id, name=name,
                                baseline_bundle_id=baseline_bundle_id, treatment_bundle_id=treatment_bundle_id)
        self._experiments[experiment_id] = experiment
        return experiment

    def assign_bundle(self, experiment_id: str, doctor_id: int) -> str:
        experiment = self._experiments.get(experiment_id)
        if experiment is None:
            raise KeyError(f"Experiment {experiment_id} not found")
        if experiment.auto_rollback_triggered:
            return experiment.baseline_bundle_id
        in_treatment = deterministic_assign(doctor_id, experiment_id, experiment.rollout_percentage)
        return experiment.treatment_bundle_id if in_treatment else experiment.baseline_bundle_id

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> list[Experiment]:
        return list(self._experiments.values())
