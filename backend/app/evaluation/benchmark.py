# -*- coding: utf-8 -*-
"""Task 14 — 可版本化临床能力基准集

BenchmarkCase 定义病例元数据（specialty、difficulty、red_flags、gold_citations 等），
BenchmarkManifest 管理版本和分割，validate_benchmark_manifest 执行完整性校验。

设计原则：
- case_id 唯一
- split 限定为 dev / test / regression / safety / benchmark
- safety split 的 case 必须有 red_flags
- gold_citations 必须存在于 source registry
- split_cases 使用固定 seed 确保可重复
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Set

logger = logging.getLogger(__name__)


# ── 合法 split ─────────────────────────────────────────────────────────────

VALID_SPLITS = frozenset({"dev", "test", "regression", "safety", "benchmark"})


# ── BenchmarkCase ──────────────────────────────────────────────────────────


@dataclass
class BenchmarkCase:
    """基准病例"""

    case_id: str
    specialty: str = ""
    difficulty: int = 3
    split: str = "test"
    required_questions: int = 5
    red_flags: list[str] = field(default_factory=list)
    expected_diagnoses: list[str] = field(default_factory=list)
    treatment_constraints: list[str] = field(default_factory=list)
    gold_citations: list[str] = field(default_factory=list)
    rubric_version: str = "v1"
    description: str = ""


# ── BenchmarkManifest ──────────────────────────────────────────────────────


@dataclass
class BenchmarkManifest:
    """基准集清单"""

    version: str = "v1.0"
    rubric_version: str = "v1"
    dataset_version: str = ""
    cases: list[BenchmarkCase] = field(default_factory=list)
    changelog: str = ""


# ── validate_benchmark_manifest ────────────────────────────────────────────


def validate_benchmark_manifest(
    manifest: BenchmarkManifest,
    known_citation_ids: Optional[Set[str]] = None,
) -> list[str]:
    """校验基准集清单完整性

    Args:
        manifest: 基准集清单
        known_citation_ids: 已知的 citation ID 集合（来自 source registry）

    Returns:
        错误列表，空列表表示通过
    """
    errors: list[str] = []

    # 1. 空 cases
    if not manifest.cases:
        errors.append("基准集为空 (empty cases)")
        return errors

    # 2. 重复 case_id
    seen_ids: set[str] = set()
    for case in manifest.cases:
        if case.case_id in seen_ids:
            errors.append(f"重复 case_id: {case.case_id} (duplicate)")
        seen_ids.add(case.case_id)

    # 3. 非法 split
    for case in manifest.cases:
        if case.split not in VALID_SPLITS:
            errors.append(
                f"非法 split '{case.split}' for case {case.case_id}，"
                f"允许值: {sorted(VALID_SPLITS)}"
            )

    # 4. safety case 必须有 red_flags
    for case in manifest.cases:
        if case.split == "safety" and not case.red_flags:
            errors.append(
                f"safety case {case.case_id} 缺少 red_flags (红旗)"
            )

    # 5. gold_citation 不存在于 registry
    if known_citation_ids is not None:
        for case in manifest.cases:
            for citation_id in case.gold_citations:
                if citation_id not in known_citation_ids:
                    errors.append(
                        f"case {case.case_id} 的 gold_citation '{citation_id}' "
                        f"不存在于 source registry"
                    )

    return errors


# ── split_cases ────────────────────────────────────────────────────────────


def split_cases(
    cases: list[BenchmarkCase],
    split: str,
    seed: int = 42,
) -> list[BenchmarkCase]:
    """按 split 筛选病例并使用固定 seed 排序

    Args:
        cases: 全部病例
        split: 目标分割
        seed: 随机种子（确保可重复）

    Returns:
        属于指定 split 的病例列表（按 seed 确定性排序）
    """
    subset = [c for c in cases if c.split == split]
    rng = random.Random(seed)
    # 使用 case_id 的 hash 做确定性 shuffle
    shuffled = sorted(subset, key=lambda c: rng.random())
    return shuffled
