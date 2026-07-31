# -*- coding: utf-8 -*-
"""评测报告 Manifest 协议 — 版本化报告类型与元数据。

每份评测报告必须知道自己是什么类型（smoke / regression / benchmark）、
由什么版本生成、包含多少病例。旧报告允许兼容读取但不自动升级为正式 regression。

用法：
    from evaluation.report_schema import load_report_manifest, ReportKind

    manifest = load_report_manifest(report_dict, allow_legacy=True)
    if manifest.report_kind is ReportKind.SMOKE:
        ...  # 不作为门禁
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── 报告类型枚举 ─────────────────────────────────────────────────────────────


class ReportKind(str, Enum):
    """报告类型 — 决定门禁行为。

    SMOKE: 冒烟/管道验证，永不阻断 pre-push。
    REGRESSION: 正式回归，阈值门禁生效。
    BENCHMARK: 基准评测，只报告不阻断。
    LEGACY_UNKNOWN: 旧报告兼容读取，不得作为正式门禁候选。
    """

    SMOKE = "smoke"
    REGRESSION = "regression"
    BENCHMARK = "benchmark"
    LEGACY_UNKNOWN = "legacy_unknown"


# ── Manifest 数据模型 ────────────────────────────────────────────────────────

# 门禁阈值默认最小病例数（与 patient_ab_thresholds.json _gate.min_cases 对齐）
_DEFAULT_MIN_CASES = 18


class ReportManifest(BaseModel):
    """评测报告元数据 — 绑定数据集、模型、Prompt、Judge、KB 和评分策略版本。"""

    report_kind: ReportKind
    report_id: str
    created_at: datetime
    case_count: int = Field(ge=0)

    # 版本绑定
    dataset_version: str
    model_version: str
    prompt_version: str
    judge_version: str
    kb_version: str
    scoring_policy_version: str

    # 可选
    seed: Optional[int] = None

    @field_validator("report_kind", mode="before")
    @classmethod
    def _validate_kind(cls, v: str) -> str:
        valid = {k.value for k in ReportKind}
        if v not in valid:
            raise ValueError(f"非法 report_kind: {v!r}，合法值: {sorted(valid)}")
        return v


# ── 加载与兼容 ───────────────────────────────────────────────────────────────


def load_report_manifest(report: dict, *, allow_legacy: bool = True) -> ReportManifest:
    """从报告字典加载 ReportManifest。

    Args:
        report: 完整报告字典（含 cases 和可选 manifest 字段）。
        allow_legacy: 是否允许无 manifest 的旧报告兼容读取。

    Returns:
        ReportManifest 实例。

    Raises:
        ValueError: allow_legacy=False 且报告无 manifest 时。
        pydantic.ValidationError: manifest 字段校验失败时。
    """
    raw_manifest = report.get("manifest")
    n_cases = len(report.get("cases", []))

    if raw_manifest is not None:
        # 有 manifest：严格校验
        manifest = ReportManifest(**raw_manifest)
        # case_count 必须与实际 cases 数量一致
        if manifest.case_count != n_cases:
            raise ValueError(
                f"manifest.case_count={manifest.case_count} 与报告实际 cases 数量 {n_cases} 不一致"
            )
        return manifest

    # 无 manifest
    if not allow_legacy:
        raise ValueError("报告缺少 manifest 字段，且 allow_legacy=False")

    # 兼容读取：按病例数推断 kind
    kind = ReportKind.SMOKE if n_cases < _DEFAULT_MIN_CASES else ReportKind.LEGACY_UNKNOWN
    return ReportManifest(
        report_kind=kind,
        report_id="legacy_unknown",
        created_at=datetime.now(timezone.utc),
        case_count=n_cases,
        dataset_version="unknown",
        model_version="unknown",
        prompt_version="unknown",
        judge_version="unknown",
        kb_version="unknown",
        scoring_policy_version="unknown",
        seed=None,
    )


# ── 校验辅助 ─────────────────────────────────────────────────────────────────


def validate_report_manifest(manifest: ReportManifest) -> list[str]:
    """对已解析的 manifest 做软校验，返回警告/错误列表（空 = 无问题）。

    不抛异常，用于报告生成后的自检和 CI 日志输出。
    """
    errors: list[str] = []

    # 版本字段为 unknown 时警告
    version_fields = [
        "dataset_version", "model_version", "prompt_version",
        "judge_version", "kb_version", "scoring_policy_version",
    ]
    for field in version_fields:
        if getattr(manifest, field) == "unknown":
            errors.append(f"{field} 为 'unknown'，正式报告应绑定具体版本")

    # regression 报告应有 seed
    if manifest.report_kind is ReportKind.REGRESSION and manifest.seed is None:
        errors.append("regression 报告建议设置 seed 以保证可复现")

    return errors
