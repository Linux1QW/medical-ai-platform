# -*- coding: utf-8 -*-
"""版本化评分策略管理"""

from pydantic import BaseModel


class ScoringPolicy(BaseModel):
    """版本化评分策略"""
    version: str = "v1"
    weights: dict[str, float] = {
        "inquiry": 0.25,
        "knowledge": 0.25,
        "humanistic": 0.20,
        "diagnosis": 0.15,
        "treatment": 0.15,
    }
    # 必需维度：如果这些维度不是 "scored" 状态，total_score=null
    required_dimensions: list[str] = []
    # 维度中文名映射
    dimension_names: dict[str, str] = {
        "inquiry": "问诊技巧评估",
        "knowledge": "医学知识评估",
        "humanistic": "人文关怀评估",
        "diagnosis": "诊断能力评估",
        "treatment": "治疗方案评估",
    }


# 策略仓库
_POLICIES: dict[str, ScoringPolicy] = {}


def register_policy(policy: ScoringPolicy):
    _POLICIES[policy.version] = policy


def get_policy(version: str = "v1") -> ScoringPolicy:
    if version not in _POLICIES:
        raise ValueError(f"未知评分策略版本: {version}")
    return _POLICIES[version]


def get_default_policy() -> ScoringPolicy:
    return get_policy("v1")


# 注册默认策略
register_policy(ScoringPolicy(version="v1"))
