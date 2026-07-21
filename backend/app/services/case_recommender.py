# -*- coding: utf-8 -*-
"""病例推荐引擎 — 基于医生能力水平推荐合适难度的病例"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.consultation import Consultation
from app.models.evaluation import Evaluation
from app.services.difficulty_model import DifficultyResult, calculate_actual_difficulty

logger = logging.getLogger(__name__)


@dataclass
class CaseRecommendation:
    case_id: str
    case_name: str
    static_difficulty: float  # 静态标签难度（目录名解析）
    actual_difficulty: Optional[float]  # 动态计算难度
    confidence: float
    reason: str
    _score: float = field(default=0.0, repr=False)  # 用于排序


async def recommend_cases(
    db: AsyncSession,
    doctor_id: Optional[int] = None,
    target_difficulty: Optional[float] = None,
    count: int = 5,
    exclude_completed: Optional[list[str]] = None,
) -> list[CaseRecommendation]:
    """为医生推荐病例

    推荐策略：
    1. 如果没有历史数据，使用静态难度标签
    2. 如果有历史数据，计算医生平均能力，推荐略高于能力的病例
    3. 遵循"最近发展区"理论：难度略高于当前水平

    Args:
        db: 异步数据库会话
        doctor_id: 医生用户 ID（用于查询历史表现）
        target_difficulty: 目标难度（如不指定，自动计算）
        count: 推荐数量
        exclude_completed: 已完成的病例 ID 列表

    Returns:
        推荐的病例列表，按匹配度排序
    """
    available_cases = _load_available_cases(exclude_completed or [])

    if not available_cases:
        return []

    # 确定目标难度
    if target_difficulty is None:
        if doctor_id:
            target_difficulty = await _estimate_target_difficulty(db, doctor_id)
        else:
            target_difficulty = 5.0  # 默认中等难度

    # 为每个候选病例计算综合难度
    scored_cases: list[CaseRecommendation] = []
    for case in available_cases:
        # 尝试从 DB 获取动态难度（通过 patient_id 关联）
        dynamic = await _try_get_dynamic_difficulty(db, case)
        effective_difficulty = (
            dynamic.difficulty
            if dynamic.difficulty is not None
            else case["static_difficulty"]
        )

        # 计算与目标难度的距离（最近发展区：略高于目标优于远低于）
        distance = effective_difficulty - target_difficulty
        if distance < 0:
            # 低于目标：惩罚稍大（太简单收益低）
            score = 1.0 / (1.0 + abs(distance) * 1.2)
        else:
            # 略高于目标：最佳挑战区
            score = 1.0 / (1.0 + distance * 0.8)

        # 构建推荐理由
        if dynamic.difficulty is not None and dynamic.confidence > 0.3:
            reason = (
                f"动态难度 {dynamic.difficulty:.1f}"
                f"（基于 {dynamic.sample_size} 次评估），"
                f"接近目标 {target_difficulty:.1f}"
            )
        else:
            reason = f"静态难度 {case['static_difficulty']:.1f}，接近目标 {target_difficulty:.1f}"

        scored_cases.append(
            CaseRecommendation(
                case_id=case["case_id"],
                case_name=case.get("case_name", case["case_id"]),
                static_difficulty=case["static_difficulty"],
                actual_difficulty=dynamic.difficulty,
                confidence=dynamic.confidence,
                reason=reason,
                _score=score,
            )
        )

    # 按匹配度排序
    scored_cases.sort(key=lambda x: x._score, reverse=True)
    return scored_cases[:count]


async def _estimate_target_difficulty(db: AsyncSession, doctor_id: int) -> float:
    """基于医生历史表现估算目标难度

    使用"最近发展区"理论：推荐略高于当前能力的难度
    """
    try:
        # 查询医生最近 20 次已完成的评估
        stmt = (
            select(Evaluation.total_score)
            .join(Consultation, Evaluation.consultation_id == Consultation.id)
            .where(
                Consultation.doctor_id == doctor_id,
                Evaluation.evaluation_status == "completed",
                Evaluation.total_score.isnot(None),
            )
            .order_by(Evaluation.created_at.desc())
            .limit(20)
        )
        result = await db.execute(stmt)
        scores = [row[0] for row in result.all()]

        if not scores:
            return 5.0  # 新医生，默认中等

        avg_score = sum(scores) / len(scores)

        # 分数映射到难度：
        # 平均分 90+ → 推荐难度 8（高难度挑战）
        # 平均分 70-90 → 推荐难度 5-7（中等偏上）
        # 平均分 50-70 → 推荐难度 3-5（中等）
        # 平均分 <50 → 推荐难度 2-3（基础练习）
        ability = avg_score / 100  # 0-1
        target = ability * 10 + 1.5  # 略高于当前能力
        return min(max(target, 1.0), 10.0)
    except Exception as e:
        logger.warning(f"Failed to estimate target difficulty: {e}")
        return 5.0


async def _try_get_dynamic_difficulty(
    db: AsyncSession,
    case: dict,
) -> DifficultyResult:
    """尝试通过虚拟患者关联获取动态难度，失败则返回空结果"""
    patient_db_id = case.get("patient_db_id")
    if patient_db_id is None:
        return DifficultyResult(difficulty=None, confidence=0, sample_size=0)
    try:
        return await calculate_actual_difficulty(db, patient_db_id)
    except Exception as e:
        logger.debug(f"Dynamic difficulty unavailable for {case['case_id']}: {e}")
        return DifficultyResult(difficulty=None, confidence=0, sample_size=0)


def _load_available_cases(
    exclude_ids: Optional[list[str]] = None,
) -> list[dict]:
    """从 dataset/ 目录加载可用病例"""
    dataset_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "dataset"
    )
    dataset_dir = os.path.normpath(dataset_dir)

    if not os.path.exists(dataset_dir):
        logger.warning(f"Dataset directory not found: {dataset_dir}")
        return []

    exclude_set = set(exclude_ids or [])
    cases: list[dict] = []

    for dirname in os.listdir(dataset_dir):
        dirpath = os.path.join(dataset_dir, dirname)
        if not os.path.isdir(dirpath):
            continue

        # 解析目录名：patient1_5 → case_id=patient1_5, difficulty=5
        parts = dirname.rsplit("_", 1)
        if len(parts) < 2:
            continue

        try:
            difficulty = float(parts[1])
        except (ValueError, IndexError):
            continue

        if dirname in exclude_set:
            continue

        # 读取病例基本信息
        case_info = _read_case_info(dirpath)

        cases.append(
            {
                "case_id": dirname,
                "case_name": case_info.get("chief_complaint") or f"病例 {parts[0]}",
                "static_difficulty": difficulty,
                "dir_path": dirpath,
            }
        )

    return cases


def _read_case_info(case_dir: str) -> dict:
    """读取病例目录中的主 JSON 文件基本信息"""
    info: dict = {}

    for filename in os.listdir(case_dir):
        if not filename.endswith(".json"):
            continue
        # 优先读取与目录同名的主 JSON（如 patient1_5/patient1_5.json）
        filepath = os.path.join(case_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue

            # 主诉在 "门诊病历"."主诉" 下
            medical_record = data.get("门诊病历", {})
            if isinstance(medical_record, dict):
                chief_complaint = medical_record.get("主诉", "")
                if chief_complaint:
                    info["chief_complaint"] = chief_complaint

            # 基础信息中的姓名可作为备选名称
            basic = data.get("基础信息", {})
            if isinstance(basic, dict):
                name = basic.get("姓名", "")
                if name:
                    info["patient_name"] = name

            # 只读第一个（主 JSON）
            break
        except Exception:
            continue

    return info
