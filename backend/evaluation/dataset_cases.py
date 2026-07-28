"""dataset/ 真实病例接入离线评测。

将根目录 dataset/patientN_XX 下的主 JSON（中文键名：基础信息/门诊病历/主诊断/
处方单/门诊对话）转换为 RagGoldCase，使 rag_eval 可直接在 150+ 个真实门诊
病例上跑评测闭环。

约定（与 case_recommender / seed_patients 的既有解析保持一致）：
- 主 JSON 与目录同名（patient1_5/patient1_5.json），人格变体文件忽略；
- 目录名后缀 XX 为门诊对话轮次数，作为静态难度分档依据；
- gold 检索期望字段（gold_relevant_sources 等）dataset 中不存在，留空，
  期望结果按 legacy 转换器默认值补齐（supports / 不拒答）。
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .datasets import RagGoldCase

logger = logging.getLogger(__name__)

# backend/evaluation/ → backend/ → 仓库根目录 / dataset
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"

# 静态难度分档：目录名后缀（≈对话轮次数）
_EASY_MAX_ROUNDS = 10.0
_MEDIUM_MAX_ROUNDS = 25.0

# 确定性 split 划分：按排序后序号取模，7 dev / 2 test / 1 regression
_SPLIT_PATTERN = ("dev",) * 7 + ("test",) * 2 + ("regression",)


def _clean_text(text: Any) -> str:
    """清理对话文本：去零宽字符与首尾空白（dataset 原文含 U+200B）"""
    if not isinstance(text, str):
        return ""
    return text.replace("\u200b", "").replace("\ufeff", "").strip()


def _parse_static_difficulty(dirname: str) -> Optional[float]:
    """解析目录名 patientN_XX 的静态难度后缀，非法则返回 None"""
    parts = dirname.rsplit("_", 1)
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def _difficulty_level(static_difficulty: float) -> str:
    if static_difficulty <= _EASY_MAX_ROUNDS:
        return "easy"
    if static_difficulty <= _MEDIUM_MAX_ROUNDS:
        return "medium"
    return "hard"


def _build_conversation_text(dialogue: List[Any]) -> str:
    """门诊对话数组 → 「医生: …\n患者: …」逐轮展开文本"""
    lines: List[str] = []
    for turn in dialogue:
        if not isinstance(turn, dict):
            continue
        doctor = _clean_text(turn.get("医生"))
        patient = _clean_text(turn.get("患者"))
        if doctor:
            lines.append(f"医生: {doctor}")
        if patient:
            lines.append(f"患者: {patient}")
    return "\n".join(lines)


def _build_patient_info(data: Dict[str, Any]) -> str:
    """基础信息 + 现病史/既往史 合成患者描述（不含姓名，去标识化）"""
    basic = data.get("基础信息", {}) if isinstance(data.get("基础信息"), dict) else {}
    record = data.get("门诊病历", {}) if isinstance(data.get("门诊病历"), dict) else {}

    parts: List[str] = []
    gender = _clean_text(basic.get("性别"))
    age = _clean_text(basic.get("年龄"))
    if gender or age:
        profile = "患者"
        if gender:
            profile += gender
        if age:
            profile += f"，{age}岁"
        parts.append(profile)

    for label, key in (("现病史", "现病史"), ("既往史", "既往史"), ("药物过敏史", "药物过敏史")):
        value = _clean_text(record.get(key))
        if value and value != "无":
            parts.append(f"{label}：{value}")

    return "；".join(parts)


def _build_treatment_plan(data: Dict[str, Any]) -> Optional[str]:
    """门诊病历.处理 + 各处方单药品明细 合成治疗方案文本"""
    lines: List[str] = []

    record = data.get("门诊病历", {}) if isinstance(data.get("门诊病历"), dict) else {}
    handling = _clean_text(record.get("处理"))
    if handling and handling != "无":
        lines.append(f"处理：{handling}")

    # 处方单键名形如「处方单1(西成方）」，按前缀匹配兼容全半角括号
    for key, value in data.items():
        if not (isinstance(key, str) and key.startswith("处方单") and isinstance(value, list)):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            name = _clean_text(item.get("药品名称"))
            if not name:
                continue
            detail_parts = [name]
            for field in ("给药方式", "频次"):
                field_value = _clean_text(item.get(field))
                if field_value:
                    detail_parts.append(field_value)
            days = _clean_text(item.get("天数"))
            if days:
                detail_parts.append(f"{days}天")
            lines.append(" ".join(detail_parts))

    return "\n".join(lines) if lines else None


def convert_patient_case(case_dir: Path, split: str = "dev") -> Optional[RagGoldCase]:
    """将单个 dataset/patientN_XX 目录转换为 RagGoldCase。

    以下情况返回 None（跳过该病例）：主 JSON 缺失/解析失败、
    目录名难度后缀非法、门诊对话为空。
    """
    dirname = case_dir.name
    static_difficulty = _parse_static_difficulty(dirname)
    if static_difficulty is None:
        return None

    main_json = case_dir / f"{dirname}.json"
    if not main_json.exists():
        logger.warning("dataset case skipped, main JSON missing: %s", main_json)
        return None

    try:
        with open(main_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("dataset case skipped, unreadable JSON %s: %s", main_json, e)
        return None
    if not isinstance(data, dict):
        return None

    dialogue = data.get("门诊对话")
    conversation_text = _build_conversation_text(dialogue) if isinstance(dialogue, list) else ""
    if not conversation_text:
        logger.warning("dataset case skipped, empty dialogue: %s", dirname)
        return None

    record = data.get("门诊病历", {}) if isinstance(data.get("门诊病历"), dict) else {}
    chief_complaint = _clean_text(record.get("主诉")) or None
    doctor_diagnosis = _clean_text(data.get("主诊断")) or None

    return RagGoldCase(
        case_id=dirname,
        split=split,
        department="未知",
        difficulty=_difficulty_level(static_difficulty),
        chief_complaint=chief_complaint,
        patient_info=_build_patient_info(data),
        conversation_text=conversation_text,
        doctor_diagnosis=doctor_diagnosis,
        treatment_plan=_build_treatment_plan(data),
        # gold 检索期望 dataset 中不存在，留空；期望结果按默认值补齐
        expected_stance="supports",
        should_refuse=False,
        notes=f"converted from dataset/{dirname}; static_difficulty={static_difficulty}",
    )


def load_dataset_cases(
    dataset_dir: Optional[Path] = None, limit: Optional[int] = None
) -> List[RagGoldCase]:
    """扫描 dataset/ 目录并转换全部病例。

    目录按名称排序后依次分配 split（7 dev / 2 test / 1 regression，
    确定性划分，同一数据集多次运行结果一致）。
    """
    root = dataset_dir or DEFAULT_DATASET_DIR
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    case_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    cases: List[RagGoldCase] = []
    for case_dir in case_dirs:
        split = _SPLIT_PATTERN[len(cases) % len(_SPLIT_PATTERN)]
        case = convert_patient_case(case_dir, split=split)
        if case is None:
            continue
        cases.append(case)
        if limit is not None and len(cases) >= limit:
            break

    logger.info("Loaded %d dataset cases from %s (scanned %d dirs)", len(cases), root, len(case_dirs))
    return cases
