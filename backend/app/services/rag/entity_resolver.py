# -*- coding: utf-8 -*-
"""医学实体归一化 — 别名映射 + 实体提取

在索引构建时为 chunk 注入实体标签，在检索时对查询做别名→规范名映射，
提升 BM25 对同义术语（如"心梗"="急性心肌梗死"="AMI"）的匹配能力。
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 加载实体词典（懒加载单例）
_entity_mapping: Optional[dict] = None


def _load_entity_mapping() -> dict:
    """加载 entity_mapping.json，首次调用时从磁盘读取并缓存"""
    global _entity_mapping
    if _entity_mapping is not None:
        return _entity_mapping

    # 从 entity_resolver.py → backend/data/entity_mapping.json
    mapping_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "data" / "entity_mapping.json"
    )
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            _entity_mapping = json.load(f)
        total = sum(len(v) for v in _entity_mapping.values())
        logger.info(
            f"实体词典加载完成: 疾病 {len(_entity_mapping.get('diseases', {}))}, "
            f"药物 {len(_entity_mapping.get('drugs', {}))}, "
            f"操作 {len(_entity_mapping.get('procedures', {}))}, "
            f"共 {total} 条"
        )
    except Exception as e:
        logger.warning(f"实体词典加载失败: {e}")
        _entity_mapping = {"diseases": {}, "drugs": {}, "procedures": {}}

    return _entity_mapping


def extract_entities(text: str) -> list:
    """从文本中提取医学实体

    遍历词典中所有实体的规范名和别名，在文本中做精确子串匹配。
    别名长度 < 2 的跳过以避免单字误匹配。

    Args:
        text: 待检测文本（chunk 内容或查询文本）

    Returns:
        [{"type": "disease", "normalized": "急性心肌梗死", "icd10": "I21.9",
          "matched_alias": "心梗", "category": "心血管"}, ...]
    """
    mapping = _load_entity_mapping()
    entities = []
    seen = set()  # 避免同一实体重复添加

    for entity_type in ["diseases", "drugs", "procedures"]:
        for normalized_name, info in mapping.get(entity_type, {}).items():
            matched_alias = None

            # 优先检查规范名
            if normalized_name in text:
                matched_alias = normalized_name
            else:
                # 检查别名（取第一个命中的）
                for alias in info.get("aliases", []):
                    if alias in text and len(alias) >= 2 and _alias_boundary_ok(alias, text):
                        matched_alias = alias
                        break

            if matched_alias is not None:
                dedup_key = (entity_type, normalized_name)
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    entities.append(
                        _build_entity(entity_type, normalized_name, info, matched_alias)
                    )

    return entities


def _alias_boundary_ok(alias: str, text: str) -> bool:
    """检查别名在文本中的匹配是否在合理的词边界内

    对纯英文/数字别名（如 "NS"、"LC"、"PTX"），要求匹配位置不在更长的
    英文单词内部，避免 "NS" 匹配到 "NSCLC" 中的子串。
    对中文别名不做额外限制。
    """
    # 纯中文别名不做边界检查
    if re.search(r'[\u4e00-\u9fff]', alias):
        return True

    # 对英文/数字别名，检查所有匹配位置是否在更长的英文 token 内部
    # 只检查 ASCII 字母数字作为边界（CJK 字符不算英文单词的一部分）
    pattern = re.compile(re.escape(alias), re.IGNORECASE)
    for m in pattern.finditer(text):
        start, end = m.start(), m.end()
        # 检查左侧：是否为 ASCII 字母或数字
        if start > 0 and text[start - 1].isascii() and text[start - 1].isalnum():
            continue
        # 检查右侧：是否为 ASCII 字母或数字
        if end < len(text) and text[end].isascii() and text[end].isalnum():
            continue
        return True  # 至少有一个匹配在词边界处
    return False


def normalize_query(text: str) -> str:
    """将查询中的别名替换为「别名(规范名)」形式，增强 BM25 匹配

    对每个命中的别名，在其后追加括号包裹的规范名，使分词后同时包含
    别名 token 和规范名 token，从而命中同时含有规范名的文档。

    Example:
        "心梗治疗" → "心梗(急性心肌梗死)治疗"
        "NSCLC靶向治疗" → "NSCLC(非小细胞肺癌)靶向治疗"

    Args:
        text: 原始查询文本

    Returns:
        注入规范名后的增强查询文本
    """
    mapping = _load_entity_mapping()

    # 收集所有需要替换的 (alias, normalized) 对，按别名长度降序排列
    # 避免短别名覆盖长别名（如 "AMI" 不应阻止 "急性心梗" 被替换）
    replacements = []
    for entity_type in ["diseases", "drugs", "procedures"]:
        for normalized_name, info in mapping.get(entity_type, {}).items():
            for alias in info.get("aliases", []):
                if (alias in text and alias != normalized_name
                        and len(alias) >= 2
                        and _alias_boundary_ok(alias, text)
                        and normalized_name not in text):
                    replacements.append((alias, normalized_name))

    # 按别名长度降序排序，防止短别名先替换导致长别名失效
    replacements.sort(key=lambda x: len(x[0]), reverse=True)

    for alias, normalized in replacements:
        if alias in text:
            text = text.replace(alias, f"{alias}({normalized})", 1)

    return text


def get_entity_tags(text: str) -> str:
    """提取文本中所有实体的规范名，以空格分隔（用于注入 chunk metadata 的 entity_names 字段）

    Args:
        text: chunk 文本内容

    Returns:
        空格分隔的规范实体名列表，如 "急性心肌梗死 阿司匹林 经皮冠状动脉介入"
    """
    entities = extract_entities(text)
    return " ".join(e["normalized"] for e in entities)


def _build_entity(entity_type: str, normalized: str, info: dict, matched_alias: str) -> dict:
    """构建单个实体字典"""
    entity = {
        "type": entity_type.rstrip("s"),  # "diseases" → "disease"
        "normalized": normalized,
        "matched_alias": matched_alias,
    }
    # 添加编码字段
    for key in ["icd10", "atc", "icd9cm3"]:
        if key in info:
            entity[key] = info[key]
    if "category" in info:
        entity["category"] = info["category"]
    return entity
