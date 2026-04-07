# -*- coding: utf-8 -*-
"""一次性脚本：将 dataset 中的病例数据构建为 FAISS 向量索引

用法：
    cd backend
    python -m app.services.rag.build_index
"""

import asyncio
import json
import logging
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 数据集根目录
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "dataset")


def _collect_prescriptions(data: dict) -> list[dict]:
    """从病例 JSON 中收集所有处方单（兼容不同 key 命名），过滤检查用药"""
    # 检查/麻醉用药关键词，这些不属于治疗方案
    EXAM_DRUG_KEYWORDS = ["丙泊酚", "达克罗宁", "利多卡因胶浆", "西甲硅油"]
    prescriptions = []
    for key, value in data.items():
        if "处方" in key and isinstance(value, list):
            for rx in value:
                name = rx.get("药品名称", "")
                if any(kw in name for kw in EXAM_DRUG_KEYWORDS):
                    continue
                prescriptions.append(rx)
    return prescriptions


def _format_prescription_text(prescriptions: list[dict]) -> str:
    """将处方列表格式化为可读文本"""
    if not prescriptions:
        return "无处方"
    lines = []
    for rx in prescriptions:
        name = rx.get("药品名称", "")
        dose = rx.get("剂量", "")
        unit = rx.get("单位", "")
        route = rx.get("给药方式", "")
        freq = rx.get("频次", "")
        days = rx.get("天数", "")
        lines.append(f"{name} {dose}{unit} {route} {freq} {days}天")
    return "\n".join(lines)


def _format_exam_text(data: dict) -> str:
    """格式化检验检查信息"""
    parts = []
    # 检验
    lab = data.get("检验")
    if lab and lab != "无":
        if isinstance(lab, dict):
            lab_name = lab.get("检验名称", "")
            items = lab.get("检验描述", [])
            details = "; ".join(
                f"{it.get('项目名称', '')}: {it.get('数值', '')} (参考{it.get('参考范围', '')})"
                for it in items
                if isinstance(it, dict)
            )
            parts.append(f"检验-{lab_name}: {details}")
        elif isinstance(lab, list):
            for item in lab:
                if isinstance(item, dict):
                    lab_name = item.get("检验名称", "")
                    items = item.get("检验描述", [])
                    details = "; ".join(
                        f"{it.get('项目名称', '')}: {it.get('数值', '')} (参考{it.get('参考范围', '')})"
                        for it in items
                        if isinstance(it, dict)
                    )
                    parts.append(f"检验-{lab_name}: {details}")
    # 检查
    exams = data.get("检查")
    if exams and exams != "无":
        if isinstance(exams, list):
            for ex in exams:
                if isinstance(ex, dict):
                    parts.append(f"{ex.get('检查名称', '')}: {ex.get('检查报告', '')}")
    return "\n".join(parts) if parts else "无"


def load_all_cases() -> list[dict]:
    """加载所有主病例 JSON，返回结构化的元数据列表"""
    dataset_path = os.path.normpath(DATASET_DIR)
    logger.info(f"扫描数据集目录: {dataset_path}")

    cases = []
    for dirname in sorted(os.listdir(dataset_path)):
        dirpath = os.path.join(dataset_path, dirname)
        if not os.path.isdir(dirpath):
            continue
        # 只取每个患者目录下的主文件（不含人格变体）
        main_file = os.path.join(dirpath, f"{dirname}.json")
        if not os.path.exists(main_file):
            continue

        with open(main_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        base = data.get("基础信息", {})
        record = data.get("门诊病历", {})
        prescriptions = _collect_prescriptions(data)
        exam_text = _format_exam_text(data)
        rx_text = _format_prescription_text(prescriptions)

        # 用于 embedding 的文本：聚合患者核心临床特征
        embed_text = (
            f"患者: {base.get('性别', '')} {base.get('年龄', '')}岁\n"
            f"主诉: {record.get('主诉', '')}\n"
            f"现病史: {record.get('现病史', '')}\n"
            f"既往史: {record.get('既往史', '')}\n"
            f"症状体征: {record.get('体格检查', '')}\n"
            f"辅助检查: {record.get('辅助检查', '')}\n"
            f"诊断: {data.get('主诊断', '')}"
        )

        meta = {
            "patient_id": dirname,
            "name": base.get("姓名", ""),
            "gender": base.get("性别", ""),
            "age": base.get("年龄", ""),
            "chief_complaint": record.get("主诉", ""),
            "history": record.get("既往史", ""),
            "present_illness": record.get("现病史", ""),
            "diagnosis": data.get("主诊断", ""),
            "prescriptions": rx_text,
            "exams": exam_text,
            "notes": record.get("注意事项", ""),
            "embed_text": embed_text,
        }
        cases.append(meta)

    logger.info(f"共加载 {len(cases)} 条主病例")
    return cases


async def build():
    """构建并保存向量索引"""
    from app.services.rag.embeddings import get_embeddings
    from app.services.rag.vector_store import CaseVectorStore

    cases = load_all_cases()
    if not cases:
        logger.error("未找到任何病例数据，请检查 dataset 目录")
        return

    # 提取所有 embed_text
    texts = [c["embed_text"] for c in cases]
    logger.info(f"开始向量化 {len(texts)} 条病例文本...")

    embeddings = await get_embeddings(texts)
    logger.info(f"向量化完成，维度: {len(embeddings[0])}")

    # 从元数据中移除 embed_text（不需要持久化）
    for c in cases:
        del c["embed_text"]

    store = CaseVectorStore()
    store.build(embeddings, cases)
    store.save()
    logger.info("索引构建完成！")


if __name__ == "__main__":
    asyncio.run(build())
