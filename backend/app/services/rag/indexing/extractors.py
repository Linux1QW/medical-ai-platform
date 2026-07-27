# -*- coding: utf-8 -*-
"""多格式文档抽取 — PDF（含表格/OCR 兜底标记）与 Word（.docx → Markdown）"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# 支持的文档格式（统一转 Markdown 后走同一套语义分块）
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".pdf", ".docx")


# ── 表格抽取辅助函数 ──────────────────────────────────────────────────────────────

def _rects_overlap(block_bbox: tuple, table_bbox: tuple, threshold: float = 0.1) -> bool:
    """检查文本块是否与表格区域重叠（重叠面积超过块自身面积 10% 则认为重叠）"""
    bx0, by0, bx1, by1 = block_bbox
    tx0, ty0, tx1, ty1 = table_bbox
    # 无交集快返
    if bx1 <= tx0 or tx1 <= bx0 or by1 <= ty0 or ty1 <= by0:
        return False
    ix = max(0.0, min(bx1, tx1) - max(bx0, tx0))
    iy = max(0.0, min(by1, ty1) - max(by0, ty0))
    intersection = ix * iy
    block_area = max((bx1 - bx0) * (by1 - by0), 1e-6)
    return intersection / block_area > threshold


def _table_to_text(rows: List, table_idx: int) -> str:
    """将 PyMuPDF 表格单元格数据转换为 Markdown 表格格式文本

    输入 rows 为 table.extract() 返回的二维列表，输出示例：
        【表格1】
        | 分期 | 表现 | 治疗方案 |
        | --- | --- | --- |
        | I期 | ... | ... |
    """
    if not rows:
        return ""

    # 清洗单元格，将 None 转为空字符串，内部换行处理
    cleaned: List[List[str]] = []
    for row in rows:
        cleaned.append([
            str(cell).replace("\n", " ").strip() if cell is not None else ""
            for cell in row
        ])

    if not cleaned:
        return ""

    # 对齐列数
    max_cols = max(len(r) for r in cleaned)
    normalized = [r + [""] * (max_cols - len(r)) for r in cleaned]

    header = normalized[0]
    data_rows = normalized[1:]

    lines = [f"【表格{table_idx}】"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in data_rows:
        if any(cell.strip() for cell in row):  # 跳过全空行
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def extract_text_from_pdf(pdf_path: Path) -> List[Dict]:
    """提取 PDF 文本，表格单独抄取并标记 content_type

    流程：
    1. 每页先检测表格区域（fitz.find_tables）
    2. 提取非表格区域的正文文本
    3. 表格转化为 Markdown 格式单独返回，避免被错误切碎
    4. 表格检测失败时自动降级为全页文本模式

    Returns:
        [{"text": ..., "page": ..., "source": ..., "content_type": "text"|"table"}, ...]
    """
    pages = []
    try:
        from app.core.config import settings
        ocr_enabled = getattr(settings, "ENABLE_OCR", False)
        ocr_threshold = getattr(settings, "OCR_MIN_TEXT_THRESHOLD", 50)
        ocr_dpi = getattr(settings, "OCR_RENDER_DPI", 200)

        doc = fitz.open(pdf_path)
        source_name = pdf_path.name

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_real_num = page_num + 1

            # 1. 尝试检测表格
            table_items: List[Dict] = []
            table_bboxes: List[tuple] = []
            try:
                table_finder = page.find_tables()
                for t_idx, table in enumerate(table_finder.tables, 1):
                    rows = table.extract()
                    table_text = _table_to_text(rows, t_idx)
                    if table_text:
                        table_items.append({
                            "text": table_text,
                            "page": page_real_num,
                            "source": source_name,
                            "content_type": "table",
                        })
                        table_bboxes.append(table.bbox)
            except Exception as e:
                logger.debug(
                    f"[{source_name}] 第{page_real_num}页表格检测失败，降级为全页文本: {e}"
                )

            # 2. 提取非表格区域的正文
            if table_bboxes:
                text_blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,...)
                non_table_parts = []
                for block in text_blocks:
                    if len(block) < 5:
                        continue
                    block_bbox = block[:4]
                    if not any(_rects_overlap(block_bbox, tb) for tb in table_bboxes):
                        block_text = block[4].strip()
                        if block_text:
                            non_table_parts.append(block_text)
                text = "\n".join(non_table_parts).strip()
            else:
                text = page.get_text().strip()

            # 3. 低文本页（疑似扫描/图片页）渲染为 PNG 并标记，交由 OCR 兜底
            ocr_image = None
            if ocr_enabled and len(text) < ocr_threshold:
                try:
                    pix = page.get_pixmap(dpi=ocr_dpi)
                    ocr_image = pix.tobytes("png")
                except Exception as e:
                    logger.debug(
                        f"[{source_name}] 第{page_real_num}页渲染 OCR 图片失败: {e}"
                    )
                    ocr_image = None

            # 4. 添加正文页（低文本但待 OCR 的页也保留，稍后回填文本）
            if text or ocr_image is not None:
                page_entry = {
                    "text": text,
                    "page": page_real_num,
                    "source": source_name,
                    "content_type": "text",
                }
                if ocr_image is not None:
                    page_entry["_ocr_image"] = ocr_image
                pages.append(page_entry)

            # 5. 添加表格块
            pages.extend(table_items)

        doc.close()
        text_cnt = sum(1 for p in pages if p.get("content_type") == "text")
        table_cnt = sum(1 for p in pages if p.get("content_type") == "table")
        logger.info(
            f"已提取 {source_name}: {text_cnt} 页正文，{table_cnt} 个表格块"
        )
        return pages

    except Exception as e:
        logger.error(f"提取 PDF 失败 {pdf_path}: {e}")
        return []


# ── Word 文档抽取（.docx → Markdown）────────────────────────────────

def _docx_heading_level(style_name: str) -> int:
    """将 Word 段落样式名映射为标题层级（0=正文）。

    兼容英文 "Heading N" / "Title" 与中文 "标题 N" / "标题" 样式。
    """
    if not style_name:
        return 0
    s = style_name.strip().lower()
    m = re.match(r'(?:heading|标题)\s*(\d+)', s)
    if m:
        return min(int(m.group(1)), 3)
    if s in ("title", "标题", "文档标题"):
        return 1
    return 0


def _iter_docx_blocks(document):
    """按文档自然顺序迭代 docx 的段落与表格块。

    Yields:
        ("paragraph", Paragraph) 或 ("table", Table)
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag  # 含命名空间，如 '{...}p' / '{...}tbl'
        if tag.endswith("}p"):
            yield "paragraph", Paragraph(child, document)
        elif tag.endswith("}tbl"):
            yield "table", Table(child, document)


def extract_text_from_docx(docx_path: Path) -> List[Dict]:
    """提取 Word 文档内容并统一为 Markdown 表示，表格单独标记 content_type。

    - 标题段落按样式转为 Markdown ATX 标题（#/##/###），保留层级结构
    - 普通段落按原文顺序拼接为正文 Markdown（空行分隔，保证段落切分）
    - 表格转为 Markdown 表格（复用 _table_to_text），作为独立 content_type='table' 块
    - Word 无稳定分页概念，正文统一归于第 1 页

    Returns:
        [{"text": ..., "page": ..., "source": ..., "content_type": "text"|"table"}, ...]
    """
    try:
        from docx import Document
    except ImportError:
        logger.error("未安装 python-docx，无法处理 .docx 文件；请 pip install python-docx")
        return []

    pages: List[Dict] = []
    try:
        document = Document(str(docx_path))
        source_name = docx_path.name

        md_lines: List[str] = []
        table_items: List[Dict] = []
        table_idx = 0

        for kind, block in _iter_docx_blocks(document):
            if kind == "paragraph":
                text = (block.text or "").strip()
                if not text:
                    continue
                style_name = block.style.name if block.style is not None else ""
                level = _docx_heading_level(style_name)
                if level > 0:
                    md_lines.append(f"{'#' * level} {text}")
                else:
                    md_lines.append(text)
            else:  # table
                rows = [[cell.text for cell in row.cells] for row in block.rows]
                table_idx += 1
                table_text = _table_to_text(rows, table_idx)
                if table_text:
                    table_items.append({
                        "text": table_text,
                        "page": 1,
                        "source": source_name,
                        "content_type": "table",
                    })

        # 段落之间以空行分隔，保证 _split_by_paragraphs 正确切分
        body_text = "\n\n".join(md_lines).strip()
        if body_text:
            pages.append({
                "text": body_text,
                "page": 1,
                "source": source_name,
                "content_type": "text",
            })
        pages.extend(table_items)

        logger.info(
            f"已提取 {source_name}: {1 if body_text else 0} 页正文，{len(table_items)} 个表格块"
        )
        return pages

    except Exception as e:
        logger.error(f"提取 Word 文档失败 {docx_path}: {e}")
        return []


def extract_document(path: Path) -> List[Dict]:
    """统一文档抽取入口：按扩展名分派到对应解析器，输出统一的页面/块结构。

    - .pdf  → extract_text_from_pdf（含表格抽取与低文本页 OCR 兜底标记）
    - .docx → extract_text_from_docx（Word → Markdown）
    其它扩展名将被跳过并告警。
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        return extract_text_from_docx(path)
    logger.warning(f"不支持的文件格式，跳过: {path.name}")
    return []
