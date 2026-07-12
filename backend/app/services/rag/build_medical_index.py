# -*- coding: utf-8 -*-
"""医学指南索引构建脚本 — 扫描 PDF 并构建 ChromaDB 向量索引

执行方式: cd backend; python -m app.services.rag.build_medical_index
"""

import asyncio
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

# 添加 backend 到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.services.rag.embeddings import get_embeddings
from app.services.rag.medical_store import get_medical_store
from app.services.rag.metadata_config import get_enriched_metadata, DocumentMetadata
from app.services.rag.entity_resolver import extract_entities

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# PDF 目录（项目根目录下的 data/）
# 从 build_medical_index.py 出发，向上 5 级到项目根目录，再进入 data/
PDF_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "data"

# 分块参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ── 标题层级识别配置（层级数字越小越高）──
HEADING_LEVELS: List[Tuple[int, re.Pattern]] = [
    # level 1: 章级
    (1, re.compile(
        r'^(第[一二三四五六七八九十百零\d]+[章部分篇]'
        r'|[一二三四五六七八九十]+、'
        r'|\d+\.\s*[\u4e00-\u9fff])',
        re.MULTILINE
    )),
    # level 2: 节级
    (2, re.compile(
        r'^(\d+\.\d+[\s\u4e00-\u9fff]'
        r'|[（\(][一二三四五六七八九十\d]+[）\)]'
        r'|【[^】]{2,20}】)',
        re.MULTILINE
    )),
    # level 3: 段落小标题
    (3, re.compile(
        r'^(\d+\.\d+\.\d+[\s\u4e00-\u9fff]'
        r'|[A-Z]\.[\s\u4e00-\u9fff])',
        re.MULTILINE
    )),
]

# 合并所有标题正则（用于 _get_heading_level）
HEADING_REGEX = re.compile(
    '|'.join(pat.pattern for _, pat in HEADING_LEVELS), re.MULTILINE
)

# 句末标点（中文和英文）
SENTENCE_END_PUNCT = r'[。！？；.!?,]'


def _get_heading_level(line: str) -> int:
    """返回标题行的层级（1=章级, 2=节级, 3=段落级, 0=非标题）"""
    stripped = line.strip()
    if not stripped:
        return 0
    for level, pattern in HEADING_LEVELS:
        if pattern.match(stripped):
            return level
    return 0


def _clean_source_name(source: str) -> str:
    """从文件名提取可读的来源标题（去掉路径和扩展名）"""
    name = Path(source).stem
    # 去掉常见前缀序号，如 "1.", "10."
    name = re.sub(r'^\d+[.\s]*', '', name).strip()
    return name or source


def _split_by_headings(text: str) -> List[Tuple[str, str, List[str]]]:
    """按章节标题分割文本，同时追踪标题层级路径

    Returns:
        列表，每项为 (当前标题, 内容, 祖先标题路径) 三元组
        - 当前标题：本节的直接标题（可为空）
        - 内容：本节文本
        - 祖先路径：从文档顶层到本节父级的标题列表（包含当前标题）
    """
    lines = text.split('\n')
    sections: List[Tuple[str, str, List[str]]] = []

    # 标题栈： [(level, heading_text), ...]
    heading_stack: List[Tuple[int, str]] = []
    current_heading = ''
    current_content_lines: List[str] = []

    def flush():
        if current_content_lines or current_heading:
            ancestor_path = [h for _, h in heading_stack]
            sections.append((
                current_heading,
                '\n'.join(current_content_lines),
                ancestor_path,
            ))

    for line in lines:
        level = _get_heading_level(line)
        if level > 0:
            flush()
            # 弹出所有层级 >= 当前层级的条目
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            current_heading = line.strip()
            heading_stack.append((level, current_heading))
            current_content_lines = []
        else:
            current_content_lines.append(line)

    flush()
    return sections


def _split_by_paragraphs(text: str) -> List[str]:
    """按段落分割文本（双换行或连续空行）"""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def _build_context_prefix(
    source_title: str,
    heading: str,
    ancestor_path: List[str],
) -> str:
    """构建 Contextual Retrieval 上下文前缀

    将文档标题 + 标题层级路径拼接为上下文摘要前缀，注入每个 chunk 开头。

    示例输出：
        「来源：非小细胞肺癌诊疗指南 > 第三章 治疗原则 > 3.1 外科治疗」
    """
    parts = [source_title] if source_title else []
    # 去掉 ancestor_path 里与 heading 重复的最后一项
    ancestors = [a for a in ancestor_path if a and a != heading]
    parts.extend(ancestors)
    if heading:
        parts.append(heading)
    if not parts:
        return ""
    return "「来源：" + " > ".join(parts) + "」"


def _build_heading_path(heading: str, ancestor_path: List[str]) -> str:
    """构建标题路径字符串（用于 metadata 存储）"""
    parts = [a for a in ancestor_path if a and a != heading]
    if heading:
        parts.append(heading)
    return " > ".join(parts) if parts else ""


def _split_by_sentences(text: str) -> List[str]:
    """按句子边界分割文本"""
    if len(text) <= CHUNK_SIZE:
        return [text]
    
    # 按句末标点分割
    parts = re.split(f'({SENTENCE_END_PUNCT})', text)
    
    # 合并标点回句子
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i]
        if i + 1 < len(parts):
            sentence += parts[i + 1]  # 加上标点
        if sentence.strip():
            sentences.append(sentence.strip())
    
    # 处理最后可能无标点的部分
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    
    return sentences if sentences else [text]


def _hard_split(text: str, chunk_size: int) -> List[str]:
    """硬切割兜底：对无标点的极长文本进行字符级分割
    
    Args:
        text: 文本内容
        chunk_size: 每块最大字符数
        
    Returns:
        文本块列表
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    
    return chunks


def _merge_units(units: List[str], chunk_size: int) -> List[str]:
    """将小单元合并为目标大小的块
    
    Args:
        units: 文本单元列表（段落或句子）
        chunk_size: 目标块大小
        
    Returns:
        合并后的文本块列表
    """
    if not units:
        return []
    
    chunks = []
    current_chunk = []
    current_len = 0
    
    for unit in units:
        unit_len = len(unit)
        
        # 如果当前块为空，直接加入
        if not current_chunk:
            current_chunk.append(unit)
            current_len = unit_len
        # 如果加入后不超过限制，继续加入
        elif current_len + unit_len + 1 <= chunk_size:  # +1 是换行符
            current_chunk.append(unit)
            current_len += unit_len + 1
        # 否则，保存当前块，开始新块
        else:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [unit]
            current_len = unit_len
    
    # 保存最后一个块
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks


def _apply_overlap(chunks: List[str], overlap: int) -> List[str]:
    """在相邻块之间应用重叠
    
    Args:
        chunks: 文本块列表
        overlap: 重叠字符数
        
    Returns:
        添加了重叠的新块列表
    """
    if len(chunks) <= 1 or overlap <= 0:
        return chunks
    
    result = [chunks[0]]
    
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        curr_chunk = chunks[i]
        
        # 从上一个块末尾取重叠内容
        if len(prev_chunk) > overlap:
            overlap_text = prev_chunk[-overlap:]
            # 尝试从完整句子/段落开始
            # 找到第一个换行符或句末标点后的位置
            newline_pos = overlap_text.find('\n')
            punct_match = re.search(SENTENCE_END_PUNCT, overlap_text)
            
            if newline_pos > 0:
                overlap_text = overlap_text[newline_pos + 1:]
            elif punct_match:
                overlap_text = overlap_text[punct_match.end():]
            
            overlap_text = overlap_text.strip()
            
            # 避免重叠文本以标题格式开头，导致与当前块的标题重复
            if overlap_text.startswith('【') and '】' in overlap_text:
                bracket_end = overlap_text.find('】')
                if bracket_end > 0:
                    overlap_text = overlap_text[bracket_end + 1:].strip()
            
            if overlap_text:
                # 检查当前块是否以标题开头，如果是，保留标题在开头
                if curr_chunk.startswith('【') and '\n' in curr_chunk:
                    # 提取标题行
                    first_newline = curr_chunk.find('\n')
                    title_line = curr_chunk[:first_newline]
                    rest_content = curr_chunk[first_newline + 1:]
                    new_chunk = title_line + '\n' + overlap_text + '\n' + rest_content
                else:
                    new_chunk = overlap_text + '\n' + curr_chunk
            else:
                new_chunk = curr_chunk
        else:
            new_chunk = curr_chunk

        result.append(new_chunk)
    
    
    return result


def _process_section(
    heading: str,
    content: str,
    chunk_size: int,
    context_prefix: str = "",
) -> List[str]:
    """处理单个章节，返回注入了上下文前缀的分块列表"""
    if not content.strip():
        return []

    paragraphs = _split_by_paragraphs(content)
    units = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            sentences = _split_by_sentences(para)
            for sent in sentences:
                if len(sent) <= chunk_size:
                    units.append(sent)
                else:
                    units.extend(_hard_split(sent, chunk_size))

    chunks = _merge_units(units, chunk_size)

    if context_prefix:
        chunks = [
            f"{context_prefix}\n{chunk}" if not chunk.startswith("「来源：")
            else chunk
            for chunk in chunks
        ]
    elif heading:
        prefix = heading if heading.startswith("【") else f"【{heading}】"
        chunks = [
            chunk if chunk.startswith("【") else f"{prefix}\n{chunk}"
            for chunk in chunks
        ]

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    source_title: str = "",
) -> List[Dict]:
    """将文本按语义感知策略分块，并注入 Contextual Retrieval 上下文前缀

    分块优先级：章节标题 > 段落 > 句子 > 硬切割
    每个 chunk 开头注入：「来源：{source_title} > {heading_path}」

    Args:
        text: 原始文本
        chunk_size: 目标块大小（字符数）
        overlap: 相邻块重叠字符数
        source_title: 文档来源标题

    Returns:
        字典列表，每项包含：
        - "text": 含上下文前缀的块内容
        - "heading_path": 标题层级路径（用于 metadata）
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        prefix = _build_context_prefix(source_title, "", [])
        return [{"text": f"{prefix}\n{text}" if prefix else text, "heading_path": ""}]

    sections = _split_by_headings(text)
    all_chunks_with_meta: List[Dict] = []
    for heading, content, ancestor_path in sections:
        ctx_prefix = _build_context_prefix(source_title, heading, ancestor_path)
        heading_path = _build_heading_path(heading, ancestor_path)
        for chunk in _process_section(heading, content, chunk_size, context_prefix=ctx_prefix):
            all_chunks_with_meta.append({"text": chunk, "heading_path": heading_path})

    if not all_chunks_with_meta:
        ctx_prefix = _build_context_prefix(source_title, "", [])
        for chunk in _process_section("", text, chunk_size, context_prefix=ctx_prefix):
            all_chunks_with_meta.append({"text": chunk, "heading_path": ""})

    texts = [item["text"] for item in all_chunks_with_meta]
    if overlap > 0 and len(texts) > 1:
        texts = _apply_overlap(texts, overlap)
        for i, item in enumerate(all_chunks_with_meta):
            item["text"] = texts[i]

    all_chunks_with_meta = [item for item in all_chunks_with_meta if item["text"].strip()]
    logger.debug(
        f"语义分块完成 [{source_title}]: {len(text)} 字 -> {len(all_chunks_with_meta)} 块"
    )
    return all_chunks_with_meta


# ── 推荐等级/证据等级提取 ──────────────────────────────────────────────────────────

# 推荐等级正则
_RECOMMENDATION_LEVEL_RE = re.compile(
    r'((?:I+|IV|V?I{1,3}|[一二三四五])级推荐|[ABC]级推荐|强推荐|弱推荐|条件性推荐)',
    re.IGNORECASE
)

# 证据等级正则
_EVIDENCE_LEVEL_RE = re.compile(
    r'(?:证据等级|证据级别|证据水平)[：:\s]*([1-5][ABC]?|[ABC])',
    re.IGNORECASE
)


def _extract_recommendation_level(text: str) -> str:
    """从 chunk 文本中提取推荐等级"""
    match = _RECOMMENDATION_LEVEL_RE.search(text)
    return match.group(1) if match else ""


def _extract_evidence_level(text: str) -> str:
    """从 chunk 文本中提取证据等级"""
    match = _EVIDENCE_LEVEL_RE.search(text)
    return match.group(1) if match else ""


def _build_embedding_text(chunk_text: str, doc_meta: DocumentMetadata, heading_path: str = "",
                          recommendation_level: str = "", evidence_level: str = "") -> str:
    """构建增强 embedding 文本，将元数据注入文本前缀

    格式：
    来源机构：CSCO
    指南：非小细胞肺癌诊疗指南
    版本：2025版
    章节：第三章 > 3.1 外科治疗
    推荐等级：I级推荐
    证据等级：1A
    正文：……
    """
    parts = []
    if doc_meta.organization:
        parts.append(f"来源机构：{doc_meta.organization}")
    if doc_meta.title:
        parts.append(f"指南：{doc_meta.title}")
    if doc_meta.version:
        parts.append(f"版本：{doc_meta.version}")
    if heading_path:
        parts.append(f"章节：{heading_path}")
    if recommendation_level:
        parts.append(f"推荐等级：{recommendation_level}")
    if evidence_level:
        parts.append(f"证据等级：{evidence_level}")
    parts.append(f"正文：{chunk_text}")
    return "\n".join(parts)


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

            # 3. 添加正文页
            if text:
                pages.append({
                    "text": text,
                    "page": page_real_num,
                    "source": source_name,
                    "content_type": "text",
                })

            # 4. 添加表格块
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


def generate_doc_id(source: str, page: int, chunk_idx: int, content: str) -> str:
    """生成文档唯一标识

    Args:
        source: 来源文件名
        page: 页码
        chunk_idx: 块索引
        content: 内容（用于生成哈希）

    Returns:
        唯一 ID 字符串
    """
    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
    return f"{source}_p{page}_c{chunk_idx}_{content_hash}"


def generate_stable_chunk_id(source: str, page: int, heading_path: str, chunk_seq: int, text: str) -> str:
    """生成稳定的 chunk ID

    格式: {file_hash_8}:{page}:{heading_hash_4}:{seq}
    确保相同内容的 chunk 在不同构建中产生相同的 ID。
    """
    file_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    heading_hash = hashlib.md5(heading_path.encode()).hexdigest()[:4] if heading_path else "0000"
    return f"{file_hash}:p{page}:h{heading_hash}:c{chunk_seq}"


def _extract_chunks_from_page(page_info: Dict) -> List[Dict]:
    """将单页 page_info 转换为 chunk 列表，区分正文和表格两种类型。

    - 正文（content_type='text'）：走 chunk_text 语义分块 + 重叠策略
    - 表格（content_type='table'）：不切分，整体作为一块，注入表格来源前缀
    """
    source_title = _clean_source_name(page_info["source"])
    page_num = page_info["page"]
    content_type = page_info.get("content_type", "text")

    if content_type == "table":
        # 表格不切分，注入㌀来源：xxx > 第N页 表格、前缀
        prefix = f"《来源：{source_title} > 第{page_num}页 表格》"
        text = page_info["text"]
        if not text.startswith("《来源："):
            text = f"{prefix}\n{text}"
        return [{"text": text, "heading_path": f"第{page_num}页 表格"}]
    else:
        return chunk_text(
            page_info["text"], CHUNK_SIZE, CHUNK_OVERLAP, source_title=source_title
        )


async def build_medical_index(target_version: str = "rag-v2"):
    """主构建流程

    Args:
        target_version: 目标索引版本（如 "rag-v2"），临时覆盖 ACTIVE_INDEX_VERSION
    """
    from app.core.config import settings
    from app.services.rag.medical_store import _reset_collection_cache, set_build_mode

    # 保存原始版本，临时覆盖
    original_version = settings.ACTIVE_INDEX_VERSION
    settings.ACTIVE_INDEX_VERSION = target_version
    set_build_mode(True)
    _reset_collection_cache()  # 清除缓存，确保使用新版本 collection

    try:
        logger.info(f"开始构建医学知识库索引，PDF 目录: {PDF_DIR}，目标版本: {target_version}")

        # 1. 验证 PDF 目录
        if not PDF_DIR.exists():
            logger.error(f"PDF 目录不存在: {PDF_DIR}")
            return

        # 2. 扫描 PDF 文件
        pdf_files = list(PDF_DIR.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"未找到 PDF 文件: {PDF_DIR}")
            return

        logger.info(f"发现 {len(pdf_files)} 个 PDF 文件")

        # 3. 提取所有文本块
        all_chunks = []  # [{"id": ..., "text": ..., "source": ..., "page": ...}]

        for pdf_path in pdf_files:
            pages = extract_text_from_pdf(pdf_path)
            for page_info in pages:
                # 使用共享分块入口，自动区分正文和表格
                chunk_items = _extract_chunks_from_page(page_info)
                for idx, item in enumerate(chunk_items):
                    chunk_content = item["text"]
                    heading_path = item.get("heading_path", "")
                    doc_id = generate_doc_id(
                        page_info["source"], page_info["page"], idx, chunk_content
                    )
                    all_chunks.append(
                        {
                            "id": doc_id,
                            "text": chunk_content,
                            "source": page_info["source"],
                            "page": page_info["page"],
                            "heading_path": heading_path,
                            "content_type": page_info.get("content_type", "text"),
                        }
                    )

        if not all_chunks:
            logger.warning("未提取到任何文本块")
            return

        logger.info(f"共提取 {len(all_chunks)} 个文本块")

        # 4. 批量生成 embedding（使用增强 embedding 文本）
        logger.info("开始生成文本向量...")

        try:
            embedding_texts = []
            for chunk in all_chunks:
                source_filename = chunk["source"]
                doc_meta = get_enriched_metadata(source_filename)
                rec_level = _extract_recommendation_level(chunk["text"])
                ev_level = _extract_evidence_level(chunk["text"])
                enhanced_text = _build_embedding_text(
                    chunk["text"], doc_meta,
                    heading_path=chunk.get("heading_path", ""),
                    recommendation_level=rec_level,
                    evidence_level=ev_level,
                )
                embedding_texts.append(enhanced_text)

            embeddings = await get_embeddings(embedding_texts)
            logger.info(f"向量生成完成: {len(embeddings)} 条")
        except Exception as e:
            logger.error(f"向量生成失败: {e}")
            return

        # 5. 准备 ChromaDB 数据（增强 metadata）
        ids = [chunk["id"] for chunk in all_chunks]
        documents = [chunk["text"] for chunk in all_chunks]
        metadatas = []
        for chunk in all_chunks:
            source_filename = chunk["source"]
            doc_meta = get_enriched_metadata(source_filename)
            rec_level = _extract_recommendation_level(chunk["text"])
            ev_level = _extract_evidence_level(chunk["text"])
            # 实体归一化：提取 chunk 中的医学实体注入 metadata
            entities = extract_entities(chunk["text"])
            entity_names = " ".join(e["normalized"] for e in entities)
            enhanced_meta = {
                "source": chunk["source"],
                "page": chunk["page"],
                "heading_path": chunk.get("heading_path", ""),
                "content_type": chunk.get("content_type", "text"),
                # 增强字段
                "organization": doc_meta.organization or "",
                "year": doc_meta.year or 0,
                "version": doc_meta.version or "",
                "document_type": doc_meta.document_type,
                "title": doc_meta.title,
                "departments": json.dumps(doc_meta.departments, ensure_ascii=False),
                "disease_tags": json.dumps(doc_meta.disease_tags, ensure_ascii=False),
                "population": json.dumps(doc_meta.population, ensure_ascii=False),
                "recommendation_level": rec_level,
                "evidence_level": ev_level,
                "index_version": target_version,
                # 实体标签（用于检索加权）
                "entities": json.dumps(entities, ensure_ascii=False) if entities else "",
                "entity_names": entity_names,
            }
            metadatas.append(enhanced_meta)

        # 6. 存入 ChromaDB
        store = get_medical_store()
        try:
            store.add_documents(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(f"医学知识库构建完成，共 {store.count()} 条文档")
        except Exception as e:
            logger.error(f"保存到 ChromaDB 失败: {e}")
            return
    finally:
        # 恢复原始 ACTIVE_INDEX_VERSION（所有退出路径都会执行）
        settings.ACTIVE_INDEX_VERSION = original_version
        set_build_mode(False)
        _reset_collection_cache()  # 清除缓存，恢复原始版本 collection


# ── 增量更新接口 ────────────────────────────────────────────────────────────────────
async def index_single_pdf(
    pdf_path: Path,
    force_replace: bool = False,
    target_version: str = None,
) -> dict:
    """对单个 PDF 进行增量索引。

    Args:
        pdf_path: PDF 文件路径
        force_replace: True 则先删除该来源的已有索引再重建；
                       False 且已有索引时直接跳过。

    Returns:
        {"source": 文件名, "status": "added"/"skipped"/"replaced", "chunks": 块数}
    """
    from app.core.config import settings
    from app.services.rag.medical_store import _reset_collection_cache, set_build_mode

    # 如果指定了 target_version，临时覆盖
    original_version = None
    if target_version is not None:
        original_version = settings.ACTIVE_INDEX_VERSION
        settings.ACTIVE_INDEX_VERSION = target_version
        set_build_mode(True)
        _reset_collection_cache()
    else:
        target_version = getattr(settings, 'ACTIVE_INDEX_VERSION', 'rag-v1')

    try:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        source_name = pdf_path.name
        store = get_medical_store()

        # 检查是否已有索引
        existing_count = store.get_source_doc_count(source_name)
        if existing_count > 0:
            if not force_replace:
                logger.info(f"跳过已索引文件 '{source_name}'（{existing_count} 条块）")
                return {"source": source_name, "status": "skipped", "chunks": existing_count}
            else:
                deleted = store.delete_by_source(source_name)
                logger.info(f"已删除 '{source_name}' 旧索引 {deleted} 条")

        # 提取并分块
        pages = extract_text_from_pdf(pdf_path)
        chunks = []
        for page_info in pages:
            chunk_items = _extract_chunks_from_page(page_info)
            for idx, item in enumerate(chunk_items):
                doc_id = generate_doc_id(
                    page_info["source"], page_info["page"], idx, item["text"]
                )
                chunks.append(
                    {
                        "id": doc_id,
                        "text": item["text"],
                        "source": page_info["source"],
                        "page": page_info["page"],
                        "heading_path": item.get("heading_path", ""),
                        "content_type": page_info.get("content_type", "text"),
                    }
                )

        if not chunks:
            logger.warning(f"'{source_name}' 未提取到任何文本块")
            return {"source": source_name, "status": "added", "chunks": 0}

        # 生成增强 embedding 文本
        doc_meta = get_enriched_metadata(source_name)
        embedding_texts = []
        for c in chunks:
            rec_level = _extract_recommendation_level(c["text"])
            ev_level = _extract_evidence_level(c["text"])
            enhanced_text = _build_embedding_text(
                c["text"], doc_meta,
                heading_path=c.get("heading_path", ""),
                recommendation_level=rec_level,
                evidence_level=ev_level,
            )
            embedding_texts.append(enhanced_text)
        embeddings = await get_embeddings(embedding_texts)

        # 构建增强 metadata 并写入 ChromaDB
        enhanced_metadatas = []
        for c in chunks:
            rec_level = _extract_recommendation_level(c["text"])
            ev_level = _extract_evidence_level(c["text"])
            # 实体归一化：提取 chunk 中的医学实体注入 metadata
            entities = extract_entities(c["text"])
            entity_names = " ".join(e["normalized"] for e in entities)
            enhanced_metadatas.append({
                "source": c["source"],
                "page": c["page"],
                "heading_path": c.get("heading_path", ""),
                "content_type": c.get("content_type", "text"),
                "organization": doc_meta.organization or "",
                "year": doc_meta.year or 0,
                "version": doc_meta.version or "",
                "document_type": doc_meta.document_type,
                "title": doc_meta.title,
                "departments": json.dumps(doc_meta.departments, ensure_ascii=False),
                "disease_tags": json.dumps(doc_meta.disease_tags, ensure_ascii=False),
                "population": json.dumps(doc_meta.population, ensure_ascii=False),
                "recommendation_level": rec_level,
                "evidence_level": ev_level,
                "index_version": target_version,
                # 实体标签（用于检索加权）
                "entities": json.dumps(entities, ensure_ascii=False) if entities else "",
                "entity_names": entity_names,
            })

        store.add_documents(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=enhanced_metadatas,
        )

        status = "replaced" if force_replace and existing_count > 0 else "added"
        logger.info(f"'{source_name}' 增量索引完成，共 {len(chunks)} 块，status={status}")
        return {"source": source_name, "status": status, "chunks": len(chunks)}
    finally:
        # 恢复原始版本（所有退出路径都会执行）
        if original_version is not None:
            settings.ACTIVE_INDEX_VERSION = original_version
            set_build_mode(False)
            _reset_collection_cache()


async def get_indexed_sources() -> List[dict]:
    """获取已建索的来源列表，含每个来源的文档块数量"""
    store = get_medical_store()
    sources = store.get_all_sources()
    result = []
    for src in sources:
        result.append({"source": src, "chunks": store.get_source_doc_count(src)})
    return result


async def switch_index_version(new_version: str, *, auto_rollback: bool = True) -> dict:
    """切换活跃索引版本，带健康检查和自动回滚

    Args:
        new_version: 新版本标识（如 "rag-v2"）
        auto_rollback: 切换后健康检查失败时是否自动回滚

    Returns:
        {"previous": str, "current": str, "doc_count": int}
        或 {"error": str} 如果切换失败
    """
    import time
    from app.core.config import settings
    from app.services.rag.medical_store import _reset_collection_cache

    previous = getattr(settings, 'ACTIVE_INDEX_VERSION', 'rag-v1')

    # 验证新版本 collection 存在
    store = get_medical_store()
    if store.client is None:
        store._init_client()

    new_collection_name = f"medical_guidelines_{new_version}"
    try:
        col = store.client.get_collection(new_collection_name)
        doc_count = col.count()
    except Exception:
        return {"error": f"Collection '{new_collection_name}' 不存在"}

    # 保存原始状态用于回滚
    original_version = previous
    original_collection = store.collection

    # 执行切换
    try:
        # 更新配置（运行时更新）
        settings.ACTIVE_INDEX_VERSION = new_version
        _reset_collection_cache()  # 清除缓存，确保新版本生效

        # 更新 medical_store 的 collection 引用
        store.collection = col

        # 重建 BM25 索引
        try:
            from app.services.rag.bm25_search import rebuild_bm25_index
            await asyncio.to_thread(rebuild_bm25_index)
            logger.info(f"BM25 索引已重建（版本: {new_version}）")
        except Exception as e:
            logger.warning(f"BM25 索引重建失败（非致命）: {e}")

        # 健康检查
        if auto_rollback:
            healthy = await _health_check_index(timeout=10.0)
            if not healthy:
                logger.warning(
                    f"Health check failed after switching to {new_version}, "
                    f"rolling back to {original_version}"
                )
                # 回滚
                settings.ACTIVE_INDEX_VERSION = original_version
                store.collection = original_collection
                _reset_collection_cache()
                # 重建 BM25 回滚版本
                try:
                    from app.services.rag.bm25_search import rebuild_bm25_index
                    await asyncio.to_thread(rebuild_bm25_index)
                except Exception:
                    pass
                return {"error": f"健康检查失败，已回滚到 {original_version}"}

        # 持久化到 .env 文件（健康检查通过后）
        env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
        if env_path.exists():
            content = env_path.read_text(encoding="utf-8")
            if "ACTIVE_INDEX_VERSION" in content:
                content = re.sub(
                    r'ACTIVE_INDEX_VERSION\s*=\s*.*',
                    f'ACTIVE_INDEX_VERSION={new_version}',
                    content,
                )
            else:
                content += f"\nACTIVE_INDEX_VERSION={new_version}\n"
            env_path.write_text(content, encoding="utf-8")
            logger.info(f"已将 ACTIVE_INDEX_VERSION={new_version} 持久化到 .env")
        else:
            # .env 不存在时创建新文件
            env_path.write_text(f"ACTIVE_INDEX_VERSION={new_version}\n", encoding="utf-8")
            logger.info(f"已创建 .env 文件并写入 ACTIVE_INDEX_VERSION={new_version}")

    except Exception as e:
        logger.error(f"Switch failed: {e}")
        # 异常时尝试回滚
        if auto_rollback:
            settings.ACTIVE_INDEX_VERSION = original_version
            store.collection = original_collection
            _reset_collection_cache()
        return {"error": f"切换失败: {e}"}

    logger.info(f"索引版本切换: {previous} → {new_version} (文档数: {doc_count})")

    return {
        "previous": previous,
        "current": new_version,
        "doc_count": doc_count,
    }


async def _health_check_index(timeout: float = 10.0) -> bool:
    """索引健康检查：执行标准查询验证索引可用性和响应时间
    
    使用简单的向量查询验证索引可用性，不调用 LLM。
    
    Args:
        timeout: 单次查询超时阈值（秒）
    
    Returns:
        True 健康，False 不健康
    """
    import time
    from app.services.rag.medical_store import get_medical_store
    
    try:
        store = get_medical_store()
        if store.collection is None:
            logger.warning("Health check: collection is None")
            return False
        
        # 使用标准测试查询验证索引
        test_queries = ["高血压 诊疗指南", "糖尿病 治疗方案"]
        
        # 获取 embedding 函数
        from app.services.rag.embeddings import get_embeddings
        embedding_fn = get_embeddings()
        
        for query in test_queries:
            start = time.time()
            
            # 生成查询向量
            query_embedding = await asyncio.to_thread(
                embedding_fn.embed_query, query
            )
            
            # 执行向量查询
            results = store.collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
            )
            
            elapsed = time.time() - start
            
            if elapsed > timeout:
                logger.warning(f"Health check query took {elapsed:.2f}s (timeout={timeout}s)")
                return False
            
            # 检查结果
            if not results or not results.get('ids') or not results['ids'][0]:
                logger.warning(f"Health check query returned no results: {query}")
                return False
        
        logger.info("Health check passed")
        return True
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(build_medical_index())
