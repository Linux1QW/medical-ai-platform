# -*- coding: utf-8 -*-
"""语义分块 — 标题层级识别、段落/句子切分、重叠策略与 chunk ID 生成"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

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

# Markdown ATX 标题正则（统一转 Markdown 后的主要标题形式：#=1级 ##=2级 ### 及更深=3级）
_ATX_HEADING_RE = re.compile(r'^(#{1,6})\s+\S')

# 句末标点（中文和英文）
SENTENCE_END_PUNCT = r'[。！？；.!?,]'


def _get_heading_level(line: str) -> int:
    """返回标题行的层级（1=章级, 2=节级, 3=段落级, 0=非标题）"""
    stripped = line.strip()
    if not stripped:
        return 0
    # Markdown ATX 标题优先（统一格式输入下 Word 标题会被转为 #/##/###）
    atx = _ATX_HEADING_RE.match(stripped)
    if atx:
        return min(len(atx.group(1)), 3)
    for level, pattern in HEADING_LEVELS:
        if pattern.match(stripped):
            return level
    return 0


def _clean_heading_text(line: str) -> str:
    """清洗标题行：去除 Markdown ATX 前缀（#）及首尾空白，返回纯标题文本"""
    return re.sub(r'^\s*#{1,6}\s+', '', line).strip()


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
            current_heading = _clean_heading_text(line)
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
    current_chunk: list[str] = []
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
