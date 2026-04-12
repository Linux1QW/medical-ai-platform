# -*- coding: utf-8 -*-
"""医学指南索引构建脚本 — 扫描 PDF 并构建 ChromaDB 向量索引

执行方式: cd backend; python -m app.services.rag.build_medical_index
"""

import asyncio
import hashlib
import logging
import sys
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF

# 添加 backend 到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.services.rag.embeddings import get_embeddings
from app.services.rag.medical_store import get_medical_store

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

# 标题识别正则模式（中文医学文档常见格式）
HEADING_PATTERNS = [
    r'^第[一二三四五六七八九十百零\d]+[章节部分]',     # 第一章、第2节、第3部分
    r'^[一二三四五六七八九十]+[、.]',              # 一、二.
    r'^[（\(][一二三四五六七八九十\d]+[）\)]',      # （一）、(1)
    r'^\d+[.、]\d*\s*',                           # 1. 1.1 2、
    r'^【.+?】',                                   # 【诊断】【治疗】
    r'^[A-Z]\.[\s\u4e00-\u9fff]',              # A. B. 后跟中文
]

# 编译正则表达式
import re
HEADING_REGEX = re.compile('|'.join(HEADING_PATTERNS), re.MULTILINE)

# 句末标点（中文和英文）
SENTENCE_END_PUNCT = r'[。！？；.!?,]'


def _is_heading(line: str) -> bool:
    """判断一行是否为标题行
    
    Args:
        line: 待检测的文本行
        
    Returns:
        是否为标题行
    """
    stripped = line.strip()
    if not stripped:
        return False
    return bool(HEADING_REGEX.match(stripped))


def _split_by_headings(text: str) -> List[tuple]:
    """按章节标题分割文本
    
    Args:
        text: 原始文本
        
    Returns:
        列表，每项为 (标题, 内容) 元组，标题可为空字符串表示文首无标题部分
    """
    lines = text.split('\n')
    sections = []
    current_heading = ''
    current_content_lines = []
    
    for line in lines:
        if _is_heading(line):
            # 保存之前的章节
            if current_content_lines or current_heading:
                sections.append((current_heading, '\n'.join(current_content_lines)))
            current_heading = line.strip()
            current_content_lines = []
        else:
            current_content_lines.append(line)
    
    # 保存最后一个章节
    if current_content_lines or current_heading:
        sections.append((current_heading, '\n'.join(current_content_lines)))
    
    return sections


def _split_by_paragraphs(text: str) -> List[str]:
    """按段落分割文本（双换行或连续空行）
    
    Args:
        text: 文本内容
        
    Returns:
        段落列表
    """
    # 使用正则分割：连续空行（>=2个换行符或包含空行的换行）
    paragraphs = re.split(r'\n\s*\n', text)
    # 过滤空段落并清理空白
    return [p.strip() for p in paragraphs if p.strip()]


def _split_by_sentences(text: str) -> List[str]:
    """按句子边界分割文本
    
    Args:
        text: 文本内容
        
    Returns:
        句子列表
    """
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


def _process_section(heading: str, content: str, chunk_size: int) -> List[str]:
    """处理单个章节，返回分块后的内容
    
    Args:
        heading: 章节标题
        content: 章节内容
        chunk_size: 目标块大小
        
    Returns:
        分块后的文本列表
    """
    if not content.strip():
        return []

    # 按段落分割
    paragraphs = _split_by_paragraphs(content)

    # 对每个段落，如果过长则进一步按句子分割
    units = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            # 按句子分割
            sentences = _split_by_sentences(para)
            for sent in sentences:
                if len(sent) <= chunk_size:
                    units.append(sent)
                else:
                    # 硬切割兜底
                    units.extend(_hard_split(sent, chunk_size))


    # 合并单元
    chunks = _merge_units(units, chunk_size)

    # 如果有标题，添加到每个块的前面作为上下文
    if heading:
        # 检查是否已经有任何标题前缀，避免重复添加
        # 如果 heading 本身已经包含【】格式，直接使用它作为前缀
        if heading.startswith("【") and heading.endswith("】"):
            prefix = heading
        else:
            prefix = f"【{heading}】"
        chunks = [
            chunk if chunk.startswith("【") else f"{prefix}\n{chunk}"
            for chunk in chunks
        ]

    return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """将文本按语义感知策略分块
    
    分块优先级策略（从粗到细）：
    1. 按章节标题分割：识别标题行，将文本拆分为若干"节"
    2. 按段落分割：对每个"节"，按双换行符分割为段落
    3. 按句子分割：对超长段落，按中文句末标点分割
    4. 硬切割兜底：对极长无标点的文本段，退化为字符级分割
    
    Args:
        text: 原始文本
        chunk_size: 目标块大小（字符数）
        overlap: 相邻块重叠字符数
        
    Returns:
        文本块列表
    """
    if not text or not text.strip():
        return []
    
    # 清理文本：移除多余空白，但保留段落分隔
    text = text.strip()
    
    # 如果文本很短，直接返回
    if len(text) <= chunk_size:
        return [text]
    
    # 按章节标题分割
    sections = _split_by_headings(text)
    
    all_chunks = []
    for heading, content in sections:
        section_chunks = _process_section(heading, content, chunk_size)
        all_chunks.extend(section_chunks)
    
    # 如果没有识别到章节，整个文本作为一个章节处理
    if not all_chunks:
        all_chunks = _process_section('', text, chunk_size)
    
    # 应用重叠
    if overlap > 0 and len(all_chunks) > 1:
        all_chunks = _apply_overlap(all_chunks, overlap)
    
    # 过滤空块
    all_chunks = [c.strip() for c in all_chunks if c.strip()]
    
    logger.debug(f"语义分块完成: 原文本 {len(text)} 字符 -> {len(all_chunks)} 个块")
    
    return all_chunks


def extract_text_from_pdf(pdf_path: Path) -> List[Dict]:
    """提取 PDF 文本，返回 [{"text": ..., "page": ..., "source": ...}, ...]

    Args:
        pdf_path: PDF 文件路径

    Returns:
        包含文本、页码和来源的列表
    """
    pages = []
    try:
        doc = fitz.open(pdf_path)
        source_name = pdf_path.name

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if text:
                pages.append(
                    {
                        "text": text,
                        "page": page_num + 1,  # 页码从 1 开始
                        "source": source_name,
                    }
                )

        doc.close()
        logger.info(f"已提取 {source_name}: {len(pages)} 页")
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


async def build_medical_index():
    """主构建流程"""
    logger.info(f"开始构建医学知识库索引，PDF 目录: {PDF_DIR}")

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
            page_text = page_info["text"]
            source = page_info["source"]
            page_num = page_info["page"]

            # 分块
            chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
            for idx, chunk_text_content in enumerate(chunks):
                doc_id = generate_doc_id(
                    source, page_num, idx, chunk_text_content
                )
                all_chunks.append(
                    {
                        "id": doc_id,
                        "text": chunk_text_content,
                        "source": source,
                        "page": page_num,
                    }
                )

    if not all_chunks:
        logger.warning("未提取到任何文本块")
        return

    logger.info(f"共提取 {len(all_chunks)} 个文本块")

    # 4. 批量生成 embedding
    texts = [chunk["text"] for chunk in all_chunks]
    logger.info("开始生成文本向量...")

    try:
        embeddings = await get_embeddings(texts)
        logger.info(f"向量生成完成: {len(embeddings)} 条")
    except Exception as e:
        logger.error(f"向量生成失败: {e}")
        return

    # 5. 准备 ChromaDB 数据
    ids = [chunk["id"] for chunk in all_chunks]
    documents = [chunk["text"] for chunk in all_chunks]
    metadatas = [
        {"source": chunk["source"], "page": chunk["page"]} for chunk in all_chunks
    ]

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


if __name__ == "__main__":
    asyncio.run(build_medical_index())
