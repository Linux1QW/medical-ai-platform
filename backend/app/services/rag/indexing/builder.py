# -*- coding: utf-8 -*-
"""索引构建 — 全量构建与单文件增量索引"""

import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.rag.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    get_embeddings,
)
from app.services.rag.entity_resolver import extract_entities
from app.services.rag.indexing.chunking import (
    _extract_chunks_from_page,
    generate_doc_id,
)
from app.services.rag.indexing.enrichment import (
    _build_embedding_text,
    _extract_evidence_level,
    _extract_recommendation_level,
)
from app.services.rag.indexing.extractors import (
    SUPPORTED_EXTENSIONS,
    extract_document,
)
from app.services.rag.indexing.manifest import (
    RAGIndexManifest,
    build_index_generation,
    compute_corpus_sha256,
    load_rag_index_manifest,
    write_rag_index_manifest,
)
from app.services.rag.lexical.artifacts import build_bm25_artifact
from app.services.rag.lexical.tokenizer import TOKENIZER_VERSION
from app.services.rag.medical_store import get_medical_store
from app.services.rag.metadata_config import get_enriched_metadata
from app.services.rag.ocr import apply_ocr_to_pages

logger = logging.getLogger(__name__)

# PDF 目录（项目根目录下的 data/）
# 从 indexing/builder.py 出发，向上 6 级到项目根目录，再进入 data/
PDF_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "data"

PARSER_VERSION = "document-extractors-v1"
CHUNKER_VERSION = "medical-chunker-v1"


def _record_source(document: dict[str, Any]) -> str:
    metadata = document.get("metadata") or {}
    return str(metadata.get("source") or document.get("source") or "")


def build_candidate_snapshot(
    active_documents: List[dict[str, Any]],
    incoming_documents: List[dict[str, Any]],
    *,
    source_name: str,
    force_replace: bool,
) -> List[dict[str, Any]]:
    """Copy active corpus and apply one source change without mutating it."""
    source_exists = any(
        _record_source(document) == source_name for document in active_documents
    )
    if source_exists and not force_replace:
        raise FileExistsError(
            f"source {source_name!r} already exists in the active generation"
        )
    snapshot = [
        deepcopy(document)
        for document in active_documents
        if not (force_replace and _record_source(document) == source_name)
    ]
    snapshot.extend(deepcopy(incoming_documents))
    return snapshot


async def _extract_candidate_records(
    document_paths: List[Path],
) -> List[dict[str, Any]]:
    chunks: List[dict[str, Any]] = []
    source_seq_counter: Dict[str, int] = {}
    for document_path in document_paths:
        pages = extract_document(document_path)
        await apply_ocr_to_pages(pages)
        for page_info in pages:
            for position, item in enumerate(_extract_chunks_from_page(page_info)):
                text = item["text"]
                source = page_info["source"]
                chunk_seq = source_seq_counter.get(source, 0)
                source_seq_counter[source] = chunk_seq + 1
                chunks.append(
                    {
                        "id": generate_doc_id(
                            source, page_info["page"], position, text
                        ),
                        "text": text,
                        "source": source,
                        "page": page_info["page"],
                        "heading_path": item.get("heading_path", ""),
                        "content_type": page_info.get("content_type", "text"),
                        "chunk_seq": chunk_seq,
                    }
                )
    if not chunks:
        raise ValueError("candidate corpus contains no chunks")

    embedding_texts = []
    records = []
    for chunk in chunks:
        doc_meta = get_enriched_metadata(chunk["source"])
        recommendation_level = _extract_recommendation_level(chunk["text"])
        evidence_level = _extract_evidence_level(chunk["text"])
        entities = extract_entities(chunk["text"])
        metadata = {
            "source": chunk["source"],
            "page": chunk["page"],
            "heading_path": chunk.get("heading_path", ""),
            "content_type": chunk.get("content_type", "text"),
            "chunk_seq": chunk.get("chunk_seq", -1),
            "organization": doc_meta.organization or "",
            "year": doc_meta.year or 0,
            "version": doc_meta.version or "",
            "document_type": doc_meta.document_type,
            "title": doc_meta.title,
            "departments": json.dumps(doc_meta.departments, ensure_ascii=False),
            "disease_tags": json.dumps(doc_meta.disease_tags, ensure_ascii=False),
            "population": json.dumps(doc_meta.population, ensure_ascii=False),
            "recommendation_level": recommendation_level,
            "evidence_level": evidence_level,
            "entities": json.dumps(entities, ensure_ascii=False) if entities else "",
            "entity_names": " ".join(entity["normalized"] for entity in entities),
        }
        records.append(
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": metadata,
                "embedding": None,
            }
        )
        embedding_texts.append(
            _build_embedding_text(
                chunk["text"],
                doc_meta,
                heading_path=chunk.get("heading_path", ""),
                recommendation_level=recommendation_level,
                evidence_level=evidence_level,
            )
        )
    embeddings = await get_embeddings(embedding_texts)
    if len(embeddings) != len(records):
        raise ValueError("embedding count does not match candidate chunk count")
    for record, embedding in zip(records, embeddings, strict=True):
        record["embedding"] = embedding
    return records


def _bm25_documents(records: List[dict[str, Any]]) -> List[dict[str, Any]]:
    return [
        {
            "id": record["id"],
            "text": record["text"],
            **dict(record.get("metadata") or {}),
        }
        for record in records
    ]


def _publish_chroma_candidate(
    generation: str,
    records: List[dict[str, Any]],
) -> str:
    store = get_medical_store()
    client = store._ensure_client()
    collection_name = f"medical_guidelines_{generation}"
    existing_names = {collection.name for collection in client.list_collections()}
    if collection_name in existing_names:
        raise FileExistsError(
            f"Chroma candidate already exists for {generation!r}"
        )
    collection = store.get_collection_for_generation(generation, create=True)
    for offset in range(0, len(records), 1000):
        batch = records[offset : offset + 1000]
        collection.add(
            ids=[record["id"] for record in batch],
            documents=[record["text"] for record in batch],
            embeddings=[record["embedding"] for record in batch],
            metadatas=[
                {**dict(record.get("metadata") or {}), "index_generation": generation}
                for record in batch
            ],
        )
    if collection.count() != len(records):
        raise RuntimeError("Chroma candidate count validation failed")
    return collection_name


def _publish_candidate_generation(
    records: List[dict[str, Any]],
    *,
    artifact_root: Optional[Path] = None,
    created_at: Optional[datetime] = None,
) -> RAGIndexManifest:
    bm25_documents = _bm25_documents(records)
    corpus_sha256 = compute_corpus_sha256(bm25_documents)
    timestamp = created_at or datetime.now(timezone.utc)
    generation = build_index_generation(corpus_sha256, timestamp)

    build_bm25_artifact(generation, bm25_documents, artifact_root)
    collection_name = _publish_chroma_candidate(generation, records)
    manifest = RAGIndexManifest(
        index_generation=generation,
        corpus_sha256=corpus_sha256,
        source_count=len({_record_source(record) for record in records}),
        chunk_count=len(records),
        parser_version=PARSER_VERSION,
        chunker_version=CHUNKER_VERSION,
        tokenizer_version=TOKENIZER_VERSION,
        embedding_model=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIM,
        chroma_collection=collection_name,
        bm25_artifact=f"{generation}/bm25",
        sparse_artifact=None,
        created_at=timestamp,
    )
    write_rag_index_manifest(manifest, artifact_root)
    return manifest


async def build_full_index_candidate(
    document_paths: Optional[List[Path]] = None,
    *,
    artifact_root: Optional[Path] = None,
    created_at: Optional[datetime] = None,
) -> RAGIndexManifest:
    """Build full dense and BM25 candidates without changing active state."""
    paths = document_paths
    if paths is None:
        paths = sorted(
            path
            for path in PDF_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    if not paths:
        raise ValueError("no supported source documents found")
    records = await _extract_candidate_records(paths)
    return _publish_candidate_generation(
        records,
        artifact_root=artifact_root,
        created_at=created_at,
    )


async def build_incremental_index_candidate(
    document_path: Path,
    *,
    active_generation: Optional[str] = None,
    force_replace: bool = False,
    artifact_root: Optional[Path] = None,
    created_at: Optional[datetime] = None,
) -> RAGIndexManifest:
    """Build a copy-on-write candidate snapshot for one source update."""
    if active_generation is None:
        from app.services.rag.indexing.versioning import get_active_index_generation

        active_generation = await get_active_index_generation()
    if not active_generation:
        raise RuntimeError("no active RAG generation pointer is available")
    load_rag_index_manifest(active_generation, artifact_root)
    active_documents = get_medical_store().export_generation_documents(
        active_generation
    )
    incoming = await _extract_candidate_records([document_path])
    snapshot = build_candidate_snapshot(
        active_documents,
        incoming,
        source_name=document_path.name,
        force_replace=force_replace,
    )
    return _publish_candidate_generation(
        snapshot,
        artifact_root=artifact_root,
        created_at=created_at,
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
        logger.info(f"开始构建医学知识库索引，文档目录: {PDF_DIR}，目标版本: {target_version}")

        # 1. 验证文档目录
        if not PDF_DIR.exists():
            logger.error(f"文档目录不存在: {PDF_DIR}")
            return

        # 2. 扫描支持的文档文件（PDF / Word）
        doc_files = sorted(
            f for f in PDF_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not doc_files:
            logger.warning(
                f"未找到可处理的文档文件（{', '.join(SUPPORTED_EXTENSIONS)}）: {PDF_DIR}"
            )
            return

        logger.info(f"发现 {len(doc_files)} 个文档文件")

        # 3. 提取所有文本块
        all_chunks = []  # [{"id": ..., "text": ..., "source": ..., "page": ...}]
        # 每个 source 的全局递增序号，用于 Small-to-Big 邻居块定位
        source_seq_counter: Dict[str, int] = {}

        for doc_path in doc_files:
            pages = extract_document(doc_path)
            await apply_ocr_to_pages(pages)  # 低文本页 OCR 兜底（未启用时零开销）
            for page_info in pages:
                # 使用共享分块入口，自动区分正文和表格
                chunk_items = _extract_chunks_from_page(page_info)
                for idx, item in enumerate(chunk_items):
                    chunk_content = item["text"]
                    heading_path = item.get("heading_path", "")
                    doc_id = generate_doc_id(
                        page_info["source"], page_info["page"], idx, chunk_content
                    )
                    src = page_info["source"]
                    chunk_seq = source_seq_counter.get(src, 0)
                    source_seq_counter[src] = chunk_seq + 1
                    all_chunks.append(
                        {
                            "id": doc_id,
                            "text": chunk_content,
                            "source": src,
                            "page": page_info["page"],
                            "heading_path": heading_path,
                            "content_type": page_info.get("content_type", "text"),
                            "chunk_seq": chunk_seq,
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
                "chunk_seq": chunk.get("chunk_seq", -1),  # Small-to-Big 邻居定位
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
        pages = extract_document(pdf_path)
        await apply_ocr_to_pages(pages)  # 低文本页 OCR 兜底（未启用时零开销）
        chunks = []
        chunk_seq = 0  # source 内全局递增序号（Small-to-Big 邻居定位，与全量构建路径一致）
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
                        "chunk_seq": chunk_seq,
                    }
                )
                chunk_seq += 1

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
                "chunk_seq": c.get("chunk_seq", -1),  # Small-to-Big 邻居定位
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
