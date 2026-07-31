"""从 embed_cache 重建医学知识库向量集合（幂等、免费、可重复）。

背景与两阶段设计
----------------
索引重建拆成两阶段，避免昂贵的 embedding 费用因 Chroma 崩溃而白花：
- 阶段一（花钱，偶尔跑）：抽取+分块+embedding，落盘为 data/embed_cache/*.npz。
  PDF 变更时才需重跑，见 app.services.rag.indexing.builder 的分块/增强逻辑。
- 阶段二（免费，本脚本）：读取 npz 灌入 Chroma 集合 medical_guidelines_<version>。
  Chroma 再损坏就删集合重跑本脚本，不花一分钱。

ChromaDB 1.5.7 注意
-------------------
该版本一旦集合规模越过 hnsw:sync_threshold 会把 HNSW 落盘为独立段，而其自身段
读取器无法再加载回来，跨进程冷读报 "Error loading hnsw index"。修复见
app/services/rag/medical_store.py 的 COLLECTION_METADATA（把阈值设为极大值，
索引常驻 WAL，进程首查时从 chroma.sqlite3 内存重建）。本脚本经该集合创建路径写入，
自动继承此配置。

用法
----
    python scripts/rebuild_kb_from_cache.py [--version rag-v1] [--fresh]

    --fresh  重灌前先删除同名集合（从零重建）。默认增量：已入库 source 自动跳过。
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time

for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_var, None)
os.environ["NO_PROXY"] = "*"

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)
CACHE_DIR = os.path.join(BACKEND, "data", "embed_cache")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rebuild_kb")


def load_npz(path):
    import numpy as np

    with np.load(path) as z:
        emb = z["embeddings"]
        payload = json.loads(bytes(z["payload"].tobytes()).decode("utf-8"))
    return payload["ids"], payload["documents"], payload["metadatas"], emb.tolist()


async def main() -> int:
    parser = argparse.ArgumentParser(description="从 embed_cache 重建向量集合")
    parser.add_argument("--version", default=None, help="索引版本（默认取 settings.ACTIVE_INDEX_VERSION）")
    parser.add_argument("--fresh", action="store_true", help="重灌前删除同名集合")
    args = parser.parse_args()

    from app.core.config import settings
    from app.services.rag.medical_store import (
        _reset_collection_cache,
        get_medical_store,
        set_build_mode,
    )

    target_version = args.version or settings.ACTIVE_INDEX_VERSION
    original_version = settings.ACTIVE_INDEX_VERSION
    settings.ACTIVE_INDEX_VERSION = target_version
    set_build_mode(True)
    _reset_collection_cache()

    try:
        if not os.path.isdir(CACHE_DIR):
            logger.error(f"缓存目录不存在: {CACHE_DIR}（请先跑阶段一 embedding 落盘）")
            return 1

        store = get_medical_store()
        client = store._ensure_client()
        name = f"medical_guidelines_{target_version}"

        if args.fresh:
            try:
                client.delete_collection(name)
                logger.info(f"已删除旧集合 {name}")
            except Exception as e:
                logger.info(f"删除集合跳过（可能不存在）: {type(e).__name__}: {e}")
            _reset_collection_cache()
            store.refresh_collection()

        files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith(".npz"))
        total = len(files)
        logger.info(f"=== 重建开始: {total} 个缓存文件 → {name} (fresh={args.fresh}) ===")

        loaded = skipped = empty = failed = 0
        chunks_total = 0
        t_start = time.time()
        for i, fname in enumerate(files, 1):
            t0 = time.time()
            try:
                ids, documents, metadatas, embeddings = load_npz(os.path.join(CACHE_DIR, fname))
                if not ids:
                    empty += 1
                    continue
                source = metadatas[0]["source"]
                if not args.fresh and store.get_source_doc_count(source) > 0:
                    skipped += 1
                    logger.info(f"[{i}/{total}] skipped  已入库 {source}")
                    continue
                store.add_documents(
                    ids=ids, documents=documents,
                    embeddings=embeddings, metadatas=metadatas,
                )
                loaded += 1
                chunks_total += len(ids)
                logger.info(f"[{i}/{total}] loaded chunks={len(ids):5d} ({time.time()-t0:.1f}s) {fname}")
            except Exception as e:
                failed += 1
                logger.error(f"[{i}/{total}] FAILED {fname}: {type(e).__name__}: {e}")

        logger.info(
            f"=== 灌库结束: loaded={loaded} skipped={skipped} empty={empty} failed={failed} "
            f"new_chunks={chunks_total} elapsed={(time.time()-t_start)/60:.1f}min ==="
        )

        # 进程内自检：真实向量检索 + 计数（顺带把 HNSW 建到内存）
        res = await store.search("非小细胞肺癌的一线治疗方案", top_k=3)
        logger.info(f"自检检索命中 {len(res)} 条，collection_count={store.count()}")
        return 0 if failed == 0 and res else 1
    finally:
        settings.ACTIVE_INDEX_VERSION = original_version
        set_build_mode(False)
        _reset_collection_cache()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
