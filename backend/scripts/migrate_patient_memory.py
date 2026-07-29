# -*- coding: utf-8 -*-
"""数据库迁移脚本：为 consultations 表增加 memory_state 列（患者智能体记忆）。

幂等：列已存在时直接跳过。
"""
import asyncio
import sys
from pathlib import Path

# 将 backend 目录加入路径以便导入 app
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text  # noqa: E402

from app.db.session import AsyncSessionLocal  # noqa: E402


async def migrate():
    async with AsyncSessionLocal() as db:
        r = await db.execute(text("SHOW COLUMNS FROM consultations LIKE 'memory_state'"))
        if r.fetchone():
            print("memory_state 列已存在，跳过迁移。")
            return
        print("[1/1] ALTER TABLE consultations ADD COLUMN memory_state ...")
        await db.execute(text(
            "ALTER TABLE consultations ADD COLUMN memory_state TEXT NULL "
            "COMMENT '患者智能体记忆状态（JSON 序列化的 MemoryState）' AFTER max_rounds"
        ))
        await db.commit()
        print("      Done.")

        # 验证
        r2 = await db.execute(text("SHOW COLUMNS FROM consultations LIKE 'memory_state'"))
        col = r2.fetchone()
        print(f"\n=== Verification ===\nColumn def: {col[1] if col else 'MISSING'}")

    print("\nMigration completed successfully!")


asyncio.run(migrate())
