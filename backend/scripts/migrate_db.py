import asyncio
import sys
from pathlib import Path

# 将 backend 目录加入路径以便导入 app
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from app.core.config import settings  # noqa: E402


async def migrate():
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.connect() as conn:
        try:
            # 1. 检查 consultations 表是否缺少 max_rounds
            result = await conn.execute(text("DESCRIBE consultations"))
            columns = [row[0] for row in result.all()]
            if 'max_rounds' not in columns:
                print("Adding max_rounds to consultations table...")
                await conn.execute(text("ALTER TABLE consultations ADD COLUMN max_rounds INT DEFAULT 20 COMMENT '最大允许问诊轮次'"))

            # 2. 提交更改
            await conn.commit()
            print("Migration completed successfully.")
        except Exception as e:
            print("Migration error:", e)

asyncio.run(migrate())
