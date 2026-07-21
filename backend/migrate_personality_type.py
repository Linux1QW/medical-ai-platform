"""
数据库迁移脚本：将 virtual_patients 表的 personality_type ENUM 从英文值迁移为中文值。

步骤：
1. 临时将列改为 VARCHAR 以允许任意值
2. 更新现有数据从英文到中文
3. 将列改回 ENUM 并使用中文值
"""
import asyncio

from sqlalchemy import text

from app.db.session import AsyncSessionLocal


async def migrate():
    async with AsyncSessionLocal() as db:
        # Step 1: 改为 VARCHAR 临时存储
        print("[1/3] ALTER COLUMN to VARCHAR...")
        await db.execute(text(
            "ALTER TABLE virtual_patients MODIFY COLUMN personality_type VARCHAR(20) NOT NULL COMMENT '人格类型'"
        ))
        await db.commit()
        print("      Done.")

        # Step 2: 更新数据
        print("[2/3] UPDATE data from English to Chinese...")
        mapping = {
            "cooperative": "配合型",
            "anxious": "焦虑型",
            "reticent": "沉默型",
            "aggressive": "对抗型",
        }
        for eng, chn in mapping.items():
            result = await db.execute(
                text("UPDATE virtual_patients SET personality_type = :chn WHERE personality_type = :eng"),
                {"chn": chn, "eng": eng},
            )
            print(f"      {eng} -> {chn}: {result.rowcount} rows updated")
        await db.commit()
        print("      Done.")

        # Step 3: 改回 ENUM 使用中文值
        print("[3/3] ALTER COLUMN back to ENUM with Chinese values...")
        await db.execute(text(
            "ALTER TABLE virtual_patients MODIFY COLUMN personality_type "
            "ENUM('配合型','焦虑型','沉默型','对抗型') NOT NULL COMMENT '人格类型'"
        ))
        await db.commit()
        print("      Done.")

        # 验证
        print("\n=== Verification ===")
        r = await db.execute(text("SHOW COLUMNS FROM virtual_patients LIKE 'personality_type'"))
        col = r.fetchone()
        print(f"Column def: {col[1]}")

        r2 = await db.execute(text("SELECT id, name, personality_type FROM virtual_patients"))
        for row in r2.fetchall():
            print(f"  id={row[0]} name={row[1]} type={row[2]}")

    print("\nMigration completed successfully!")


asyncio.run(migrate())
