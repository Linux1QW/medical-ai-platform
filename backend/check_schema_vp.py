import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

async def check_schema():
    engine = create_async_engine('mysql+aiomysql://root:qjr3225365@localhost:3306/medical_ai')
    async with engine.connect() as conn:
        try:
            result = await conn.execute(text("DESCRIBE virtual_patients"))
            columns = [row[0] for row in result.all()]
            print("Virtual Patients columns:", columns)
        except Exception as e:
            print("Error checking schema:", e)

asyncio.run(check_schema())
