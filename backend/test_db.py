import asyncio

from sqlalchemy.ext.asyncio import create_async_engine


async def test():
    engine = create_async_engine('mysql+aiomysql://root:qjr3225365@localhost:3306/medical_ai')
    try:
        async with engine.connect():
            print('DB Connection OK')
    except Exception as e:
        print('DB Connection Error:', e)

asyncio.run(test())
