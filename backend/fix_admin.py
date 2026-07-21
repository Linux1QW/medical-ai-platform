
import asyncio

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Assuming the script is run from the root of the backend directory
from app.core.config import settings
from app.core.security import hash_password
from app.models.user import User

DATABASE_URL = settings.DATABASE_URL

async def fix_admin_password():
    engine = create_async_engine(DATABASE_URL, echo=True)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with AsyncSessionLocal() as db:
        # Find the admin user
        result = await db.execute(select(User).where(User.username == 'admin'))
        admin_user = result.scalar_one_or_none()

        if admin_user:
            print("Found admin user. Updating password...")
            new_hashed_password = hash_password('admin123')

            # Update the password
            await db.execute(
                update(User)
                .where(User.username == 'admin')
                .values(hashed_password=new_hashed_password)
            )
            await db.commit()
            print(f"Admin password updated successfully. New hash: {new_hashed_password}")
        else:
            print("Admin user not found.")

if __name__ == "__main__":
    asyncio.run(fix_admin_password())
