import asyncio
import getpass
import os
import sys

# 将当前目录添加到路径以便导入 app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.security import hash_password
from app.models.user import User


async def init_admin():
    print("正在连接数据库...")
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # 检查是否已有管理员
        result = await session.execute(select(User).where(User.role == "admin"))
        admin = result.scalar_one_or_none()

        if admin:
            print(f"提示：管理员已存在 - {admin.username} ({admin.email})")
            return

        print("\n" + "="*30)
        print("   初始管理员账户初始化")
        print("="*30)

        username = input("请输入管理员用户名 [默认 admin]: ").strip() or "admin"
        email = input("请输入管理员邮箱: ").strip()
        while not email:
            email = input("错误：邮箱不能为空，请输入管理员邮箱: ").strip()

        password = getpass.getpass("请输入管理员密码: ")
        while not password:
            password = getpass.getpass("错误：密码不能为空，请输入管理员密码: ")

        confirm_password = getpass.getpass("请再次输入密码以确认: ")
        if password != confirm_password:
            print("错误：两次输入的密码不一致，请重新运行脚本。")
            return

        hashed_password = hash_password(password)

        new_admin = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            real_name="系统管理员",
            role="admin",
            department="技术部"
        )

        try:
            session.add(new_admin)
            await session.commit()
            print("\n" + "-"*30)
            print("✅ 管理员账号创建成功！")
            print(f"用户名: {username}")
            print(f"邮箱: {email}")
            print("登录地址: http://localhost:5173/login")
            print("-"*30)
        except Exception as e:
            print(f"\n❌ 创建失败: {str(e)}")
            await session.rollback()

if __name__ == "__main__":
    try:
        asyncio.run(init_admin())
    except KeyboardInterrupt:
        print("\n已取消操作")
        sys.exit(0)
