# -*- coding: utf-8 -*-
"""Alembic 迁移环境：接入项目 Settings 与 SQLAlchemy 模型 metadata

- 连接串来自 app.core.config.settings（读取 backend/.env），不在 alembic.ini 中硬编码
- autogenerate 基于 app.models 中全部模型（import app.models 即完成注册）
"""

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

import app.models  # noqa: F401  确保全部模型注册到 Base.metadata
from alembic import context
from app.core.config import settings
from app.models.base import Base

config = context.config

# 迁移使用同步驱动（pymysql），与应用运行时的 aiomysql 互不影响；
# ALEMBIC_DATABASE_URL 环境变量可覆盖（如用临时空库生成 baseline）
config.set_main_option(
    "sqlalchemy.url",
    os.environ.get("ALEMBIC_DATABASE_URL") or settings.DATABASE_URL_SYNC,
)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：仅生成 SQL 脚本，不连接数据库"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连接数据库执行迁移"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # 不比对 server_default：存量库由手写 SQL 建表，DEFAULT 子句与 ORM
            # 模型定义存在大量无害差异，开启会淹没真实变更
            compare_server_default=False,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
