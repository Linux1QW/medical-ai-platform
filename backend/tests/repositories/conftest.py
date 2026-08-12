"""Fixtures for repository tests using in-memory async SQLite."""

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Import all models to ensure they are registered with Base.metadata
from app.models import (  # noqa: F401
    AgentTraceEvent,
    CoachDecision,
    CoachSession,
    CoachStreamEvent,
    ExperimentAssignment,
    PromptBundle,
    TraineeMemory,
    TraineeMemoryConsent,
)
from app.models.base import Base


@pytest.fixture(scope="session")
def async_engine():
    """Create an async SQLite engine for the test session."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    return engine


@pytest.fixture(scope="session")
async def setup_tables(async_engine):
    """Create all tables once per test session."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await async_engine.dispose()


@pytest.fixture
async def db_session(async_engine, setup_tables) -> AsyncSession:
    """Provide a transactional async session for each test.

    Uses a connection-level transaction that is rolled back after each test
    to ensure test isolation.
    """
    _session_factory = async_sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_engine.connect() as conn:
        # Begin a transaction that will be rolled back
        await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            yield session
        finally:
            await session.close()
            await conn.rollback()
