import asyncio
import types
import pytest
import pytest_asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app import transcriber as tr
from app.models.base import Base


pytestmark = pytest.mark.asyncio


async def test_utcnow_timezone_aware():
    now = tr.utcnow()
    assert now.tzinfo is not None and now.tzinfo.utcoffset(now) is not None


async def test_get_db_session_factory_returns_sessionmaker(monkeypatch):
    # Используем отдельный in-memory SQLite
    monkeypatch.setenv("DB_DSN", "sqlite+aiosqlite:///:memory:")
    # settings уже инициализированы — подменим прямо поле
    tr.settings.DB_DSN = "sqlite+aiosqlite:///:memory:"

    Session = await tr.get_db_session_factory()
    assert isinstance(Session, async_sessionmaker)

    # Проверим, что из фабрики можно открыть сессию и создать таблицы
    engine = Session.kw["bind"]  # type: ignore[index]
    async with engine.begin() as conn:  # type: ignore[attr-defined]
        await conn.run_sync(Base.metadata.create_all)
    async with Session() as db:
        assert isinstance(db, AsyncSession)

    await engine.dispose()  # type: ignore[attr-defined]


async def test_get_redis_uses_from_url(monkeypatch):
    captured = {}

    class FakeRedis:
        def __init__(self, dsn):
            captured["dsn"] = dsn
        async def close(self): pass

    def fake_from_url(dsn, decode_responses=True):
        captured["decode"] = decode_responses
        return FakeRedis(dsn)

    monkeypatch.setattr(tr.aioredis, "from_url", fake_from_url, raising=True)
    tr.settings.REDIS_DSN = "redis://example:6379/5"
    r = await tr.get_redis()
    assert isinstance(r, FakeRedis)
    assert captured["dsn"] == "redis://example:6379/5"
    assert captured["decode"] is True


async def test_s3_client_builds_with_aioboto3(monkeypatch):
    # перехватываем вызов Session().client(...)
    called = {}

    class FakeSession:
        def client(self, service, **kwargs):
            called["service"] = service
            called.update(kwargs)
            class _C: pass
            return _C()

    monkeypatch.setattr(tr, "aioboto3", types.SimpleNamespace(Session=lambda: FakeSession()))
    tr.settings.S3_ENDPOINT = "http://minio:9000"
    tr.settings.S3_ACCESS_KEY = "k"
    tr.settings.S3_SECRET_KEY = "s"
    tr.settings.S3_REGION = "us-east-1"

    c = tr.s3_client()
    assert called["service"] == "s3"
    assert called["endpoint_url"] == "http://minio:9000"
    assert called["aws_access_key_id"] == "k"
    assert called["aws_secret_access_key"] == "s"
    assert called["region_name"] == "us-east-1"
    assert c is not None

