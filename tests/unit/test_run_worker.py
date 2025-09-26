import asyncio
import pytest
import pytest_asyncio

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app import transcriber as tr
from app.models.base import Base


pytestmark = pytest.mark.asyncio


class _FakeRedis:
    def __init__(self, queue_name: str, first_job: str):
        self.queue_name = queue_name
        self.first_job = first_job
        self.closed = False
        self._emitted = False

    async def blpop(self, key: str, timeout: int = 0):
        # один раз отдаём job, потом «блокируемся» вечно — мы же прервём воркер
        if not self._emitted and key == self.queue_name:
            self._emitted = True
            return (key, self.first_job)
        await asyncio.sleep(3600)  # никогда не дойдёт
        return None

    async def close(self):
        self.closed = True


@pytest_asyncio.fixture
async def Session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    await engine.dispose()


async def test_run_worker_pops_one_and_calls_process_job(monkeypatch, Session):
    calls: list[str] = []

    async def _fake_get_db_session_factory():
        return Session

    async def _fake_get_redis():
        return _FakeRedis(queue_name="stt_jobs", first_job="job-xyz")

    async def _fake_process_job(job_id: str, **kwargs):
        calls.append(job_id)
        # прерываем воркер после первой задачи
        raise asyncio.CancelledError

    # патчим фабрики + сам обработчик
    monkeypatch.setattr(tr, "get_db_session_factory", _fake_get_db_session_factory, raising=True)
    monkeypatch.setattr(tr, "get_redis", _fake_get_redis, raising=True)
    monkeypatch.setattr(tr, "process_job", _fake_process_job, raising=True)

    # запускаем воркер и ловим отмену
    with pytest.raises(asyncio.CancelledError):
        await tr.run_worker()

    assert calls == ["job-xyz"]

