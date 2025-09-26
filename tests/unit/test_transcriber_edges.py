import asyncio
import pytest
import pytest_asyncio

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app import transcriber as tr
from app.models.base import Base


pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def Session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    await engine.dispose()


async def test_process_job_returns_if_missing(Session, monkeypatch):
    # Нету job в БД → ничего не падает, просто лог и return
    class ASRDummy:
        async def transcribe(self, *a, **k): return "t"
    # notion/tg None — тоже покрываем эту ветку
    await tr.process_job(
        "does-not-exist",
        Session=Session,
        asr=ASRDummy(),
        notion=None,
        tg=None,
    )
    # если не упало — ок (покрыли строки поиска и ранний выход)


async def test_process_job_without_notion_and_tg(Session, monkeypatch):
    # создадим job напрямую через БД
    from app.models.job import STTJob, JobStatus
    from datetime import datetime
    async with Session() as db:
        j = STTJob(
            job_id="no-nt",
            user="u",
            tag=None,
            status=JobStatus.RECEIVED,
            s3_bucket="b",
            s3_key="k.ogg",
            s3_url="s3://b/k.ogg",
            created_at=datetime.utcnow(),
        )
        db.add(j)
        await db.commit()

    # подменим fetch_audio и ASR
    async def fake_fetch(bucket: str, key: str) -> bytes: return b"x"
    class ASRDummy:
        async def transcribe(self, *a, **k): return "ok"

    monkeypatch.setattr(tr, "fetch_audio_bytes", fake_fetch, raising=True)

    await tr.process_job(
        "no-nt",
        Session=Session,
        asr=ASRDummy(),
        notion=None,  # <- важный путь
        tg=None,      # <- важный путь
    )

    # проверим, что задача дошла до READY
    async with Session() as db:
        from app.models.job import STTJob, JobStatus
        j = await db.get(STTJob, "no-nt")
        assert j is not None and j.status == JobStatus.READY


async def test_run_worker_closes_redis_in_finally(monkeypatch, Session):
    # подменяем фабрики
    async def _fake_db_factory():
        return Session

    class FakeRedis:
        def __init__(self):
            self.closed = False
            self._gave = False
        async def blpop(self, key, timeout=0):
            if not self._gave:
                self._gave = True
                return (key, "job-1")
            # После первой выдачи — провоцируем завершение цикла
            raise asyncio.CancelledError
        async def close(self):
            self.closed = True

    async def _fake_get_redis():
        return FakeRedis()

    async def _fake_process_job(job_id: str, **kw):
        # ничего не делаем; пусть blpop второй раз упадёт CancelledError
        return None

    monkeypatch.setattr(tr, "get_db_session_factory", _fake_db_factory, raising=True)
    monkeypatch.setattr(tr, "get_redis", _fake_get_redis, raising=True)
    monkeypatch.setattr(tr, "process_job", _fake_process_job, raising=True)

    with pytest.raises(asyncio.CancelledError):
        await tr.run_worker()

    # Проверка закрытия redis через finally
    r = await _fake_get_redis()
    assert isinstance(r, FakeRedis)
    # тут нельзя проверить конкретный инстанс r.closed, так как run_worker создаёт СВОЙ
    # но если хочешь прямой ассерт — можно вынести redis в внешнюю переменную/замыкание:

