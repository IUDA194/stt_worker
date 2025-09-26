import os
import pytest
import pytest_asyncio
from datetime import datetime
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# импортируем код воркера
from app import transcriber as worker
from app.models.base import Base
from app.models.job import STTJob, JobStatus


# ------------ базовые фикстуры ------------

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture(scope="session")
async def engine():
    os.environ["DB_DSN"] = "sqlite+aiosqlite:///:memory:"
    eng = create_async_engine(os.environ["DB_DSN"], echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture(scope="session")
async def Session(engine):
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@pytest_asyncio.fixture
async def make_job(Session):
    async def _make(
        *,
        job_id: str = "job-1",
        user: str = "alice",
        tag: str | None = "voice",
        status: JobStatus = JobStatus.RECEIVED,
        s3_bucket: str = "stt-audio",
        s3_key: str = "alice/job-1.ogg",
        s3_url: str | None = "s3://stt-audio/alice/job-1.ogg",
        created_at: datetime | None = None,
    ) -> STTJob:
        created_at = created_at or datetime.utcnow()
        async with Session() as db:
            j = STTJob(
                job_id=job_id,
                user=user,
                tag=tag,
                status=status,
                s3_bucket=s3_bucket,
                s3_key=s3_key,
                s3_url=s3_url,
                created_at=created_at,
                transcribed_at=None,
                transcript=None,
            )
            db.add(j)
            await db.commit()
            return j
    return _make


# Подменяем скачивание аудио из S3 — всегда отдаём фиктивные байты
@pytest_asyncio.fixture
async def patch_fetch_audio(monkeypatch):
    async def _fake_fetch(bucket: str, key: str) -> bytes:
        return b"FAKE_OGG_BYTES"
    monkeypatch.setattr(worker, "fetch_audio_bytes", _fake_fetch, raising=True)


# Лёгкие фейки для зависимостей
class FakeNotion:
    def __init__(self):
        self.last: dict = {}

    async def create_page(self, **kwargs) -> str:
        self.last = kwargs
        return "https://notion.test/page"


class FakeTelegram:
    def __init__(self):
        self.last_text: str | None = None
        self.last_chat_id: str | None = None

    async def send(self, text: str, *, chat_id: str | None = None) -> None:
        self.last_text = text
        self.last_chat_id = chat_id


@pytest_asyncio.fixture
async def notion():
    return FakeNotion()


@pytest_asyncio.fixture
async def telegram():
    return FakeTelegram()
