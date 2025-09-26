import os
import asyncio
from datetime import datetime
from typing import AsyncIterator, Dict

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# импортируем воркер
from app import transcriber as worker
from app.config import settings
from app.models.job import STTJob, JobStatus
from app.models.base import Base


# ---- Фейковые внешние сервисы ----

class FakeNotion:
    def __init__(self):
        self.last = {}

    async def create_page(self, **kwargs) -> str:
        self.last = kwargs
        return "https://notion.test/page"


class FakeTelegram:
    def __init__(self):
        self.last_text = None
        self.last_chat_id = None

    async def send(self, text: str, *, chat_id: str | None = None) -> None:
        self.last_text = text
        self.last_chat_id = chat_id


class DummyASR:
    def __init__(self, text: str = "Привет, мир!"):
        self.text = text
        self.calls = 0

    async def transcribe(self, audio_bytes: bytes, filename: str, *, language: str, task: str, timeout: float) -> str:
        self.calls += 1
        return self.text


class BrokenASR:
    async def transcribe(self, *a, **kw):
        raise RuntimeError("ASR down")


# ---- Общие фикстуры ----

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


@pytest_asyncio.fixture
async def patch_fetch_audio(monkeypatch):
    """Подменяем скачивание из S3 — возвращаем фиктивные байты."""
    async def _fake_fetch(bucket: str, key: str) -> bytes:
        return b"FAKE_OGG_BYTES"
    monkeypatch.setattr(worker, "fetch_audio_bytes", _fake_fetch, raising=True)


@pytest_asyncio.fixture
async def notion():
    return FakeNotion()


@pytest_asyncio.fixture
async def telegram():
    return FakeTelegram()
