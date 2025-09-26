import os
import pytest
import pytest_asyncio
from datetime import datetime

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app import transcriber as tr
from app.models.base import Base
from app.models.job import STTJob, JobStatus


pytestmark = pytest.mark.asyncio


# ---------- помощники: S3 и чтение байт ----------
class _Body:
    def __init__(self, data: bytes):
        self._data = data
    async def read(self) -> bytes:
        return self._data

class _FakeS3:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return False
    async def get_object(self, Bucket: str, Key: str):
        # возвращаем тело с байтами
        return {"Body": _Body(b"FAKE_AUDIO")}


def _fake_s3_client():
    # контекстный менеджер как у aioboto3
    return _FakeS3()


async def test_fetch_audio_bytes_reads_body(monkeypatch):
    # патчим фабрику клиента S3 внутри модуля transcriber
    monkeypatch.setattr(tr, "s3_client", _fake_s3_client, raising=True)
    data = await tr.fetch_audio_bytes("b", "k")
    assert data == b"FAKE_AUDIO"


# ---------- База и хелперы статусов ----------
@pytest_asyncio.fixture
async def Session():
    os.environ["DB_DSN"] = "sqlite+aiosqlite:///:memory:"
    engine = create_async_engine(os.environ["DB_DSN"], echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        yield Session
    finally:
        await engine.dispose()


async def test_update_and_set_transcript(Session):
    async with Session() as db:
        j = STTJob(
            job_id="j1",
            user="u",
            tag=None,
            status=JobStatus.RECEIVED,
            s3_bucket="b",
            s3_key="k",
            s3_url="s3://b/k",
            created_at=datetime.utcnow(),
        )
        db.add(j)
        await db.commit()

        await tr.update_status(db, j, JobStatus.STARTED)
        assert j.status == JobStatus.STARTED

        await tr.set_transcript(db, j, "text")
        assert j.status == JobStatus.TRANSCRIBED
        assert j.transcript == "text"
        assert j.transcribed_at is not None

