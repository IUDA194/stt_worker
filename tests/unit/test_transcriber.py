import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app import transcriber as worker
from app.models.job import JobStatus, STTJob

pytestmark = pytest.mark.asyncio


async def test_process_job_success(Session: async_sessionmaker[AsyncSession], make_job, patch_fetch_audio, notion, telegram):
    # Arrange: есть job в БД
    await make_job(job_id="ok-1")

    class ASRDummy:
        async def transcribe(self, *a, **kw):
            return "Это тестовая транскрипция."

    # Act
    await worker.process_job(
        "ok-1",
        Session=Session,
        asr=ASRDummy(),
        notion=notion,
        tg=telegram,
    )

    # Assert: статус/поля
    async with Session() as db:
        j: STTJob | None = await db.get(STTJob, "ok-1")
        assert j is not None
        assert j.status == JobStatus.READY
        assert j.transcribed_at is not None
        assert j.transcript == "Это тестовая транскрипция."
    # Notion/Telegram вызваны
    assert notion.last.get("job_id") == "ok-1"
    assert telegram.last_text and "Транскрипция готова" in telegram.last_text
    assert telegram.last_chat_id == "alice"


async def test_process_job_asr_error(Session, make_job, patch_fetch_audio, notion, telegram):
    await make_job(job_id="fail-1")

    class ASRBroken:
        async def transcribe(self, *a, **kw):
            raise RuntimeError("boom")

    await worker.process_job(
        "fail-1",
        Session=Session,
        asr=ASRBroken(),
        notion=notion,
        tg=telegram,
    )

    async with Session() as db:
        j = await db.get(STTJob, "fail-1")
        assert j is not None
        assert j.status == JobStatus.ERROR
        assert j.transcript and j.transcript.startswith("[ERROR]")
