import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app import transcriber as worker
from app.models.job import JobStatus, STTJob


pytestmark = pytest.mark.asyncio


async def test_process_job_happy_path(Session: async_sessionmaker[AsyncSession], make_job, patch_fetch_audio, notion, telegram):
    # Given: job в БД
    job = await make_job(job_id="job-ok")

    # ASR вернёт заранее известный текст
    asr = worker.ASRClient("http://fake")  # не будет вызван по сети
    # Подмена методов через класс-заглушку
    class ASRDummy:
        async def transcribe(self, *a, **kw):
            return "Это тестовая транскрипция."

    # When
    await worker.process_job(
        "job-ok",
        Session=Session,
        asr=ASRDummy(),
        notion=notion,
        tg=telegram,
    )

    # Then: статусы и поля обновлены
    async with Session() as db:
        refreshed: STTJob | None = await db.get(STTJob, "job-ok")
        assert refreshed is not None
        assert refreshed.status == JobStatus.READY
        assert refreshed.transcribed_at is not None
        assert refreshed.transcript == "Это тестовая транскрипция."

    # Проверяем, что Notion и Telegram вызывались
    assert notion.last.get("job_id") == "job-ok"
    assert "Транскрипция готова" in telegram.last_text
    assert "job-ok" in telegram.last_text
    assert telegram.last_chat_id == job.user


async def test_process_job_asr_error(Session, make_job, patch_fetch_audio, telegram, notion):
    await make_job(job_id="job-fail")
    class ASRBroken:
        async def transcribe(self, *a, **kw):
            raise RuntimeError("boom")

    await worker.process_job(
        "job-fail",
        Session=Session,
        asr=ASRBroken(),
        notion=notion,
        tg=telegram,
    )

    async with Session() as db:
        refreshed = await db.get(STTJob, "job-fail")
        assert refreshed is not None
        assert refreshed.status == JobStatus.ERROR
        assert refreshed.transcript is not None
        assert refreshed.transcript.startswith("[ERROR]")
