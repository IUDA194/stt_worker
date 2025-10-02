from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aioboto3
import httpx
from botocore.config import Config
from redis import asyncio as aioredis
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.asr_client import ASRClient
from app.config import settings
from app.models.job import JobStatus, STTJob
from app.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

# ---------- Notion: мягкая зависимость + явный лог причины импорта ----------
try:  # pragma: no cover - soft dependency for optional Notion integration
    from app.notion import NotionUploader
except Exception as e:  # pragma: no cover
    NotionUploader = None  # type: ignore[assignment]
    logger.warning("Notion import failed: %s", e)


# ---------------------- Вспомогательные функции ----------------------


def utcnow() -> datetime:
    """Возвращает timezone-aware текущее время в UTC."""
    return datetime.now(timezone.utc)


async def get_db_session_factory() -> async_sessionmaker[AsyncSession]:
    """Создаёт фабрику асинхронных сессий SQLAlchemy."""
    dsn = settings.DB_DSN
    engine_kwargs: dict[str, Any]
    if dsn.startswith("sqlite+aiosqlite://"):
        engine_kwargs = {"echo": False}
    else:
        engine_kwargs = {"echo": False, "pool_size": 5, "max_overflow": 10}

    engine = create_async_engine(dsn, **engine_kwargs)
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_redis():
    """Инициализирует подключение к Redis (asyncio-обёртка)."""
    return aioredis.from_url(settings.REDIS_DSN, decode_responses=True)


def s3_client():
    """Фабрика асинхронного S3-клиента для aioboto3."""
    cfg = Config(
        s3={"addressing_style": "path"},
        signature_version="s3v4",
        retries={"max_attempts": 3, "mode": "standard"},
    )
    session = aioboto3.Session()
    return session.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=getattr(settings, "S3_REGION", None),
        config=cfg,
    )


async def fetch_audio_bytes(bucket: str, key: str) -> bytes:
    """Скачивает аудиофайл из S3 и возвращает его как bytes."""
    async with s3_client() as client:
        obj = await client.get_object(Bucket=bucket, Key=key)
        body = obj["Body"]
        return await body.read()


async def update_status(db: AsyncSession, job: STTJob, status: JobStatus) -> None:
    """Обновляет статус задачи и сохраняет изменения."""
    job.status = status
    await db.commit()


async def set_transcript(db: AsyncSession, job: STTJob, transcript: str) -> None:
    """Сохраняет текст транскрипции и помечает задачу как завершённую."""
    job.transcript = transcript
    job.transcribed_at = utcnow()
    job.status = JobStatus.TRANSCRIBED
    await db.commit()


# ---------------------- Поддержка нормализации аудио ----------------------


SUPPORTED_DIRECT_EXTS = {".ogg", ".wav", ".flac", ".webm"}  # можно слать как есть
NEEDS_CONVERT_EXTS = {".mp3", ".m4a", ".aac", ".mp4", ".mka", ".oga"}  # раньше конвертили в WAV


def _lower_ext(name: str) -> str:
    try:
        return Path(name).suffix.lower()
    except Exception:
        return ""


async def _ensure_ffmpeg_available() -> None:
    """Проверяет наличие ffmpeg, логирует подсказку если отсутствует."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-version",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        if proc.returncode != 0:
            logger.warning("ffmpeg seems unavailable (rc=%s). Audio conversion may fail.", proc.returncode)
    except FileNotFoundError:
        logger.warning("ffmpeg not found in PATH. Audio conversion may fail.")


async def convert_to_wav_16k_mono(src_bytes: bytes, src_name: str) -> tuple[bytes, str]:
    """
    (Старая версия) Конвертирует в WAV 16k mono — ОСТАВЛЕНО для совместимости, но больше не используется по умолчанию.
    """
    await _ensure_ffmpeg_available()

    tmpdir = tempfile.mkdtemp(prefix="stt_norm_")
    in_path = os.path.join(tmpdir, os.path.basename(src_name))
    out_path = os.path.join(tmpdir, "audio.wav")

    try:
        with open(in_path, "wb") as f:
            f.write(src_bytes)

        cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", out_path]

        def _run():
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        await asyncio.to_thread(_run)

        with open(out_path, "rb") as f:
            out_bytes = f.read()

        return out_bytes, "audio.wav"
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


async def convert_to_ogg_opus(src_bytes: bytes, src_name: str, bitrate: str = "48k") -> tuple[bytes, str]:
    """
    Новая нормализация по умолчанию:
    Конвертирует произвольный аудио-байт-поток в OGG/Opus (моно, VBR, заданный битрейт).
    На выходе значительно меньше размер (обычно 3–6 МБ вместо десятков).
    """
    await _ensure_ffmpeg_available()

    tmpdir = tempfile.mkdtemp(prefix="stt_norm_")
    in_path = os.path.join(tmpdir, os.path.basename(src_name))
    out_path = os.path.join(tmpdir, "audio.ogg")

    try:
        with open(in_path, "wb") as f:
            f.write(src_bytes)

        # Преобразование в Opus mono, VBR; ffmpeg сам подберёт частоту дискретизации
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-ac", "1",                    # моно
            "-c:a", "libopus",
            "-b:a", bitrate,               # например, 48k
            "-vbr", "on",
            out_path,
        ]

        def _run():
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        await asyncio.to_thread(_run)

        with open(out_path, "rb") as f:
            out_bytes = f.read()

        return out_bytes, "audio.ogg"
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


async def maybe_normalize_audio(src_bytes: bytes, src_filename: str) -> tuple[bytes, str, bool]:
    """
    (Старая стратегия) — оставлено для обратной совместимости: конвертация в WAV.
    Теперь по умолчанию используем convert_to_ogg_opus() в process_job().
    """
    ext = _lower_ext(src_filename)
    if ext in SUPPORTED_DIRECT_EXTS:
        return src_bytes, src_filename, False
    if ext in NEEDS_CONVERT_EXTS or not ext:
        try:
            dst_bytes, dst_name = await convert_to_wav_16k_mono(src_bytes, src_filename or "audio.bin")
            return dst_bytes, dst_name, True
        except Exception as e:
            logger.warning("Audio normalization (WAV) failed (%s). Will send original bytes: %s", src_filename, e)
            return src_bytes, src_filename or "audio.ogg", False
    return src_bytes, src_filename or "audio.ogg", False


# ---------------------- Обработка задач ----------------------


def _resolve_asr(asr: Optional[Any]) -> Any:
    return asr or ASRClient()


def _resolve_notion(notion: Optional[Any]):
    """
    Возвращает NotionUploader, если заданы:
      - NOTION_API_KEY
      - NOTION_ROOT_PAGE_ID
    Пишет подробный лог диагностики окружения.
    """
    if notion is not None:
        return notion
    if NotionUploader is None:
        logger.info("Notion uploader unavailable: import failed")
        return None

    token = getattr(settings, "NOTION_API_KEY", None)
    root_page_id = getattr(settings, "NOTION_ROOT_PAGE_ID", None)
    debug = bool(getattr(settings, "NOTION_DEBUG", False))

    logger.info(
        "Notion env check: token=%s root=%s debug=%s",
        ("set" if token else "missing"),
        (root_page_id[:6] + "..." if root_page_id else "missing"),
        debug,
    )

    if not token or not root_page_id:
        return None

    try:
        return NotionUploader(token=token, root_page_id=root_page_id, debug=debug)
    except Exception as e:
        logger.warning("Notion uploader init failed: %s", e)
        return None


def _resolve_telegram(tg: Optional[Any]):
    """
    Возвращает TelegramNotifier, если есть TELEGRAM_BOT_TOKEN.
    TELEGRAM_CHAT_ID используется как опциональный дефолт (фолбэк).
    Основной адресат берётся из job.user при отправке.
    """
    if tg is not None:
        return tg
    token = getattr(settings, "TELEGRAM_BOT_TOKEN", None)
    default_chat_id = getattr(settings, "TELEGRAM_CHAT_ID", None)  # fallback only
    if not token:
        return None
    return TelegramNotifier(token, default_chat_id)


async def process_job(
    job_id: str,
    *,
    Session: Optional[async_sessionmaker[AsyncSession]] = None,
    asr: Optional[Any] = None,
    notion: Optional[Any] = None,
    tg: Optional[Any] = None,
) -> None:
    session_factory = Session or await get_db_session_factory()
    async with session_factory() as db:
        job = await db.get(STTJob, job_id)
        if job is None:
            logger.warning("Job %s not found", job_id)
            return

        logger.info("Start processing %s", job_id)

        asr_client = _resolve_asr(asr)
        notion_client = _resolve_notion(notion)
        telegram_client = _resolve_telegram(tg)

        # Диагностика доступности интеграций
        if telegram_client is None:
            logger.info("Job %s: Telegram integration disabled (missing TELEGRAM_BOT_TOKEN)", job_id)
        else:
            token_tail = getattr(settings, "TELEGRAM_BOT_TOKEN", "")[-4:]
            logger.info("Job %s: Telegram enabled (token=****%s)", job_id, token_tail)

        if notion_client is None:
            logger.info("Job %s: Notion integration disabled", job_id)

        try:
            logger.info("Job %s: setting status STARTED", job_id)
            await update_status(db, job, JobStatus.STARTED)

            logger.info(
                "Job %s: downloading audio from s3://%s/%s",
                job_id,
                job.s3_bucket,
                job.s3_key,
            )
            audio_bytes = await fetch_audio_bytes(job.s3_bucket, job.s3_key)
            logger.info("Job %s: audio downloaded (%d bytes)", job_id, len(audio_bytes))
            filename = job.s3_key.split("/")[-1] or "audio.ogg"
            language = getattr(settings, "ASR_LANGUAGE", "ru")
            task = getattr(settings, "ASR_TASK", "transcribe")
            timeout = float(getattr(settings, "ASR_TIMEOUT", 6000.0))

            # === Новая нормализация: OGG/Opus (моно, VBR, ~48k) ===
            opus_bytes, opus_name = await convert_to_ogg_opus(audio_bytes, filename, bitrate="48k")
            # Лог размера после конвертации для контроля выигрыша
            try:
                logger.info(
                    "Job %s: audio normalized to OGG Opus (%.2f → %.2f MB) (%s -> %s)",
                    job_id,
                    len(audio_bytes) / (1024 * 1024),
                    len(opus_bytes) / (1024 * 1024),
                    filename,
                    opus_name,
                )
            except Exception:
                logger.info("Job %s: audio normalized to OGG Opus (%s -> %s)", job_id, filename, opus_name)

            content_type = "audio/ogg"

            logger.info(
                "Job %s: sending audio to ASR (task=%s, language=%s, timeout=%.1f)",
                job_id,
                task,
                language,
                timeout,
            )
            transcript = await asr_client.transcribe(
                opus_bytes,
                filename=opus_name,
                language=language,
                task=task,
                timeout=timeout,
                content_type=content_type,  # ВАЖНО: правильный content-type
            )
            logger.info("Job %s: transcription received (%d chars)", job_id, len(transcript))

            logger.info("Job %s: saving transcript", job_id)
            await set_transcript(db, job, transcript)

            notion_url: Optional[str] = None
            if notion_client is not None:
                try:
                    logger.info("Job %s: sending transcript to Notion", job_id)
                    notion_url = await notion_client.create_page(
                        job_id=job.job_id,
                        transcript=transcript,
                        user=job.user,
                        tag=job.tag,
                        s3_url=job.s3_url,
                    )
                    logger.info("Job %s: Notion page created at %s", job_id, notion_url)
                except Exception as exc:  # подробный стек в лог
                    logger.exception("Job %s: Notion notification failed: %s", job_id, exc)
            else:
                logger.info("Job %s: Notion integration disabled", job_id)

            if telegram_client is not None:
                try:
                    logger.info("Job %s: sending Telegram notification", job_id)
                    lines = [
                        f"Транскрипция готова для задачи {job.job_id}",
                        f"Пользователь: {job.user}",
                    ]
                    if job.tag:
                        lines.append(f"Тег: {job.tag}")
                    if job.s3_url:
                        lines.append(f"S3: {job.s3_url}")
                    if notion_url:
                        lines.append(f"Notion: {notion_url}")
                    message = "\n".join(lines)

                    # Основной получатель — user из БД; если пусто — уйдёт на default_chat_id в TelegramNotifier
                    chat_override = job.user or None
                    try:
                        await telegram_client.send(message, chat_id=chat_override)
                        logger.info("Job %s: Telegram notification sent", job_id)
                    except httpx.HTTPStatusError as e:
                        # типовой случай: 403, если пользователь не писал боту/нет прав
                        if e.response is not None and e.response.status_code == 403:
                            logger.warning("Job %s: Telegram 403 to user %s, retrying to default chat", job_id, job.user)
                            await telegram_client.send(message)  # fallback на дефолтный чат
                            logger.info("Job %s: Telegram fallback notification sent", job_id)
                        else:
                            raise
                except Exception as exc:  # pragma: no cover
                    logger.warning("Job %s: Telegram notification failed: %s", job_id, exc)
            else:
                logger.info("Job %s: Telegram integration disabled", job_id)

            logger.info("Job %s: marking status READY", job_id)
            await update_status(db, job, JobStatus.READY)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Job %s failed: %s", job_id, exc)
            job.status = JobStatus.ERROR
            job.transcript = f"[ERROR] {exc}"
            job.transcribed_at = utcnow()
            await db.commit()


async def run_worker() -> None:
    """Основной цикл воркера: слушает очередь и обрабатывает задачи."""
    Session = await get_db_session_factory()
    redis = await get_redis()
    queue = settings.REDIS_QUEUE

    logger.info("Service is started")
    logger.info("Listening on Redis queue %s", queue)

    try:
        while True:
            item = await redis.blpop(queue, timeout=0)
            if not item:
                continue

            _, job_id = item
            logger.info("New job in redis: %s", job_id)
            try:
                await process_job(job_id, Session=Session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - логируем и идём дальше
                logger.exception("Processing of job %s failed: %s", job_id, exc)
    finally:
        await redis.close()

