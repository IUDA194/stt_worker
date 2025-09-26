from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import String, DateTime, Text, Enum as SAEnum, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class JobStatus(str, Enum):
    RECEIVED = "Получен"
    STARTED = "Транскрипция началась"
    TRANSCRIBED = "Транскрипция завершена"
    READY = "Готово"
    ERROR = "Ошибка"


class STTJob(Base):
    __tablename__ = "stt_jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user: Mapped[str] = mapped_column(String(128), index=True)
    tag: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    status: Mapped[JobStatus] = mapped_column(SAEnum(JobStatus), index=True)

    s3_bucket: Mapped[str] = mapped_column(String(255))
    s3_key: Mapped[str] = mapped_column(String(1024))
    s3_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    transcribed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    transcript: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


Index(
    "ix_stt_jobs_user_status_created",
    STTJob.user,
    STTJob.status,
    STTJob.created_at.desc(),
)
