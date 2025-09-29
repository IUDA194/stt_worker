# worker/app/config.py
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # ищем .env в нескольких местах: рядом, уровнем выше и в back/
        env_file=(".env", "../.env", "../back/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === БД ===
    DB_DSN: str = "postgresql+asyncpg://stt_user:strongpassword@localhost:5432/stt"

    # === Redis ===
    REDIS_DSN: str = "redis://localhost:6379/0"
    REDIS_QUEUE: str = "stt_jobs"

    # === S3 (MinIO или Yandex Object Storage) ===
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_REGION: str = "us-east-1"
    S3_BUCKET: str = "stt-audio"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"

    # === ASR: Синхронный API (v1) ===
    ASR_URL: str = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"

    # === ASR: Асинхронный API (v3) ===
    # Обычно всё доступно через https://stt.api.cloud.yandex.net
    ASR_LONG_BASE_URL: str = "https://stt.api.cloud.yandex.net"
    ASR_LONG_START_ENDPOINT: str = "speechkit/v3/recognizeFile"
    ASR_LONG_OPS_ENDPOINT: str = "operations"

    # === Авторизация в SpeechKit ===
    SPEECHKIT_API_KEY: str | None = None           # либо Api-Key
    SPEECHKIT_IAM_TOKEN: str | None = None         # либо IAM-токен (требует folderId)
    SPEECHKIT_FOLDER_ID: str | None = None

    # === Настройки распознавания ===
    SPEECHKIT_LANG: str = "ru-RU"                  # ru-RU / en-US / ...
    SPEECHKIT_TOPIC: str = "general"               # general, notes, calls, etc.
    SPEECHKIT_MODEL: str | None = None             # можно указать конкретную модель
    SPEECHKIT_PROFANITY_FILTER: bool = False
    SPEECHKIT_RAW_RESULTS: bool = False
    SPEECHKIT_FORMAT: str = "oggopus"              # oggopus или lpcm
    SPEECHKIT_SAMPLE_RATE_HZ: int | None = None    # требуется для lpcm

    # === Интеграции (опционально) ===
    NOTION_API_KEY: str | None = None
    NOTION_ROOT_PAGE_ID: str | None = None
    NOTION_DEBUG: bool = False

    TELEGRAM_BOT_TOKEN: str | None = None
    TELEGRAM_CHAT_ID: str | None = None

    # === Логирование ===
    LOG_LEVEL: str = "INFO"


settings = Settings()

