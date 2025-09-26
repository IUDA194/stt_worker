from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # ищем .env в нескольких местах: в папке воркера, уровнем выше и в back/
        env_file=(".env", "../.env", "../back/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # БД
    DB_DSN: str = "postgresql+asyncpg://stt_user:strongpassword@localhost:5432/stt"

    # Redis
    REDIS_DSN: str = "redis://localhost:6379/0"
    REDIS_QUEUE: str = "stt_jobs"

    # S3
    S3_ENDPOINT: str = "http://localhost:9000"  # <— переопределится из .env
    S3_REGION: str = "us-east-1"
    S3_BUCKET: str = "stt-audio"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"

    # ASR
    ASR_URL: str = "http://localhost:9000/asr"

    # Notion / TG (если используешь)
    NOTION_API_KEY: str | None = None
    NOTION_ROOT_PAGE_ID: str | None = None
    NOTION_DEBUG: bool = False

    TELEGRAM_BOT_TOKEN: str | None = None
    TELEGRAM_CHAT_ID: str | None = None

    # Логирование
    LOG_LEVEL: str = "INFO"


settings = Settings()
