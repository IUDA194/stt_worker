# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Базовые утилиты + шрифты + ffmpeg
# - без рекомендованных зависимостей, максимально компактно
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    fontconfig fonts-dejavu-core fonts-symbola \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Установка uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ENV PYTHONPATH=/app

WORKDIR /app
COPY . .

# Устанавливаем зависимости проекта (poetry/requirements управляет uv через pyproject.toml/uv.lock)
RUN uv sync --frozen

# Запуск воркера
CMD ["uv", "run", "main.py"]

