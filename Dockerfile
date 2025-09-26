# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# curl, сертификаты, базовые шрифты, fontconfig, И ЭМОДЖИ
# - fonts-dejavu-core: базовая латиница/кириллица
# - fonts-symbola: монохромные emoji-глифы (если есть в репозитории)
# - fontconfig: для fc-cache (обновить кеш шрифтов после ручной установки)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates fontconfig fonts-dejavu-core fonts-symbola \
 && rm -rf /var/lib/apt/lists/*

# Установка uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ENV PYTHONPATH=/app

WORKDIR /app
COPY . .

RUN uv sync --frozen

# Запуск API
CMD ["uv", "run", "main.py"]

