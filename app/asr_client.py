# worker/app/asr_client.py
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from app.config import settings

__all__ = ["ASRClient", "transcribe"]


DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0, read=50.0)
_MAX_BODY_SNIPPET = 1024


logger = logging.getLogger(__name__)


class ASRClient:
    """HTTP-клиент для общения с ASR-сервисом."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        endpoint: str = "/asr",
        client: Optional[httpx.AsyncClient] = None,
        timeout: Optional[float | httpx.Timeout] = None,
    ) -> None:
        raw_url = base_url or settings.ASR_URL
        self._url = self._normalise_url(raw_url, endpoint)
        self._client = client
        self._timeout = timeout

    @staticmethod
    def _normalise_url(base: str, endpoint: str) -> str:
        base = base.rstrip("/")
        clean_endpoint = endpoint.strip("/")
        if clean_endpoint and base.endswith(f"/{clean_endpoint}"):
            return base
        if clean_endpoint:
            return f"{base}/{clean_endpoint}"
        return base

    @staticmethod
    def _resolve_timeout(value: Optional[float | httpx.Timeout]) -> httpx.Timeout:
        if isinstance(value, httpx.Timeout):
            return value
        if isinstance(value, (int, float)):
            return httpx.Timeout(value)
        return DEFAULT_TIMEOUT

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.ogg",
        content_type: str = "audio/ogg",
        language: Optional[str] = None,
        task: Optional[str] = None,
        timeout: Optional[float | httpx.Timeout] = None,
    ) -> str:
        params = {
            "task": task or getattr(settings, "ASR_TASK", "transcribe"),
            "language": language or getattr(settings, "ASR_LANGUAGE", "ru"),
        }
        files = {"audio_file": (filename, audio_bytes, content_type)}

        client = self._client
        response: httpx.Response

        if client is not None:
            response = await client.post(self._url, params=params, files=files)
        else:
            timeout_conf = self._resolve_timeout(timeout or self._timeout)
            async with httpx.AsyncClient(timeout=timeout_conf) as http_client:
                response = await http_client.post(self._url, params=params, files=files)

        content_type = response.headers.get("content-type", "")
        body_text = response.text
        body_snippet = body_text[:_MAX_BODY_SNIPPET]

        if response.status_code != 200:
            logger.error(
                "ASR request failed: status=%s content_type=%s body=%r",
                response.status_code,
                content_type,
                body_snippet,
            )
            raise RuntimeError(
                f"ASR request failed: {response.status_code} {body_snippet}"
            )

        if "json" in content_type.lower():
            try:
                data: dict[str, Any] = response.json()
            except ValueError as exc:  # pragma: no cover - defensive
                transcript = body_text.strip()
                if transcript:
                    logger.warning(
                        "ASR response JSON decode failed, falling back to raw text: "
                        "status=%s error=%s",
                        response.status_code,
                        exc,
                    )
                    return transcript

                logger.error(
                    "ASR response is not valid JSON: status=%s content_type=%s body=%r",
                    response.status_code,
                    content_type,
                    body_snippet,
                )
                raise ValueError(
                    "ASR response is not valid JSON and body is empty"
                ) from exc

            text = data.get("text")
            if not isinstance(text, str):
                logger.error(
                    "ASR response JSON missing 'text': payload=%s", data
                )
                raise ValueError("ASR response JSON has no 'text' field")

            return text

        transcript = body_text.strip()
        if not transcript:
            logger.error(
                "ASR response body is empty: status=%s content_type=%s",
                response.status_code,
                content_type,
            )
            raise ValueError("ASR response body is empty")

        logger.debug(
            "ASR returned plain text response: status=%s content_type=%s",
            response.status_code,
            content_type,
        )
        return transcript


async def transcribe(
    audio_bytes: bytes,
    *,
    filename: str = "audio.ogg",
    content_type: str = "audio/ogg",
    language: Optional[str] = None,
    task: Optional[str] = None,
    timeout: Optional[float | httpx.Timeout] = None,
) -> str:
    """Совместимость со старым API — прокси к :class:`ASRClient`."""
    client = ASRClient()
    return await client.transcribe(
        audio_bytes,
        filename=filename,
        content_type=content_type,
        language=language,
        task=task,
        timeout=timeout,
    )
