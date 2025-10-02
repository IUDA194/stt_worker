# worker/app/asr_client.py
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from typing import Any, Optional, Dict, List

import httpx

from app.config import settings

__all__ = ["ASRClient", "transcribe"]

logger = logging.getLogger(__name__)

# ===== HTTP/таймауты =====
DEFAULT_CLIENT_TIMEOUT = httpx.Timeout(3600.0, connect=10.0, write=6000.0, read=60.0)
_MAX_BODY_SNIPPET = 2048  # используется только когда полное логирование отключено

# ===== Опрос операции =====
DEFAULT_INITIAL_WAIT_SEC = 2.5
DEFAULT_POLL_INTERVAL_SEC = 2.0
MAX_POLL_INTERVAL_SEC = 10.0
POLL_BACKOFF_FACTOR = 1.35
DEFAULT_TOTAL_TIMEOUT_SEC = 7200.0  # 2 часа на длинные файлы

# ===== Флаги/параметры логирования и авто-завершения =====
ASR_LOG_FULL_BODIES: bool = bool(getattr(settings, "ASR_LOG_FULL_BODIES", False))
ASR_AUTO_FINALIZE_POLL_LIMIT: int = int(getattr(settings, "ASR_AUTO_FINALIZE_POLL_LIMIT", 2))


def _norm_join(base: str, *parts: str) -> str:
    base = (base or "").rstrip("/")
    for p in parts:
        if not p:
            continue
        p = p.strip("/")
        base = f"{base}/{p}"
    return base


def _iter_ndjson(text: str) -> List[dict]:
    """Парс NDJSON (построчно), игнор битых строк."""
    items: List[dict] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
        except json.JSONDecodeError:
            # бывает мусор — игнорируем
            pass
    return items


def _tail(s: str, n: int = 200) -> str:
    """Хвостовая вырезка строки для логов (когда полное логирование выключено)."""
    if not s:
        return ""
    return s[-n:]


def _tail_text_preview(text: str, n: int = 80) -> str:
    """Хвостовая вырезка текста сегмента с многоточием слева."""
    if not text:
        return ""
    t = text[-n:]
    return ("…" if len(text) > n else "") + t


def _text_signature(text: str, words: Optional[List[dict]] = None) -> str:
    """
    Сигнатура сегмента: SHA1( text + |start-end|... ) — стабильная между повторами.
    Если нет слов — используем только текст.
    """
    h = hashlib.sha1()
    h.update((text or "").strip().encode("utf-8"))
    if isinstance(words, list) and words:
        spans: List[str] = []
        for w in words:
            try:
                s = str(w.get("startTimeMs"))
                e = str(w.get("endTimeMs"))
                spans.append(f"{s}-{e}")
            except Exception:
                continue
        if spans:
            h.update(("|" + "|".join(spans)).encode("utf-8"))
    return h.hexdigest()


def _log_body(prefix: str, body: str) -> None:
    """Логировать тело ответа целиком (если включено), иначе — хвост."""
    if ASR_LOG_FULL_BODIES:
        logger.info("%s: body_len=%d body=%r", prefix, len(body or ""), body)
    else:
        logger.info("%s: body_tail=%r", prefix, _tail(body or "", _MAX_BODY_SNIPPET))


class ASRClient:
    """
    Yandex SpeechKit STT v3 (long-running) клиент.
    Старт: POST  {BASE}/stt/v3/recognizeFileAsync
    Опрос: GET   {BASE}/stt/v3/getRecognition?operationId=...
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        start_endpoint: Optional[str] = None,
        ops_endpoint: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
        client_timeout: Optional[httpx.Timeout | float] = None,
    ) -> None:
        base_url_cfg = (
            base_url
            or getattr(settings, "ASR_LONG_BASE_URL", None)
            or getattr(settings, "ASR_URL", None)  # fallback
            or "https://stt.api.cloud.yandex.net"
        )
        start_ep_cfg = (
            start_endpoint
            or getattr(settings, "ASR_LONG_START_ENDPOINT", None)
            or "stt/v3/recognizeFileAsync"
        )
        ops_ep_cfg = (
            ops_endpoint
            or getattr(settings, "ASR_LONG_OPS_ENDPOINT", None)
            or "stt/v3/getRecognition"
        )

        self._start_url = _norm_join(base_url_cfg, start_ep_cfg)
        self._ops_url = _norm_join(base_url_cfg, ops_ep_cfg)
        self._client = client
        self._timeout = self._resolve_timeout(client_timeout)

    # ------- helpers

    @staticmethod
    def _resolve_timeout(value: Optional[httpx.Timeout | float]) -> httpx.Timeout:
        if isinstance(value, httpx.Timeout):
            return value
        if isinstance(value, (int, float)):
            if value <= 0:
                return httpx.Timeout(None)
            return httpx.Timeout(value)
        return DEFAULT_CLIENT_TIMEOUT

    def _auth_headers(self) -> dict[str, str]:
        api_key = getattr(settings, "SPEECHKIT_API_KEY", "") or ""
        iam = getattr(settings, "SPEECHKIT_IAM_TOKEN", "") or ""
        if api_key:
            return {"Authorization": f"Api-Key {api_key}"}
        if iam:
            return {"Authorization": f"Bearer {iam}"}
        raise RuntimeError("SpeechKit auth is not configured: set SPEECHKIT_API_KEY or SPEECHKIT_IAM_TOKEN")

    def _common_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        iam = getattr(settings, "SPEECHKIT_IAM_TOKEN", "") or ""
        folder_id = getattr(settings, "SPEECHKIT_FOLDER_ID", "") or ""
        if iam:
            if not folder_id:
                raise RuntimeError("SPEECHKIT_FOLDER_ID is required when using SPEECHKIT_IAM_TOKEN")
            params["folderId"] = folder_id
        return params

    async def _http_post(self, url: str, **kwargs) -> httpx.Response:
        if self._client is not None:
            return await self._client.post(url, **kwargs)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.post(url, **kwargs)

    async def _http_get(self, url: str, **kwargs) -> httpx.Response:
        if self._client is not None:
            return await self._client.get(url, **kwargs)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.get(url, **kwargs)

    # ------- формат аудио

    def _detect_audio_format(self, audio_bytes: bytes, content_type: str) -> dict[str, Any]:
        ct = (content_type or "").lower()
        is_wav_bytes = (
            len(audio_bytes) >= 12
            and audio_bytes[0:4] == b"RIFF"
            and audio_bytes[8:12] == b"WAVE"
        )
        if is_wav_bytes or ct in ("audio/wav", "audio/x-wav"):
            return {"containerAudio": {"containerAudioType": "WAV"}}
        if ct in ("audio/mpeg", "audio/mp3"):
            return {"containerAudio": {"containerAudioType": "MP3"}}
        if ct in ("audio/l16", "audio/pcm"):
            rate = getattr(settings, "SPEECHKIT_SAMPLE_RATE_HZ", None) or 16000
            return {
                "rawAudio": {
                    "audioEncoding": "LINEAR16_PCM",
                    "sampleRateHertz": int(rate),
                    "audioChannelCount": 1,
                }
            }
        # по умолчанию считаем OGG OPUS
        return {"containerAudio": {"containerAudioType": "OGG_OPUS"}}

    # ------- старт операции

    async def start_by_bytes(
        self,
        *,
        audio_bytes: bytes,
        content_type: str = "audio/ogg",
        language: Optional[str] = None,
    ) -> str:
        # Для большого base64 тело может парситься/валидироваться дольше 60с.
        # Дадим старту более мягкий read-timeout и один ретрай на ReadTimeout.
        start_timeout = httpx.Timeout(3600.0, connect=10.0, write=6000.0, read=300.0)
        start_attempts = 2

        headers = self._auth_headers()
        headers["Content-Type"] = "application/json"
        params = self._common_params()

        lang = language or getattr(settings, "SPEECHKIT_LANG", "ru-RU")
        mdl = getattr(settings, "SPEECHKIT_MODEL", None) or "general"
        profanity = bool(getattr(settings, "SPEECHKIT_PROFANITY_FILTER", False))

        audio_format = self._detect_audio_format(audio_bytes, content_type)

        payload = {
            "content": base64.b64encode(audio_bytes).decode("ascii"),
            "recognitionModel": {
                "model": mdl,
                "audioFormat": audio_format,
                "textNormalization": {
                    "textNormalization": "TEXT_NORMALIZATION_ENABLED",
                    "profanityFilter": profanity,
                },
                "languageRestriction": {
                    "restrictionType": "WHITELIST",
                    "languageCode": [lang],
                },
            },
        }

        last_exc: Optional[Exception] = None
        for i in range(1, start_attempts + 1):
            try:
                resp = await self._http_post(
                    self._start_url,
                    headers=headers,
                    params=params,
                    json=payload,
                    timeout=start_timeout,  # перезаписываем client timeout только для этого запроса
                )
                break
            except httpx.ReadTimeout as e:
                last_exc = e
                if i < start_attempts:
                    logger.warning("ASR start_by_bytes ReadTimeout, retrying (%d/%d)…", i + 1, start_attempts)
                    await asyncio.sleep(1.0 * i)
                    continue
                raise

        body = resp.text
        if resp.status_code >= 300:
            if ASR_LOG_FULL_BODIES:
                logger.warning(
                    "ASR start_by_bytes failed: HTTP %s body_len=%d body=%r",
                    resp.status_code,
                    len(body or ""),
                    body,
                )
            else:
                logger.warning(
                    "ASR start_by_bytes failed: HTTP %s body_tail=%r",
                    resp.status_code,
                    _tail(body or "", _MAX_BODY_SNIPPET),
                )
            raise RuntimeError(f"ASR start_by_bytes failed: {resp.status_code}")

        data = resp.json()
        op_id = data.get("operationId") or data.get("id") or data.get("name")
        if not op_id:
            if ASR_LOG_FULL_BODIES:
                logger.error("ASR start_by_bytes: cannot extract operation id. body_len=%d body=%r", len(body or ""), body)
            else:
                logger.error("ASR start_by_bytes: cannot extract operation id. body_tail=%r", _tail(body or "", _MAX_BODY_SNIPPET))
            raise RuntimeError("Cannot extract operation id")

        initial_wait = float(getattr(settings, "ASR_LONG_INITIAL_WAIT_SEC", DEFAULT_INITIAL_WAIT_SEC))
        logger.info("ASR operation %s started, waiting %.1fs before first poll", op_id, initial_wait)
        if initial_wait > 0:
            await asyncio.sleep(initial_wait)
        return op_id

    # ------- опрос + накопление сегментов с дедупликацией и авто-завершением

    async def poll_collect_texts(
        self,
        operation_id: str,
        *,
        poll_interval: Optional[float] = None,
        timeout_total: Optional[float] = None,
    ) -> List[str]:
        """
        Собираем ВСЕ финальные куски (finalRefinement.normalizedText и/или final.alternatives),
        по индексу finalIndex, с защитой от дубликатов. Ждём статус COMPLETED,
        либо авто-завершаем если eou>=received и N опросов подряд без изменений.
        """
        headers = self._auth_headers()
        params = self._common_params()
        params["operationId"] = operation_id

        interval = float(poll_interval or getattr(settings, "ASR_LONG_POLL_INTERVAL_SEC", DEFAULT_POLL_INTERVAL_SEC))
        total_timeout = float(timeout_total or getattr(settings, "ASR_LONG_TIMEOUT_TOTAL_SEC", DEFAULT_TOTAL_TIMEOUT_SEC))
        started_at = time.monotonic()

        # Храним сегменты и сигнатуры для дедупа
        segments: Dict[int, str] = {}
        sig_by_index: Dict[int, str] = {}
        idx_by_sig: Dict[str, int] = {}  # для final без индекса
        seen_signatures: set[str] = set()
        completed = False

        # Для авто-завершения
        idle_polls = 0
        last_progress_key = None

        while True:
            elapsed = time.monotonic() - started_at
            if elapsed > total_timeout:
                raise TimeoutError(f"ASR operation {operation_id} timed out after {elapsed:.1f}s")

            resp = await self._http_get(self._ops_url, params=params, headers=headers)
            body = resp.text
            ctype = (resp.headers.get("content-type") or "").lower()

            # Всегда логируем тело (полностью/хвост), как просил
            _log_body(f"ASR {operation_id} GET body", body)

            # Значения прогресса по умолчанию на этот цикл
            rec_ms = fin_ms = par_ms = eou_ms = 0
            fi_cur = None

            if resp.status_code == 404:
                logger.info("ASR operation %s not ready yet (404), retry in %.1fs", operation_id, interval)
            elif resp.status_code >= 300:
                logger.warning("ASR poll %s: HTTP %s (see previous body log)", operation_id, resp.status_code)
            else:
                # Унифицированный парсинг NDJSON/JSON
                objs = _iter_ndjson(body)
                if not objs:
                    try:
                        one = resp.json()
                        if isinstance(one, dict):
                            objs = [one]
                    except Exception:
                        objs = []

                for obj in objs:
                    carrier = obj.get("result") if isinstance(obj.get("result"), dict) else obj

                    # прогресс-лог по курсорам (если есть)
                    cursors = carrier.get("audioCursors") or {}
                    try:
                        rec_ms = int(cursors.get("receivedDataMs") or 0)
                        fin_ms = int(cursors.get("finalTimeMs") or 0)
                        par_ms = int(cursors.get("partialTimeMs") or 0)
                        eou_ms = int(cursors.get("eouTimeMs") or 0)
                    except Exception:
                        pass
                    fi_cur = (carrier.get("finalRefinement") or {}).get("finalIndex")

                    if any([rec_ms, fin_ms, par_ms, fi_cur is not None]):
                        logger.info(
                            "ASR %s: progress receivedDataMs=%s, finalTimeMs=%s, partialTimeMs=%s, finalIndex=%s",
                            operation_id, rec_ms, fin_ms, par_ms, fi_cur
                        )

                    # ===== 1) finalRefinement.normalizedText (имеет finalIndex) =====
                    final_ref = (carrier.get("finalRefinement") or {})
                    fi = final_ref.get("finalIndex")
                    norm = final_ref.get("normalizedText") or {}
                    alts = norm.get("alternatives") if isinstance(norm, dict) else None
                    if isinstance(fi, int) and isinstance(alts, list) and alts:
                        best_text = ""
                        best_words = None
                        for a in alts:
                            if isinstance(a, dict) and isinstance(a.get("text"), str) and a["text"].strip():
                                best_text = a["text"].strip()
                                best_words = a.get("words")
                                break
                        if best_text:
                            sig = _text_signature(best_text, best_words)
                            prev_sig = sig_by_index.get(fi)
                            if prev_sig != sig:
                                segments[fi] = best_text
                                sig_by_index[fi] = sig
                                seen_signatures.add(sig)
                                total_chars = sum(len(t) for t in segments.values())
                                logger.info(
                                    "ASR %s: received segment(refined) #%s len=%d, tail=%r | total=%d seg, %d chars",
                                    operation_id, fi, len(best_text), _tail_text_preview(best_text, 80),
                                    len(segments), total_chars
                                )

                    # ===== 1b) final.alternatives (может не иметь finalIndex) =====
                    final_blk = carrier.get("final") or {}
                    final_alts = final_blk.get("alternatives")
                    if isinstance(final_alts, list) and final_alts:
                        best_text = ""
                        best_words = None
                        for a in final_alts:
                            if isinstance(a, dict) and isinstance(a.get("text"), str) and a["text"].strip():
                                best_text = a["text"].strip()
                                best_words = a.get("words")
                                break
                        if best_text:
                            sig = _text_signature(best_text, best_words)
                            if sig not in seen_signatures:
                                if sig in idx_by_sig:
                                    use_idx = idx_by_sig[sig]
                                else:
                                    if isinstance(fi, int) and fi not in segments:
                                        use_idx = fi
                                    else:
                                        use_idx = (max(segments.keys()) + 1) if segments else 0
                                    idx_by_sig[sig] = use_idx

                                segments[use_idx] = best_text
                                sig_by_index[use_idx] = sig
                                seen_signatures.add(sig)
                                total_chars = sum(len(t) for t in segments.values())
                                logger.info(
                                    "ASR %s: received segment(final)  #%s len=%d, tail=%r | total=%d seg, %d chars",
                                    operation_id, use_idx, len(best_text), _tail_text_preview(best_text, 80),
                                    len(segments), total_chars
                                )

                    # ===== 2) статус завершения =====
                    status_code = carrier.get("statusCode") or {}
                    message = status_code.get("message") if isinstance(status_code, dict) else None
                    if isinstance(message, str) and message.upper().startswith("COMPLETED"):
                        completed = True
                    # альтернативные признаки завершения
                    if not completed and str(carrier.get("done", "")).lower() in {"true", "1"}:
                        completed = True

            # Эвристика авто-завершения: если eou>=received и опросы повторяют одно и то же
            progress_key = (rec_ms, eou_ms, fin_ms, par_ms, fi_cur, tuple(sorted(segments.items())))
            if not completed:
                stable_final_idx = (fi_cur is not None) and (last_progress_key is not None) and (last_progress_key[4] == fi_cur)
                if last_progress_key == progress_key and eou_ms >= rec_ms and (len(segments) > 0 or stable_final_idx):
                    idle_polls += 1
                else:
                    idle_polls = 0
                last_progress_key = progress_key

                if idle_polls >= ASR_AUTO_FINALIZE_POLL_LIMIT:
                    logger.info(
                        "ASR operation %s: auto-finalize after idle=%d polls (eou=%dms, received=%dms, seg=%d, finalIndex=%s)",
                        operation_id, idle_polls, eou_ms, rec_ms, len(segments), fi_cur
                    )
                    completed = True

            if completed:
                ordered = [segments[k] for k in sorted(segments.keys())]
                total_chars = sum(len(t) for t in ordered)
                logger.info(
                    "ASR operation %s COMPLETED: %d segments, %d chars, %.1fs total. Final tail=%r",
                    operation_id, len(ordered), total_chars, elapsed, _tail_text_preview(" ".join(ordered), 120)
                )
                return ordered

            logger.info("ASR operation %s: waiting %.1fs before next poll", operation_id, interval)
            await asyncio.sleep(interval)
            interval = min(MAX_POLL_INTERVAL_SEC, interval * POLL_BACKOFF_FACTOR)

    # ------- извлечение текста (склейка)

    @staticmethod
    def join_segments(segments: List[str]) -> str:
        text = " ".join(s.strip() for s in segments if isinstance(s, str) and s.strip())
        return " ".join(text.split())

    # ------- высокоуровневые обёртки

    async def transcribe_long_by_bytes(
        self,
        *,
        audio_bytes: bytes,
        content_type: str = "audio/ogg",
        language: Optional[str] = None,
    ) -> str:
        op_id = await self.start_by_bytes(audio_bytes=audio_bytes, content_type=content_type, language=language)
        segments = await self.poll_collect_texts(op_id)
        return self.join_segments(segments)

    # ------- совместимая сигнатура (как у старого кода)

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.bin",                 # совместимость
        content_type: str = "audio/ogg",
        language: Optional[str] = None,
        task: Optional[str] = None,                  # совместимость, не используется
        timeout: Optional[float | httpx.Timeout] = None,  # совместимость, не используется
        **extra_cfg: Any,
    ) -> str:
        _ = filename, task, timeout, extra_cfg
        lang = language or getattr(settings, "SPEECHKIT_LANG", "ru-RU")
        return await self.transcribe_long_by_bytes(
            audio_bytes=audio_bytes, content_type=content_type, language=lang
        )


# ===== глобальная функция-обёртка =====

async def transcribe(
    audio_bytes: bytes,
    *,
    filename: str = "audio.bin",
    content_type: str = "audio/ogg",
    language: Optional[str] = None,
    task: Optional[str] = None,
    timeout: Optional[float | httpx.Timeout] = None,
    **extra_cfg: Any,
) -> str:
    client = ASRClient()
    return await client.transcribe(
        audio_bytes,
        filename=filename,
        content_type=content_type,
        language=language,
        task=task,
        timeout=timeout,
        **extra_cfg,
    )

