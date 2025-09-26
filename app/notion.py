import json
import re
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"

# --- утилиты ---

def _sanitize_title(s: Optional[str], fallback: str = "Без тега") -> str:
    s = (s or "").strip()
    if not s:
        return fallback
    s = re.sub(r"[\r\n\t]", " ", s)
    return s[:200]

def _split_text(s: str, max_len: int) -> list[str]:
    out: list[str] = []
    cur = 0
    n = len(s)
    while cur < n:
        tail = min(cur + max_len, n)
        if tail == n:
            out.append(s[cur:tail])
            break
        cut = s.rfind(" ", cur, tail)
        if cut == -1 or cut <= cur + max_len * 0.6:
            cut = tail
        out.append(s[cur:cut].rstrip())
        cur = cut + 1
    return out

def _redact_token(token: Optional[str]) -> str:
    if not token:
        return "missing"
    if len(token) <= 8:
        return "****"
    return f"{token[:3]}***{token[-4:]}"

def _short(s: str, n: int = 500) -> str:
    return s if len(s) <= n else (s[:n] + f"...({len(s)-n} more)")

@dataclass
class _RetryConfig:
    attempts: int = 3
    backoff_sec: float = 0.8  # экспоненциальный backoff

class NotionHTTPError(Exception):
    pass

class NotionUploader:
    """
    Логирует каждый шаг:
      - параметры окружения (маскированно)
      - поиск/создание страниц
      - HTTP статус, URL, кванты тела (усечённо), rate-limit заголовки
      - детальные тексты ошибок сервера
    """

    def __init__(self, token: str, root_page_id: str, *, debug: bool = False, retry: _RetryConfig = _RetryConfig()):
        if not token:
            raise ValueError("NOTION_API_KEY is empty")
        if not root_page_id:
            raise ValueError("NOTION_ROOT_PAGE_ID is empty")

        self.token = token
        self.root_page_id = root_page_id
        self.debug = debug
        self.retry = retry

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

        logger.info(
            "NotionUploader init: token=%s root_page_id=%s debug=%s attempts=%s backoff=%.2fs",
            _redact_token(self.token),
            f"{self.root_page_id[:6]}...{self.root_page_id[-4:]}",
            self.debug,
            self.retry.attempts,
            self.retry.backoff_sec,
        )

    # ---- низкоуровневый вызов с ретраями и логами ----

    async def _req(self, client: httpx.AsyncClient, method: str, path: str, *, json_body: Optional[dict] = None) -> httpx.Response:
        url = f"{NOTION_API_BASE}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.retry.attempts + 1):
            try:
                if self.debug:
                    logger.debug("Notion request [%s] %s body=%s", method, url, _short(json.dumps(json_body or {}, ensure_ascii=False)))
                t0 = time.monotonic()
                resp = await client.request(method, url, headers=self.headers, json=json_body, timeout=30.0)
                dt = (time.monotonic() - t0) * 1000
                rl = {
                    "rl-remaining": resp.headers.get("x-ratelimit-remaining"),
                    "rl-reset": resp.headers.get("x-ratelimit-reset"),
                }
                logger.info("Notion response [%s] %s -> %s in %.1fms %s", method, url, resp.status_code, dt, rl)

                if resp.status_code >= 400:
                    # попробуем вытащить сообщение
                    text = _short(await _safe_text(resp))
                    logger.warning("Notion HTTP %s for %s: %s", resp.status_code, path, text)
                    # 429/5xx — имеет смысл ретраить
                    if resp.status_code in (429, 502, 503, 504) and attempt < self.retry.attempts:
                        sleep = self.retry.backoff_sec * (2 ** (attempt - 1))
                        logger.warning("Retrying Notion in %.1fs (attempt %s/%s)", sleep, attempt + 1, self.retry.attempts)
                        await _async_sleep(sleep)
                        continue
                    raise NotionHTTPError(f"Notion {resp.status_code}: {text}")
                return resp
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
                last_exc = e
                if attempt < self.retry.attempts:
                    sleep = self.retry.backoff_sec * (2 ** (attempt - 1))
                    logger.warning("Notion timeout/protocol error: %s. Retrying in %.1fs (attempt %s/%s)", e, sleep, attempt + 1, self.retry.attempts)
                    await _async_sleep(sleep)
                    continue
                logger.error("Notion request failed after retries: %s", e)
                raise
        # теоретически не дойдём сюда
        if last_exc:
            raise last_exc  # type: ignore[misc]
        raise RuntimeError("Unknown Notion request loop state")

    # ---- поиски/создание страниц ----

    async def _search_page_by_title_under_parent(
        self, client: httpx.AsyncClient, title: str, parent_page_id: str
    ) -> Optional[str]:
        payload = {
            "query": title,
            "filter": {"value": "page", "property": "object"},
            "sort": {"direction": "descending", "timestamp": "last_edited_time"},
        }
        resp = await self._req(client, "POST", "/search", json_body=payload)
        data = resp.json()
        if self.debug:
            logger.debug("Notion /search results=%s", _short(json.dumps(data, ensure_ascii=False)))
        for res in data.get("results", []):
            if res.get("object") != "page":
                continue
            parent = res.get("parent") or {}
            if parent.get("type") == "page_id" and parent.get("page_id") == parent_page_id:
                props = res.get("properties", {})
                title_prop = props.get("title") or props.get("Name")
                if title_prop and title_prop.get("type") == "title":
                    texts = title_prop.get("title", [])
                    full_title = "".join([t.get("plain_text", "") for t in texts]).strip()
                    if full_title == title:
                        page_id = res.get("id")
                        logger.info("Found tag page '%s' under %s: %s", title, parent_page_id, page_id)
                        return page_id
        logger.info("Tag page '%s' not found under %s", title, parent_page_id)
        return None

    async def _create_page_under_parent(
        self,
        client: httpx.AsyncClient,
        parent_page_id: str,
        title: str,
        children_blocks: Optional[list] = None,
        icon_emoji: Optional[str] = None,
    ) -> dict:
        payload = {
            "parent": {"type": "page_id", "page_id": parent_page_id},
            "properties": {"title": {"title": [{"type": "text", "text": {"content": title}}]}},
        }
        if children_blocks:
            payload["children"] = children_blocks
        if icon_emoji:
            payload["icon"] = {"type": "emoji", "emoji": icon_emoji}
        resp = await self._req(client, "POST", "/pages", json_body=payload)
        data = resp.json()
        logger.info("Created page '%s' under %s -> id=%s url=%s", title, parent_page_id, data.get("id"), data.get("url"))
        if self.debug:
            logger.debug("Created page payload: %s", _short(json.dumps(data, ensure_ascii=False)))
        return data

    async def _get_or_create_tag_page(self, client: httpx.AsyncClient, tag: str) -> str:
        title = _sanitize_title(tag, "Без тега")
        page_id = await self._search_page_by_title_under_parent(client, title, self.root_page_id)
        if page_id:
            return page_id
        created = await self._create_page_under_parent(
            client, parent_page_id=self.root_page_id, title=title, children_blocks=None, icon_emoji="🗂️"
        )
        return created["id"]

    # ---- публичный API ----

    async def create_page(
        self,
        *,
        job_id: str,
        transcript: str,
        user: Optional[str],
        tag: Optional[str],
        s3_url: Optional[str],
    ) -> str:
        logger.info(
            "Create Notion page: job_id=%s user=%s tag=%s s3=%s",
            job_id,
            user,
            tag,
            s3_url,
        )
        now_utc = datetime.now(timezone.utc)
        try:
            from zoneinfo import ZoneInfo
            now_local = now_utc.astimezone(ZoneInfo("Europe/Berlin"))
        except Exception:
            now_local = now_utc

        page_title = f"{now_local.strftime('%Y-%m-%d %H:%M')} — {job_id}"

        meta_lines = []
        if user:
            meta_lines.append(f"Пользователь: {user}")
        if tag:
            meta_lines.append(f"Тег: {tag}")
        if s3_url:
            meta_lines.append(f"S3: {s3_url}")
        meta_text = "\n".join(meta_lines) if meta_lines else " "

        children = [
            {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Транскрипция"}}]}},
            {"object": "block", "type": "callout", "callout": {"icon": {"emoji": "📝"}, "rich_text": [{"type": "text", "text": {"content": meta_text}}]}},
        ]
        if s3_url:
            children.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {"type": "text", "text": {"content": "Исходный файл: "}},
                            {"type": "text", "text": {"content": s3_url, "link": {"url": s3_url}}},
                        ]
                    },
                }
            )

        transcript = (transcript or "").strip()
        if transcript:
            for chunk in _split_text(transcript, 1800):
                children.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]}})

        async with httpx.AsyncClient() as client:
            tag_page_id = await self._get_or_create_tag_page(client, tag or "Без тега")
            created = await self._create_page_under_parent(
                client, parent_page_id=tag_page_id, title=page_title, children_blocks=children, icon_emoji="🎧"
            )
        url = created.get("url", "")
        logger.info("Notion page created: %s", url)
        return url

# --- helpers (async sleep, safe text) ---

async def _async_sleep(sec: float) -> None:
    # без asyncio импортов вверху файла
    import asyncio
    await asyncio.sleep(sec)

async def _safe_text(resp: httpx.Response) -> str:
    try:
        return await resp.aread().decode("utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        try:
            return resp.text
        except Exception:
            return "<unreadable>"

