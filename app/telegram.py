import httpx
from typing import Optional

class TelegramNotifier:
    def __init__(self, bot_token: str, default_chat_id: Optional[str] = None):
        self.bot_token = bot_token
        self.default_chat_id = default_chat_id

    async def send(self, text: str, chat_id: Optional[str] = None) -> None:
        target_chat_id = chat_id or self.default_chat_id
        if not target_chat_id:
            raise ValueError("Telegram chat_id is not provided")

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": target_chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()

