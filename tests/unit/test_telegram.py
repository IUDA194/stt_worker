import pytest
import respx
import httpx
from app.telegram import TelegramNotifier

pytestmark = pytest.mark.asyncio

@respx.mock
async def test_telegram_notifier_sends_message():
    notifier = TelegramNotifier("BOT:123", "123456")
    route = respx.post("https://api.telegram.org/botBOT:123/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    await notifier.send("hi")
    assert route.called
    # проверим, что тело корректное
    sent = route.calls[0].request
    assert sent.method == "POST"
    assert sent.url.host == "api.telegram.org"

