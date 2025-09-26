import pytest
import respx
import httpx

from app.asr_client import ASRClient

pytestmark = pytest.mark.asyncio


@respx.mock
async def test_asr_client_transcribe_ok():
    client = ASRClient("http://asr.example.com")

    route = respx.post("http://asr.example.com/asr").mock(
        return_value=httpx.Response(200, json={"text": "hello world"})
    )

    out = await client.transcribe(b"FAKE_BYTES", filename="a.ogg", language="ru", task="transcribe")
    assert out == "hello world"
    assert route.called


@respx.mock
async def test_asr_client_bad_payload_raises():
    client = ASRClient("http://asr.example.com")
    respx.post("http://asr.example.com/asr").mock(
        return_value=httpx.Response(200, json={"foo": "bar"})
    )
    with pytest.raises(ValueError):
        await client.transcribe(b"...", filename="a.ogg")


@respx.mock
async def test_asr_client_plain_text_response():
    client = ASRClient("http://asr.example.com")

    respx.post("http://asr.example.com/asr").mock(
        return_value=httpx.Response(
            200,
            text="транскрипция",
            headers={"content-type": "text/plain; charset=utf-8"},
        )
    )

    out = await client.transcribe(b"FAKE_BYTES", filename="a.ogg")
    assert out == "транскрипция"
