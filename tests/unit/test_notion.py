import pytest
import pytest_asyncio

from app.notion import NotionUploader


pytestmark = pytest.mark.asyncio


class _DummyPages:
    def __init__(self, bucket: list[dict]):
        self.bucket = bucket

    async def create(self, **kwargs):
        # сохраняем то, что отправили в create, для проверок
        self.bucket.append(kwargs)
        # вернём URL страницы — как в реальном Notion ответе
        return {"url": "https://notion.example/page"}


class _DummyClient:
    def __init__(self, bucket: list[dict]):
        self.pages = _DummyPages(bucket)


@pytest_asyncio.fixture
async def notion_uploader(monkeypatch):
    # создаём uploader и подменяем внутренний AsyncClient на наш dummy
    sent: list[dict] = []
    uploader = NotionUploader(token="x", database_id="db1")
    uploader.client = _DummyClient(sent)  # type: ignore[assignment]
    return uploader, sent


async def test_notion_create_page_success(notion_uploader):
    uploader, sent = notion_uploader
    url = await uploader.create_page(
        title="My Title",
        transcript="Hello world",
        s3_url="s3://bucket/key",
        tag="voice",
        user="alice",
        job_id="job-123",
    )
    assert url == "https://notion.example/page"
    assert len(sent) == 1
    payload = sent[0]
    # Проверяем, что нужные поля уходят
    assert payload["parent"]["database_id"] == "db1"
    props = payload["properties"]
    assert props["Name"]["title"][0]["text"]["content"] == "My Title"
    assert props["JobID"]["rich_text"][0]["text"]["content"] == "job-123"
    assert props["User"]["rich_text"][0]["text"]["content"] == "alice"
    assert props["Tag"]["rich_text"][0]["text"]["content"] == "voice"
    assert props["S3"]["url"] == "s3://bucket/key"
    # и что тело транскрипта попало в блоки
    blocks = payload["children"]
    assert blocks[0]["type"] == "heading_2"
    assert "Transcript" in blocks[0]["heading_2"]["rich_text"][0]["text"]["content"]
    assert "Hello world" in blocks[1]["paragraph"]["rich_text"][0]["text"]["content"]


async def test_notion_create_page_without_url(notion_uploader):
    # если Notion вернёт объект без url — функция должна вернуть пустую строку
    uploader, _ = notion_uploader

    class _NoUrlPages:
        async def create(self, **kwargs):
            return {}

    class _NoUrlClient:
        def __init__(self):
            self.pages = _NoUrlPages()

    uploader.client = _NoUrlClient()  # type: ignore[assignment]
    url = await uploader.create_page(
        title="t", transcript="x", s3_url=None, tag=None, user="u", job_id="id"
    )
    assert url == ""

