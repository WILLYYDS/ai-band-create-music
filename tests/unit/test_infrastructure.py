import logging

from app.infrastructure.cache import NullCache
from app.infrastructure.events import GenerationEvent, NullEventPublisher
from app.infrastructure.logging import configure_logging
from app.infrastructure.queue import InlineTaskDispatcher


async def test_local_infrastructure_needs_no_external_service() -> None:
    cache = NullCache()
    await cache.set("key", b"value", 60)
    assert await cache.get("key") is None
    await cache.delete("key")

    await NullEventPublisher().publish(GenerationEvent("test", "job", "request"))
    result = await InlineTaskDispatcher().submit("job", lambda: _answer())
    assert result == 42


async def _answer() -> int:
    return 42


def test_logging_suppresses_httpx_info_and_allows_explicit_debug(monkeypatch, caplog):
    httpx_logger = logging.getLogger("httpx")
    monkeypatch.setattr(httpx_logger, "level", httpx_logger.level)
    caplog.set_level(logging.INFO)
    configure_logging()
    logging.getLogger("app").info("application event")
    httpx_logger.info("outbound request")
    httpx_logger.warning("network warning")
    assert [record.message for record in caplog.records] == [
        "application event",
        "network warning",
    ]
    configure_logging(httpx_level=logging.DEBUG)
    assert httpx_logger.isEnabledFor(logging.INFO)
