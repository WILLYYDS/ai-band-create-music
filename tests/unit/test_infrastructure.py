from app.infrastructure.cache import NullCache
from app.infrastructure.events import GenerationEvent, NullEventPublisher
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
