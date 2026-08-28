from __future__ import annotations

from typing import Protocol


class CacheBackend(Protocol):
    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None: ...

    async def delete(self, key: str) -> None: ...


class NullCache:
    """Local default. It deliberately stores nothing and needs no Redis service."""

    async def get(self, key: str) -> bytes | None:
        return None

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        return None

    async def delete(self, key: str) -> None:
        return None
