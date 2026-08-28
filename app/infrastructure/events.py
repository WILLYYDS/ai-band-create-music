from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    name: str
    job_id: str
    request_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EventPublisher(Protocol):
    async def publish(self, event: GenerationEvent) -> None: ...


class NullEventPublisher:
    """Reserved hook for a future broker-backed event publisher."""

    async def publish(self, event: GenerationEvent) -> None:
        return None
