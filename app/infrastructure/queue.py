from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Protocol, TypeVar

T = TypeVar("T")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskDispatcher(Protocol):
    async def submit(self, job_id: str, task: Callable[[], Awaitable[T]]) -> T: ...


class InlineTaskDispatcher:
    """Executes locally while preserving the seam for a future durable queue."""

    async def submit(self, job_id: str, task: Callable[[], Awaitable[T]]) -> T:
        return await task()
