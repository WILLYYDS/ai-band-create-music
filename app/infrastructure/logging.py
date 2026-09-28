from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LogContext:
    request_id: str
    job_id: str | None = None
    provider: str | None = None


def configure_logging(level: int = logging.INFO) -> None:
    """Small local default; handlers can be replaced by structured logging later."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def log_extra(context: LogContext, **values: object) -> dict[str, object]:
    return {
        "request_id": context.request_id,
        "job_id": context.job_id,
        "provider": context.provider,
        **values,
    }
