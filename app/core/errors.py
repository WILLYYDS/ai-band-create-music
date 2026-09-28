import os
from json import JSONDecodeError


class UnsupportedOutputLayoutError(ValueError):
    """Cleanup cannot safely traverse symbolic links in the output tree."""


def public_error_reason(exc: Exception, *, safe_message: str | None = None) -> str:
    """Describe failures without exposing paths or exception input."""
    reason = type(exc).__name__
    if isinstance(exc, OSError) and exc.errno is not None:
        reason += f": errno={exc.errno} ({os.strerror(exc.errno)})"
    elif isinstance(exc, JSONDecodeError):
        reason += f": {exc.msg} (line {exc.lineno} column {exc.colno})"
    elif isinstance(exc, UnsupportedOutputLayoutError):
        reason += ": refusing symlinked directory"
    elif safe_message is not None:
        reason += f": {safe_message}"
    return reason


class GenerationError(RuntimeError):
    """A user-facing failure in the music generation pipeline."""


class CapacityExceededError(GenerationError):
    """Raised when all local generation slots are occupied."""
