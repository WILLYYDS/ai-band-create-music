class GenerationError(RuntimeError):
    """A user-facing failure in the music generation pipeline."""


class CapacityExceededError(GenerationError):
    """Raised when all local generation slots are occupied."""
