from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str
    durationMinutes: int | float | str | None = None
    # Exact short length (e.g. 30 s previews); takes precedence over durationMinutes.
    durationSeconds: int | None = Field(default=None, ge=10, le=360)
    # Display name chosen in the group chat; shown by the history page instead of the lyrics.
    title: str | None = Field(default=None, max_length=100)
    provider: Literal["minimax_music", "elevenlabs_music"] | None = None
    count: Literal[1, 2] = 1


class ErrorResponse(BaseModel):
    success: bool = False
    message: str


class PlaybackUrls(BaseModel):
    fullTrack: str | None = Field(default=None, exclude_if=lambda value: value is None)
    replacedVocal: str | None = Field(default=None, exclude_if=lambda value: value is None)
    mixedTrack: str | None = Field(default=None, exclude_if=lambda value: value is None)
    stems: dict[str, str] | None = Field(default=None, exclude_if=lambda value: value is None)


class MusicOutput(BaseModel):
    fullTrack: str
    playback: PlaybackUrls | None = Field(default=None, exclude_if=lambda value: value is None)
    mixedTrack: str | None = Field(default=None, exclude_if=lambda value: value is None)
    replacedVocal: str | None = Field(default=None, exclude_if=lambda value: value is None)
    durationSeconds: float | None = None
    stems: dict[str, str]
    stemUrls: list[str]
    waveforms: dict[str, list[float]]
    splitEnabled: bool
    debug: dict[str, Any]


class GenerateResponse(MusicOutput):
    success: bool = True
    jobId: str
    createdAt: str | None = None
    provider: str | None = None
    requestedCount: int | None = None
    requestedDurationSeconds: float | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    prompt: str
    durationMinutes: int | float | Literal["auto"]
    structuredPrompt: str
    lyrics: str
    count: Literal[1, 2] = 1
    alternatives: list[MusicOutput] = Field(default_factory=list)
    # Backward-compatible response field used by released clients.
    warning: str | None = None


class CreateGenerationJobResponse(BaseModel):
    jobId: str
    status: Literal["pending", "running"]


class UpdateGenerationJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["cancelled"]


class GenerationJobResponse(BaseModel):
    jobId: str
    createdAt: str
    title: str | None = None
    prompt: str
    structuredPrompt: str | None = None
    lyrics: str | None = None
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    stage: str
    progress: int | None = None
    step: int | None = None
    totalSteps: int | None = None
    message: str
    # Backward-compatible response field used by released clients.
    warning: str | None = None
    result: GenerateResponse | None = None
    error: str | None = None
    splitStatus: Literal["pending", "running", "succeeded", "failed", "cancelled"] | None = None
    splitSong: int | None = None
    splitError: str | None = None
    replaceStatus: Literal["pending", "running", "succeeded", "failed", "cancelled"] | None = None
    replaceSong: int | None = None
    replaceError: str | None = None
    mixStatus: Literal["pending", "running", "succeeded", "failed", "cancelled"] | None = None
    mixSong: int | None = None
    mixError: str | None = None
