from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str
    durationMinutes: int | float | str | None = None
    provider: Literal["minimax_music", "elevenlabs_music"] | None = None
    count: Literal[1, 2] = 1


class ErrorResponse(BaseModel):
    success: bool = False
    message: str


class MusicOutput(BaseModel):
    fullTrack: str
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
    requestedDurationSeconds: float | None = None
    prompt: str
    durationMinutes: int | Literal["auto"]
    structuredPrompt: str
    lyrics: str
    count: Literal[1, 2] = 1
    alternatives: list[MusicOutput] = Field(default_factory=list)
    warning: str | None = None


class CreateGenerationJobResponse(BaseModel):
    jobId: str
    status: Literal["pending", "running"]


class UpdateGenerationJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["cancelled"]


class GenerationJobResponse(BaseModel):
    jobId: str
    prompt: str
    structuredPrompt: str | None = None
    lyrics: str | None = None
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    stage: str
    progress: int
    message: str
    warning: str | None = None
    result: GenerateResponse | None = None
    error: str | None = None
