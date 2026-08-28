from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str
    durationMinutes: int | float | str | None = None


class ErrorResponse(BaseModel):
    success: bool = False
    message: str


class GenerateResponse(BaseModel):
    success: bool = True
    jobId: str
    prompt: str
    durationMinutes: int
    structuredPrompt: str
    fullTrack: str
    stems: dict[str, str]
    stemUrls: list[str]
    waveforms: dict[str, list[float]]
    splitEnabled: bool
    debug: dict[str, Any]


class CreateGenerationJobResponse(BaseModel):
    jobId: str
    status: Literal["pending", "running"]


class UpdateGenerationJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["cancelled"]


class GenerationJobResponse(BaseModel):
    jobId: str
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    stage: str
    progress: int
    message: str
    result: GenerateResponse | None = None
    error: str | None = None
