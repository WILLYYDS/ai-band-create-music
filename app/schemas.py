from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer


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

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None


class ErrorResponse(BaseModel):
    success: bool = False
    message: str


class PlaybackUrls(BaseModel):
    fullTrack: str | None = Field(default=None, exclude_if=lambda value: value is None)
    replacedVocal: str | None = Field(default=None, exclude_if=lambda value: value is None)
    mixedTrack: str | None = Field(default=None, exclude_if=lambda value: value is None)
    stems: dict[str, str] | None = Field(default=None, exclude_if=lambda value: value is None)


class MusicOutput(BaseModel):
    songNumber: int | None = Field(default=None, ge=1, le=2, exclude_if=lambda value: value is None)
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
    title: str | None = None
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


class SongGenerationState(BaseModel):
    songNumber: int = Field(ge=1, le=2)
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    stage: str
    message: str = ""
    progress: int | None = Field(default=None, ge=0, le=100)
    step: int | None = None
    totalSteps: int | None = None
    receivedAudioSeconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    expectedAudioSeconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    error: str | None = None


class GenerationJobResponse(BaseModel):
    @model_serializer(mode="wrap")
    def serialize_retention(self, handler):
        response = handler(self)
        if self.retentionState == "disabled":
            for name in ("expiresAt", "retentionState", "audioAvailable"):
                response.pop(name, None)
        return response

    jobId: str
    createdAt: str
    expiresAt: str | None = None
    retentionState: Literal[
        "disabled", "dry_run", "unmanaged", "retained", "expired_active", "expired"
    ] = "disabled"
    audioAvailable: bool = False
    title: str | None = None
    prompt: str
    structuredPrompt: str | None = None
    lyrics: str | None = None
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    stage: str
    progress: int | None = None
    step: int | None = None
    totalSteps: int | None = None
    receivedAudioSeconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    expectedAudioSeconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    currentSong: int | None = Field(default=None, ge=1, le=2)
    songStates: list[SongGenerationState] = Field(default_factory=list)
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
