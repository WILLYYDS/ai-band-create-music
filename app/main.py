from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import Settings
from app.core.errors import CapacityExceededError, GenerationError
from app.infrastructure.cache import NullCache
from app.infrastructure.events import NullEventPublisher
from app.infrastructure.logging import configure_logging
from app.infrastructure.queue import InlineTaskDispatcher
from app.schemas import (
    CreateGenerationJobResponse,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    GenerationJobResponse,
    UpdateGenerationJobRequest,
)
from app.services.audio_files import build_public_audio_url, detect_audio_content_type
from app.services.job_files import read_job_diagnostics
from app.services.orchestrator import GenerationOrchestrator
from app.services.prompt import OpenAICompatiblePromptExpander, effective_llm_output_tokens
from app.services.providers import create_music_provider
from app.services.stems import DemucsStemSeparator
from app.services.voice import RVCEngine, install_voice_api
from app.services.waveforms import extract_waveforms

logger = logging.getLogger(__name__)


class RequestSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, default_limit: int, voice_limit: int) -> None:
        self.app = app
        self.default_limit = default_limit
        self.voice_limit = voice_limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self.voice_limit if scope["path"] == "/api/voice/convert" else self.default_limit
        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
                if declared_size < 0:
                    raise ValueError
            except ValueError:
                await JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"success": False, "message": "Content-Length 无效。"},
                )(scope, receive, send)
                return
            if declared_size > limit:
                await JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={"success": False, "message": "请求体过大。"},
                )(scope, receive, send)
                return

        received_size = 0
        too_large = False

        async def limited_receive():
            nonlocal received_size, too_large
            if too_large:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received_size += len(message.get("body", b""))
                if received_size > limit:
                    too_large = True
                    return {"type": "http.disconnect"}
            return message

        async def limited_send(message):
            if not too_large:
                await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except Exception:
            if not too_large:
                raise
        if too_large:
            await JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content={"success": False, "message": "请求体过大。"},
            )(scope, receive, send)


@dataclass(slots=True)
class GenerationJob:
    job_id: str
    prompt: str
    structured_prompt: str | None = None
    lyrics: str | None = None
    status: str = "pending"
    stage: str = "pending"
    progress: int | None = None
    step: int | None = None
    total_steps: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message: str = "任务已创建"
    warning: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    task: asyncio.Task[None] | None = None
    deleted_stems: dict[str, dict[str, Any]] = field(default_factory=dict)

    def response(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "createdAt": self.created_at,
            "step": self.step,
            "totalSteps": self.total_steps,
            "prompt": self.prompt,
            "structuredPrompt": self.structured_prompt,
            "lyrics": self.lyrics,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "warning": self.warning,
            "result": self.result,
            "error": self.error,
        }

    def save(self, output_dir: Path) -> None:
        target = output_dir / "jobs" / self.job_id / "job.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"job.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                payload = {**self.response(), "deletedStems": self.deleted_stems}
                diagnostics = read_job_diagnostics(output_dir, self.job_id)
                if diagnostics:
                    payload["diagnostics"] = diagnostics
                json.dump(payload, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def load_jobs(output_dir: Path) -> dict[str, GenerationJob]:
    jobs = {}
    for target in (output_dir / "jobs").glob("*/job.json"):
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            validated = GenerationJobResponse.model_validate(data)
            if validated.jobId != target.parent.name:
                raise ValueError("jobId does not match directory")
            job = GenerationJob(
                job_id=validated.jobId,
                prompt=validated.prompt,
                created_at=validated.createdAt,
                structured_prompt=validated.structuredPrompt,
                lyrics=validated.lyrics,
                status=validated.status,
                stage=validated.stage,
                progress=validated.progress,
                step=validated.step,
                total_steps=validated.totalSteps,
                message=validated.message,
                warning=validated.warning,
                error=validated.error,
                result=data.get("result"),
                deleted_stems=data.get("deletedStems", {}),
            )
            if job.status in {"pending", "running"}:
                job.status = job.stage = "failed"
                job.progress = job.step = job.total_steps = None
                job.error = job.message = "服务器重启，生成任务已中断。"
                job.save(output_dir)
            jobs[job.job_id] = job
        except (OSError, ValueError, TypeError):
            logger.exception("Unable to load job metadata: %s", target)
    return jobs


async def import_legacy_songs(
    jobs: dict[str, GenerationJob], settings: Settings, candidates: list[Path]
) -> None:
    represented = set()
    fingerprints = set()
    legacy_prompts = {}
    for prompt_file in (settings.output_dir / "jobs").glob("*/prompts.json"):
        try:
            metadata = json.loads(prompt_file.read_text(encoding="utf-8"))
            if isinstance(metadata, dict):
                for stem in prompt_file.parent.rglob("full_song_*"):
                    legacy_prompts[stem.stem.rsplit("_", 1)[0]] = metadata
                legacy_prompts[prompt_file.parent.name] = metadata
        except (OSError, ValueError):
            logger.warning("Unable to read legacy prompts: %s", prompt_file)

    def fingerprint(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    for job in jobs.values():
        if job.result:
            for result in [job.result, *job.result.get("alternatives", [])]:
                try:
                    path = _output_path_from_url(result["fullTrack"], settings)
                    represented.add(path)
                    fingerprints.add(fingerprint(path))
                except (KeyError, ValueError, OSError):
                    pass
    for path in candidates:
        if path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            continue
        job_id = "legacy_" + hashlib.sha256(path.name.encode()).hexdigest()[:20]
        if path.resolve() in represented or job_id in jobs or not path.is_file():
            continue
        try:
            digest = fingerprint(path)
        except OSError:
            logger.exception("Unable to read legacy audio: %s", path)
            continue
        if digest in fingerprints or path.stat().st_size == 0:
            continue
        created = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        metadata = legacy_prompts.get(path.stem, {})
        if not metadata and path.stem.startswith("full_song_job_"):
            metadata = legacy_prompts.get(
                path.stem.removeprefix("full_song_").rsplit("_", 1)[0], {}
            )
        job = GenerationJob(
            job_id=job_id,
            prompt=metadata.get("prompt") or path.stem,
            structured_prompt=metadata.get("structuredPrompt"),
            lyrics=metadata.get("lyrics"),
            created_at=created,
            status="succeeded",
            stage="completed",
            progress=100,
            message="已导入历史完整歌曲",
        )
        job.result = {
            "success": True,
            "jobId": job_id,
            "createdAt": created,
            "prompt": job.prompt,
            "structuredPrompt": job.structured_prompt or "",
            "lyrics": job.lyrics or "",
            "durationMinutes": "auto",
            "count": 1,
            "alternatives": [],
            "fullTrack": path.relative_to(settings.output_dir).as_posix(),
            "stems": {},
            "stemUrls": [],
            "waveforms": await extract_waveforms({"full": path}),
            "splitEnabled": False,
            "debug": {"imported": True},
        }
        job.save(settings.output_dir)
        jobs[job_id] = job
        fingerprints.add(digest)


def _song_result(job: GenerationJob, song: int) -> dict[str, Any] | None:
    if job.result is None or song < 0:
        return None
    if song == 0:
        return job.result
    alternatives = job.result.get("alternatives")
    if not isinstance(alternatives, list) or song > len(alternatives):
        return None
    result = alternatives[song - 1]
    return result if isinstance(result, dict) else None


def build_orchestrator(
    settings: Settings,
    client: httpx.AsyncClient,
    direct_client: httpx.AsyncClient | None = None,
) -> GenerationOrchestrator:
    # Instantiate the reserved local cache so the wiring point stays explicit. The
    # current pipeline intentionally performs no cache reads or writes.
    NullCache()
    music_providers = {
        name: create_music_provider(settings, client, direct_client, name)
        for name in ("minimax_music", "elevenlabs_music")
    }
    return GenerationOrchestrator(
        settings=settings,
        prompt_expander=OpenAICompatiblePromptExpander(settings, client),
        music_provider=music_providers.get(settings.music_provider)
        or create_music_provider(settings, client, direct_client),
        stem_separator=DemucsStemSeparator(settings),
        task_dispatcher=InlineTaskDispatcher(),
        events=NullEventPublisher(),
        music_providers=music_providers,
    )


_SUPPORTED_PROXY_SCHEMES = frozenset({"http", "https", "socks5", "socks5h"})


def _sanitize_proxy_environment() -> None:
    """Make proxy environment variables consumable by httpx.

    httpx only accepts ``http``/``https``/``socks5``/``socks5h`` proxy schemes and
    raises ``ValueError`` at client construction time for anything else. Real
    machines often export proxy variables with schemes httpx rejects, e.g. the
    curl-style ``socks://host:port`` alias (urllib can surface it from either the
    uppercase or lowercase ``*_proxy`` variable, whichever wins iteration order)
    or ``socks4://``. Rewrite ``socks://`` to ``socks5://`` and drop variables
    whose scheme httpx cannot use, so the application starts successfully whether
    or not a proxy is configured.
    """
    for name in list(os.environ):
        if not name.lower().endswith("_proxy"):
            continue
        value = os.environ[name]
        if not value:
            continue
        lowered = value.lower()
        if lowered.startswith("socks://"):
            os.environ[name] = "socks5://" + value[len("socks://") :]
        elif "://" in lowered:
            scheme = lowered.split("://", 1)[0]
            if scheme not in _SUPPORTED_PROXY_SCHEMES:
                del os.environ[name]


def create_app(
    settings: Settings | None = None,
    orchestrator: GenerationOrchestrator | None = None,
    voice_engine: RVCEngine | None = None,
) -> FastAPI:
    application_settings = settings or Settings()
    application_settings.output_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if orchestrator is not None:
            application.state.orchestrator = orchestrator
            yield
            return
        # Tolerate proxy environments httpx would otherwise reject at client
        # construction (e.g. `socks://...` exported by common proxy clients).
        _sanitize_proxy_environment()
        async with (
            httpx.AsyncClient() as client,
            httpx.AsyncClient(trust_env=False) as direct_client,
        ):
            application.state.orchestrator = build_orchestrator(
                application_settings, client, direct_client
            )
            yield

    application = FastAPI(
        title="AI Band Music Generation API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    application.state.settings = application_settings
    application.state.jobs = load_jobs(application_settings.output_dir)
    application.state.job_subscribers = {}
    history_lock = asyncio.Lock()
    legacy_imported = False
    legacy_candidates = sorted(application_settings.output_dir.glob("full_song_*"))
    install_voice_api(application, application_settings, voice_engine)
    if orchestrator is not None:
        application.state.orchestrator = orchestrator

    cors_origins = application_settings.cors_origin_list
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials="*" not in cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(
        RequestSizeLimitMiddleware,
        default_limit=application_settings.request_max_bytes,
        voice_limit=application_settings.rvc_max_upload_bytes + 1024 * 1024,
    )

    @application.get("/api/health")
    async def health(request: Request) -> dict[str, object]:
        active_orchestrator = _orchestrator(request)
        base_url = _public_base_url(request, application_settings)
        return {
            "ok": True,
            "service": "AI Band System Backend",
            "outputUrl": f"{base_url}/output",
            "maxConcurrentGenerations": active_orchestrator.capacity.maximum,
            "activeGenerationCount": active_orchestrator.capacity.active,
            "duration": {
                "defaultMinutes": application_settings.default_duration_minutes,
                "minMinutes": application_settings.min_duration_minutes,
                "maxMinutes": application_settings.max_duration_minutes,
            },
            "llm": {
                "model": application_settings.llm_model,
                "timeoutSeconds": application_settings.llm_timeout_seconds,
                "maxTokens": application_settings.llm_max_tokens,
                "initialMaxTokens": effective_llm_output_tokens(
                    application_settings.llm_max_tokens, strict=False
                ),
                "retryMaxTokens": effective_llm_output_tokens(
                    application_settings.llm_max_tokens, strict=True
                ),
                "disableThinking": application_settings.llm_disable_thinking,
            },
            "elevenLabs": {
                "modelId": application_settings.elevenlabs_music_model_id,
                "outputFormat": application_settings.elevenlabs_music_output_format,
                "useCompositionPlan": application_settings.elevenlabs_use_composition_plan,
                "clearChineseVocalMode": application_settings.elevenlabs_clear_chinese_vocal_mode,
            },
            "splitting": {
                "enabled": False,
                "pythonCommand": sys.executable,
                "profile": application_settings.split_profile,
                "model": application_settings.demucs_model or None,
                "device": application_settings.demucs_device or "auto",
                "jobs": application_settings.demucs_jobs or None,
                "segment": application_settings.demucs_segment or None,
            },
            "infrastructure": {
                "taskBackend": application_settings.task_backend,
                "cacheBackend": application_settings.cache_backend,
                "eventBackend": application_settings.event_backend,
            },
        }

    @application.post(
        "/api/generate",
        response_model=GenerateResponse,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def generate(payload: GenerateRequest, request: Request):
        try:
            prompt, duration = _generation_parameters(payload, application_settings)
        except ValueError as exc:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "message": str(exc)},
            )
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        job = GenerationJob(job_id=f"job_{uuid4().hex}", prompt=prompt)
        job.save(application_settings.output_dir)
        request.app.state.jobs[job.job_id] = job

        async def report(stage, progress, message, structured, lyrics, step, total):
            job.status = "running"
            job.stage, job.progress, job.message = stage, progress, message
            job.step, job.total_steps = step, total
            if structured is not None:
                job.structured_prompt = structured
            if lyrics is not None:
                job.lyrics = lyrics
            job.save(application_settings.output_dir)

        try:
            result = await _orchestrator(request).generate(
                prompt,
                duration,
                request_id,
                job_id=job.job_id,
                progress=report,
                provider=payload.provider,
                count=payload.count,
            )
            job.result = result
            job.status, job.stage, job.progress = "succeeded", "completed", 100
            job.message = "音乐生成完成"
            result["createdAt"] = job.created_at
            job.save(application_settings.output_dir)
            request.app.state.jobs[job.job_id] = job
            return _render_result_urls(
                result, _public_base_url(request, application_settings), application_settings
            )
        except HTTPException as exc:
            job.error = str(exc)
            raise
        except CapacityExceededError as exc:
            job.error = str(exc)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"success": False, "message": str(exc)},
            )
        except GenerationError as exc:
            job.error = str(exc)
            logger.exception("generation failed request_id=%s", request_id)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "message": str(exc)},
            )
        except Exception as exc:
            job.error = str(exc)
            logger.exception("unexpected generation failure request_id=%s", request_id)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "message": "生成失败，请查看后端日志。"},
            )
        finally:
            if job.result is None:
                job.status = job.stage = "failed"
                job.progress = None
                job.error = job.error or "生成任务已中断。"
                job.message = job.error
            job.step = job.total_steps = None
            job.save(application_settings.output_dir)

    @application.post(
        "/api/jobs",
        response_model=CreateGenerationJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        responses={400: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    )
    async def create_generation_job(payload: GenerateRequest, request: Request):
        try:
            prompt, duration = _generation_parameters(payload, application_settings)
        except ValueError as exc:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "message": str(exc)},
            )

        active_orchestrator = _orchestrator(request)
        try:
            await active_orchestrator.capacity.acquire()
        except CapacityExceededError:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"success": False, "message": "当前生成任务过多，请稍后重试。"},
            )

        jobs: dict[str, GenerationJob] = request.app.state.jobs
        job_id = f"job_{int(asyncio.get_running_loop().time() * 1000)}_{uuid4().hex[:8]}"
        selected_provider = payload.provider or application_settings.music_provider
        warning = (
            "ElevenLabs Music 暂不支持单次生成两首，已按一首生成。"
            if payload.count == 2 and selected_provider == "elevenlabs_music"
            else None
        )
        job = GenerationJob(job_id=job_id, prompt=prompt, warning=warning)

        def publish() -> None:
            for queue in request.app.state.job_subscribers.get(job_id, ()):
                if queue.empty():
                    queue.put_nowait(None)

        def save_and_publish() -> None:
            job.save(application_settings.output_dir)
            publish()

        try:
            job.save(application_settings.output_dir)
        except Exception:
            await active_orchestrator.capacity.release()
            raise
        jobs[job_id] = job
        request_id = request.headers.get("X-Request-ID") or uuid4().hex

        async def report(
            stage: str,
            progress: int | None,
            message: str,
            structured_prompt: str | None,
            lyrics: str | None,
            step: int | None = None,
            total_steps: int | None = None,
        ) -> None:
            job.status = "running"
            job.stage = stage
            job.progress = progress
            job.step = step
            job.total_steps = total_steps
            job.message = message
            if structured_prompt is not None:
                job.structured_prompt = structured_prompt
            if lyrics is not None:
                job.lyrics = lyrics
            save_and_publish()

        async def execute() -> None:
            try:
                job.status = "running"
                job.message = "服务器正在生成音乐"
                save_and_publish()
                job.result = await active_orchestrator.generate(
                    prompt,
                    duration,
                    request_id,
                    job_id=job_id,
                    progress=report,
                    capacity_reserved=True,
                    provider=payload.provider,
                    count=payload.count,
                )
                job.status = "succeeded"
                job.result["createdAt"] = job.created_at
                job.stage = "completed"
                job.progress = 100
                job.message = "音乐生成完成"
            except asyncio.CancelledError:
                job.status = "cancelled"
                job.stage = "cancelled"
                job.message = "任务已取消"
            except Exception as exc:
                logger.exception("background generation failed job_id=%s", job_id)
                job.status = "failed"
                job.stage = "failed"
                job.message = "音乐生成失败"
                job.error = str(exc)
            finally:
                job.step = job.total_steps = None
                if job.status != "succeeded":
                    job.progress = None
                await active_orchestrator.capacity.release()
                save_and_publish()

        job.task = asyncio.create_task(execute(), name=job_id)
        return {"jobId": job_id, "status": job.status}

    @application.get("/api/jobs")
    async def generation_history(request: Request):
        nonlocal legacy_imported
        async with history_lock:
            if not legacy_imported:
                await import_legacy_songs(
                    request.app.state.jobs,
                    application_settings,
                    legacy_candidates,
                )
                legacy_imported = True
        return {
            "jobs": [
                _job_response(job, request, application_settings)
                for job in sorted(
                    request.app.state.jobs.values(), key=lambda job: job.created_at, reverse=True
                )
            ]
        }

    @application.get("/api/jobs/{job_id}/events")
    async def generation_job_events(job_id: str, request: Request):
        job = request.app.state.jobs.get(job_id)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )

        async def events() -> AsyncIterator[str]:
            queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
            subscribers = request.app.state.job_subscribers.setdefault(job_id, set())
            subscribers.add(queue)
            try:
                while True:
                    payload = _job_response(job, request, application_settings)
                    if job.status not in {"pending", "running"}:
                        yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        return
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    try:
                        await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                subscribers.discard(queue)
                if not subscribers:
                    request.app.state.job_subscribers.pop(job_id, None)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.get(
        "/api/jobs/{job_id}",
        response_model=GenerationJobResponse,
        responses={404: {"model": ErrorResponse}},
    )
    async def generation_job(job_id: str, request: Request):
        job = request.app.state.jobs.get(job_id)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )
        return _job_response(job, request, application_settings)

    @application.patch(
        "/api/jobs/{job_id}",
        response_model=GenerationJobResponse,
        responses={404: {"model": ErrorResponse}},
    )
    async def update_generation_job(
        job_id: str,
        payload: UpdateGenerationJobRequest,
        request: Request,
    ):
        job = request.app.state.jobs.get(job_id)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )
        if job.task is not None and not job.task.done():
            job.task.cancel()
            job.status = "cancelled"
            job.stage = "cancelled"
            job.message = "任务已取消"
            job.progress = job.step = job.total_steps = None
            job.save(application_settings.output_dir)
            for queue in request.app.state.job_subscribers.get(job_id, ()):
                if queue.empty():
                    queue.put_nowait(None)
        return _job_response(job, request, application_settings)

    @application.delete(
        "/api/jobs/{job_id}/stems/{stem_name}",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    async def delete_generation_stem(job_id: str, stem_name: str, request: Request, song: int = 0):
        job = request.app.state.jobs.get(job_id)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )
        result = _song_result(job, song)
        if job.status != "succeeded" or result is None:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "生成任务尚未完成，无法删除音轨。"},
            )
        if not result.get("splitEnabled"):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "完整混音不能作为分轨删除。"},
            )

        stems = result.get("stems", {})
        stem_url = stems.get(stem_name)
        if not stem_url:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "音轨不存在或已被删除。"},
            )
        try:
            target = _output_path_from_url(stem_url, application_settings)
            if not target.is_file():
                raise OSError("stem file is missing")
            trash = _stem_trash_path(target, application_settings, job_id, song)
            trash.parent.mkdir(parents=True, exist_ok=True)
            trash.unlink(missing_ok=True)
            target.replace(trash)
        except (OSError, ValueError):
            logger.exception("failed to delete stem job_id=%s stem=%s", job_id, stem_name)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "message": "删除音轨文件失败。"},
            )
        stem_index = list(stems).index(stem_name)
        waveforms = result.get("waveforms")
        deleted_key = f"{song}:{stem_name}"
        job.deleted_stems[deleted_key] = {
            "url": stem_url,
            "index": stem_index,
            "waveform": waveforms.get(stem_name) if isinstance(waveforms, dict) else None,
        }
        del stems[stem_name]
        result["stemUrls"] = list(stems.values())
        if isinstance(waveforms, dict):
            waveforms.pop(stem_name, None)
        job.message = f"音轨 {stem_name} 已删除"
        job.save(application_settings.output_dir)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.put(
        "/api/jobs/{job_id}/stems/{stem_name}",
        response_model=GenerationJobResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    async def restore_generation_stem(job_id: str, stem_name: str, request: Request, song: int = 0):
        job = request.app.state.jobs.get(job_id)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )
        result = _song_result(job, song)
        if job.status != "succeeded" or result is None:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "生成任务尚未完成，无法恢复音轨。"},
            )

        stems = result.get("stems", {})
        if stem_name in stems:
            return _job_response(job, request, application_settings)
        deleted_key = f"{song}:{stem_name}"
        deleted = job.deleted_stems.get(deleted_key)
        if deleted is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "没有可恢复的音轨。"},
            )

        try:
            target = _output_path_from_url(deleted["url"], application_settings)
            trash = _stem_trash_path(target, application_settings, job_id, song)
            if not trash.is_file():
                del job.deleted_stems[deleted_key]
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"success": False, "message": "删除的音轨文件已不存在，无法恢复。"},
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            trash.replace(target)
        except (OSError, ValueError):
            logger.exception("failed to restore stem job_id=%s stem=%s", job_id, stem_name)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "message": "恢复音轨文件失败。"},
            )

        items = list(stems.items())
        items.insert(min(deleted["index"], len(items)), (stem_name, deleted["url"]))
        result["stems"] = dict(items)
        result["stemUrls"] = list(result["stems"].values())
        waveforms = result.get("waveforms")
        if isinstance(waveforms, dict) and deleted["waveform"] is not None:
            waveforms[stem_name] = deleted["waveform"]
        del job.deleted_stems[deleted_key]
        job.message = f"音轨 {stem_name} 已恢复"
        job.save(application_settings.output_dir)
        return _job_response(job, request, application_settings)

    @application.get("/output/{file_path:path}")
    async def audio_file(file_path: str, request: Request) -> StreamingResponse:
        root = application_settings.output_dir.resolve()
        target = (root / file_path).resolve()
        try:
            relative = target.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Audio file not found") from exc
        if ".trash" in relative.parts or relative.parts[:1] == ("rvc",):
            raise HTTPException(status_code=404, detail="Audio file not found")
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Audio file not found")
        file_size = target.stat().st_size
        start, end = _parse_byte_range(request.headers.get("range"), file_size)
        response_status = status.HTTP_206_PARTIAL_CONTENT if request.headers.get("range") else 200
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
            "Content-Length": str(end - start + 1),
        }
        if response_status == status.HTTP_206_PARTIAL_CONTENT:
            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        return StreamingResponse(
            _stream_file(target, start, end),
            status_code=response_status,
            media_type=detect_audio_content_type(target),
            headers=headers,
        )

    return application


def _orchestrator(request: Request) -> GenerationOrchestrator:
    active = getattr(request.app.state, "orchestrator", None)
    if active is None:
        raise HTTPException(status_code=503, detail="Application is not ready")
    return active


def _generation_parameters(payload: GenerateRequest, settings: Settings) -> tuple[str, int | None]:
    prompt = payload.prompt.strip()
    if not prompt:
        raise ValueError("prompt 不能为空，请输入歌曲风格描述。")
    if len(prompt) > settings.prompt_max_chars:
        raise ValueError(f"prompt 不能超过 {settings.prompt_max_chars} 个字符。")
    if payload.durationMinutes is None or (
        isinstance(payload.durationMinutes, str)
        and payload.durationMinutes.strip().lower() == "auto"
    ):
        return prompt, None
    return prompt, settings.normalize_duration(payload.durationMinutes)


def _public_base_url(request: Request, settings: Settings) -> str:
    return settings.public_base_url or str(request.base_url).rstrip("/")


def _job_response(job: GenerationJob, request: Request, settings: Settings) -> dict[str, Any]:
    response = job.response()
    if job.result is not None:
        response["result"] = _render_result_urls(
            job.result, _public_base_url(request, settings), settings
        )
    return response


def _render_result_urls(
    result: dict[str, Any], base_url: str, settings: Settings
) -> dict[str, Any]:
    rendered = copy.deepcopy(result)
    for output in [rendered, *rendered.get("alternatives", [])]:
        if not isinstance(output, dict):
            continue
        full_track = output.get("fullTrack")
        if isinstance(full_track, str):
            output["fullTrack"] = _public_audio_url(full_track, base_url, settings)
        stems = output.get("stems")
        if isinstance(stems, dict):
            output["stems"] = {
                name: _public_audio_url(url, base_url, settings)
                for name, url in stems.items()
                if isinstance(url, str)
            }
            output["stemUrls"] = list(output["stems"].values())
    return rendered


def _public_audio_url(audio_url: str, base_url: str, settings: Settings) -> str:
    try:
        target = _output_path_from_url(audio_url, settings)
    except ValueError:
        return audio_url
    return build_public_audio_url(base_url, target.relative_to(settings.output_dir.resolve()))


def _output_path_from_url(audio_url: str, settings: Settings):
    path = unquote(urlsplit(audio_url).path)
    marker = "/output/"
    if marker in path:
        path = path.split(marker, 1)[1]
    elif urlsplit(audio_url).scheme or path.startswith("/"):
        raise ValueError("Not an output URL")
    root = settings.output_dir.resolve()
    target = (root / path).resolve()
    target.relative_to(root)
    return target


def _stem_trash_path(target, settings: Settings, job_id: str, song: int = 0):
    root = settings.output_dir.resolve() / ".trash" / job_id
    return root / (f"song_{song + 1}" if song else "") / target.name


def _parse_byte_range(value: str | None, file_size: int) -> tuple[int, int]:
    if not value:
        return 0, file_size - 1
    if not value.startswith("bytes=") or "," in value:
        raise HTTPException(status_code=416, detail="Invalid byte range")
    start_text, separator, end_text = value[6:].partition("-")
    if not separator:
        raise HTTPException(status_code=416, detail="Invalid byte range")
    try:
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                raise ValueError
            start = max(0, file_size - suffix_length)
            end = max(0, file_size - 1)
        else:
            start = int(start_text)
            end = int(end_text) if end_text else max(0, file_size - 1)
    except ValueError as exc:
        raise HTTPException(status_code=416, detail="Invalid byte range") from exc
    if start < 0 or end < start or start >= file_size:
        raise HTTPException(status_code=416, detail="Invalid byte range")
    return start, min(end, file_size - 1)


async def _stream_file(path, start: int, end: int) -> AsyncIterator[bytes]:
    remaining = end - start + 1
    with path.open("rb") as source:
        source.seek(start)
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def run() -> None:
    settings = Settings()
    configure_logging()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)


app = create_app()
