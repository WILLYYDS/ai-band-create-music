from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
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
from app.services.audio_files import (
    build_public_audio_url,
    detect_audio_content_type,
    output_path_from_url,
)
from app.services.job_files import job_song_dir, read_job_diagnostics
from app.services.orchestrator import GenerationOrchestrator
from app.services.prompt import OpenAICompatiblePromptExpander, effective_llm_output_tokens
from app.services.providers import create_music_provider
from app.services.stems import DemucsStemSeparator
from app.services.voice import RVCEngine, install_voice_api, replacement_filename
from app.services.waveforms import extract_waveforms, merge_waveform_sets

logger = logging.getLogger(__name__)
SPLIT_PROGRESS = 76
WAVEFORM_PROGRESS = 90
SPLIT_COMPLETE_PROGRESS = 100
REPLACE_PROGRESS = 90
REPLACE_COMPLETE_PROGRESS = 100
# 替换已经进入终态（超时/取消），但 RVC 推理线程还没退出、因而仍占着一个生成并发额度的数量。
# 额度按设计等到线程真正退出才释放，这里只把它暴露出来，让 /api/health 与日志能发现卡住的替换。
_replacement_wind_down_active = 0


class RequestSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, default_limit: int) -> None:
        self.app = app
        self.default_limit = default_limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self.default_limit
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
    deleted_replaced_vocals: dict[str, dict[str, str]] = field(default_factory=dict)
    split_task: asyncio.Task[None] | None = field(default=None, repr=False)
    split_song: int | None = field(default=None, repr=False)
    split_status: str | None = field(default=None, repr=False)
    split_stage: str | None = field(default=None, repr=False)
    split_progress: int | None = field(default=None, repr=False)
    split_message: str | None = field(default=None, repr=False)
    split_error: str | None = field(default=None, repr=False)
    replace_task: asyncio.Task[None] | None = field(default=None, repr=False)
    replace_song: int | None = field(default=None, repr=False)
    replace_status: str | None = field(default=None, repr=False)
    replace_stage: str | None = field(default=None, repr=False)
    replace_progress: int | None = field(default=None, repr=False)
    replace_message: str | None = field(default=None, repr=False)
    replace_error: str | None = field(default=None, repr=False)
    replace_cancel_requested: bool = field(default=False, repr=False)

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
                payload = {
                    **self.response(),
                    "deletedStems": self.deleted_stems,
                }
                if self.deleted_replaced_vocals:
                    payload["deletedReplacedVocals"] = self.deleted_replaced_vocals
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
                deleted_replaced_vocals=data.get("deletedReplacedVocals", {}),
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
        expose_headers=[
            "Content-Disposition",
            "X-Mix-Output",
            "X-RVC-Model",
            "X-RVC-Output",
        ],
    )
    application.add_middleware(
        RequestSizeLimitMiddleware,
        default_limit=application_settings.request_max_bytes,
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
                "enabled": True,
                "mode": "on_demand",
                "profile": application_settings.split_profile,
                "model": application_settings.demucs_model or None,
                "device": application_settings.demucs_device or "auto",
                "jobs": application_settings.demucs_jobs or None,
                "segment": application_settings.demucs_segment or None,
            },
            "replacement": {
                "conversionTimeoutSeconds": application_settings.rvc_conversion_timeout_seconds,
                # 已进入终态但 RVC 线程仍未退出的替换数量；这些替换按设计继续占着生成并发额度。
                "workersHoldingCapacityAfterTerminal": _replacement_wind_down_active,
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
        return {
            "jobs": [
                _job_response(
                    job,
                    request,
                    application_settings,
                    include_split=False,
                    include_operations=False,
                    include_waveforms=False,
                )
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

            def frame() -> tuple[bool, str]:
                """按"即将推送给客户端的状态"生成一帧，并返回它是否为终态。

                判据与帧体必须在同一次同步读取里定下来：分轨/换声会覆盖 job.status，若先判终态
                再取帧体，操作恰好在本帧之前收尾时（例如 /replace 在客户端连上 SSE 前就跑完了）
                就会推出 status="running" 与 replaceStatus="succeeded" 自相矛盾的 done 帧，
                客户端要么误判失败、要么永远等不到终态。
                """
                status = _operation_status_override(job) or job.status
                final = status not in {"pending", "running"}
                # 任务结束前 result 不会变：分轨只在收尾那一刻一次性写入 stems 和波形，
                # 紧接着 status 就变成终态。所以中间帧只推阶段进度，真实波形留给 done
                # 帧——否则每次 15 秒保活超时都会重推一整份 640-bin 波形。
                payload = _job_response(
                    job, request, application_settings, include_waveforms=final
                )
                return final, json.dumps(payload, ensure_ascii=False)

            try:
                while True:
                    final, body = frame()
                    if final:
                        yield f"event: done\ndata: {body}\n\n"
                        return
                    yield f"data: {body}\n\n"
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

    @application.post(
        "/api/jobs/{job_id}/split",
        response_model=GenerationJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        responses={
            200: {"model": GenerationJobResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
        },
    )
    async def split_generation_job(
        job_id: str, request: Request, response: Response, song: int = 0
    ):
        job = request.app.state.jobs.get(job_id)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )
        if job.status != "succeeded":
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "生成任务尚未完成，无法拆轨。"},
            )
        result = _song_result(job, song)
        if result is None or not isinstance(result.get("fullTrack"), str):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "该歌曲没有可供拆轨的完整音频。"},
            )
        if job.replace_task is not None and not job.replace_task.done():
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "success": False,
                    "message": f"第 {(job.replace_song or 0) + 1} 首歌曲正在替换人声。",
                },
            )
        if job.split_status in {"pending", "running"}:
            if job.split_song == song:
                return _job_response(job, request, application_settings)
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "success": False,
                    "message": f"第 {(job.split_song or 0) + 1} 首歌曲正在拆轨。",
                },
            )
        stems = result.get("stems")
        if isinstance(stems, dict) and stems:
            response.status_code = status.HTTP_200_OK
            return _job_response(job, request, application_settings)
        try:
            full_path = _output_path_from_url(result["fullTrack"], application_settings)
            if not full_path.is_file():
                raise OSError(f"missing full track: {full_path}")
        except (OSError, ValueError):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "完整音频文件不存在，无法拆轨。"},
            )

        active_orchestrator = _orchestrator(request)
        job.split_song = song
        job.split_status = "pending"
        job.split_stage = "splitting"
        job.split_progress = SPLIT_PROGRESS
        job.split_message = "Demucs 正在分离音轨"
        job.split_error = None
        try:
            await active_orchestrator.capacity.acquire()
        except CapacityExceededError:
            job.split_song = None
            job.split_status = None
            job.split_stage = None
            job.split_progress = None
            job.split_message = None
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"success": False, "message": "当前拆轨任务过多，请稍后重试。"},
            )
        except asyncio.CancelledError:
            job.split_song = None
            job.split_status = None
            job.split_stage = None
            job.split_progress = None
            job.split_message = None
            raise

        def publish() -> None:
            for queue in request.app.state.job_subscribers.get(job_id, ()):
                if queue.empty():
                    queue.put_nowait(None)

        async def execute() -> None:
            try:
                job.split_status = "running"
                publish()
                output_dir = job_song_dir(application_settings.output_dir, job_id, song)
                relative_output = output_dir.relative_to(application_settings.output_dir)
                split_result = await active_orchestrator.stem_separator.split(full_path, output_dir)
                stems = {
                    name: (relative_output / file_name).as_posix()
                    for name, file_name in split_result.files.items()
                }
                job.split_stage = "waveform"
                job.split_progress = WAVEFORM_PROGRESS
                job.split_message = "正在提取真实波形"
                publish()
                # 连同完整混音一起重新提取：本 PR 之前的任务把 "full" 存成 64 个 bin，
                # 直接与 640 个 bin 的分轨合并会让客户端拿到长度不一致的波形。
                fresh_waveforms = await extract_waveforms(
                    {
                        "full": full_path,
                        **{
                            name: application_settings.output_dir / relative_output / file_name
                            for name, file_name in split_result.files.items()
                        },
                    }
                )
                result["stems"] = stems
                result["stemUrls"] = list(stems.values())
                result["waveforms"] = merge_waveform_sets(result.get("waveforms"), fresh_waveforms)
                result["splitEnabled"] = bool(stems)
                debug = result.get("debug")
                if not isinstance(debug, dict):
                    debug = result["debug"] = {}
                debug["splitterDurationMs"] = split_result.duration_ms
                deleted_prefix = f"{song}:"
                for key, deleted in list(job.deleted_stems.items()):
                    if key.startswith(deleted_prefix):
                        try:
                            target = _output_path_from_url(deleted["url"], application_settings)
                            _stem_trash_path(target, application_settings, job_id, song).unlink(
                                missing_ok=True
                            )
                        except (KeyError, OSError, ValueError):
                            logger.warning(
                                "failed to clear replaced stem trash job_id=%s key=%s",
                                job_id,
                                key,
                            )
                        del job.deleted_stems[key]
                job.save(application_settings.output_dir)
                job.split_status = "succeeded"
                job.split_stage = "completed"
                job.split_progress = SPLIT_COMPLETE_PROGRESS
                job.split_message = "音轨分离完成"
            except asyncio.CancelledError:
                job.split_status = "cancelled"
                job.split_stage = "cancelled"
                job.split_progress = None
                job.split_message = "音轨分离已取消"
            except Exception:
                logger.exception("history split failed job_id=%s song=%s", job_id, song)
                job.split_status = "failed"
                job.split_stage = "failed"
                job.split_progress = None
                job.split_message = "音轨分离失败"
                job.split_error = "音轨分离失败，请检查服务配置后重试。"
            finally:
                await active_orchestrator.capacity.release()
                publish()

        job.split_task = asyncio.create_task(execute(), name=f"split-{job_id}-{song}")
        return _job_response(job, request, application_settings)

    @application.post(
        "/api/jobs/{job_id}/replace",
        response_model=GenerationJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        responses={
            200: {"model": GenerationJobResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
        },
    )
    async def replace_generation_vocal(
        job_id: str, request: Request, response: Response, song: int = 0
    ):
        job = request.app.state.jobs.get(job_id)
        if job is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "生成任务不存在。"},
            )
        if job.status != "succeeded":
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "生成任务尚未完成，无法替换人声。"},
            )
        result = _song_result(job, song)
        if result is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "歌曲不存在。"},
            )
        if job.split_status in {"pending", "running"}:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "success": False,
                    "message": f"第 {(job.split_song or 0) + 1} 首歌曲正在拆轨。",
                },
            )
        if job.replace_task is not None and not job.replace_task.done():
            if job.replace_status in {"failed", "cancelled"}:
                return JSONResponse(
                    status_code=status.HTTP_409_CONFLICT,
                    content={
                        "success": False,
                        "message": "上一次人声替换仍在安全收尾，请稍后重试。",
                    },
                )
            if job.replace_song == song:
                return _job_response(job, request, application_settings)
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "success": False,
                    "message": f"第 {(job.replace_song or 0) + 1} 首歌曲正在替换人声。",
                },
            )

        stems = result.get("stems")
        vocal_url = next(
            (
                value
                for name, value in (stems.items() if isinstance(stems, dict) else ())
                if name.lower() in {"vocal", "vocals", "voice"} and isinstance(value, str)
            ),
            None,
        )
        if vocal_url is None:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "该歌曲没有人声音轨，请先完成拆轨。"},
            )
        try:
            output_dir = job_song_dir(application_settings.output_dir, job_id, song)
            vocal_path = _output_path_from_url(vocal_url, application_settings)
            if vocal_path.parent != output_dir.resolve() or not vocal_path.is_file():
                raise OSError("missing vocal stem")
        except (OSError, ValueError):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"success": False, "message": "人声音轨文件不存在，无法替换。"},
            )
        if vocal_path.stat().st_size > application_settings.rvc_max_upload_bytes:
            return JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content={"success": False, "message": "人声音轨文件过大，无法替换。"},
            )

        cached_url = result.get("replacedVocal")
        if isinstance(cached_url, str):
            try:
                cached_path = _output_path_from_url(cached_url, application_settings)
                if (
                    result.get("_replacedVocalModel")
                    == _rvc_model_fingerprint(application_settings)
                    and cached_path.parent == output_dir.resolve()
                    and cached_path.is_file()
                ):
                    response.status_code = status.HTTP_200_OK
                    return _job_response(job, request, application_settings)
            except ValueError:
                pass

        active_orchestrator = _orchestrator(request)
        job.replace_song = song
        job.replace_status = "pending"
        job.replace_stage = "replacing_vocal"
        job.replace_progress = REPLACE_PROGRESS
        job.replace_message = "RVC 正在替换人声"
        job.replace_error = None
        job.replace_cancel_requested = False
        try:
            await active_orchestrator.capacity.acquire()
        except CapacityExceededError:
            job.replace_song = None
            job.replace_status = None
            job.replace_stage = None
            job.replace_progress = None
            job.replace_message = None
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"success": False, "message": "当前人声替换任务过多，请稍后重试。"},
            )
        except asyncio.CancelledError:
            job.replace_song = None
            job.replace_status = None
            job.replace_stage = None
            job.replace_progress = None
            job.replace_message = None
            raise

        if isinstance(cached_url, str):
            # 缓存失效（模型指纹/文件/路径任一不符）：只把引用从结果里摘掉以触发重新推理，
            # 磁盘上的旧文件保持不动。它与新产物同名，重跑成功时 output_path.replace 会原子
            # 覆盖它；重跑失败/超时/取消时它则是"上一版还能听"的唯一退路——留着只是不再被任何
            # 结果引用、也不会对外暴露，同曲最多一份，不会堆积。
            result.pop("replacedVocal", None)
            result.pop("_replacedVocalModel", None)
            job.save(application_settings.output_dir)

        def publish() -> None:
            for queue in request.app.state.job_subscribers.get(job_id, ()):
                if queue.empty():
                    queue.put_nowait(None)

        async def execute() -> None:
            try:
                job.replace_status = "running"
                publish()
                result_path = output_dir / replacement_filename(vocal_path.name)
                # 这里刻意不用 TemporaryDirectory 上下文：它在 await 被取消时会立刻 rmtree，
                # 而 RVC 推理线程无法强停、仍在写这个目录，清理与写文件会互相打架（真实 RVC 还会
                # 在推理中途读取输入/写入输出）。改成手动创建、等线程确认退出后再删。
                temp_dir = Path(tempfile.mkdtemp(prefix=".rvc-", dir=output_dir))
                try:
                    output_path = temp_dir / "converted.wav"
                    if job.replace_cancel_requested:
                        raise asyncio.CancelledError
                    conversion_task = asyncio.create_task(
                        request.app.state.voice_engine.convert(vocal_path, output_path)
                    )
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(conversion_task),
                            timeout=application_settings.rvc_conversion_timeout_seconds,
                        )
                    except asyncio.CancelledError:
                        # 外层被取消（PATCH 取消任务或进程收尾）时不能直接返回：RVC 推理线程无法
                        # 强停，这里必须等它真正退出，否则临时目录会被提前删除、线程后续写文件失败，
                        # 引擎锁也会在推理仍占用 GPU 时被释放。conversion_task 用 shield 保护，
                        # 所以拿到的异常只反映线程自身的失败，可以安全折叠。
                        await _await_stuck_replacement_worker(
                            conversion_task, job_id=job_id, song=song, reason="cancelled"
                        )
                        raise
                    except asyncio.TimeoutError:
                        logger.error(
                            "vocal replacement timed out job_id=%s song=%s timeout_seconds=%s",
                            job_id,
                            song,
                            application_settings.rvc_conversion_timeout_seconds,
                        )
                        job.replace_status = "failed"
                        job.replace_stage = "failed"
                        job.replace_progress = None
                        job.replace_message = "人声替换超时"
                        job.replace_error = "人声替换超时，请重试。"
                        publish()
                        # 失败状态已经对客户端生效，但线程仍在跑：这里继续等它退出，只为了让锁和
                        # 临时目录与线程寿命对齐。等待时长不受超时约束，若线程永不返回，这个替换
                        # 会一直占着并发额度——保持与 README 记载的语义一致，不做强制释放，
                        # 只通过日志和 /api/health 把"卡住的收尾"暴露出来。
                        await _await_stuck_replacement_worker(
                            conversion_task, job_id=job_id, song=song, reason="timed out"
                        )
                        return
                    if job.replace_cancel_requested:
                        raise asyncio.CancelledError
                    output_path.replace(result_path)
                finally:
                    # 走到这里 conversion_task 一定已经结束（成功/失败/两条收尾分支都等过它），
                    # 所以删除临时目录不会再和推理线程抢文件。
                    shutil.rmtree(temp_dir, ignore_errors=True)
                result["replacedVocal"] = result_path.relative_to(
                    application_settings.output_dir.resolve()
                ).as_posix()
                result["_replacedVocalModel"] = _rvc_model_fingerprint(application_settings)
                job.deleted_replaced_vocals.pop(str(song), None)
                job.save(application_settings.output_dir)
                job.replace_status = "succeeded"
                job.replace_stage = "completed"
                job.replace_progress = REPLACE_COMPLETE_PROGRESS
                job.replace_message = "人声替换完成"
            except asyncio.CancelledError:
                job.replace_status = "cancelled"
                job.replace_stage = "cancelled"
                job.replace_progress = None
                job.replace_message = "人声替换已取消"
            except Exception:
                logger.exception("vocal replacement failed job_id=%s song=%s", job_id, song)
                job.replace_status = "failed"
                job.replace_stage = "failed"
                job.replace_progress = None
                job.replace_message = "人声替换失败"
                job.replace_error = "人声替换失败，请检查 RVC 配置后重试。"
            finally:
                job.replace_cancel_requested = False
                await active_orchestrator.capacity.release()
                publish()

        job.replace_task = asyncio.create_task(execute(), name=f"replace-{job_id}-{song}")
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
        if job.split_task is not None and not job.split_task.done():
            job.split_task.cancel()
            job.split_status = "cancelled"
            job.split_stage = "cancelled"
            job.split_progress = None
            job.split_message = "音轨分离已取消"
        if (
            job.replace_status in {"pending", "running"}
            and job.replace_task is not None
            and not job.replace_task.done()
        ):
            # ponytail: RVC runs in a worker thread and has no safe stop API; mark cancellation
            # now, keep capacity reserved, then discard its output when inference returns.
            _mark_replace_cancelled(job)
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
        if stem_name.lower() in {"vocal", "vocals", "voice"}:
            if (
                job.replace_song == song
                and job.replace_status in {"pending", "running"}
                and job.replace_task is not None
                and not job.replace_task.done()
            ):
                _mark_replace_cancelled(job)
            elif job.replace_song == song and (
                job.replace_task is None or job.replace_task.done()
            ):
                job.replace_song = None
                job.replace_status = None
                job.replace_stage = None
                job.replace_progress = None
                job.replace_message = None
                job.replace_error = None
            replaced_url = result.pop("replacedVocal", None)
            result.pop("_replacedVocalModel", None)
            job.deleted_replaced_vocals.pop(str(song), None)
            if isinstance(replaced_url, str):
                try:
                    replaced_path = _output_path_from_url(replaced_url, application_settings)
                    if replaced_path.parent == target.parent:
                        replaced_path.unlink(missing_ok=True)
                except ValueError:
                    pass
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


def _split_status_override(job: GenerationJob) -> str | None:
    """返回进行中的分轨强加给任务响应的状态；没有分轨在跑时返回 None。

    分轨接口（`POST /api/jobs/{job_id}/split`）只在任务结束后才允许调用，所以 Demucs
    工作期间任务自身早已是终态（"succeeded"），客户端——包括必须在整个分轨期间保持
    打开的 SSE 流——要看到的是分轨的 "pending"/"running"。规则集中放在这里，避免各个
    调用点各自推导。
    """
    if job.split_status in {"pending", "running"}:
        return job.split_status
    return None


def _operation_status_override(job: GenerationJob) -> str | None:
    if job.replace_status in {"pending", "running"}:
        return job.replace_status
    return _split_status_override(job)


def _mark_replace_cancelled(job: GenerationJob) -> None:
    job.replace_cancel_requested = True
    job.replace_status = "cancelled"
    job.replace_stage = "cancelled"
    job.replace_progress = None
    job.replace_message = "人声替换已取消"
    job.replace_error = None


def _reported_replace_status(
    job: GenerationJob, settings: Settings, *, fingerprint: str | None = None
) -> str | None:
    """对外汇报的替换状态：有在途操作就报它，否则从结果里推导。

    只依赖 job.replace_status 会有两个洞：它不落盘（重启后回到 None），也不被"恢复被删除的
    替换结果"这条路径重新赋值。于是客户端会收到"result 里有 replacedVocal、replaceStatus 却
    是 null"的矛盾响应，前端的替换状态机会直接判失败（"生成服务未返回人声替换状态"），
    撤回删除后再点替换就再也走不通。这里以结果为准补齐终态：

    - 有 replacedVocal、且模型指纹与当前模型一致（或这条旧数据没有指纹）→ succeeded；
    - 指纹不一致说明缓存已失效，交给 /replace 重新推理，不在这里谎报成功。
    """
    if job.replace_status is not None:
        return job.replace_status
    result = job.result
    if not isinstance(result, dict) or not isinstance(result.get("replacedVocal"), str):
        return None
    recorded = result.get("_replacedVocalModel")
    if isinstance(recorded, str):
        if fingerprint is None:
            fingerprint = _rvc_model_fingerprint(settings)
        if recorded != fingerprint:
            return None
    return "succeeded"


async def _await_stuck_replacement_worker(
    conversion_task: asyncio.Task[None], *, job_id: str, song: int, reason: str
) -> None:
    """等待已经进入终态、但 RVC 线程仍在跑的替换收尾。

    额度按设计保留到线程真正退出（详见 README），所以这里的等待没有上限；为了让运维能发现
    "线程不返回、额度一直被占"的情况，等待期间计数并写日志，线程退出后再撤销。
    """
    global _replacement_wind_down_active
    _replacement_wind_down_active += 1
    logger.warning(
        "vocal replacement worker still running after %s; holding capacity until it exits "
        "job_id=%s song=%s",
        reason,
        job_id,
        song,
    )
    try:
        await conversion_task
    except Exception:
        logger.warning(
            "vocal replacement worker failed while winding down job_id=%s song=%s",
            job_id,
            song,
            exc_info=True,
        )
    finally:
        _replacement_wind_down_active = max(0, _replacement_wind_down_active - 1)
        logger.warning(
            "vocal replacement worker exited job_id=%s song=%s", job_id, song
        )


@lru_cache(maxsize=8)
def _file_sha256(path: str, _size: int, _mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rvc_model_fingerprint(settings: Settings) -> str:
    paths = [settings.rvc_model_path, settings.rvc_index_path]
    parts = [settings.rvc_model_version]
    for path in paths:
        if path is None:
            parts.append("")
            continue
        try:
            stat = path.stat()
            parts.append(_file_sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        except OSError:
            parts.append("missing")
    return f"v1:{hashlib.sha256(chr(0).join(parts).encode()).hexdigest()}"


def _job_response(
    job: GenerationJob,
    request: Request,
    settings: Settings,
    *,
    include_split: bool = True,
    include_operations: bool = True,
    include_waveforms: bool = True,
) -> dict[str, Any]:
    response = job.response()
    split_status = _split_status_override(job)
    if include_operations and include_split and split_status is not None:
        response.update(
            status=split_status,
            stage=job.split_stage,
            progress=job.split_progress,
            message=job.split_message,
        )
    if include_operations and job.split_status is not None:
        response["splitStatus"] = job.split_status
        response["splitSong"] = job.split_song
        response["splitError"] = job.split_error
    replace_status = _reported_replace_status(job, settings)
    if include_operations and replace_status in {"pending", "running"}:
        response.update(
            status=replace_status,
            stage=job.replace_stage,
            progress=job.replace_progress,
            message=job.replace_message,
        )
    if include_operations and replace_status is not None:
        response["replaceStatus"] = replace_status
        response["replaceSong"] = job.replace_song
        response["replaceError"] = job.replace_error
    if job.result is not None:
        response["result"] = _render_result_urls(
            job.result,
            _public_base_url(request, settings),
            settings,
            include_waveforms=include_waveforms,
        )
    return response


def _render_result_urls(
    result: dict[str, Any],
    base_url: str,
    settings: Settings,
    *,
    include_waveforms: bool = True,
) -> dict[str, Any]:
    rendered = copy.deepcopy(result if include_waveforms else _without_waveforms(result))
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
        replaced_vocal = output.get("replacedVocal")
        if isinstance(replaced_vocal, str):
            output["replacedVocal"] = _public_audio_url(replaced_vocal, base_url, settings)
        for key in [key for key in output if isinstance(key, str) and key.startswith("_")]:
            del output[key]
    return rendered


def _without_waveforms(result: dict[str, Any]) -> dict[str, Any]:
    """浅拷贝一份任务结果，把波形包络清空。

    两个调用方都只需要拿一次波形：`GET /api/jobs` 会投影每个已存任务（每个最多两首
    歌），且不绘制分轨编辑器车道；SSE 流则因为任务进入终态前 result 不会变化，却在
    整个分轨期间每 15 秒保活一次就重发一帧。这两处带上波形只会放大响应体和上面的
    深拷贝。这里清空键而不是删除，是为了保持文档化的响应结构稳定；需要真实波形请走
    `GET /api/jobs/{job_id}` 或终态 done 帧。拷贝是浅拷贝，不会改动任务自己存的波形。
    """
    projected = {key: value for key, value in result.items() if key != "waveforms"}
    projected["waveforms"] = {}
    alternatives = result.get("alternatives")
    if isinstance(alternatives, list):
        projected["alternatives"] = [
            _without_waveforms(item) if isinstance(item, dict) else item for item in alternatives
        ]
    return projected


def _public_audio_url(audio_url: str, base_url: str, settings: Settings) -> str:
    try:
        target = _output_path_from_url(audio_url, settings)
    except ValueError:
        return audio_url
    return build_public_audio_url(base_url, target.relative_to(settings.output_dir.resolve()))


def _output_path_from_url(audio_url: str, settings: Settings):
    return output_path_from_url(audio_url, settings.output_dir)


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
