from __future__ import annotations

import asyncio
import time
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

import httpx

from app.core.config import PROJECT_ROOT, Settings
from app.core.errors import GenerationError
from app.services.audio_files import download_audio, require_readable_file, write_stream_atomically
from app.services.job_files import update_provider_diagnostic
from app.services.prompt import (
    OpenAICompatiblePromptExpander,
    _http_failure_message,
    build_elevenlabs_planning_prompt,
    enhance_elevenlabs_composition_plan,
    looks_like_chinese_music_request,
    split_generation_prompt,
)

MINIMAX_GENERATE_PATH = "/v1/audio/jobs"
ProviderProgressCallback = Callable[[str, int | None, int | None], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class MusicResult:
    audio_path: Path
    debug: dict[str, Any]


class MusicProvider(Protocol):
    async def generate(
        self,
        structured_prompt: str,
        duration_seconds: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress: ProviderProgressCallback | None = None,
        job_id: str | None = None,
    ) -> MusicResult: ...

def _response_diagnostic(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return {
        "statusCode": response.status_code,
        "headers": dict(response.headers),
        "body": body,
    }


def extract_task_id(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    nested = data.get("data") if isinstance(data.get("data"), dict) else {}
    return next(
        (
            str(value)
            for value in (
                data.get("taskId"),
                data.get("task_id"),
                data.get("id"),
                nested.get("taskId"),
                nested.get("task_id"),
                nested.get("id"),
            )
            if value
        ),
        None,
    )


def extract_task_status(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    nested = data.get("data") if isinstance(data.get("data"), dict) else {}
    return str(
        data.get("status") or data.get("state") or nested.get("status") or nested.get("state") or ""
    ).lower()


def extract_generated_audio_url(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    nested = data.get("data")
    candidates: list[Any] = [data]
    if isinstance(nested, dict):
        candidates.append(nested)
    elif isinstance(nested, list) and nested and isinstance(nested[0], dict):
        candidates.append(nested[0])
    for candidate in candidates:
        for key in ("audioUrl", "audio_url", "url"):
            if candidate.get(key):
                return str(candidate[key])
    return None


def _suno_clips(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("data", "clips", "items"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def extract_suno_clip_ids(data: Any) -> list[str]:
    return [str(clip["id"]) for clip in _suno_clips(data) if clip.get("id")]


def extract_suno_audio_url(data: Any) -> str | None:
    clips = _suno_clips(data)
    for clip in clips:
        status = str(clip.get("status", "")).lower()
        if clip.get("audio_url") and status in {"streaming", "complete", "completed", "succeeded"}:
            return str(clip["audio_url"])
    for clip in clips:
        if clip.get("audio_url"):
            return str(clip["audio_url"])
    return extract_generated_audio_url(data)


class MockMusicProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def generate(
        self,
        structured_prompt: str,
        duration_seconds: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress: ProviderProgressCallback | None = None,
        job_id: str | None = None,
    ) -> MusicResult:
        require_readable_file(
            self._settings.mock_full_song_path,
            "音乐生成 mock 文件不存在，请放置本地完整歌曲或设置 MOCK_FULL_SONG_PATH",
        )
        return MusicResult(self._settings.mock_full_song_path, {"provider": "mock"})


class GenericMusicProvider:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    def _headers(self) -> dict[str, str]:
        if self._settings.music_api_key is None:
            raise GenerationError("音乐生成失败：MUSIC_PROVIDER=generic 时必须配置 MUSIC_API_KEY。")
        return {
            "Authorization": f"Bearer {self._settings.music_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        structured_prompt: str,
        duration_seconds: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress: ProviderProgressCallback | None = None,
        job_id: str | None = None,
    ) -> MusicResult:
        if not self._settings.music_api_base_url:
            raise GenerationError(
                "音乐生成失败：MUSIC_API_MODE=real 时必须配置 MUSIC_API_BASE_URL。"
            )
        try:
            response = await self._client.post(
                f"{self._settings.music_api_base_url}{self._settings.music_generate_path}",
                json={
                    "prompt": structured_prompt,
                    "duration_minutes": duration_seconds / 60,
                    "duration_seconds": duration_seconds,
                    "model": self._settings.music_model or None,
                    "callback_url": self._settings.music_callback_url or None,
                },
                headers=self._headers(),
                timeout=self._settings.music_api_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            audio_url = extract_generated_audio_url(data)
            if audio_url:
                audio_path = await self._download(audio_url, f"full_song_{time.time_ns()}.mp3")
                return MusicResult(audio_path, {"provider": "generic", "mode": "direct_audio_url"})
            task_id = extract_task_id(data)
            if not task_id:
                raise GenerationError(
                    "音乐生成接口未返回 taskId 或 audioUrl，请检查代理接口响应结构。"
                )
            audio_path = await self._poll(task_id)
            return MusicResult(
                audio_path, {"provider": "generic", "mode": "polling", "taskId": task_id}
            )
        except GenerationError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise GenerationError(f"音乐生成失败：{_http_failure_message(exc)}") from exc

    async def _poll(self, task_id: str) -> Path:
        url = f"{self._settings.music_api_base_url}{self._settings.music_status_path}"
        for _ in range(self._settings.music_max_poll_attempts):
            await asyncio.sleep(self._settings.music_poll_interval_seconds)
            response = await self._client.get(
                url,
                params={"taskId": task_id},
                headers=self._headers(),
                timeout=self._settings.music_api_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            status = extract_task_status(data)
            audio_url = extract_generated_audio_url(data)
            if audio_url and status in {
                "success",
                "succeeded",
                "completed",
                "complete",
                "done",
                "",
            }:
                return await self._download(audio_url, f"full_song_{task_id}.mp3")
            if status in {"failed", "error", "canceled", "cancelled"}:
                raise GenerationError(f"音乐生成任务失败，taskId={task_id}，status={status}")
        raise GenerationError(f"音乐生成任务超时，taskId={task_id}")

    async def _download(self, audio_url: str, name: str) -> Path:
        return await download_audio(
            self._client,
            audio_url,
            self._settings.output_dir / name,
            timeout=self._settings.music_download_timeout_seconds,
            trusted_local_root=PROJECT_ROOT,
        )


class ElevenLabsMusicProvider:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    def _api_key(self) -> str:
        secret = self._settings.elevenlabs_api_key or self._settings.music_api_key
        if secret is None:
            raise GenerationError("ElevenLabs 音乐生成失败：缺少 ELEVENLABS_API_KEY 环境变量。")
        return secret.get_secret_value()

    def _headers(self, accept_audio: bool = False) -> dict[str, str]:
        headers = {"xi-api-key": self._api_key(), "Content-Type": "application/json"}
        if accept_audio:
            headers["Accept"] = "audio/mpeg"
        return headers

    async def generate(
        self,
        structured_prompt: str,
        duration_seconds: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress: ProviderProgressCallback | None = None,
        job_id: str | None = None,
    ) -> MusicResult:
        music_length_ms = duration_seconds * 1000
        clear_chinese = (
            self._settings.elevenlabs_clear_chinese_vocal_mode
            and looks_like_chinese_music_request(user_prompt, structured_prompt)
        )
        use_plan = (
            self._settings.elevenlabs_use_composition_plan
            and not self._settings.elevenlabs_force_instrumental
        )
        lyrics, _ = split_generation_prompt(user_prompt)
        planning_prompt = build_elevenlabs_planning_prompt(
            structured_prompt, duration_seconds / 60, clear_chinese, lyrics
        )
        try:
            if use_plan:
                plan_response = await self._client.post(
                    f"{self._settings.elevenlabs_music_base_url}/v1/music/plan",
                    json={
                        "prompt": planning_prompt,
                        "music_length_ms": music_length_ms,
                        "model_id": self._settings.elevenlabs_music_model_id,
                    },
                    headers=self._headers(),
                    timeout=self._settings.music_api_timeout_seconds,
                )
                plan_response.raise_for_status()
                request_body = {
                    "composition_plan": enhance_elevenlabs_composition_plan(
                        plan_response.json(), clear_chinese
                    ),
                    "model_id": self._settings.elevenlabs_music_model_id,
                }
            else:
                request_body = {
                    "prompt": planning_prompt,
                    "music_length_ms": music_length_ms,
                    "model_id": self._settings.elevenlabs_music_model_id,
                    "force_instrumental": self._settings.elevenlabs_force_instrumental,
                }

            target = self._settings.output_dir / f"full_song_elevenlabs_{time.time_ns()}.mp3"
            async with self._client.stream(
                "POST",
                f"{self._settings.elevenlabs_music_base_url}/v1/music",
                json=request_body,
                params={"output_format": self._settings.elevenlabs_music_output_format},
                headers=self._headers(accept_audio=True),
                timeout=self._settings.music_api_timeout_seconds,
            ) as response:
                if not response.is_success:
                    await response.aread()
                response.raise_for_status()
                await write_stream_atomically(
                    response.aiter_bytes(), target, "ElevenLabs 音乐生成接口返回空音频。"
                )
            return MusicResult(
                target,
                {
                    "provider": "elevenlabs_music",
                    "modelId": self._settings.elevenlabs_music_model_id,
                    "outputFormat": self._settings.elevenlabs_music_output_format,
                    "mode": "composition_plan" if use_plan else "prompt",
                    "clearChineseVocalMode": clear_chinese,
                },
            )
        except GenerationError:
            raise
        except httpx.HTTPStatusError as exc:
            raise GenerationError(self._failure_message(exc)) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise GenerationError(f"ElevenLabs 音乐生成失败：{_http_failure_message(exc)}") from exc

    @staticmethod
    def _failure_message(error: httpx.HTTPStatusError) -> str:
        detail = _http_failure_message(error)
        status = error.response.status_code
        if status == 401:
            return f"ElevenLabs 音乐生成失败：{detail}。请检查 API Key 的 music_generation 权限。"
        if status == 402:
            return f"ElevenLabs 音乐生成失败：{detail}。当前套餐可能不支持 Music API。"
        if status == 403:
            return f"ElevenLabs 音乐生成失败：{detail}。请检查账号套餐、区域和 Music API 权限。"
        return f"ElevenLabs 音乐生成失败：{detail}"


DEFAULT_SUNO_MODELS = (
    "v4.5-all",
    "chirp-v4-5-all",
    "chirp-v4-5",
    "chirp-v4",
    "chirp-v3-5",
    "chirp-v3-0",
)


class SunoMusicProvider:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._settings.music_api_key:
            headers["Authorization"] = f"Bearer {self._settings.music_api_key.get_secret_value()}"
        return headers

    def _models(self) -> list[str]:
        configured = [
            item.strip() for item in self._settings.suno_model_fallbacks.split(",") if item.strip()
        ]
        values = [self._settings.suno_model, *configured]
        if self._settings.suno_use_default_model_fallbacks:
            values.extend(DEFAULT_SUNO_MODELS)
        return list(dict.fromkeys(value for value in values if value))

    async def generate(
        self,
        structured_prompt: str,
        duration_seconds: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress: ProviderProgressCallback | None = None,
        job_id: str | None = None,
    ) -> MusicResult:
        if not self._settings.music_api_base_url:
            raise GenerationError(
                "音乐生成失败：MUSIC_PROVIDER=suno_api 时必须配置 MUSIC_API_BASE_URL。"
            )
        duration_minutes = duration_seconds / 60
        failures: list[str] = []
        data: Any = None
        for model in self._models():
            try:
                response = await self._client.post(
                    f"{self._settings.music_api_base_url}{self._settings.music_generate_path}",
                    json={
                        "prompt": structured_prompt,
                        "duration": duration_minutes,
                        "duration_minutes": duration_minutes,
                        "make_instrumental": self._settings.suno_make_instrumental,
                        "model": model,
                        "wait_audio": False,
                    },
                    headers=self._headers(),
                    timeout=self._settings.suno_create_timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                break
            except (httpx.HTTPError, ValueError) as exc:
                failures.append(f"{model}: {_http_failure_message(exc)}")
        if data is None:
            raise GenerationError(f"Suno 音乐生成失败：无法创建任务。{' | '.join(failures)}")

        audio_url = extract_suno_audio_url(data)
        if audio_url:
            path = await self._download(audio_url, f"full_song_suno_{time.time_ns()}.mp3")
            return MusicResult(path, {"provider": "suno_api", "mode": "direct_audio_url"})
        clip_ids = extract_suno_clip_ids(data)
        if not clip_ids:
            raise GenerationError("Suno 音乐生成失败：suno-api 未返回 clip id。")
        path = await self._poll(clip_ids)
        return MusicResult(path, {"provider": "suno_api", "mode": "polling", "clipIds": clip_ids})

    async def _poll(self, clip_ids: list[str]) -> Path:
        ids = ",".join(clip_ids)
        url = f"{self._settings.music_api_base_url}{self._settings.music_status_path}"
        for _ in range(self._settings.music_max_poll_attempts):
            await asyncio.sleep(self._settings.music_poll_interval_seconds)
            response = await self._client.get(
                url,
                params={"ids": ids},
                headers=self._headers(),
                timeout=self._settings.music_api_timeout_seconds,
            )
            response.raise_for_status()
            audio_url = extract_suno_audio_url(response.json())
            if audio_url:
                return await self._download(audio_url, f"full_song_suno_{clip_ids[0]}.mp3")
        raise GenerationError(f"suno-api 生成任务超时，ids={ids}")

    async def _download(self, url: str, name: str) -> Path:
        return await download_audio(
            self._client,
            url,
            self._settings.output_dir / name,
            timeout=self._settings.music_download_timeout_seconds,
            trusted_local_root=PROJECT_ROOT,
        )


class MiniMaxMusicProvider:
    """Client for the self-hosted MiniMax Music 3 service."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        lyrics_writer: OpenAICompatiblePromptExpander | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._lyrics_writer = lyrics_writer or OpenAICompatiblePromptExpander(settings, client)

    async def generate(
        self,
        structured_prompt: str,
        duration_seconds: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress: ProviderProgressCallback | None = None,
        job_id: str | None = None,
    ) -> MusicResult:
        if not self._settings.minimax_base_url:
            raise GenerationError("MiniMax 音乐生成失败：MINIMAX_BASE_URL 不能为空。")
        try:
            lyrics, _ = split_generation_prompt(user_prompt)
            if not lyrics:
                lyrics = await self._lyrics_writer.write_lyrics(
                    structured_prompt,
                    user_prompt,
                    (duration_seconds or self._settings.default_duration_minutes * 60) / 60,
                )
            if len(lyrics) < 10:
                raise GenerationError("MiniMax 音乐生成失败：生成的歌词不足 10 个字符。")
            target = self._settings.output_dir / f"full_song_minimax_{time.time_ns()}.wav"
            request_body = {
                "model": self._settings.minimax_model,
                "input": lyrics,
                "instructions": structured_prompt,
                "seed": self._settings.minimax_seed + variation,
                "num_inference_steps": self._settings.minimax_num_inference_steps,
                "response_format": "wav",
                "stream": False,
            }
            if job_id:
                request_body["jobId"] = uuid5(NAMESPACE_URL, f"{job_id}:{variation}").hex
            if duration_seconds is not None:
                request_body["audio_duration"] = duration_seconds
                timing_instruction = (
                    f"[Timing: The track must be exactly {duration_seconds:g} seconds long. "
                    "Pace the arrangement so the final lyric ends 2-5 seconds before that exact "
                    "end. After the final lyric, stop all vocals completely and use no more than "
                    "5 seconds for the outro.]"
                )
            else:
                timing_instruction = (
                    "[Timing: Choose the natural complete song duration. Sing every supplied lyric "
                    "without cutting off the ending, then stop all vocals and finish with only a "
                    "short natural outro.]"
                )
            request_body["instructions"] += (
                "\n[Lyric Fidelity: Sing every supplied non-tag lyric line exactly once, verbatim, "
                "and in order. Never omit, repeat, paraphrase, invent, or replace any lyric words.]"
                f"\n{timing_instruction}"
                "\n[Vocal Ending: Never fill unused time with repeated or invented vocals, "
                "humming, chants, ad-libs, or an extended instrumental outro.]"
            )
            url = f"{self._settings.minimax_base_url}{MINIMAX_GENERATE_PATH}"
            request_diagnostic = {
                "method": "POST",
                "url": url,
                "timeoutSeconds": self._settings.minimax_timeout_seconds,
                "body": request_body,
            }
            update_provider_diagnostic(
                self._settings.output_dir,
                job_id,
                variation,
                provider="minimax_music",
                request=request_diagnostic,
            )
            response = await self._client.post(
                url,
                json=request_body,
                timeout=httpx.Timeout(self._settings.minimax_timeout_seconds, connect=10),
            )
            create_response = _response_diagnostic(response)
            update_provider_diagnostic(
                self._settings.output_dir,
                job_id,
                variation,
                createResponse=create_response,
            )
            response.raise_for_status()
            create_body = create_response["body"]
            remote_id = create_body.get("jobId") if isinstance(create_body, dict) else None
            if not isinstance(remote_id, str) or not remote_id:
                raise GenerationError("MiniMax 未返回 jobId。")

            job_url = f"{url}/{quote(str(remote_id), safe='')}"
            started = time.monotonic()
            try:
                while True:
                    remaining = self._settings.minimax_timeout_seconds - (
                        time.monotonic() - started
                    )
                    if remaining <= 0:
                        raise GenerationError("MiniMax 音乐生成任务超时。")
                    response = await self._client.get(job_url, timeout=min(30, remaining))
                    status_response = _response_diagnostic(response)
                    update_provider_diagnostic(
                        self._settings.output_dir,
                        job_id,
                        variation,
                        statusResponse=status_response,
                    )
                    response.raise_for_status()
                    state = status_response["body"]
                    if not isinstance(state, dict):
                        raise GenerationError("MiniMax 返回的任务状态不是 JSON 对象。")
                    step, total = state.get("step"), state.get("totalSteps")
                    if step is not None and (type(step) is not int or step < 0):
                        raise GenerationError("MiniMax 返回无效 step。")
                    if total is not None and (type(total) is not int or total < 0):
                        raise GenerationError("MiniMax 返回无效 totalSteps。")
                    if progress:
                        await progress(state.get("stage") or "generating_music", step, total)
                    if state["status"] == "succeeded":
                        break
                    if state["status"] in {"failed", "cancelled"}:
                        raise GenerationError(state.get("error") or "MiniMax 音乐生成失败。")
                    if state["status"] not in {"pending", "running"}:
                        raise GenerationError("MiniMax 返回未知任务状态。")
                    await asyncio.sleep(min(self._settings.music_poll_interval_seconds, remaining))
            except (asyncio.CancelledError, Exception):
                try:
                    await self._client.delete(job_url, timeout=10)
                except httpx.HTTPError:
                    pass
                raise
            try:
                async with self._client.stream(
                    "GET",
                    f"{job_url}/audio",
                    timeout=httpx.Timeout(self._settings.minimax_timeout_seconds, connect=10),
                ) as response:
                    if not response.is_success:
                        await response.aread()
                    audio_response = {
                        "statusCode": response.status_code,
                        "headers": dict(response.headers),
                    }
                    if not response.is_success:
                        audio_response["body"] = response.text
                    update_provider_diagnostic(
                        self._settings.output_dir,
                        job_id,
                        variation,
                        audioResponse=audio_response,
                    )
                    response.raise_for_status()
                    await write_stream_atomically(
                        response.aiter_bytes(), target, "MiniMax Music 3 服务返回空音频。"
                    )
                with wave.open(str(target), "rb") as audio:
                    actual_duration_seconds = round(
                        audio.getnframes() / audio.getframerate(), 3
                    )
                duration_diagnostic = {"actualDurationSeconds": actual_duration_seconds}
                if duration_seconds is not None:
                    duration_diagnostic["requestedDurationSeconds"] = duration_seconds
                update_provider_diagnostic(
                    self._settings.output_dir, job_id, variation, **duration_diagnostic
                )
            finally:
                # The durable local copy belongs to history; release the remote temporary WAV.
                try:
                    await self._client.delete(job_url, timeout=10)
                except httpx.HTTPError:
                    pass
            debug = {
                "provider": "minimax_music",
                "modelId": self._settings.minimax_model,
                "mode": "self_hosted_wav",
                "jobId": remote_id,
                "step": step,
                "totalSteps": total,
                "durationSeconds": actual_duration_seconds,
                "seed": self._settings.minimax_seed + variation,
                "numInferenceSteps": self._settings.minimax_num_inference_steps,
                "request": request_diagnostic,
                "createResponse": create_response,
                "statusResponse": status_response,
                "audioResponse": audio_response,
            }
            if duration_seconds is not None:
                debug["requestedDurationSeconds"] = duration_seconds
            return MusicResult(target, debug)
        except GenerationError:
            raise
        except httpx.ConnectError as exc:
            raise GenerationError(
                "MiniMax Music 3 服务无法直连："
                f"{self._settings.minimax_base_url}{MINIMAX_GENERATE_PATH}。"
                "请求已绕过系统代理；请确认服务监听地址、端口和防火墙配置。"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise GenerationError(
                f"MiniMax Music 3 服务调用失败：{_http_failure_message(exc)}"
            ) from exc
        except (httpx.HTTPError, ValueError, wave.Error) as exc:
            raise GenerationError(
                f"MiniMax Music 3 服务调用失败：{_http_failure_message(exc)}"
            ) from exc


def create_music_provider(
    settings: Settings,
    client: httpx.AsyncClient,
    direct_client: httpx.AsyncClient | None = None,
    provider: str | None = None,
) -> MusicProvider:
    if settings.music_api_mode == "mock":
        return MockMusicProvider(settings)
    selected = provider or settings.music_provider
    if selected == "elevenlabs_music":
        elevenlabs_client = (
            direct_client
            if settings.elevenlabs_bypass_global_proxy and direct_client is not None
            else client
        )
        return ElevenLabsMusicProvider(settings, elevenlabs_client)
    if selected == "suno_api":
        return SunoMusicProvider(settings, client)
    if selected == "minimax_music":
        return MiniMaxMusicProvider(
            settings,
            direct_client or client,
            OpenAICompatiblePromptExpander(settings, client),
        )
    return GenericMusicProvider(settings, client)
