from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import Settings
from app.core.errors import CapacityExceededError
from app.infrastructure.events import EventPublisher, GenerationEvent
from app.infrastructure.queue import TaskDispatcher
from app.services.audio_files import build_public_audio_url, ensure_file_under_root
from app.services.prompt import PromptExpander, split_generation_prompt
from app.services.providers import MusicProvider
from app.services.stems import STEM_NAMES, StemSeparator
from app.services.waveforms import extract_waveforms

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, int, str, str | None, str | None], Awaitable[None]]


def save_job_prompts(
    output_dir: Path,
    job_id: str,
    prompt: str,
    structured_prompt: str | None = None,
    lyrics: str | None = None,
) -> Path:
    target = output_dir / "jobs" / job_id / "prompts.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "jobId": job_id,
                "prompt": prompt,
                "structuredPrompt": structured_prompt,
                "lyrics": lyrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


class GenerationCapacity:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    async def acquire(self) -> None:
        async with self._lock:
            if self._active >= self.maximum:
                raise CapacityExceededError("当前生成任务过多，请稍后重试。")
            self._active += 1

    async def release(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        await self.acquire()
        try:
            yield
        finally:
            await self.release()


class GenerationOrchestrator:
    def __init__(
        self,
        settings: Settings,
        prompt_expander: PromptExpander,
        music_provider: MusicProvider,
        stem_separator: StemSeparator,
        task_dispatcher: TaskDispatcher,
        events: EventPublisher,
        music_providers: dict[str, MusicProvider] | None = None,
    ) -> None:
        self.settings = settings
        self.prompt_expander = prompt_expander
        self.music_provider = music_provider
        self.music_providers = music_providers or {}
        self.stem_separator = stem_separator
        self.task_dispatcher = task_dispatcher
        self.events = events
        self.capacity = GenerationCapacity(settings.max_concurrent_generations)

    async def generate(
        self,
        user_prompt: str,
        duration_minutes: int | None,
        public_base_url: str,
        request_id: str,
        *,
        job_id: str | None = None,
        progress: ProgressCallback | None = None,
        capacity_reserved: bool = False,
        provider: str | None = None,
        count: int = 1,
    ) -> dict[str, Any]:
        job_id = job_id or f"job_{int(time.time() * 1000)}_{uuid4().hex[:8]}"
        selected_provider = provider or self.settings.music_provider
        effective_count = count if selected_provider == "minimax_music" else 1
        warning = (
            "ElevenLabs Music 暂不支持单次生成两首，已按一首生成。"
            if count == 2 and selected_provider == "elevenlabs_music"
            else None
        )

        async def report(
            stage: str,
            value: int,
            message: str,
            structured_prompt: str | None = None,
            lyrics: str | None = None,
        ) -> None:
            if progress is not None:
                await progress(stage, value, message, structured_prompt, lyrics)

        async def execute() -> dict[str, Any]:
            save_job_prompts(self.settings.output_dir, job_id, user_prompt)
            await self.events.publish(GenerationEvent("generation.started", job_id, request_id))
            await report("expanding_prompt", 10, "正在扩写音乐创作提示")
            _, style = split_generation_prompt(user_prompt)
            structured_prompt, lyrics = await self.prompt_expander.prepare(
                user_prompt,
                duration_minutes or self.settings.default_duration_minutes,
            )
            provider_prompt = f"[歌词与创作内容]\n{lyrics}"
            if style:
                provider_prompt += f"\n\n[风格要求]\n{style}"
            save_job_prompts(
                self.settings.output_dir,
                job_id,
                user_prompt,
                structured_prompt,
                lyrics,
            )
            outputs: list[dict[str, Any]] = []
            music_provider = self.music_providers.get(selected_provider, self.music_provider)
            for index in range(effective_count):
                song_number = index + 1
                await report(
                    "generating_music",
                    25 + index * 35,
                    f"音乐模型正在生成第 {song_number}/{effective_count} 首",
                    structured_prompt,
                    lyrics,
                )
                music_result = await music_provider.generate(
                    structured_prompt,
                    duration_minutes,
                    provider_prompt,
                    variation=index,
                )
                await report("saving_audio", 45 + index * 35, f"正在保存第 {song_number} 首")
                full_path = await ensure_file_under_root(
                    music_result.audio_path,
                    self.settings.output_dir,
                    f"full_song_{job_id}_{song_number}.mp3",
                )
                full_relative = full_path.relative_to(self.settings.output_dir)
                full_url = build_public_audio_url(public_base_url, full_relative)

                if not self.settings.enable_audio_splitting:
                    stems = {name: full_url for name in STEM_NAMES}
                    waveform_paths = {"full": full_path}
                    split_debug: dict[str, Any] = {}
                    split_enabled = False
                else:
                    await report(
                        "splitting",
                        50 + index * 35,
                        f"Demucs 正在分离第 {song_number} 首",
                    )
                    relative_output = Path("jobs") / job_id / f"song_{song_number}"
                    split_result = await self.stem_separator.split(
                        full_path, self.settings.output_dir / relative_output
                    )
                    stems = {
                        name: build_public_audio_url(public_base_url, relative_output / file_name)
                        for name, file_name in split_result.files.items()
                    }
                    waveform_paths = {
                        name: self.settings.output_dir / relative_output / file_name
                        for name, file_name in split_result.files.items()
                    }
                    split_debug = {"splitterDurationMs": split_result.duration_ms}
                    split_enabled = True

                outputs.append(
                    {
                        "fullTrack": full_url,
                        "stems": stems,
                        "stemUrls": list(stems.values()),
                        "waveforms": await extract_waveforms(waveform_paths),
                        "splitEnabled": split_enabled,
                        "debug": {"music": music_result.debug, **split_debug},
                    }
                )

            await report("finalizing", 95, "正在校验并整理输出文件")
            primary = outputs[0]
            response = {
                "success": True,
                "jobId": job_id,
                "prompt": user_prompt,
                "durationMinutes": duration_minutes if duration_minutes is not None else "auto",
                "structuredPrompt": structured_prompt,
                "lyrics": lyrics,
                "count": effective_count,
                "alternatives": outputs[1:],
                "warning": warning,
                **primary,
            }
            await self.events.publish(GenerationEvent("generation.succeeded", job_id, request_id))
            logger.info("generation succeeded job_id=%s request_id=%s", job_id, request_id)
            return response

        async def dispatch() -> dict[str, Any]:
            try:
                return await self.task_dispatcher.submit(job_id, execute)
            except Exception:
                await self.events.publish(GenerationEvent("generation.failed", job_id, request_id))
                raise

        if capacity_reserved:
            return await dispatch()
        async with self.capacity.slot():
            return await dispatch()
