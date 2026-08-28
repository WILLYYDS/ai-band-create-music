from __future__ import annotations

import asyncio
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
from app.services.prompt import PromptExpander
from app.services.providers import MusicProvider
from app.services.stems import STEM_NAMES, StemSeparator
from app.services.waveforms import extract_waveforms

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, int, str], Awaitable[None]]


class GenerationCapacity:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._lock:
            if self._active >= self.maximum:
                raise CapacityExceededError("当前生成任务过多，请稍后重试。")
            self._active += 1
        try:
            yield
        finally:
            async with self._lock:
                self._active = max(0, self._active - 1)


class GenerationOrchestrator:
    def __init__(
        self,
        settings: Settings,
        prompt_expander: PromptExpander,
        music_provider: MusicProvider,
        stem_separator: StemSeparator,
        task_dispatcher: TaskDispatcher,
        events: EventPublisher,
    ) -> None:
        self.settings = settings
        self.prompt_expander = prompt_expander
        self.music_provider = music_provider
        self.stem_separator = stem_separator
        self.task_dispatcher = task_dispatcher
        self.events = events
        self.capacity = GenerationCapacity(settings.max_concurrent_generations)

    async def generate(
        self,
        user_prompt: str,
        duration_minutes: int,
        public_base_url: str,
        request_id: str,
        *,
        job_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        job_id = job_id or f"job_{int(time.time() * 1000)}_{uuid4().hex[:8]}"

        async def report(stage: str, value: int, message: str) -> None:
            if progress is not None:
                await progress(stage, value, message)

        async def execute() -> dict[str, Any]:
            await self.events.publish(GenerationEvent("generation.started", job_id, request_id))
            await report("expanding_prompt", 10, "正在扩写音乐创作提示")
            structured_prompt = await self.prompt_expander.expand(user_prompt)
            await report("generating_music", 25, "ElevenLabs 正在生成完整音乐")
            music_result = await self.music_provider.generate(
                structured_prompt, duration_minutes, user_prompt
            )
            await report("saving_audio", 60, "正在保存完整音乐")
            full_path = await ensure_file_under_root(
                music_result.audio_path,
                self.settings.output_dir,
                f"full_song_{job_id}.mp3",
            )
            full_relative = full_path.relative_to(self.settings.output_dir)
            full_url = build_public_audio_url(public_base_url, full_relative)

            if not self.settings.enable_audio_splitting:
                stems = {name: full_url for name in STEM_NAMES}
                waveform_paths = {"full": full_path}
                split_debug: dict[str, Any] = {
                    "splitterStdout": "",
                    "splitterStderr": "",
                }
                split_enabled = False
            else:
                await report("splitting", 65, "Demucs 正在分离音轨")
                relative_output = Path("jobs") / job_id
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
                split_debug = {
                    "splitterStdout": split_result.stdout,
                    "splitterStderr": split_result.stderr,
                    "splitterDurationMs": split_result.duration_ms,
                }
                split_enabled = True

            await report("finalizing", 95, "正在校验并整理输出文件")
            waveforms = await extract_waveforms(waveform_paths)

            response = {
                "success": True,
                "jobId": job_id,
                "prompt": user_prompt,
                "durationMinutes": duration_minutes,
                "structuredPrompt": structured_prompt,
                "fullTrack": full_url,
                "stems": stems,
                "stemUrls": list(stems.values()),
                "waveforms": waveforms,
                "splitEnabled": split_enabled,
                "debug": {"music": music_result.debug, **split_debug},
            }
            await self.events.publish(GenerationEvent("generation.succeeded", job_id, request_id))
            logger.info("generation succeeded job_id=%s request_id=%s", job_id, request_id)
            return response

        async with self.capacity.slot():
            try:
                return await self.task_dispatcher.submit(job_id, execute)
            except Exception:
                await self.events.publish(GenerationEvent("generation.failed", job_id, request_id))
                raise
