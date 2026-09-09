from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.config import Settings
from app.infrastructure.events import NullEventPublisher
from app.infrastructure.queue import InlineTaskDispatcher
from app.services.orchestrator import GenerationOrchestrator
from app.services.providers import MusicResult
from app.services.stems import SplitResult, stem_output_files


class StubPromptExpander:
    async def expand(self, user_prompt: str) -> str:
        return f"[Genre: Test], [Source: {user_prompt}]"

    async def prepare(
        self, user_prompt: str, duration_minutes: int | None = None
    ) -> tuple[str, str]:
        if user_prompt.startswith("[歌词与创作内容]"):
            lyrics = user_prompt.split("\n\n[风格要求]", 1)[0].split("\n", 1)[1]
            return await self.expand(user_prompt), f"[Verse]\n{lyrics}"
        return await self.expand(user_prompt), "[Verse]\n自动生成的测试歌词"

    async def write_lyrics(
        self, structured_prompt: str, user_prompt: str, duration_minutes: int
    ) -> str:
        return "[Verse]\n自动生成的测试歌词"


class BlockingPromptExpander:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def expand(self, user_prompt: str) -> str:
        self.started.set()
        await self.release.wait()
        return "[Genre: Test]"

    async def prepare(
        self, user_prompt: str, duration_minutes: int | None = None
    ) -> tuple[str, str]:
        return await self.expand(user_prompt), "[Verse]\n自动生成的测试歌词"

    async def write_lyrics(
        self, structured_prompt: str, user_prompt: str, duration_minutes: int
    ) -> str:
        return "[Verse]\n自动生成的测试歌词"


class StubMusicProvider:
    def __init__(self, source: Path, name: str = "stub") -> None:
        self.source = source
        self.name = name
        self.user_prompt = ""
        self.variations: list[int] = []

    async def generate(
        self,
        structured_prompt: str,
        duration_minutes: int | None,
        user_prompt: str,
        *,
        variation: int = 0,
        progress=None,
        job_id=None,
    ) -> MusicResult:
        self.user_prompt = user_prompt
        self.variations.append(variation)
        return MusicResult(
            self.source, {"provider": self.name, "durationMinutes": duration_minutes}
        )


class StubStemSeparator:
    async def split(self, input_path: Path, output_dir: Path) -> SplitResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = stem_output_files(input_path)
        for file_name in output_files.values():
            (output_dir / file_name).write_bytes(b"ID3-stem-audio")
        return SplitResult("stub split", "", 5, output_files)


def make_settings(tmp_path: Path, **updates: object) -> Settings:
    rvc_model = tmp_path / "rvc-model.pth"
    rvc_model.write_bytes(b"test model")
    values: dict[str, object] = {
        "output_dir": tmp_path / "output",
        "mock_full_song_path": tmp_path / "source.mp3",
        "public_base_url": "",
        "enable_audio_splitting": True,
        "rvc_model_path": rvc_model,
        "rvc_index_path": None,
        "rvc_base_model_dir": None,
        "rvc_result_dir": tmp_path / "output" / "rvc",
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def make_orchestrator(
    settings: Settings,
    *,
    prompt_expander=None,
) -> GenerationOrchestrator:
    source = settings.mock_full_song_path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"ID3-full-audio")
    return GenerationOrchestrator(
        settings=settings,
        prompt_expander=prompt_expander or StubPromptExpander(),
        music_provider=StubMusicProvider(source),
        stem_separator=StubStemSeparator(),
        task_dispatcher=InlineTaskDispatcher(),
        events=NullEventPublisher(),
    )
