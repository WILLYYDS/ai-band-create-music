from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.core.errors import GenerationError
from app.services.providers import (
    ElevenLabsMusicProvider,
    GenericMusicProvider,
    extract_generated_audio_url,
    extract_suno_audio_url,
    extract_task_id,
)
from tests.helpers import make_settings


class StreamingErrorBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"detail":{"message":"unauthorized"}}'


def test_provider_response_extractors_cover_legacy_shapes() -> None:
    assert extract_task_id({"data": {"task_id": "task-1"}}) == "task-1"
    assert (
        extract_generated_audio_url({"data": [{"audio_url": "https://audio"}]}) == "https://audio"
    )
    assert (
        extract_suno_audio_url([{"id": "clip", "status": "streaming", "audio_url": "https://suno"}])
        == "https://suno"
    )


async def test_elevenlabs_uses_composition_plan_and_streams_audio(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/plan"):
            return httpx.Response(
                200,
                json={"sections": [{"duration_ms": 10_000, "lines": ["第一句", "第二句"]}]},
            )
        return httpx.Response(200, content=b"ID3-generated-audio")

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="elevenlabs_music",
        elevenlabs_api_key="secret",
        elevenlabs_music_base_url="https://eleven.test",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ElevenLabsMusicProvider(settings, client).generate(
            "[Genre: Rock]", 2, "普通话摇滚"
        )

    assert result.audio_path.read_bytes() == b"ID3-generated-audio"
    assert result.debug["mode"] == "composition_plan"
    assert len(requests) == 2
    music_body = json.loads(requests[1].content)
    assert "composition_plan" in music_body
    assert "prompt" not in music_body
    assert "clear vocal articulation" in music_body["composition_plan"]["positive_global_styles"]


async def test_elevenlabs_reads_streaming_error_before_building_message(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, stream=StreamingErrorBody())

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="elevenlabs_music",
        elevenlabs_api_key="invalid",
        elevenlabs_music_base_url="https://eleven.test",
        elevenlabs_use_composition_plan=False,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="music_generation 权限"):
            await ElevenLabsMusicProvider(settings, client).generate("[Genre: Rock]", 2, "rock")


async def test_generic_provider_accepts_direct_audio_url(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"audioUrl": "https://provider.test/song.mp3"})
        return httpx.Response(200, content=b"ID3-generic")

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="generic",
        music_api_key="secret",
        music_api_base_url="https://provider.test",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await GenericMusicProvider(settings, client).generate("[Genre: Folk]", 1, "folk")
    assert result.audio_path.read_bytes() == b"ID3-generic"
    assert result.debug["mode"] == "direct_audio_url"
