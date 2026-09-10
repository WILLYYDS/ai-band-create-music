from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import httpx
import pytest

from app.core.errors import GenerationError
from app.services.providers import (
    ElevenLabsMusicProvider,
    GenericMusicProvider,
    MiniMaxMusicProvider,
    create_music_provider,
    extract_generated_audio_url,
    extract_suno_audio_url,
    extract_task_id,
)
from tests.helpers import make_settings


class StreamingErrorBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"detail":{"message":"unauthorized"}}'


class StubLyricsWriter:
    async def write_lyrics(
        self, structured_prompt: str, user_prompt: str, duration_minutes: int
    ) -> str:
        return "[Verse]\ntest lyrics"


def wav_bytes(seconds: float = 1) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0" * round(seconds * 8000 * 4))
    return output.getvalue()


def test_provider_response_extractors_cover_legacy_shapes() -> None:
    assert extract_task_id({"data": {"task_id": "task-1"}}) == "task-1"
    assert (
        extract_generated_audio_url({"data": [{"audio_url": "https://audio"}]}) == "https://audio"
    )
    assert (
        extract_suno_audio_url([{"id": "clip", "status": "streaming", "audio_url": "https://suno"}])
        == "https://suno"
    )


async def test_request_provider_overrides_configured_default(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="elevenlabs_music",
    )
    async with httpx.AsyncClient() as client:
        provider = create_music_provider(settings, client, provider="minimax_music")
        assert isinstance(provider, MiniMaxMusicProvider)


async def test_minimax_provider_uses_direct_client(tmp_path: Path) -> None:
    proxied_requests: list[httpx.Request] = []
    direct_requests: list[httpx.Request] = []

    def proxied_handler(request: httpx.Request) -> httpx.Response:
        proxied_requests.append(request)
        raise AssertionError("MiniMax must not use the proxy-aware client")

    def direct_handler(request: httpx.Request) -> httpx.Response:
        direct_requests.append(request)
        return minimax_response(request)

    settings = make_settings(tmp_path, music_api_mode="real")
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(proxied_handler)) as client,
        httpx.AsyncClient(transport=httpx.MockTransport(direct_handler)) as direct_client,
    ):
        provider = create_music_provider(
            settings,
            client,
            direct_client,
            provider="minimax_music",
        )
        await provider.generate(
            "[Genre: Rock]",
            60,
            "[歌词与创作内容]\n[Verse]\ntest lyrics",
        )

    assert not proxied_requests
    assert direct_requests[0].url == "http://127.0.0.1:8111/v1/audio/jobs"


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
            "[Genre: Rock]",
            120,
            "[歌词与创作内容]\n第一句原歌词\n第二句原歌词\n\n[风格要求]\n普通话摇滚",
        )

    assert result.audio_path.read_bytes() == b"ID3-generated-audio"
    assert result.debug["mode"] == "composition_plan"
    assert len(requests) == 2
    plan_body = json.loads(requests[0].content)
    assert "第一句原歌词\n第二句原歌词" in plan_body["prompt"]
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
            await ElevenLabsMusicProvider(settings, client).generate("[Genre: Rock]", 120, "rock")


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
        result = await GenericMusicProvider(settings, client).generate("[Genre: Folk]", 60, "folk")
    assert result.audio_path.read_bytes() == b"ID3-generic"
    assert result.debug["mode"] == "direct_audio_url"


async def test_minimax_provider_streams_self_hosted_wav(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return minimax_response(request)

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="minimax_music",
        minimax_base_url="https://minimax.test",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await MiniMaxMusicProvider(settings, client, StubLyricsWriter()).generate(
            "[Genre: Rock]", 150, "rock"
        )

    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/audio/jobs"
    assert "authorization" not in requests[0].headers
    assert body["model"] == "MiniMaxAI/MiniMax-Music3"
    assert "Finish singing every lyric line by 140 seconds" in body["instructions"]
    assert body["input"] == "[Verse]\ntest lyrics"
    assert body["seed"] == 42
    assert body["num_inference_steps"] == 30
    assert body["response_format"] == "wav"
    assert body["stream"] is False
    assert body["audio_duration"] == 150
    assert "auto_duration_hint" not in body
    assert "max_new_tokens" not in body
    assert result.audio_path.suffix == ".wav"
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.debug["durationSeconds"] == 1
    assert result.debug["requestedDurationSeconds"] == 150
    assert result.debug["mode"] == "self_hosted_wav"


async def test_minimax_provider_sends_selected_duration(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return minimax_response(request)

    settings = make_settings(tmp_path, minimax_base_url="https://minimax.test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await MiniMaxMusicProvider(settings, client, StubLyricsWriter()).generate(
            "[Genre: Rock]", 60, "rock", job_id="local-job"
        )

    body = json.loads(requests[0].content)
    assert body["audio_duration"] == 60
    assert "Finish singing every lyric line by 50 seconds" in body["instructions"]
    assert "reserve the final 10 seconds" in body["instructions"]
    assert "After the final provided lyric line, stop all vocals completely" in body["instructions"]
    assert "do not invent or repeat lyrics" in body["instructions"]
    diagnostics = json.loads(
        (settings.output_dir / "jobs/local-job/prompts.json").read_text(encoding="utf-8")
    )
    provider_request = diagnostics["providerRequests"][0]
    assert provider_request["requestedDurationSeconds"] == 60
    assert provider_request["actualDurationSeconds"] == 1
    assert provider_request["request"]["body"] == body
    assert provider_request["request"]["url"] == "https://minimax.test/v1/audio/jobs"
    assert provider_request["createResponse"]["body"] == {"jobId": "remote-job"}
    assert provider_request["statusResponse"]["body"]["status"] == "succeeded"
    assert provider_request["audioResponse"]["statusCode"] == 200


async def test_minimax_provider_offsets_seed_for_alternatives(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return minimax_response(request)

    settings = make_settings(tmp_path, minimax_base_url="https://minimax.test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await MiniMaxMusicProvider(settings, client, StubLyricsWriter()).generate(
            "[Genre: Rock]", 60, "rock", variation=1
        )

    assert json.loads(requests[0].content)["seed"] == settings.minimax_seed + 1


async def test_minimax_provider_preserves_create_page_lyrics(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return minimax_response(request)

    settings = make_settings(tmp_path, minimax_base_url="https://minimax.test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await MiniMaxMusicProvider(settings, client, StubLyricsWriter()).generate(
            "[Genre: Dream Pop]",
            60,
            "[歌词与创作内容]\n夜色落进空荡站台\n最后一班车没有回来\n\n[风格要求]\n梦幻流行、空灵女声",
        )

    body = json.loads(requests[0].content)
    assert body["input"] == "夜色落进空荡站台\n最后一班车没有回来"
    assert "梦幻流行、空灵女声" not in body["instructions"]


async def test_minimax_provider_reports_server_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "invalid request"})

    settings = make_settings(tmp_path, minimax_base_url="https://minimax.test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="422.*invalid request"):
            await MiniMaxMusicProvider(settings, client, StubLyricsWriter()).generate(
                "[Genre: Rock]", 120, "rock", job_id="failed-job"
            )

    diagnostics = json.loads(
        (settings.output_dir / "jobs/failed-job/prompts.json").read_text(encoding="utf-8")
    )["providerRequests"][0]
    assert diagnostics["request"]["body"]["audio_duration"] == 120
    assert diagnostics["createResponse"]["statusCode"] == 422
    assert diagnostics["createResponse"]["body"] == {"detail": "invalid request"}


async def test_minimax_provider_reports_direct_connection_target(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    settings = make_settings(tmp_path, minimax_base_url="http://192.168.1.4:8111")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(
            GenerationError,
            match=r"192\.168\.1\.4:8111/v1/audio/jobs.*绕过系统代理",
        ):
            await MiniMaxMusicProvider(settings, client, StubLyricsWriter()).generate(
                "[Genre: Rock]", 60, "rock"
            )


def minimax_response(request):
    if request.method == "POST":
        return httpx.Response(202, json={"jobId": "remote-job"})
    if request.url.path.endswith("/audio"):
        return httpx.Response(200, content=wav_bytes())
    return httpx.Response(
        200, json={"status": "succeeded", "stage": "completed", "step": None, "totalSteps": None}
    )
