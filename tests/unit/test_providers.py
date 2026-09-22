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
    _wrap_long_lyric_lines,
    create_music_provider,
    extract_generated_audio_url,
    extract_suno_audio_url,
    extract_task_id,
)
from tests.helpers import make_settings


class StreamingErrorBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"detail":{"message":"unauthorized"}}'


MINIMAX_USER_PROMPT = "[歌词与创作内容]\n[Verse]\ntest lyrics\n\n[风格要求]\nrock"


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


def test_minimax_wraps_long_lyrics_without_empty_lines_or_lost_text() -> None:
    samples = (
        "短句一，" + "啊" * 40 + "。",
        "啊" * 80,
        "Mr. Smith walked down the long and winding road to nowhere at all tonight.",
    )
    for lyrics in samples:
        wrapped = _wrap_long_lyric_lines(lyrics)
        assert not wrapped.endswith("\n")
        assert max(map(len, wrapped.splitlines())) <= 32
        assert wrapped.replace("\n", "") == lyrics


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


async def test_elevenlabs_sends_complete_prompt_and_streams_audio(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
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
            job_id="job-1",
        )

    assert result.audio_path.read_bytes() == b"ID3-generated-audio"
    assert result.audio_path == settings.output_dir / "jobs/job-1/song_1/full_song_job-1_1.mp3"
    assert not list(settings.output_dir.glob("full_song_*"))
    assert result.debug["mode"] == "prompt"
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/music"
    music_body = json.loads(requests[0].content)
    assert "[Genre: Rock]" in music_body["prompt"]
    assert "第一句原歌词\n第二句原歌词" in music_body["prompt"]
    assert "Mandarin Chinese lead vocals" in music_body["prompt"]
    assert music_body["music_length_ms"] == 120_000
    assert "composition_plan" not in music_body


async def test_elevenlabs_rejects_oversized_complete_prompt_before_request(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"unexpected")

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="elevenlabs_music",
        elevenlabs_api_key="secret",
        elevenlabs_music_base_url="https://eleven.test",
    )
    lyrics = "[Verse]\n" + "很长的完整歌词" * 700
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="超过接口上限 4100"):
            await ElevenLabsMusicProvider(settings, client).generate(
                "[Genre: Rock]",
                120,
                f"[歌词与创作内容]\n{lyrics}",
                job_id="oversized-job",
            )

    assert not requests
    diagnostics = json.loads(
        (settings.output_dir / "jobs/oversized-job/prompts.json").read_text(encoding="utf-8")
    )
    provider_request = diagnostics["providerRequests"][0]
    assert provider_request["request"]["body"]["prompt"].endswith(lyrics)
    assert "超过接口上限 4100" in provider_request["validationError"]


async def test_elevenlabs_instrumental_mode_omits_chinese_vocal_direction(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"ID3-generated-audio")

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="elevenlabs_music",
        elevenlabs_api_key="secret",
        elevenlabs_music_base_url="https://eleven.test",
        elevenlabs_force_instrumental=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ElevenLabsMusicProvider(settings, client).generate(
            "[Genre: dark techno]", 180, ""
        )

    body = json.loads(requests[0].content)
    assert body["force_instrumental"] is True
    assert "Mandarin Chinese lead vocals" not in body["prompt"]
    assert result.debug["clearChineseVocalMode"] is False


async def test_elevenlabs_reads_streaming_error_before_building_message(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, stream=StreamingErrorBody())

    settings = make_settings(
        tmp_path,
        music_api_mode="real",
        music_provider="elevenlabs_music",
        elevenlabs_api_key="invalid",
        elevenlabs_music_base_url="https://eleven.test",
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
        result = await GenericMusicProvider(settings, client).generate(
            "[Genre: Folk]", 60, "folk", job_id="job-1"
        )
    assert result.audio_path.read_bytes() == b"ID3-generic"
    assert result.audio_path == settings.output_dir / "jobs/job-1/song_1/full_song_job-1_1.mp3"
    assert result.debug["mode"] == "direct_audio_url"


async def test_minimax_provider_omits_duration_for_auto(tmp_path: Path) -> None:
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
        result = await MiniMaxMusicProvider(settings, client).generate(
            "[Genre: Rock]", None, MINIMAX_USER_PROMPT
        )

    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/audio/jobs"
    assert "authorization" not in requests[0].headers
    assert body["model"] == "MiniMaxAI/MiniMax-Music3"
    assert "Choose the natural complete song duration" in body["instructions"]
    assert "Sing every supplied non-tag lyric line exactly once" in body["instructions"]
    assert "Never fill unused time with repeated or invented vocals" in body["instructions"]
    assert body["input"] == "[Verse]\ntest lyrics"
    assert body["seed"] == 42
    assert body["num_inference_steps"] == 30
    assert body["response_format"] == "wav"
    assert body["stream"] is False
    assert "audio_duration" not in body
    assert "auto_duration_hint" not in body
    assert "max_new_tokens" not in body
    assert result.audio_path.suffix == ".wav"
    assert result.audio_path.read_bytes()[:4] == b"RIFF"
    assert result.debug["durationSeconds"] == 1
    assert "requestedDurationSeconds" not in result.debug
    assert result.debug["mode"] == "self_hosted_wav"


async def test_minimax_provider_ignores_selected_duration(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return minimax_response(request)

    settings = make_settings(tmp_path, minimax_base_url="https://minimax.test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await MiniMaxMusicProvider(settings, client).generate(
            "[Genre: Rock]", 60, MINIMAX_USER_PROMPT, job_id="local-job"
        )

    body = json.loads(requests[0].content)
    assert "audio_duration" not in body
    assert "Choose the natural complete song duration" in body["instructions"]
    assert result.audio_path == (
        settings.output_dir / "jobs/local-job/song_1/full_song_local-job_1.wav"
    )
    diagnostics = json.loads(
        (settings.output_dir / "jobs/local-job/prompts.json").read_text(encoding="utf-8")
    )
    provider_request = diagnostics["providerRequests"][0]
    assert "requestedDurationSeconds" not in provider_request
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
        await MiniMaxMusicProvider(settings, client).generate(
            "[Genre: Rock]", 60, MINIMAX_USER_PROMPT, variation=1
        )

    assert json.loads(requests[0].content)["seed"] == settings.minimax_seed + 1


async def test_minimax_provider_preserves_create_page_lyrics(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return minimax_response(request)

    settings = make_settings(tmp_path, minimax_base_url="https://minimax.test")
    lyrics = "夜色落进空荡站台，最后一班车没有回来，月光沿着铁轨沉默地延伸，我仍在原地等待。"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await MiniMaxMusicProvider(settings, client).generate(
            "[Genre: Dream Pop]",
            60,
            f"[歌词与创作内容]\n[Verse]\n{lyrics}\n\n[风格要求]\n梦幻流行、空灵女声",
        )

    body = json.loads(requests[0].content)
    assert body["input"].splitlines()[0] == "[Verse]"
    assert len(body["input"].splitlines()) > 2
    assert body["input"].replace("\n", "") == f"[Verse]{lyrics}"
    assert "梦幻流行、空灵女声" not in body["instructions"]


async def test_minimax_provider_reports_server_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "invalid request"})

    settings = make_settings(tmp_path, minimax_base_url="https://minimax.test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="422.*invalid request"):
            await MiniMaxMusicProvider(settings, client).generate(
                "[Genre: Rock]", 120, MINIMAX_USER_PROMPT, job_id="failed-job"
            )

    diagnostics = json.loads(
        (settings.output_dir / "jobs/failed-job/prompts.json").read_text(encoding="utf-8")
    )["providerRequests"][0]
    assert "audio_duration" not in diagnostics["request"]["body"]
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
            await MiniMaxMusicProvider(settings, client).generate(
                "[Genre: Rock]", 60, MINIMAX_USER_PROMPT
            )


def minimax_response(request):
    if request.method == "POST":
        return httpx.Response(202, json={"jobId": "remote-job"})
    if request.url.path.endswith("/audio"):
        return httpx.Response(200, content=wav_bytes())
    return httpx.Response(
        200, json={"status": "succeeded", "stage": "completed", "step": None, "totalSteps": None}
    )
