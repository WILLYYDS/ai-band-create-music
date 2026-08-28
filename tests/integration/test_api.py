from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import pytest

from app.main import create_app
from tests.helpers import BlockingPromptExpander, make_orchestrator, make_settings


async def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )


def _clear_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset every proxy variable inherited from the host shell.

    Keeps the test deterministic regardless of the machine running it: a
    developer's shell often exports ``http_proxy``/``all_proxy`` etc., and the
    uppercase/lowercase variants can shadow the variables a test sets.
    """
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name, raising=False)


async def test_health_preserves_legacy_contract(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        response = await client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["maxConcurrentGenerations"] == 1
    assert body["outputUrl"] == "http://testserver/output"
    assert body["llm"]["timeoutSeconds"] == 120
    assert body["llm"]["initialMaxTokens"] == 1024
    assert body["llm"]["retryMaxTokens"] == 2048
    assert body["llm"]["disableThinking"] is True
    assert body["splitting"]["device"] == "auto"
    assert body["infrastructure"] == {
        "taskBackend": "inline",
        "cacheBackend": "none",
        "eventBackend": "none",
    }


async def test_generate_validates_prompt_boundaries(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, prompt_max_chars=2000)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        blank = await client.post("/api/generate", json={"prompt": "   "})
        maximum = await client.post("/api/generate", json={"prompt": "乐" * 2000})
        too_long = await client.post("/api/generate", json={"prompt": "乐" * 2001})
    assert blank.status_code == 400
    assert maximum.status_code == 200
    assert too_long.status_code == 400
    assert "2000" in too_long.json()["message"]


async def test_generate_preserves_not_ready_response(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))
    async with await _client(app) as client:
        response = await client.post("/api/generate", json={"prompt": "ambient"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Application is not ready"


async def test_generate_returns_stems_and_downloadable_audio(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, enable_audio_splitting=True)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        response = await client.post(
            "/api/generate",
            json={"prompt": "明亮的普通话摇滚", "durationMinutes": 3},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["splitEnabled"] is True
        assert list(body["stems"]) == ["vocal", "drums", "bass", "other"]
        assert body["stems"]["vocal"].endswith("_vocal.mp3")
        assert body["stems"]["drums"].endswith("_drums.mp3")
        assert body["waveforms"] == {}
        audio = await client.get(body["stems"]["vocal"])
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/mpeg")
    assert audio.headers["cache-control"] == "no-store"
    assert audio.content == b"ID3-stem-audio"


async def test_async_job_reports_real_stage_and_result(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, enable_audio_splitting=True)
    blocker = BlockingPromptExpander()
    app = create_app(settings, make_orchestrator(settings, prompt_expander=blocker))
    async with await _client(app) as client:
        created = await client.post("/api/jobs", json={"prompt": "真实进度"})
        assert created.status_code == 202
        job_id = created.json()["jobId"]
        await blocker.started.wait()

        running = await client.get(f"/api/jobs/{job_id}")
        assert running.json()["status"] == "running"
        assert running.json()["stage"] == "expanding_prompt"
        assert running.json()["progress"] == 10

        blocker.release.set()
        for _ in range(20):
            completed = await client.get(f"/api/jobs/{job_id}")
            if completed.json()["status"] == "succeeded":
                break
            await asyncio.sleep(0)

    body = completed.json()
    assert body["progress"] == 100
    assert body["result"]["splitEnabled"] is True
    assert set(body["result"]["stems"]) == {"vocal", "drums", "bass", "other"}


async def test_async_job_can_be_cancelled(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    blocker = BlockingPromptExpander()
    orchestrator = make_orchestrator(settings, prompt_expander=blocker)
    app = create_app(settings, orchestrator)
    async with await _client(app) as client:
        created = await client.post("/api/jobs", json={"prompt": "取消任务"})
        job_id = created.json()["jobId"]
        await blocker.started.wait()
        cancelled = await client.patch(
            f"/api/jobs/{job_id}", json={"status": "cancelled"}
        )
        await asyncio.sleep(0)

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert orchestrator.capacity.active == 0


async def test_async_job_deleted_stem_stays_deleted(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, enable_audio_splitting=True)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        created = await client.post("/api/jobs", json={"prompt": "删除分轨"})
        job_id = created.json()["jobId"]
        for _ in range(20):
            completed = await client.get(f"/api/jobs/{job_id}")
            if completed.json()["status"] == "succeeded":
                break
            await asyncio.sleep(0)

        vocal_url = completed.json()["result"]["stems"]["vocal"]
        deleted = await client.delete(f"/api/jobs/{job_id}/stems/vocal")
        refreshed = await client.get(f"/api/jobs/{job_id}")
        missing_file = await client.get(vocal_url)
        hidden_trash = await client.get(
            f"/output/.trash/{job_id}/{Path(vocal_url).name}"
        )
        restored = await client.put(f"/api/jobs/{job_id}/stems/vocal")
        restored_file = await client.get(vocal_url)

    assert deleted.status_code == 204
    assert deleted.content == b""
    assert "vocal" not in refreshed.json()["result"]["stems"]
    assert missing_file.status_code == 404
    assert hidden_trash.status_code == 404
    assert restored.status_code == 200
    assert "vocal" in restored.json()["result"]["stems"]
    assert restored_file.status_code == 200


async def test_split_disabled_returns_full_track_compatibility_stems(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, enable_audio_splitting=False)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        response = await client.post("/api/generate", json={"prompt": "ambient"})
    body = response.json()
    assert body["splitEnabled"] is False
    assert set(body["stems"].values()) == {body["fullTrack"]}


async def test_request_size_limit_returns_413(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, request_max_bytes=1024)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        response = await client.post(
            "/api/generate",
            content=b"x" * 1025,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json()["success"] is False


async def test_concurrent_generation_returns_429_without_queueing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_concurrent_generations=1)
    blocker = BlockingPromptExpander()
    orchestrator = make_orchestrator(settings, prompt_expander=blocker)
    app = create_app(settings, orchestrator)
    async with await _client(app) as client:
        first = asyncio.create_task(client.post("/api/generate", json={"prompt": "first"}))
        await blocker.started.wait()
        second = await client.post("/api/generate", json={"prompt": "second"})
        blocker.release.set()
        first_response = await first
    assert second.status_code == 429
    assert first_response.status_code == 200
    assert orchestrator.capacity.active == 0


async def test_audio_route_rejects_missing_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        response = await client.get("/output/not-found.mp3")
    assert response.status_code == 404


async def test_audio_route_supports_byte_ranges(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, enable_audio_splitting=False)
    app = create_app(settings, make_orchestrator(settings))
    async with await _client(app) as client:
        generated = await client.post("/api/generate", json={"prompt": "range test"})
        response = await client.get(generated.json()["fullTrack"], headers={"Range": "bytes=0-2"})
    assert response.status_code == 206
    assert response.content == b"ID3"
    assert response.headers["content-range"].startswith("bytes 0-2/")


async def test_audio_route_returns_empty_file_with_zero_length(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    (settings.output_dir / "empty.mp3").touch()
    async with await _client(app) as client:
        response = await client.get("/output/empty.mp3")
    assert response.status_code == 200
    assert response.headers["content-length"] == "0"
    assert response.content == b""


@pytest.mark.parametrize(
    "proxy_variables",
    [
        {},
        {"ALL_PROXY": "socks5://127.0.0.1:7897"},
        {"ALL_PROXY": "socks://127.0.0.1:7897"},
        {"all_proxy": "socks://127.0.0.1:7897"},
        {"ALL_PROXY": "socks5://127.0.0.1:7897", "all_proxy": "socks://127.0.0.1:7897"},
        {"HTTP_PROXY": "http://127.0.0.1:7897", "HTTPS_PROXY": "http://127.0.0.1:7897"},
        {"ALL_PROXY": "http://127.0.0.1:7897"},
        {"ALL_PROXY": "socks4://127.0.0.1:7897"},
    ],
    ids=[
        "no-proxy",
        "socks5",
        "socks-alias",
        "lowercase-socks-alias",
        "mixed-case-socks",
        "http-https",
        "http-all",
        "unsupported-socks4-dropped",
    ],
)
async def test_application_lifespan_accepts_proxy_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proxy_variables: dict[str, str],
) -> None:
    settings = make_settings(tmp_path, enable_audio_splitting=False)
    _clear_proxy_environment(monkeypatch)
    for name, value in proxy_variables.items():
        monkeypatch.setenv(name, value)
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        assert app.state.orchestrator is not None
