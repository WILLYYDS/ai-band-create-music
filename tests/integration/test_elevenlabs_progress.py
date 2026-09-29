import asyncio
import json
import shutil
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.requests import Request

from app.main import GenerationJob, create_app, load_jobs
from app.services.providers import ElevenLabsMusicProvider
from tests.helpers import make_orchestrator, make_settings


class BlockingAudio(httpx.AsyncByteStream):
    def __init__(self, fail=False):
        self.reached = asyncio.Event()
        self.release = asyncio.Event()
        self.fail = fail
        self.closed = False

    async def __aiter__(self):
        yield b"\0" * 32_000  # One second of stereo PCM at 8 kHz.
        self.reached.set()
        await self.release.wait()
        if self.fail:
            raise httpx.ReadError("audio stream interrupted")
        yield b"\0" * (32_000 * 9)

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("duration", [None, 10])
@pytest.mark.parametrize("outcome", ["succeeded", "failed", "cancelled"])
async def test_elevenlabs_dual_song_progress_over_sse(tmp_path, monkeypatch, duration, outcome):
    reports = []
    original_save = GenerationJob.save

    def save(job, output_dir):
        reports.append((job.status, job.stage, job.progress))
        original_save(job, output_dir)

    monkeypatch.setattr(GenerationJob, "save", save)
    settings = make_settings(
        tmp_path,
        music_provider="elevenlabs_music",
        elevenlabs_api_key="test",
        elevenlabs_music_base_url="https://eleven.test",
        elevenlabs_music_output_format="pcm_8000",
    )
    # Exercise provider progress without ffmpeg or waveform subprocesses.
    monkeypatch.setattr("app.services.orchestrator.make_playback_mp3", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "app.services.orchestrator.extract_waveforms", AsyncMock(return_value={"full": [0.25, 0.5]})
    )
    streams = [BlockingAudio(), BlockingAudio(fail=outcome == "failed")]
    headers_reached = [asyncio.Event(), asyncio.Event()]
    headers_release = [asyncio.Event(), asyncio.Event()]
    requests = []

    async def handler(request):
        requests.append(request)
        assert request.url.path == "/v1/music/stream"
        index = len(requests) - 1
        headers_reached[index].set()
        await headers_release[index].wait()
        return httpx.Response(200, stream=streams[index])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as provider_client:
        orchestrator = make_orchestrator(settings)
        orchestrator.music_providers["elevenlabs_music"] = ElevenLabsMusicProvider(
            settings, provider_client
        )
        app = create_app(settings, orchestrator)
        endpoint = next(
            route.endpoint for route in app.routes if route.path == "/api/jobs/{job_id}/events"
        )
        payload = {"prompt": "普通话摇滚", "count": 2}
        if duration is not None:
            payload["durationSeconds"] = duration
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://testserver"
        ) as client:
            job_id = (await client.post("/api/jobs", json=payload)).json()["jobId"]
            job = app.state.jobs[job_id]
            events = None
            try:
                for index, stream in enumerate(streams):
                    await asyncio.wait_for(headers_reached[index].wait(), 2)
                    waiting = (await client.get(f"/api/jobs/{job_id}")).json()
                    assert waiting["currentSong"] == index + 1
                    assert waiting["songStates"][index]["status"] == "running"
                    if index == 1:
                        assert waiting["result"]["count"] == 1
                        assert waiting["result"]["requestedCount"] == 2
                        assert waiting["result"]["songNumber"] == 1
                        assert waiting["result"]["alternatives"] == []
                        assert waiting["songStates"][0]["status"] == "succeeded"
                        assert waiting["songStates"][0]["progress"] == 100
                        published = await asyncio.wait_for(anext(events.body_iterator), 2)
                        first_ready = json.loads(published.split("data: ", 1)[1])
                        assert first_ready["status"] == "running"
                        assert first_ready["result"]["count"] == 1
                        assert first_ready["result"]["songNumber"] == 1
                        assert first_ready["currentSong"] == 2
                        restart_dir = tmp_path / "restart"
                        shutil.copytree(settings.output_dir, restart_dir)
                        repaired = load_jobs(restart_dir)[job_id]
                        assert repaired.status == "failed"
                        assert repaired.current_song is None
                        assert repaired.result["count"] == 1
                        assert repaired.song_states[0]["status"] == "succeeded"
                        assert repaired.song_states[1]["status"] == "failed"
                        full_track = waiting["result"]["fullTrack"]
                        assert (await client.get(full_track)).status_code == 200
                    else:
                        assert waiting["result"] is None
                    assert waiting["stage"] == "generating_music"
                    assert waiting["receivedAudioSeconds"] is None
                    assert waiting["expectedAudioSeconds"] == duration
                    assert waiting["progress"] == (index * 50 if duration else None)
                    assert f"第 {index + 1}/2 首：等待 ElevenLabs 返回音频" == waiting["message"]
                    headers_release[index].set()
                    await asyncio.wait_for(stream.reached.wait(), 2)
                    row = (await client.get(f"/api/jobs/{job_id}")).json()
                    assert row["status"] == "running"
                    assert row["stage"] == "receiving_audio"
                    assert row["receivedAudioSeconds"] == 1
                    assert row["expectedAudioSeconds"] == duration
                    assert row["progress"] == (index * 50 + 5 if duration else None)
                    assert row["songStates"][index]["receivedAudioSeconds"] == 1
                    assert row["songStates"][index]["progress"] == (10 if duration else None)
                    assert row["step"] is row["totalSteps"] is None
                    assert f"第 {index + 1}/2 首" in row["message"]
                    assert "已接收 1 秒音频" in row["message"]
                    saved = json.loads(
                        (settings.output_dir / "jobs" / job_id / "job.json").read_text()
                    )
                    assert saved["receivedAudioSeconds"] == 1
                    if events:
                        await events.body_iterator.aclose()
                    request = Request(
                        {
                            "type": "http",
                            "app": app,
                            "headers": [],
                            "scheme": "http",
                            "server": ("testserver", 80),
                            "path": f"/api/jobs/{job_id}/events",
                            "root_path": "",
                            "query_string": b"",
                            "method": "GET",
                        }
                    )
                    events = await endpoint(job_id, request)
                    frame = await anext(events.body_iterator)
                    snapshot = json.loads(frame.split("data: ", 1)[1])
                    assert snapshot["message"] == row["message"]
                    assert snapshot["songStates"] == row["songStates"]
                    if index == 1:
                        assert row["result"]["waveforms"] == {"full": [0.25, 0.5]}
                        assert snapshot["result"] == {**row["result"], "waveforms": {}}
                    if index == 1 and outcome == "cancelled":
                        cancelled = await client.patch(
                            f"/api/jobs/{job_id}", json={"status": "cancelled"}
                        )
                        assert cancelled.json()["receivedAudioSeconds"] is None
                    else:
                        stream.release.set()
                await asyncio.wait_for(job.task, 2)
                done = await asyncio.wait_for(anext(events.body_iterator), 2)
                assert done.startswith("event: done\n")
                terminal = json.loads(done.split("data: ", 1)[1])
                status = "cancelled" if outcome == "cancelled" else "succeeded"
                assert terminal["status"] == status
                assert terminal["currentSong"] is None
                assert [song["status"] for song in terminal["songStates"]] == ["succeeded", outcome]
                assert terminal["progress"] == (100 if status == "succeeded" else None)
                assert terminal["receivedAudioSeconds"] is None
                assert terminal["expectedAudioSeconds"] is None
                assert terminal["step"] is terminal["totalSteps"] is None
                if outcome == "succeeded":
                    assert terminal["result"]["count"] == 2
                    assert terminal["result"]["alternatives"][0]["durationSeconds"] == 10
                    assert terminal["result"]["alternatives"][0]["songNumber"] == 2
                else:
                    assert terminal["result"]["count"] == 1
                    assert terminal["result"]["songNumber"] == 1
                    assert terminal["result"]["alternatives"] == []
                    if outcome == "failed":
                        assert terminal["error"] is None
                        assert "第 2 首生成失败" in terminal["warning"]
                        assert "audio stream interrupted" in terminal["songStates"][1]["error"]
                    song_dir = settings.output_dir / "jobs" / job_id / "song_2"
                    assert not list(song_dir.glob("*.wav"))
                restored = load_jobs(settings.output_dir)[job_id]
                assert restored.status == status
                assert restored.result["count"] == terminal["result"]["count"]
                assert restored.result["songNumber"] == 1
                assert restored.song_states == terminal["songStates"]
                history = (await client.get("/api/jobs")).json()["jobs"]
                assert history[0]["result"]["count"] == terminal["result"]["count"]
                assert history[0]["songStates"] == terminal["songStates"]
                assert all(stream.closed for stream in streams)
                assert len(requests) == 2  # No retries on cancellation or a broken audio stream.
                if duration is not None:
                    values = [
                        value
                        for status, stage, value in reports
                        if status == "running" and stage not in {"pending", "expanding_prompt"}
                    ]
                    assert all(value is not None for value in values)
                    assert values == sorted(values)
                    assert ("running", "saving_audio", 50) in reports
                    assert ("running", "waveform", 50) in reports
                    if status == "succeeded":
                        assert ("running", "finalizing", 99) in reports
                assert not list(settings.output_dir.rglob("*.part"))
            finally:
                for release in headers_release:
                    release.set()
                for stream in streams:
                    stream.release.set()
                if events:
                    await events.body_iterator.aclose()
                await asyncio.wait_for(job.task, 2)


@pytest.mark.parametrize("duration", [None, 10])
async def test_elevenlabs_sync_generate_clears_audio_progress(tmp_path, monkeypatch, duration):
    settings = make_settings(
        tmp_path,
        music_provider="elevenlabs_music",
        elevenlabs_api_key="test",
        elevenlabs_music_output_format="pcm_8000",
    )
    monkeypatch.setattr("app.services.orchestrator.make_playback_mp3", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "app.services.orchestrator.extract_waveforms", AsyncMock(return_value={"full": [0.25, 0.5]})
    )
    stream = BlockingAudio()
    stream.release.set()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
    ) as provider_client:
        orchestrator = make_orchestrator(settings)
        orchestrator.music_providers["elevenlabs_music"] = ElevenLabsMusicProvider(
            settings, provider_client
        )
        app = create_app(settings, orchestrator)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://testserver"
        ) as client:
            payload = {"prompt": "普通话摇滚"}
            if duration is not None:
                payload["durationSeconds"] = duration
            response = await client.post("/api/generate", json=payload)
            assert response.status_code == 200
            row = (await client.get(f"/api/jobs/{response.json()['jobId']}")).json()
            assert row["status"] == "succeeded"
            assert row["progress"] == 100
            assert row["receivedAudioSeconds"] is row["expectedAudioSeconds"] is None
            assert row["result"]["durationSeconds"] == 10
