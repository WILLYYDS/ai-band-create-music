from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.errors import GenerationError, ProviderGlobalError
from app.main import GenerationJob, create_app, load_jobs
from tests.helpers import make_orchestrator, make_settings


@pytest.mark.parametrize("provider", ["minimax_music", "elevenlabs_music"])
@pytest.mark.parametrize("endpoint", ["/api/generate", "/api/jobs"])
@pytest.mark.parametrize("failed_songs", [{0}, {1}, {0, 1}])
async def test_song_failures_preserve_other_outputs(
    tmp_path, monkeypatch, provider, endpoint, failed_songs
):
    settings = make_settings(tmp_path, music_provider=provider)
    orchestrator = make_orchestrator(settings)
    original_generate = orchestrator.music_provider.generate
    monkeypatch.setattr("app.services.orchestrator.make_playback_mp3", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "app.services.orchestrator.extract_waveforms", AsyncMock(return_value={"full": [0.5]})
    )
    calls = []
    snapshots = []

    async def generate(*args, variation=0, progress=None, job_id=None, **kwargs):
        calls.append(variation)
        snapshots.append(app.state.jobs[job_id].response())
        await progress("generating_music", 2, 5)
        if variation in failed_songs:
            raise GenerationError(f"song {variation + 1} rejected")
        return await original_generate(*args, variation=variation, **kwargs)

    orchestrator.music_provider.generate = generate
    app = create_app(settings, orchestrator)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        response = await client.post(endpoint, json={"prompt": "rock", "count": 2})
        if endpoint == "/api/jobs":
            assert response.status_code == 202
            job_id = response.json()["jobId"]
            await app.state.jobs[job_id].task
        else:
            assert response.status_code == (500 if len(failed_songs) == 2 else 200)
            job_id = next(iter(app.state.jobs))
        row = (await client.get(f"/api/jobs/{job_id}")).json()
        assert calls == [0, 1]
        assert row["currentSong"] is None
        assert [song["status"] for song in row["songStates"]] == [
            "failed" if index in failed_songs else "succeeded" for index in range(2)
        ]
        if len(failed_songs) == 2:
            assert row["status"] == "failed"
            assert row["result"] is None
            assert "song 1 rejected" in row["error"]
            assert "song 2 rejected" in row["error"]
        else:
            assert row["status"] == "succeeded"
            assert row["progress"] == 100
            assert row["error"] is None
            result = row["result"]
            assert result["count"] == 1
            assert result["requestedCount"] == 2
            assert result["alternatives"] == []
            assert result["songNumber"] == (2 if 0 in failed_songs else 1)
            assert "生成失败" in row["warning"]
            assert row["warning"] == result["warning"]
            assert (await client.get(result["fullTrack"])).content == b"ID3-full-audio"
            if 1 in failed_songs:
                assert snapshots[1]["status"] == "running"
                assert snapshots[1]["currentSong"] == 2
                assert snapshots[1]["result"]["count"] == 1
                assert snapshots[1]["songStates"][0]["status"] == "succeeded"
            # Output index stays zero even if only the originally second song succeeded.
            split = await client.post(f"/api/jobs/{job_id}/split?song=0")
            assert split.status_code == 202
            await app.state.jobs[job_id].split_task
            split_job = (await client.get(f"/api/jobs/{job_id}")).json()
            assert split_job["splitStatus"] == "succeeded"
            directory = f"/song_{result['songNumber']}/"
            assert all(directory in url for url in split_job["result"]["stems"].values())
            if 0 in failed_songs:
                # Delete/restore uses the available-output index, but the file is in song_2.
                song_dir = settings.output_dir / "jobs" / job_id / "song_2"
                mixed = song_dir / "vocal_rvc_mix.wav"
                mixed.write_bytes(b"mixed audio")
                mixed_url = mixed.relative_to(settings.output_dir).as_posix()
                app.state.jobs[job_id].result["mixedTrack"] = mixed_url
                form = {
                    "filename": mixed.name,
                    "job_id": job_id,
                    "song": "0",
                    # 首次拆轨已经推进过文件版本，派生音频的删除/撤回要带上当前版本。
                    "audio_revision": "1",
                }
                assert (
                    await client.request("DELETE", "/api/voice/result", data=form)
                ).status_code == 200
                assert not mixed.exists()
                trash_dir = settings.output_dir / ".trash" / job_id / "song_2"
                assert (trash_dir / mixed.name).read_bytes() == b"mixed audio"
                assert not (trash_dir.parent / "song_1").exists()
                assert (await client.put("/api/voice/result", data=form)).status_code == 200
                assert mixed.read_bytes() == b"mixed audio"
                assert not (trash_dir / mixed.name).exists()
                assert app.state.jobs[job_id].result["mixedTrack"] == mixed_url
                replacement = song_dir / "vocal_rvc_vocal.wav"
                replacement.write_bytes(b"replaced audio")
                stored_result = app.state.jobs[job_id].result
                stored_result["replacedVocal"] = replacement.relative_to(
                    settings.output_dir
                ).as_posix()
                stem = settings.output_dir / stored_result["stems"]["vocal"]
                assert (
                    await client.delete(
                        f"/api/jobs/{job_id}/stems/vocal?song=0",
                        headers={"X-Audio-Revision": "1"},
                    )
                ).status_code == 204
                assert not stem.exists()
                assert replacement.read_bytes() == b"replaced audio"
                assert (trash_dir / stem.name).exists()
                assert not (trash_dir / replacement.name).exists()
                assert (
                    await client.put(
                        f"/api/jobs/{job_id}/stems/vocal?song=0",
                        headers={"X-Audio-Revision": "1"},
                    )
                ).status_code == 200
                assert stem.exists()
                assert replacement.read_bytes() == b"replaced audio"
                assert not (trash_dir / stem.name).exists()
                assert not (trash_dir / replacement.name).exists()
        restored = load_jobs(settings.output_dir)[job_id]
        assert restored.status == row["status"]
        assert restored.song_states == row["songStates"]
        assert (restored.result is None) == (row["result"] is None)


@pytest.mark.parametrize("status", ["pending", "running", "failed", "cancelled"])
@pytest.mark.parametrize("method", ["DELETE", "PUT"])
async def test_voice_result_edits_require_completed_generation(tmp_path, status, method):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    relative = "jobs/partial/song_1/vocal_rvc_mix.wav"
    track = settings.output_dir / relative
    track.parent.mkdir(parents=True)
    track.write_bytes(b"ready audio")
    app.state.jobs["partial"] = GenerationJob(
        job_id="partial", prompt="rock", status=status, result={"fullTrack": relative}
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        response = await client.request(
            method, "/api/voice/result", data={"filename": track.name, "job_id": "partial"}
        )
    assert response.status_code == 409
    assert "生成任务尚未完成" in response.json()["detail"]
    assert track.read_bytes() == b"ready audio"
    assert not (settings.output_dir / ".trash").exists()


@pytest.mark.parametrize("endpoint", ["/api/generate", "/api/jobs"])
@pytest.mark.parametrize("failed_song", [0, 1])
async def test_global_provider_failure_stops_song_requests(
    tmp_path, monkeypatch, endpoint, failed_song
):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    original_generate = orchestrator.music_provider.generate
    orchestrator.events.publish = AsyncMock()
    monkeypatch.setattr("app.services.orchestrator.make_playback_mp3", AsyncMock(return_value=None))
    monkeypatch.setattr("app.services.orchestrator.extract_waveforms", AsyncMock(return_value={}))
    calls = []

    async def generate(*args, variation=0, **kwargs):
        calls.append(variation)
        if variation == failed_song:
            raise ProviderGlobalError("provider quota exhausted")
        return await original_generate(*args, variation=variation, **kwargs)

    orchestrator.music_provider.generate = generate
    app = create_app(settings, orchestrator)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        response = await client.post(endpoint, json={"prompt": "rock", "count": 2})
        if endpoint == "/api/jobs":
            assert response.status_code == 202
            job_id = response.json()["jobId"]
            await app.state.jobs[job_id].task
        else:
            assert response.status_code == 500
            job_id = next(iter(app.state.jobs))
        row = (await client.get(f"/api/jobs/{job_id}")).json()
        assert calls == list(range(failed_song + 1))
        assert row["status"] == "failed"
        assert row["currentSong"] is None
        assert row["error"] == "provider quota exhausted"
        assert [song["status"] for song in row["songStates"]] == (
            ["succeeded", "failed"] if failed_song else ["failed", "failed"]
        )
        if failed_song:
            assert row["result"]["count"] == 1
            assert row["result"]["songNumber"] == 1
            assert (await client.get(row["result"]["fullTrack"])).content == b"ID3-full-audio"
        else:
            assert row["result"] is None
        restored = load_jobs(settings.output_dir)[job_id]
        assert restored.status == "failed"
        assert restored.song_states == row["songStates"]
        assert (restored.result is None) == (row["result"] is None)
    events = [call.args[0].name for call in orchestrator.events.publish.await_args_list]
    assert events == ["generation.started", "generation.failed"]
