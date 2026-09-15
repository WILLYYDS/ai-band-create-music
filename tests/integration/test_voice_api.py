from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx

from app.core.errors import GenerationError
from app.main import create_app
from tests.helpers import make_orchestrator, make_settings


class FakeVoiceEngine:
    loaded = True

    async def convert(self, input_path: Path, output_path: Path, **params) -> None:
        assert input_path.is_file()
        assert params["f0_method"] == "rmvpe"
        assert params["index_rate"] == 0.5
        assert params["rms_mix_rate"] == 1.0
        output_path.write_bytes(b"RIFF-converted-wav")


def seed_job(app, settings, job_id: str) -> Path:
    song_dir = settings.output_dir / "jobs" / job_id / "song_1"
    song_dir.mkdir(parents=True, exist_ok=True)
    full_track = song_dir / "full_song.wav"
    full_track.write_bytes(b"RIFF-full")
    app.state.jobs[job_id] = SimpleNamespace(
        result={
            "fullTrack": full_track.relative_to(settings.output_dir).as_posix(),
            "alternatives": [],
        }
    )
    return song_dir


async def test_voice_conversion_download_delete_and_restore(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = "job-1"
    app = create_app(settings, make_orchestrator(settings), FakeVoiceEngine())
    seed_job(app, settings, job_id)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        converted = await client.post(
            "/api/voice/convert",
            files={"file": ("real_song_vocal.mp3", b"ID3" + b"0" * 20_000, "audio/mpeg")},
            data={"job_id": job_id, "index_rate": "0.5", "song_name": "AI 生成曲目"},
        )
        direct_download = await client.get(converted.json()["url"])
        deleted = await client.request(
            "DELETE",
            "/api/voice/result",
            data={"job_id": job_id, "filename": "real_song_rvc_vocal.wav"},
        )
        missing = await client.get(f"/output/jobs/{job_id}/song_1/real_song_rvc_vocal.wav")
        restored = await client.request(
            "PUT",
            "/api/voice/result",
            data={"job_id": job_id, "filename": "real_song_rvc_vocal.wav"},
        )

    assert converted.status_code == 200
    assert converted.headers["x-rvc-output"] == "real_song_rvc_vocal.wav"
    assert converted.json() == {
        "success": True,
        "filename": "real_song_rvc_vocal.wav",
        "url": f"/output/jobs/{job_id}/song_1/real_song_rvc_vocal.wav",
    }
    assert direct_download.content == b"RIFF-converted-wav"
    assert deleted.status_code == 200
    assert missing.status_code == 404
    assert restored.status_code == 200
    assert (settings.output_dir / "jobs" / job_id / "song_1" / "real_song_rvc_vocal.wav").is_file()


async def test_voice_result_rejects_path_traversal(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings), FakeVoiceEngine())
    seed_job(app, settings, "job-1")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.request(
            "DELETE",
            "/api/voice/result",
            data={"job_id": "job-1", "filename": "../secret.wav"},
        )
    assert response.status_code == 400

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.request(
            "DELETE",
            "/api/voice/result",
            data={"job_id": "job-1", "filename": "legacy.mp3"},
        )
    assert response.status_code == 400

    missing_job_dir = settings.output_dir / "jobs" / "job-does-not-exist"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/voice/convert",
            files={"file": ("voice.wav", b"RIFF-audio", "audio/wav")},
            data={"job_id": "job-does-not-exist"},
        )
    assert response.status_code == 404
    assert not missing_job_dir.exists()

    app = create_app(settings, make_orchestrator(settings), FakeVoiceEngine())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/voice/convert",
            files={"file": ("voice.wav", b"RIFF-audio", "audio/wav")},
            data={"job_id": ".."},
        )
    assert response.status_code == 400


async def test_voice_mix_uses_server_side_stems(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    job_id = "job-1"
    app = create_app(settings, make_orchestrator(settings), FakeVoiceEngine())
    song_dir = seed_job(app, settings, job_id)
    vocal = song_dir / "real_song_rvc_vocal.wav"
    vocal.write_bytes(b"RIFF-vocal")
    stems = {}
    for name in ("drums", "bass", "other"):
        path = song_dir / f"real_song_{name}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ID3-stem")
        stems[name] = f"/output/jobs/{job_id}/song_1/{path.name}"

    async def fake_mix(
        inputs: list[Path],
        reference_path: Path,
        output_path: Path,
        timeout_seconds: float,
    ) -> None:
        assert inputs == [
            vocal,
            song_dir / "real_song_drums.mp3",
            song_dir / "real_song_bass.mp3",
            song_dir / "real_song_other.mp3",
        ]
        assert reference_path == song_dir / "full_song.wav"
        assert timeout_seconds == settings.rvc_mix_timeout_seconds
        output_path.write_bytes(b"RIFF-mixed")

    monkeypatch.setattr("app.services.voice._mix_tracks", fake_mix)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/voice/mix",
            data={"job_id": job_id, "vocal_filename": vocal.name, **stems},
            headers={"Origin": "https://frontend.example"},
        )
        direct_download = await client.get(response.json()["url"])

        (song_dir / "real_song_drums.mp3").unlink()
        missing = await client.post(
            "/api/voice/mix",
            data={"job_id": job_id, "vocal_filename": vocal.name, **stems},
        )
        (song_dir / "real_song_drums.mp3").write_bytes(b"ID3-stem")

        async def unavailable(*_args, **_kwargs):
            raise GenerationError("音频处理需要 ffmpeg 和 ffprobe")

        monkeypatch.setattr("app.services.voice._mix_tracks", unavailable)
        unavailable_response = await client.post(
            "/api/voice/mix",
            data={"job_id": job_id, "vocal_filename": vocal.name, **stems},
        )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "filename": "real_song_rvc_mix.wav",
        "url": f"/output/jobs/{job_id}/song_1/real_song_rvc_mix.wav",
    }
    assert response.headers["x-mix-output"] == "real_song_rvc_mix.wav"
    exposed_headers = {
        value.strip().lower()
        for value in response.headers["access-control-expose-headers"].split(",")
    }
    assert {"content-disposition", "x-mix-output", "x-rvc-model", "x-rvc-output"} <= (
        exposed_headers
    )
    assert direct_download.content == b"RIFF-mixed"
    assert missing.status_code == 404
    assert missing.json()["detail"] == "混音输入文件不可读"
    assert str(song_dir) not in missing.text
    assert unavailable_response.status_code == 503
    assert unavailable_response.json()["detail"] == "音频处理需要 ffmpeg 和 ffprobe"
    assert (song_dir / "real_song_rvc_mix.wav").read_bytes() == b"RIFF-mixed"


async def test_voice_mix_respects_shared_capacity(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator, FakeVoiceEngine())
    song_dir = seed_job(app, settings, "job-1")
    vocal = song_dir / "voice_rvc_vocal.wav"
    vocal.write_bytes(b"RIFF-vocal")
    stems = {}
    for name in ("drums", "bass", "other"):
        path = song_dir / f"song_{name}.mp3"
        path.write_bytes(b"ID3-stem")
        stems[name] = f"/output/jobs/job-1/song_1/{path.name}"
    await orchestrator.capacity.acquire()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/api/voice/mix",
                data={"job_id": "job-1", "vocal_filename": vocal.name, **stems},
            )
        assert orchestrator.capacity.active == 1
    finally:
        await orchestrator.capacity.release()
    assert response.status_code == 429
