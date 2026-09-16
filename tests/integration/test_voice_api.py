from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx

from app.main import create_app
from tests.helpers import make_orchestrator, make_settings


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


async def test_voice_result_download_delete_and_restore(tmp_path: Path) -> None:
    """替换结果的下载 / 删除 / 恢复生命周期（前端 replace-complete 正在用这条路径）。"""
    settings = make_settings(tmp_path)
    job_id = "job-1"
    app = create_app(settings, make_orchestrator(settings))
    song_dir = seed_job(app, settings, job_id)
    result_path = song_dir / "real_song_rvc_vocal.wav"
    result_path.write_bytes(b"RIFF-converted-wav")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        direct_download = await client.get(
            f"/output/jobs/{job_id}/song_1/real_song_rvc_vocal.wav"
        )
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

    assert direct_download.content == b"RIFF-converted-wav"
    assert deleted.status_code == 200
    assert missing.status_code == 404
    assert restored.status_code == 200
    assert result_path.is_file()


async def test_voice_result_rejects_path_traversal(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
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


async def test_voice_result_accepts_url_shaped_filenames(tmp_path: Path) -> None:
    """前端会原样回传结果 URL，DELETE/PUT 必须容忍它（历史上前端代理层做过这层归一化）。

    只接受裸文件名会让"删除替换人声 → 撤回"在代理层不再归一化时直接 400，
    进而连撤回后的重新替换也一起失败。
    """
    settings = make_settings(tmp_path)
    job_id = "job-1"
    app = create_app(settings, make_orchestrator(settings))
    song_dir = seed_job(app, settings, job_id)
    result_path = song_dir / "song_rvc_vocal.wav"
    result_path.write_bytes(b"RIFF-converted-wav")
    forms = (
        "song_rvc_vocal.wav",
        f"/api/music/output/jobs/{job_id}/song_1/song_rvc_vocal.wav",
        f"http://127.0.0.1:8010/output/jobs/{job_id}/song_1/song_rvc_vocal.wav",
        f"http://localhost:3000/api/music/output/jobs/{job_id}/song_1/song_rvc_vocal.wav",
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        for value in forms:
            result_path.write_bytes(b"RIFF-converted-wav")
            deleted = await client.request(
                "DELETE", "/api/voice/result", data={"job_id": job_id, "filename": value}
            )
            assert deleted.status_code == 200, value
            assert not result_path.exists(), value
            restored = await client.request(
                "PUT", "/api/voice/result", data={"job_id": job_id, "filename": value}
            )
            assert restored.status_code == 200, value
            assert result_path.is_file(), value

        # 归一化之后仍然禁止路径穿越
        escaped = await client.request(
            "DELETE", "/api/voice/result", data={"job_id": job_id, "filename": "../secret.wav"}
        )
        assert escaped.status_code == 400
