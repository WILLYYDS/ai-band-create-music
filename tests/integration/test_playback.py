import os
import shutil
import sys
import wave
from pathlib import Path

import httpx
import pytest

from app.main import GenerationJob, _render_result_urls, create_app
from app.services.audio_files import make_playback_mp3
from app.services.job_files import capture_mix_artifact, drop_mix_artifact, restore_mix_artifact
from tests.helpers import make_orchestrator, make_settings


def wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        output.writeframes(b"\0\0" * 8000)


async def test_four_preview_urls_range_and_stale_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    folder = settings.output_dir / "jobs/j/song_1"
    names = {
        "fullTrack": "full.wav",
        "vocal": "vocal.wav",
        "replacedVocal": "replaced.wav",
        "mixedTrack": "mix.wav",
    }
    paths = {key: folder / name for key, name in names.items()}
    for path in paths.values():
        wav(path)
    previews = {
        key: await make_playback_mp3(path, settings.output_dir) for key, path in paths.items()
    }
    assert all(previews.values())
    assert all(Path(preview).parent.name == "playtrack" for preview in previews.values())
    assert all((settings.output_dir / preview).is_file() for preview in previews.values())
    result = {
        "fullTrack": paths["fullTrack"].relative_to(settings.output_dir).as_posix(),
        "stems": {"vocal": paths["vocal"].relative_to(settings.output_dir).as_posix()},
        "replacedVocal": paths["replacedVocal"].relative_to(settings.output_dir).as_posix(),
        "mixedTrack": paths["mixedTrack"].relative_to(settings.output_dir).as_posix(),
        "playback": {
            "fullTrack": previews["fullTrack"],
            "stems": {"vocal": previews["vocal"]},
            "replacedVocal": previews["replacedVocal"],
            "mixedTrack": previews["mixedTrack"],
        },
    }
    rendered = _render_result_urls(result, "http://testserver", settings)
    assert set(rendered["playback"]) == {"fullTrack", "stems", "replacedVocal", "mixedTrack"}
    assert rendered["fullTrack"].endswith(".wav")
    original_open = Path.open

    def reject_wav_read(path: Path, *args, **kwargs):
        if path.suffix == ".wav":
            raise AssertionError("result rendering must not read WAV bytes")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", reject_wav_read)
        assert _render_result_urls(result, "http://testserver", settings)["playback"] == rendered[
            "playback"
        ]
    app = create_app(settings, make_orchestrator(settings))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        response = await client.get(
            rendered["playback"]["fullTrack"], headers={"Range": "bytes=0-2"}
        )
    assert response.status_code == 206
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.headers["content-length"] == "3"
    assert response.headers["content-range"].startswith("bytes 0-2/")
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert len(response.content) == 3

    previous_mtime = paths["replacedVocal"].stat().st_mtime_ns
    replacement = paths["replacedVocal"].with_suffix(".new")
    wav(replacement)
    replacement.write_bytes(replacement.read_bytes()[:-2] + b"\1\0")
    os.utime(replacement, ns=(previous_mtime + 1_000_000, previous_mtime + 1_000_000))
    replacement.replace(paths["replacedVocal"])
    assert (
        "replacedVocal"
        not in _render_result_urls(result, "http://testserver", settings)["playback"]
    )
    fresh = await make_playback_mp3(paths["replacedVocal"], settings.output_dir)
    assert fresh != previews["replacedVocal"]
    assert not (settings.output_dir / previews["replacedVocal"]).exists()
    saved = capture_mix_artifact(result)
    assert saved["mixPlayback"] == previews["mixedTrack"]
    drop_mix_artifact(result)
    assert "mixedTrack" not in result["playback"]
    assert restore_mix_artifact(result, saved)
    assert result["playback"]["mixedTrack"] == previews["mixedTrack"]


async def test_generation_and_split_publish_previews(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, mock_full_song_path=tmp_path / "source.wav")
    orchestrator = make_orchestrator(settings)
    wav(settings.mock_full_song_path)
    app = create_app(settings, orchestrator)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        started = await client.post("/api/jobs", json={"prompt": "test"})
        job_id = started.json()["jobId"]
        await app.state.jobs[job_id].task
        generated = (await client.get(f"/api/jobs/{job_id}")).json()["result"]
        assert generated["fullTrack"].endswith(".wav")
        assert generated["playback"]["fullTrack"].endswith(".mp3")
        assert (await client.post(f"/api/jobs/{job_id}/split")).status_code == 202
        await app.state.jobs[job_id].split_task
        split = (await client.get(f"/api/jobs/{job_id}")).json()["result"]
        assert "stems" not in split["playback"]
        assert all(url.endswith(".wav") for url in split["stems"].values())


async def test_failed_encoding_leaves_wav_and_no_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "full.wav"
    wav(source)
    original = source.read_bytes()

    def unavailable() -> None:
        raise RuntimeError("ffmpeg unavailable")

    monkeypatch.setattr("app.services.stems.prepare_ffmpeg_environment", unavailable)
    assert await make_playback_mp3(source, tmp_path) is None
    assert source.read_bytes() == original
    assert not list(tmp_path.rglob("*.mp3"))


async def test_encoder_timeout_kills_child_and_removes_temp(tmp_path: Path, monkeypatch) -> None:
    import asyncio

    source = tmp_path / "full.wav"
    wav(source)
    original = source.read_bytes()
    spawned = []
    create = asyncio.create_subprocess_exec

    async def sleeping_encoder(*_args, **kwargs):
        process = await create(sys.executable, "-c", "import time; time.sleep(10)", **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr("app.services.audio_files.PLAYBACK_ENCODE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr("app.services.audio_files.asyncio.create_subprocess_exec", sleeping_encoder)
    assert await make_playback_mp3(source, tmp_path) is None
    assert spawned[0].returncode is not None
    assert source.read_bytes() == original
    assert not list((tmp_path / "playtrack").iterdir())


async def test_replacement_waveform_and_delete_restore_preview(tmp_path: Path) -> None:
    class CopyVoice:
        loaded = True

        async def convert(self, source: Path, target: Path, **_kwargs: object) -> None:
            shutil.copy2(source, target)

    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings), CopyVoice())
    folder = settings.output_dir / "jobs/j/song_1"
    full, vocal = folder / "full.wav", folder / "vocal.wav"
    wav(full)
    wav(vocal)
    result = {
        "success": True,
        "jobId": "j",
        "prompt": "test",
        "durationMinutes": "auto",
        "structuredPrompt": "test",
        "lyrics": "test",
        "count": 1,
        "alternatives": [],
        "fullTrack": full.relative_to(settings.output_dir).as_posix(),
        "stems": {"vocal": vocal.relative_to(settings.output_dir).as_posix()},
        "stemUrls": [vocal.relative_to(settings.output_dir).as_posix()],
        "waveforms": {},
        "splitEnabled": True,
        "debug": {},
    }
    job = GenerationJob(job_id="j", prompt="test", status="succeeded", result=result)
    app.state.jobs["j"] = job
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        accepted = await client.post("/api/jobs/j/replace")
        assert accepted.status_code == 202
        await job.replace_task
        completed = (await client.get("/api/jobs/j")).json()["result"]
        assert len(completed["waveforms"]["replaced"]) == 640
        assert completed["playback"]["replacedVocal"].endswith(".mp3")
        original_preview = completed["playback"]["replacedVocal"]
        filename = completed["replacedVocal"]
        payload = {"filename": filename, "job_id": "j", "song": "0"}
        assert (
            await client.request("DELETE", "/api/voice/result", data=payload)
        ).status_code == 200
        deleted = (await client.get("/api/jobs/j")).json()["result"]
        assert "replacedVocal" not in deleted
        assert "replacedVocal" not in deleted["playback"]
        assert "replaced" not in deleted["waveforms"]
        assert (await client.request("PUT", "/api/voice/result", data=payload)).status_code == 200
        restored = (await client.get("/api/jobs/j")).json()["result"]
        assert restored["playback"]["replacedVocal"] == original_preview
        assert len(restored["waveforms"]["replaced"]) == 640
