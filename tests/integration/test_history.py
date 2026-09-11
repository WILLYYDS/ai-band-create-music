import asyncio
import json
import wave
from unittest.mock import AsyncMock

import httpx

from app.main import GenerationJob, create_app, load_jobs
from tests.helpers import make_orchestrator, make_settings


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver")


async def test_direct_history_restart_and_duplicate_import(tmp_path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    orchestrator.stem_separator.split = AsyncMock(side_effect=AssertionError("no splitting"))
    app = create_app(settings, orchestrator)
    async with client(app) as http:
        result = (await http.post("/api/generate", json={"prompt": "rock"})).json()
        history = (await http.get("/api/jobs")).json()["jobs"]
    assert len(history) == 1
    assert history[0]["createdAt"] == result["createdAt"]
    assert history[0]["result"]["fullTrack"] == result["fullTrack"]
    stored = json.loads(
        (settings.output_dir / "jobs" / result["jobId"] / "job.json").read_text()
    )
    assert stored["diagnostics"]["prompt"] == "rock"
    assert stored["diagnostics"]["provider"] == "minimax_music"
    assert stored["diagnostics"]["durationSource"] == "provider"
    assert "effectiveDurationSeconds" not in stored["diagnostics"]
    assert "effectiveDurationMinutes" not in stored["diagnostics"]
    orchestrator.stem_separator.split.assert_not_called()
    copy = settings.output_dir / "full_song_minimax_duplicate.wav"
    copy.write_bytes(b"ID3-full-audio")
    restarted = create_app(settings, orchestrator)
    async with client(restarted) as http:
        restored = (await http.get("/api/jobs")).json()["jobs"]
        assert len(restored) == 1
        assert (await http.get(result["fullTrack"])).content == b"ID3-full-audio"
    assert copy.exists()


async def test_history_urls_do_not_persist_request_host(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with client(app) as http:
        result = (
            await http.post(
                "/api/generate", json={"prompt": "rock"}, headers={"Host": "attacker.test"}
            )
        ).json()
        stored = json.loads(
            (settings.output_dir / "jobs" / result["jobId"] / "job.json").read_text()
        )
        history = (await http.get("/api/jobs")).json()["jobs"]

    assert result["fullTrack"].startswith("http://attacker.test/output/")
    assert stored["result"]["fullTrack"].endswith(f"{result['jobId']}_1.mp3")
    assert "://" not in stored["result"]["fullTrack"]
    assert history[0]["result"]["fullTrack"].startswith("http://testserver/output/")


def test_interrupted_and_corrupt_metadata(tmp_path):
    settings = make_settings(tmp_path)
    job = GenerationJob(
        job_id="interrupted", prompt="rock", status="running", step=4, total_steps=30
    )
    job.save(settings.output_dir)
    corrupt = settings.output_dir / "jobs" / "corrupt" / "job.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{broken")
    restored = load_jobs(settings.output_dir)
    assert restored["interrupted"].status == "failed"
    assert restored["interrupted"].step is None
    assert "重启" in restored["interrupted"].error
    assert (
        json.loads((corrupt.parent.parent / "interrupted" / "job.json").read_text())["status"]
        == "failed"
    )
    assert corrupt.read_text() == "{broken"
    assert not list(settings.output_dir.rglob("*.tmp"))


async def test_legacy_import_real_waveform_and_restart(tmp_path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    song = settings.output_dir / "full_song_old.wav"
    with wave.open(str(song), "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x10" * 8000)
    original = song.read_bytes()
    app = create_app(settings, orchestrator)
    async with client(app) as http:
        rows = (await http.get("/api/jobs")).json()["jobs"]
    assert len(rows) == 1
    assert rows[0]["result"]["waveforms"]["full"] == [1.0] * 64
    async with client(create_app(settings, orchestrator)) as http:
        assert (await http.get("/api/jobs")).json()["jobs"] == rows
    assert song.read_bytes() == original


async def test_job_forwards_actual_provider_counts(tmp_path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    reached, release = asyncio.Event(), asyncio.Event()
    original = orchestrator.music_provider.generate

    async def generate(*args, progress=None, **kwargs):
        await progress("denoising", 7, 90)
        reached.set()
        await release.wait()
        return await original(*args, **kwargs)

    orchestrator.music_provider.generate = generate
    app = create_app(settings, orchestrator)
    async with client(app) as http:
        job_id = (await http.post("/api/jobs", json={"prompt": "rock"})).json()["jobId"]
        await asyncio.wait_for(reached.wait(), 2)
        row = (await http.get(f"/api/jobs/{job_id}")).json()
        assert (row["stage"], row["step"], row["totalSteps"], row["progress"]) == (
            "denoising",
            7,
            90,
            None,
        )
        release.set()
        await app.state.jobs[job_id].task
        assert load_jobs(settings.output_dir)[job_id].status == "succeeded"


async def test_direct_failure_is_persisted(tmp_path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    orchestrator.music_provider.generate = AsyncMock(side_effect=RuntimeError("provider died"))
    app = create_app(settings, orchestrator)
    async with client(app) as http:
        assert (await http.post("/api/generate", json={"prompt": "rock"})).status_code == 500
    row = next(iter(load_jobs(settings.output_dir).values()))
    assert row.status == "failed"
    assert row.error == "provider died"
