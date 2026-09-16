from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from app.main import GenerationJob, create_app
from tests.helpers import make_orchestrator, make_settings


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver")


def seed_job(app, settings, job_id: str = "job-replace") -> tuple[GenerationJob, Path]:
    song_dir = settings.output_dir / "jobs" / job_id / "song_1"
    song_dir.mkdir(parents=True, exist_ok=True)
    full_track = song_dir / "full_song.wav"
    vocal = song_dir / "demo_vocal.mp3"
    full_track.write_bytes(b"RIFF-full")
    vocal.write_bytes(b"ID3-vocal")
    job = GenerationJob(
        job_id=job_id,
        prompt="rock",
        structured_prompt="[Genre: Rock]",
        lyrics="[Verse]\ntest",
        status="succeeded",
        stage="completed",
        progress=100,
        message="音乐生成完成",
        result={
            "success": True,
            "jobId": job_id,
            "prompt": "rock",
            "durationMinutes": "auto",
            "structuredPrompt": "[Genre: Rock]",
            "lyrics": "[Verse]\ntest",
            "count": 1,
            "alternatives": [],
            "fullTrack": full_track.relative_to(settings.output_dir).as_posix(),
            "stems": {"vocal": vocal.relative_to(settings.output_dir).as_posix()},
            "stemUrls": [vocal.relative_to(settings.output_dir).as_posix()],
            "waveforms": {"vocal": [0.5]},
            "splitEnabled": True,
            "debug": {},
        },
    )
    app.state.jobs[job_id] = job
    job.save(settings.output_dir)
    return job, vocal


class BlockingVoiceEngine:
    loaded = True

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def convert(self, input_path: Path, output_path: Path, **_params) -> None:
        self.calls += 1
        assert input_path.name == "demo_vocal.mp3"
        self.started.set()
        await self.release.wait()
        output_path.write_bytes(b"RIFF-replaced")


class FailingVoiceEngine:
    loaded = True

    async def convert(self, *_args, **_params) -> None:
        raise RuntimeError("secret /tmp/model failure")


async def test_replace_runs_as_job_operation_and_caches_result(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    engine = BlockingVoiceEngine()
    app = create_app(settings, orchestrator, engine)
    job, _ = seed_job(app, settings)

    async with client(app) as http:
        accepted = await http.post(f"/api/jobs/{job.job_id}/replace?song=0")
        await asyncio.wait_for(engine.started.wait(), 2)
        running = (await http.get(f"/api/jobs/{job.job_id}")).json()
        history_running = (await http.get("/api/jobs")).json()["jobs"][0]
        duplicate = await http.post(f"/api/jobs/{job.job_id}/replace?song=0")
        blocked_split = await http.post(f"/api/jobs/{job.job_id}/split?song=0")
        events_task = asyncio.create_task(http.get(f"/api/jobs/{job.job_id}/events"))
        await asyncio.sleep(0)
        assert not events_task.done()
        engine.release.set()
        await job.replace_task
        events = await asyncio.wait_for(events_task, 2)
        completed = (await http.get(f"/api/jobs/{job.job_id}")).json()
        audio = await http.get(completed["result"]["replacedVocal"])
        cached = await http.post(f"/api/jobs/{job.job_id}/replace?song=0")

    assert accepted.status_code == duplicate.status_code == 202
    assert (running["status"], running["stage"], running["progress"]) == (
        "running",
        "replacing_vocal",
        90,
    )
    assert running["replaceStatus"] == "running"
    assert running["replaceSong"] == 0
    assert running["message"] == "RVC 正在替换人声"
    assert history_running["status"] == "succeeded"
    assert "replaceStatus" not in history_running
    assert blocked_split.status_code == 409
    assert completed["status"] == completed["replaceStatus"] == "succeeded"
    assert completed["result"]["replacedVocal"].endswith("/demo_rvc_vocal.wav")
    assert audio.content == b"RIFF-replaced"
    assert cached.status_code == 200
    assert engine.calls == 1
    assert events.text.startswith("data: ")
    assert "event: done" in events.text
    assert json.loads(events.text.rsplit("data: ", 1)[1])["replaceStatus"] == "succeeded"

    restarted = create_app(settings, orchestrator, engine)
    async with client(restarted) as http:
        restored = (await http.get(f"/api/jobs/{job.job_id}")).json()
    assert restored["result"]["replacedVocal"].endswith("/demo_rvc_vocal.wav")


async def test_replace_failure_and_cancel_preserve_generation(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    failed_app = create_app(settings, orchestrator, FailingVoiceEngine())
    failed_job, _ = seed_job(failed_app, settings, "job-failed")

    async with client(failed_app) as http:
        assert (await http.post(f"/api/jobs/{failed_job.job_id}/replace")).status_code == 202
        await failed_job.replace_task
        failed = (await http.get(f"/api/jobs/{failed_job.job_id}")).json()
        history = (await http.get("/api/jobs")).json()["jobs"]

    assert failed["status"] == "succeeded"
    assert failed["replaceStatus"] == "failed"
    assert failed["replaceError"] == "人声替换失败，请检查 RVC 配置后重试。"
    assert "secret" not in json.dumps(failed)
    assert all("replaceStatus" not in item for item in history)

    engine = BlockingVoiceEngine()
    app = create_app(settings, orchestrator, engine)
    job, _ = seed_job(app, settings, "job-cancelled")
    async with client(app) as http:
        await http.post(f"/api/jobs/{job.job_id}/replace")
        await asyncio.wait_for(engine.started.wait(), 2)
        cancelled = await http.patch(f"/api/jobs/{job.job_id}", json={"status": "cancelled"})
        assert not job.replace_task.done()
        assert orchestrator.capacity.active == 1
        engine.release.set()
        await job.replace_task
        after_cancel = (await http.get(f"/api/jobs/{job.job_id}")).json()

    assert cancelled.json()["status"] == "succeeded"
    assert cancelled.json()["replaceStatus"] == "cancelled"
    assert after_cancel["replaceStatus"] == "cancelled"
    assert "replacedVocal" not in after_cancel["result"]
    assert job.status == "succeeded"
    assert orchestrator.capacity.active == 0


async def test_replace_rejects_missing_inputs_and_capacity(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator, BlockingVoiceEngine())
    job, vocal = seed_job(app, settings)

    async with client(app) as http:
        assert (await http.post("/api/jobs/missing/replace")).status_code == 404
        assert (await http.post(f"/api/jobs/{job.job_id}/replace?song=1")).status_code == 404
        job.result["stems"] = {}
        no_vocal = await http.post(f"/api/jobs/{job.job_id}/replace")
        job.result["stems"] = {"vocal": vocal.relative_to(settings.output_dir).as_posix()}
        vocal.unlink()
        missing_file = await http.post(f"/api/jobs/{job.job_id}/replace")
        vocal.write_bytes(b"ID3-vocal")
        await orchestrator.capacity.acquire()
        try:
            limited = await http.post(f"/api/jobs/{job.job_id}/replace")
        finally:
            await orchestrator.capacity.release()

    assert no_vocal.status_code == missing_file.status_code == 409
    assert limited.status_code == 429
    assert job.replace_status is None
