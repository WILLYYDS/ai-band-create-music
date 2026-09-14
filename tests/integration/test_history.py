import asyncio
import json
import wave
from unittest.mock import AsyncMock

import httpx
from starlette.requests import Request

from app.core.errors import CapacityExceededError
from app.main import GenerationJob, create_app, load_jobs
from tests.helpers import make_orchestrator, make_settings


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver")


async def test_direct_history_restart_and_duplicate_import(tmp_path):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    # 生成路径只产出完整混音。
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


async def test_history_song_starts_async_split_and_updates_result(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)

    async with client(app) as http:
        generated = (await http.post("/api/generate", json={"prompt": "rock"})).json()
        job_id = generated["jobId"]
        original_split = orchestrator.stem_separator.split
        started, release = asyncio.Event(), asyncio.Event()

        async def split(*args, **kwargs):
            started.set()
            await release.wait()
            return await original_split(*args, **kwargs)

        monkeypatch.setattr(orchestrator.stem_separator, "split", split)
        monkeypatch.setattr(
            "app.main.extract_waveforms",
            AsyncMock(
                return_value={
                    name: [0.5] for name in ("vocal", "drums", "bass", "other")
                }
            ),
        )
        accepted = await http.post(f"/api/jobs/{job_id}/split?song=0")
        await asyncio.wait_for(started.wait(), 2)
        running = (await http.get(f"/api/jobs/{job_id}")).json()
        duplicate = await http.post(f"/api/jobs/{job_id}/split?song=0")
        release.set()
        await app.state.jobs[job_id].split_task
        completed = (await http.get(f"/api/jobs/{job_id}")).json()
        cache_guard = AsyncMock(
            side_effect=AssertionError("cached split must not run again")
        )
        monkeypatch.setattr(orchestrator.stem_separator, "split", cache_guard)
        cached = await http.post(f"/api/jobs/{job_id}/split?song=0")

        for stem in list(completed["result"]["stems"]):
            assert (await http.delete(f"/api/jobs/{job_id}/stems/{stem}")).status_code == 204
        assert (await http.get(f"/api/jobs/{job_id}")).json()["result"]["stems"] == {}
        monkeypatch.setattr(orchestrator.stem_separator, "split", original_split)
        resplit = await http.post(f"/api/jobs/{job_id}/split?song=0")
        await app.state.jobs[job_id].split_task
        resplit_result = (await http.get(f"/api/jobs/{job_id}")).json()

    assert accepted.status_code == duplicate.status_code == 202
    assert (running["status"], running["stage"], running["progress"]) == (
        "running",
        "splitting",
        76,
    )
    assert completed["status"] == "succeeded"
    assert completed["result"]["splitEnabled"] is True
    assert sorted(completed["result"]["stems"]) == ["bass", "drums", "other", "vocal"]
    assert completed["result"]["waveforms"]["vocal"] == [0.5]
    assert cached.status_code == 200
    assert cached.json()["status"] == "succeeded"
    cache_guard.assert_not_called()
    assert resplit.status_code == 202
    assert sorted(resplit_result["result"]["stems"]) == ["bass", "drums", "other", "vocal"]
    assert app.state.jobs[job_id].deleted_stems == {}
    assert not list((settings.output_dir / ".trash" / job_id).glob("*"))


async def test_split_reserves_job_before_capacity_await(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, max_concurrent_generations=3)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    original_acquire = orchestrator.capacity.acquire
    entered, release = asyncio.Event(), asyncio.Event()
    acquire_calls = 0

    async def acquire():
        nonlocal acquire_calls
        acquire_calls += 1
        entered.set()
        await release.wait()
        await original_acquire()

    monkeypatch.setattr(
        "app.main.extract_waveforms",
        AsyncMock(return_value={name: [0.5] for name in ("vocal", "drums", "bass", "other")}),
    )
    async with client(app) as http:
        job_id = (
            await http.post("/api/generate", json={"prompt": "rock", "count": 2})
        ).json()["jobId"]
        monkeypatch.setattr(orchestrator.capacity, "acquire", acquire)
        first = asyncio.create_task(http.post(f"/api/jobs/{job_id}/split?song=0"))
        await asyncio.wait_for(entered.wait(), 2)
        duplicate = await http.post(f"/api/jobs/{job_id}/split?song=0")
        other_song = await http.post(f"/api/jobs/{job_id}/split?song=1")
        release.set()
        accepted = await first
        await app.state.jobs[job_id].split_task

    assert accepted.status_code == duplicate.status_code == 202
    assert acquire_calls == 1
    assert other_song.status_code == 409
    assert "第 1 首" in other_song.json()["message"]
    assert app.state.jobs[job_id].status == "succeeded"


async def test_split_failure_and_cancel_preserve_completed_generation(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)

    async with client(app) as http:
        job_id = (await http.post("/api/generate", json={"prompt": "rock"})).json()["jobId"]
        monkeypatch.setattr(
            orchestrator.stem_separator,
            "split",
            AsyncMock(side_effect=RuntimeError("secret /tmp/model failed")),
        )
        assert (await http.post(f"/api/jobs/{job_id}/split?song=0")).status_code == 202
        await app.state.jobs[job_id].split_task
        failed = (await http.get(f"/api/jobs/{job_id}")).json()

    assert failed["status"] == "succeeded"
    assert failed["splitStatus"] == "failed"
    assert failed["splitError"] == "音轨分离失败，请检查服务配置后重试。"
    assert "secret" not in json.dumps(failed)
    assert app.state.jobs[job_id].status == "succeeded"
    assert load_jobs(settings.output_dir)[job_id].status == "succeeded"
    async with client(create_app(settings, orchestrator)) as http:
        assert (await http.get(f"/api/jobs/{job_id}")).json()["status"] == "succeeded"

    started = asyncio.Event()

    async def blocking_split(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(orchestrator.stem_separator, "split", blocking_split)
    async with client(app) as http:
        await http.post(f"/api/jobs/{job_id}/split?song=0")
        await asyncio.wait_for(started.wait(), 2)
        await http.patch(f"/api/jobs/{job_id}", json={"status": "cancelled"})
        assert app.state.jobs[job_id].split_status == "cancelled"
        await asyncio.sleep(0)
        cancelled = (await http.get(f"/api/jobs/{job_id}")).json()

    assert cancelled["status"] == "succeeded"
    assert cancelled["splitStatus"] == "cancelled"
    assert app.state.jobs[job_id].status == "succeeded"
    assert load_jobs(settings.output_dir)[job_id].status == "succeeded"


async def test_split_rejects_invalid_or_unavailable_jobs(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator)
    async with client(app) as http:
        assert (await http.post("/api/jobs/missing/split")).status_code == 404
        job_id = (await http.post("/api/generate", json={"prompt": "rock"})).json()["jobId"]
        job = app.state.jobs[job_id]
        full_track = job.result.pop("fullTrack")
        assert (await http.post(f"/api/jobs/{job_id}/split")).status_code == 409
        job.result["fullTrack"] = full_track
        full_file = next(settings.output_dir.glob(f"full_song_{job_id}*"))
        full_file.unlink()
        assert (await http.post(f"/api/jobs/{job_id}/split")).status_code == 409
        full_file.write_bytes(b"ID3-full-audio")
        monkeypatch.setattr(
            orchestrator.capacity,
            "acquire",
            AsyncMock(side_effect=CapacityExceededError("full")),
        )
        limited = await http.post(f"/api/jobs/{job_id}/split")

    assert limited.status_code == 429
    assert job.split_status is None


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


async def test_invalid_persisted_output_url_does_not_break_history(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    async with client(app) as http:
        job_id = (await http.post("/api/generate", json={"prompt": "rock"})).json()["jobId"]

    job_file = settings.output_dir / "jobs" / job_id / "job.json"
    stored = json.loads(job_file.read_text())
    stored["result"]["fullTrack"] = "/etc/passwd"
    job_file.write_text(json.dumps(stored))

    async with client(create_app(settings, make_orchestrator(settings))) as http:
        history = await http.get("/api/jobs")
        detail = await http.get(f"/api/jobs/{job_id}")

    assert history.status_code == detail.status_code == 200
    assert detail.json()["result"]["fullTrack"] == "/etc/passwd"


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


async def test_history_preserves_failed_jobs_and_events_send_done(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    failed = GenerationJob(job_id="failed", prompt="bad", status="failed", stage="failed")
    failed.error = failed.message = "provider died"
    app.state.jobs[failed.job_id] = failed

    async with client(app) as http:
        assert (await http.get("/api/jobs")).json()["jobs"][0]["error"] == "provider died"
        assert (await http.get("/api/jobs/failed")).json()["error"] == "provider died"
        events = await http.get("/api/jobs/failed/events")

    assert events.headers["content-type"].startswith("text/event-stream")
    assert events.text.startswith("event: done\ndata: ")
    assert json.loads(events.text.split("data: ", 1)[1])["status"] == "failed"


async def test_unstarted_job_events_do_not_register_subscriber(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    app.state.jobs["active"] = GenerationJob(
        job_id="active", prompt="rock", status="running", stage="music"
    )
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/jobs/{job_id}/events"
    )
    request = Request({"type": "http", "app": app, "headers": []})

    await endpoint("active", request)

    assert app.state.job_subscribers == {}


async def test_job_events_keep_alive_after_timeout(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = GenerationJob(job_id="active", prompt="rock", status="running", stage="music")
    app.state.jobs[job.job_id] = job
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/jobs/{job_id}/events"
    )
    request = Request({
        "type": "http",
        "app": app,
        "headers": [],
        "scheme": "http",
        "server": ("testserver", 80),
        "path": "/api/jobs/active/events",
        "root_path": "",
        "query_string": b"",
        "method": "GET",
    })

    async def time_out(awaitable, *_args, **_kwargs):
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", time_out)
    response = await endpoint("active", request)
    assert (await anext(response.body_iterator)).startswith("data: ")
    assert await anext(response.body_iterator) == ": keep-alive\n\n"
    await response.body_iterator.aclose()


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
