import asyncio
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import httpx
import pytest

from app.main import GenerationJob, create_app
from app.services.retention import (
    cleanup_expired_jobs,
    is_expired,
    refresh_retention_inventory,
    retention_status,
    seconds_until_cleanup,
)
from tests.helpers import make_orchestrator, make_settings
from tests.integration.test_history import events_request


def add_job(app, job_id, created_at):
    job = GenerationJob(
        job_id=job_id,
        prompt="rock",
        status="succeeded",
        stage="completed",
        created_at=created_at,
    )
    root = app.state.settings.output_dir
    for song in (1, 2):
        folder = root / "jobs" / job_id / f"song_{song}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "full.mp3").write_bytes(b"audio")
        trash = root / ".trash" / job_id / f"song_{song}"
        trash.mkdir(parents=True, exist_ok=True)
        (trash / "vocal.wav").write_bytes(b"deleted audio")
    job.save(root)
    app.state.jobs[job_id] = job
    return job


async def wait_for_inventory(app):
    async def ready():
        while getattr(app.state, "retention_inventory", None) is None:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(ready(), timeout=2)


def test_expiry_boundary_and_beijing_schedule():
    now = datetime(2026, 9, 28, 19, tzinfo=timezone.utc)  # Beijing 03:00, Sep 29.
    job = GenerationJob(job_id="j", prompt="rock", created_at=(now - timedelta(days=3)).isoformat())
    assert is_expired(job, 3, now)
    assert not is_expired(job, 3, now - timedelta(microseconds=1))
    job.created_at = "2026-09-25T19:00:00Z"
    assert is_expired(job, 3, now)
    job.created_at = "2026-09-26T03:00:00+08:00"
    assert is_expired(job, 3, now)
    for invalid in ("bad", "2026-09-25T19:00:00", "9999-12-31T00:00:00+00:00"):
        job.created_at = invalid
        assert not is_expired(job, 3, now)
    assert seconds_until_cleanup(now) == 86400
    assert seconds_until_cleanup(now - timedelta(seconds=1)) == 1
    assert seconds_until_cleanup(now + timedelta(hours=1)) == 23 * 3600


async def test_cleanup_logging_busy_tasks_and_retry_after_restart(tmp_path, caplog):
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=4)).isoformat()
    add_job(app, "old", old)
    add_job(app, "recent", now.isoformat())
    add_job(app, "invalid", "unknown")
    busy = add_job(app, "busy", old)
    busy.replace_status = "failed"  # Timed out, but the RVC worker is still running.
    release = asyncio.Event()
    busy.replace_task = asyncio.create_task(release.wait())
    caplog.set_level(logging.INFO, logger="app.services.retention")
    try:
        with patch("app.services.retention.shutil.rmtree", side_effect=PermissionError("denied")):
            await cleanup_expired_jobs(app, reason="test", now=now)
        assert "old" not in app.state.jobs
        assert (settings.output_dir / ".expired/old/job/job.json").is_file()
        assert (settings.output_dir / ".expired/old/trash/song_2/vocal.wav").is_file()
        assert (settings.output_dir / "jobs/busy/job.json").is_file()
        assert (settings.output_dir / "jobs/recent/job.json").is_file()
        assert (settings.output_dir / "jobs/invalid/job.json").is_file()
        records = [json.loads(record.message) for record in caplog.records]
        failure = next(record for record in records if record["event"] == "job_delete_failed")
        assert failure["jobId"] == "old" and "denied" in failure["error"]
        assert datetime.fromisoformat(failure["time"]).tzinfo is not None
        assert records[-1]["failedJobIds"] == ["old"]
        assert records[-1]["skippedJobIds"] == ["busy", "invalid"]
    finally:
        release.set()
        await busy.replace_task
    restarted = create_app(settings, make_orchestrator(settings))
    await cleanup_expired_jobs(restarted, reason="restart", now=now)
    assert not (settings.output_dir / ".expired/old").exists()
    assert not (settings.output_dir / "jobs/busy").exists()
    assert not (settings.output_dir / ".trash/busy").exists()
    records = [json.loads(record.message) for record in caplog.records]
    assert records[-1]["deletedJobIds"] == ["busy", "old"]


@pytest.mark.parametrize("failure", ["jobs", ".trash", "target_symlink", "target_conflict"])
async def test_staging_validates_all_paths_before_creating_or_moving(tmp_path, caplog, failure):
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    add_job(app, "victim", (datetime.now(timezone.utc) - timedelta(days=5)).isoformat())
    root = settings.output_dir
    pending = root / ".expired/victim"
    if failure in {"jobs", ".trash"}:
        outside = tmp_path / "outside"
        (root / failure).rename(outside)
        (root / failure).symlink_to(outside, target_is_directory=True)
    else:
        pending.mkdir(parents=True)
        if failure == "target_symlink":
            outside = tmp_path / "outside"
            outside.mkdir()
            (pending / "trash").symlink_to(outside, target_is_directory=True)
        else:
            (pending / "trash").mkdir()
        (pending / "trash/keep.mp3").write_bytes(b"keep")
    caplog.set_level(logging.INFO, logger="app.services.retention")
    await cleanup_expired_jobs(app, reason="test")
    assert (root / "jobs/victim/song_1/full.mp3").read_bytes() == b"audio"
    assert (root / ".trash/victim/song_1/vocal.wav").read_bytes() == b"deleted audio"
    assert "victim" in app.state.jobs and not (pending / "job").exists()
    if failure in {"jobs", ".trash"}:
        assert not pending.exists()
    else:
        assert (pending / "trash/keep.mp3").read_bytes() == b"keep"
    assert json.loads(caplog.records[-1].message)["failedJobIds"] == ["victim"]


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("invalid_timestamp", [False, True])
async def test_pending_entry_cannot_expire_loaded_job_after_policy_change(
    tmp_path, caplog, dry_run, invalid_timestamp
):
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=dry_run)
    app = create_app(settings, make_orchestrator(settings))
    now = datetime.now(timezone.utc)
    created_at = "invalid" if invalid_timestamp else (now - timedelta(days=5)).isoformat()
    add_job(app, "victim", created_at)
    pending = settings.output_dir / ".expired/victim"
    pending.mkdir(parents=True)  # Simulate an empty staging directory from the old implementation.
    settings.song_retention_days = 7
    restarted = create_app(settings, make_orchestrator(settings))
    caplog.set_level(logging.INFO, logger="app.services.retention")
    await cleanup_expired_jobs(restarted, reason="restart", now=now)
    assert "victim" in restarted.state.jobs
    assert (settings.output_dir / "jobs/victim/song_1/full.mp3").read_bytes() == b"audio"
    assert (settings.output_dir / ".trash/victim/song_1/vocal.wav").is_file()
    summary = json.loads(caplog.records[-1].message)
    assert summary["deletedJobIds"] == summary["wouldDeleteJobIds"] == []
    assert summary["skippedCount"] == 1
    expected_reason = "invalid_created_at" if invalid_timestamp else "not_expired"
    assert any(
        json.loads(record.message).get("reason") == expected_reason for record in caplog.records
    )


async def test_partial_staging_retry_survives_longer_policy_after_restart(tmp_path, monkeypatch):
    from pathlib import Path

    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    now = datetime.now(timezone.utc)
    add_job(app, "victim", (now - timedelta(days=5)).isoformat())
    root = settings.output_dir
    original_rename = Path.rename

    def fail_trash_move(path, target):
        if path == root / ".trash/victim":
            raise PermissionError("cannot move trash")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_trash_move)
    await cleanup_expired_jobs(app, reason="test", now=now)
    assert not (root / "jobs/victim").exists()
    assert (root / ".expired/victim/job/job.json").is_file()
    assert (root / ".trash/victim").is_dir()
    monkeypatch.undo()
    settings.song_retention_days = 7
    restarted = create_app(settings, make_orchestrator(settings))
    assert "victim" not in restarted.state.jobs
    await cleanup_expired_jobs(restarted, reason="restart", now=now)
    assert not (root / ".expired/victim").exists()
    assert not (root / ".trash/victim").exists()


async def test_expired_access_cache_lifecycle_and_persistent_audit(tmp_path):
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    now = datetime.now(timezone.utc)
    add_job(app, "old", (now - timedelta(days=4)).isoformat())
    add_job(app, "recent", now.isoformat())
    preview = settings.output_dir / "jobs/recent/song_1/playtrack/full.playback-1-2.mp3"
    preview.parent.mkdir()
    preview.write_bytes(b"preview")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        assert [job["jobId"] for job in (await client.get("/api/jobs")).json()["jobs"]] == [
            "recent"
        ]
        for path in ("/api/jobs/old", "/api/jobs/old/events", "/output/jobs/old/song_1/full.mp3"):
            assert (await client.get(path)).status_code == 404
        for operation in ("split", "replace", "mix"):
            assert (await client.post(f"/api/jobs/old/{operation}", json={})).status_code == 404
        assert (
            await client.patch("/api/jobs/old", json={"status": "cancelled"})
        ).status_code == 404
        for method in ("DELETE", "PUT"):
            assert (await client.request(method, "/api/jobs/old/stems/vocal")).status_code == 404
            assert (
                await client.request(
                    method,
                    "/api/voice/result",
                    data={"job_id": "old", "filename": "vocal.wav"},
                )
            ).status_code == 404
        response = await client.get("/output/jobs/recent/song_1/playtrack/full.playback-1-2.mp3")
        assert response.status_code == 200
        max_age = int(response.headers["cache-control"].split("max-age=")[1].split(",")[0])
        assert 0 < max_age <= 3 * 86400
        # Expired files remain on disk until the next cleanup, but cannot be served.
        assert (settings.output_dir / "jobs/old/song_1/full.mp3").is_file()
        async with app.router.lifespan_context(app):

            async def wait_for_cleanup():
                while (settings.output_dir / "jobs/old").exists():
                    await asyncio.sleep(0.005)

            await asyncio.wait_for(wait_for_cleanup(), timeout=2)
            assert "old" not in app.state.jobs
            assert not (settings.output_dir / "jobs/old").exists()
            assert not (settings.output_dir / ".trash/old").exists()
            assert (await client.get("/output/logs/retention.log")).status_code == 404
    assert not any(task.get_name() == "song-retention" for task in asyncio.all_tasks())
    records = [
        json.loads(line)
        for line in (settings.output_dir / "logs/retention.log").read_text().splitlines()
    ]
    deletion = next(record for record in records if record["event"] == "job_deleted")
    assert deletion["jobId"] == "old" and datetime.fromisoformat(deletion["time"]).tzinfo
    summary = next(record for record in records if record["event"] == "cleanup_finished")
    assert summary["deletedJobIds"] == ["old"]


async def test_cleanup_rejects_symlinked_job_directory(tmp_path, caplog):
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    old = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
    job = GenerationJob(job_id="linked", prompt="rock", status="failed", created_at=old)
    app.state.jobs[job.job_id] = job
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.mp3").write_bytes(b"keep")
    (settings.output_dir / "jobs").mkdir()
    (settings.output_dir / "jobs/linked").symlink_to(outside, target_is_directory=True)
    caplog.set_level(logging.INFO, logger="app.services.retention")
    await cleanup_expired_jobs(app, reason="test")
    assert (outside / "keep.mp3").read_bytes() == b"keep"
    assert "linked" in app.state.jobs
    assert json.loads(caplog.records[-1].message)["failedJobIds"] == ["linked"]
    assert "refusing symlinked directory" in retention_status(app)["lastCleanupError"]
    assert str(outside) not in retention_status(app)["lastCleanupError"]


@pytest.mark.parametrize("directory", ["jobs", ".expired"])
@pytest.mark.parametrize("enabled", [False, True])
async def test_symlinked_layout_is_reported_at_startup_without_deleting_target(
    tmp_path, caplog, directory, enabled
):
    settings = make_settings(tmp_path, song_retention_enabled=enabled, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    add_job(app, "victim", (datetime.now(timezone.utc) - timedelta(days=5)).isoformat())
    linked_root = settings.output_dir / directory
    outside = tmp_path / "other_volume"
    if directory == "jobs":
        linked_root.rename(outside)
    else:
        outside.mkdir()
        (outside / "keep.mp3").write_bytes(b"keep")
    linked_root.symlink_to(outside, target_is_directory=True)
    caplog.set_level(logging.INFO, logger="app.services.retention")
    assert retention_status(app)["unsupportedLayout"] is None
    async with app.router.lifespan_context(app):
        await wait_for_inventory(app)
        await refresh_retention_inventory(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://testserver"
        ) as client:
            policy = (await client.get("/api/health")).json()["retention"]
        assert policy["unsupportedLayout"] is True
        assert policy["unsupportedLayoutDirectories"] == [directory]
        assert policy["inventoryStale"] is True
        assert "refusing symlinked directory" in policy["inventoryError"]
        assert str(settings.output_dir) not in policy["inventoryError"]
        if enabled:
            assert policy["lastCleanupResult"] == "failed"
            assert "refusing symlinked directory" in policy["lastCleanupError"]
            assert str(settings.output_dir) not in policy["lastCleanupError"]
        assert "victim" in app.state.jobs
        kept = outside / "victim/song_1/full.mp3" if directory == "jobs" else outside / "keep.mp3"
        assert kept.read_bytes() == (b"audio" if directory == "jobs" else b"keep")
        warnings = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING
            and '"event": "unsupported_layout"' in record.message
        ]
        assert len(warnings) == 1
        assert json.loads(warnings[0].message)["directories"] == [directory]
        linked_root.unlink()
        outside.rename(linked_root)
        await refresh_retention_inventory(app)
        recovered = retention_status(app)
        assert recovered["unsupportedLayout"] is False
        assert recovered["unsupportedLayoutDirectories"] == []
        assert recovered["inventoryError"] is None


@pytest.mark.parametrize("staging_blocked", [False, True])
async def test_pending_scan_failure_continues_known_jobs_and_reports_result(
    tmp_path, monkeypatch, caplog, staging_blocked
):
    import errno
    from pathlib import Path

    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    old = add_job(app, "old", (datetime.now(timezone.utc) - timedelta(days=5)).isoformat())
    add_job(app, "recent", datetime.now(timezone.utc).isoformat())
    pending_root = settings.output_dir / ".expired"
    if staging_blocked:
        pending_root.write_text("not a directory")
    else:
        pending_root.mkdir()
        original_iterdir = Path.iterdir

        def unreadable(path):
            if path == pending_root:
                raise PermissionError(errno.EACCES, "denied", str(path))
            return original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", unreadable)
    caplog.set_level(logging.INFO, logger="app.services.retention")
    assert retention_status(app)["lastCleanupResult"] is None
    await cleanup_expired_jobs(app, reason="test")
    records = [json.loads(record.message) for record in caplog.records]
    scan_failure = next(record for record in records if record["event"] == "cleanup_failed")
    assert scan_failure["stage"] == "pendingDeletionScan"
    summary = records[-1]
    assert summary["event"] == "cleanup_finished" and summary["pendingDeletionScanFailed"] is True
    assert summary["failedJobIds"] == (["old"] if staging_blocked else [])
    assert summary["deletedJobIds"] == ([] if staging_blocked else ["old"])
    assert (settings.output_dir / "jobs/old").exists() is staging_blocked
    assert ("old" in app.state.jobs) is staging_blocked
    if staging_blocked:
        assert app.state.jobs["old"] is old and pending_root.read_text() == "not a directory"
    assert (settings.output_dir / "jobs/recent").is_dir()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        policy = (await client.get("/api/health")).json()["retention"]
    assert policy["cleanupRunning"] is False and policy["remainingJobs"] == 0
    assert policy["lastCleanupResult"] == ("failed" if staging_blocked else "partial_failure")
    assert policy["lastCleanupReason"] == "test"
    assert datetime.fromisoformat(policy["lastCleanupAt"]).tzinfo is not None
    assert policy["pendingDeletionScanFailed"] is True
    assert str(settings.output_dir) not in policy["lastCleanupError"]
    monkeypatch.undo()
    if staging_blocked:
        pending_root.unlink()
    await cleanup_expired_jobs(app, reason="retry")
    recovered = retention_status(app)
    assert recovered["lastCleanupResult"] == "success" and recovered["lastCleanupError"] is None
    assert recovered["pendingDeletionScanFailed"] is False
    assert not (settings.output_dir / "jobs/old").exists()


async def test_daily_cleanup_logs_and_stops_with_lifespan(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    monkeypatch.setattr("app.services.retention.seconds_until_cleanup", lambda now: 0.01)
    async with app.router.lifespan_context(app):
        await wait_for_inventory(app)  # Let the empty startup sweep finish first.
        add_job(app, "scheduled", (datetime.now(timezone.utc) - timedelta(days=4)).isoformat())

        async def wait_for_deletion():
            while "scheduled" in app.state.jobs:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(wait_for_deletion(), timeout=2)
    records = [
        json.loads(line)
        for line in (settings.output_dir / "logs/retention.log").read_text().splitlines()
    ]
    summaries = [record for record in records if record["event"] == "cleanup_finished"]
    assert summaries[0]["reason"] == "startup" and summaries[0]["deletedJobIds"] == []
    assert summaries[-1]["reason"] == "scheduled"
    assert summaries[-1]["deletedJobIds"] == ["scheduled"]
    assert not any(task.get_name() == "song-retention" for task in asyncio.all_tasks())


async def test_retention_disabled_by_default_preserves_history_and_files(tmp_path, caplog):
    settings = make_settings(tmp_path)
    assert settings.song_retention_enabled is False
    app = create_app(settings, make_orchestrator(settings))
    add_job(app, "old", (datetime.now(timezone.utc) - timedelta(days=4)).isoformat())
    caplog.set_level(logging.INFO, logger="app.services.retention")
    async with app.router.lifespan_context(app):
        await cleanup_expired_jobs(app, reason="test")
        assert not any(task.get_name() == "song-retention" for task in asyncio.all_tasks())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://testserver"
        ) as client:
            assert (await client.get("/api/jobs")).json()["jobs"][0]["expiresAt"] is None
            assert (await client.get("/api/jobs/old")).json()["expiresAt"] is None
            assert (await client.get("/output/jobs/old/song_1/full.mp3")).content == b"audio"
            policy = (await client.get("/api/health")).json()["retention"]
            assert policy["enabled"] is False and policy["nextCleanupAt"] is None
            assert not policy["cleanupRunning"] and policy["remainingJobs"] == 0
    assert (settings.output_dir / "jobs/old/job.json").is_file()
    assert json.loads(caplog.records[0].message)["event"] == "retention_policy"
    assert json.loads(caplog.records[0].message)["enabled"] is False


async def test_expiry_visible_in_details_history_and_health(tmp_path):
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    now = datetime.now(timezone.utc)
    add_job(app, "recent", now.isoformat())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        expected = (now + timedelta(days=3)).isoformat()
        assert (await client.get("/api/jobs")).json()["jobs"][0]["expiresAt"] == expected
        assert (await client.get("/api/jobs/recent")).json()["expiresAt"] == expected
        policy = (await client.get("/api/health")).json()["retention"]
    assert policy["enabled"] is True and policy["days"] == 3
    next_run = datetime.fromisoformat(policy["nextCleanupAt"])
    assert next_run > now
    assert next_run.astimezone(timezone(timedelta(hours=8))).hour == 3


async def test_sse_expiring_during_stream_uses_done_contract(tmp_path):
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    job = add_job(app, "live", datetime.now(timezone.utc).isoformat())
    job.status = job.stage = "running"
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/jobs/{job_id}/events"
    )
    response = await endpoint("live", events_request(app, "/api/jobs/live/events"))
    stream = response.body_iterator
    first = await anext(stream)
    assert first.startswith("data: ") and json.loads(first.removeprefix("data: "))["expiresAt"]
    job.created_at = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
    job.status = job.stage = "failed"  # The operation has now finished.
    for queue in app.state.job_subscribers["live"]:
        queue.put_nowait(None)
    frame = await anext(stream)
    assert frame.startswith("event: done\ndata: ")
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload == {
        "success": False,
        "status": "expired",
        "jobId": "live",
        "message": "歌曲已过期。",
    }
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert "live" not in app.state.job_subscribers


@pytest.mark.parametrize("failure", ["mkdir", "handler"])
async def test_file_logging_failure_does_not_block_startup(tmp_path, caplog, failure):
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    caplog.set_level(logging.INFO, logger="app.services.retention")
    if failure == "mkdir":
        (settings.output_dir / "logs").write_text("a file, not a directory")
    with patch("app.services.retention.RotatingFileHandler") as constructor:
        constructor.side_effect = PermissionError("log unavailable")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://testserver"
            ) as client:
                assert (await client.get("/api/health")).status_code == 200
    assert any("using application logs" in record.message for record in caplog.records)
    assert any('"event": "cleanup_finished"' in record.message for record in caplog.records)


async def test_cancelled_shutdown_always_removes_and_closes_handler(tmp_path, monkeypatch):
    from app.services.retention import logger

    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    started, entered = asyncio.Event(), asyncio.Event()

    async def blocked_cleanup(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("app.services.retention.cleanup_expired_jobs", blocked_cleanup)
    baseline = list(logger.handlers)

    async def session():
        async with app.router.lifespan_context(app):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(session())
    await asyncio.wait_for(entered.wait(), timeout=2)
    await asyncio.wait_for(started.wait(), timeout=2)
    handler = next(handler for handler in logger.handlers if handler not in baseline)
    task.cancel()
    await asyncio.sleep(0)  # Shutdown is now waiting for the blocked cleanup task.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert logger.handlers == baseline and handler.stream is None
    assert not any(task.get_name() == "song-retention" for task in asyncio.all_tasks())


async def test_slow_startup_cleanup_does_not_block_health(tmp_path, monkeypatch):
    from app.services.retention import shutil

    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    add_job(app, "old", (datetime.now(timezone.utc) - timedelta(days=4)).isoformat())
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original = shutil.rmtree

    def slow_remove(path):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=2)
        original(path)

    monkeypatch.setattr("app.services.retention.shutil.rmtree", slow_remove)
    async with app.router.lifespan_context(app):
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://testserver"
            ) as client:
                response = await client.get("/api/health")
                assert response.status_code == 200
                assert response.json()["retention"]["cleanupRunning"] is True
                assert response.json()["retention"]["remainingJobs"] == 1
        finally:
            release.set()
    assert not (settings.output_dir / ".expired/old").exists()


async def test_in_flight_range_download_survives_cleanup(tmp_path, monkeypatch):
    import os

    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    job = add_job(app, "live", datetime.now(timezone.utc).isoformat())
    path = settings.output_dir / "jobs/live/song_1/full.mp3"
    path.write_bytes(b"music" * 20000)
    handles = []
    original = os.fdopen

    def capture_open(descriptor, *args, **kwargs):
        assert os.get_blocking(descriptor) is True
        handle = original(descriptor, *args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(os, "fdopen", capture_open)

    async def race(scope, receive, send):
        async def send_after_cleanup(message):
            if message["type"] == "http.response.start":
                assert sum(not handle.closed for handle in handles) == 1
                job.created_at = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
                await cleanup_expired_jobs(app, reason="download_race")
                assert not path.exists()
            await send(message)

        await app(scope, receive, send_after_cleanup)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(race), base_url="http://testserver"
    ) as client:
        response = await client.get(
            "/output/jobs/live/song_1/full.mp3", headers={"Range": "bytes=0-70000"}
        )
    assert response.status_code == 206 and response.headers["content-length"] == "70001"
    assert response.content == (b"music" * 20000)[:70001]
    assert handles and all(handle.closed for handle in handles)


@pytest.mark.parametrize("enabled", [True, False])
async def test_orphan_preview_access_depends_on_retention_policy(tmp_path, enabled):
    settings = make_settings(tmp_path, song_retention_enabled=enabled, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    path = settings.output_dir / "jobs/orphan/song_1/playtrack/full.playback-1-2.mp3"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"preview")
    await refresh_retention_inventory(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        response = await client.get("/output/jobs/orphan/song_1/playtrack/full.playback-1-2.mp3")
        inventory = (await client.get("/api/health")).json()["retention"]
    assert inventory["orphanJobIds"] == ["orphan"]
    assert response.status_code == (404 if enabled else 200)
    if not enabled:
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"


async def test_first_enable_defaults_to_dry_run_and_reports_impact(tmp_path, caplog):
    settings = make_settings(tmp_path, song_retention_enabled=True)
    assert settings.song_retention_dry_run is True
    app = create_app(settings, make_orchestrator(settings))
    old = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
    add_job(app, "old", old)
    pending = settings.output_dir / ".expired/previous/job"
    pending.mkdir(parents=True)
    (pending / "full.mp3").write_bytes(b"partial deletion")
    caplog.set_level(logging.INFO, logger="app.services.retention")
    async with app.router.lifespan_context(app):
        await wait_for_inventory(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://testserver"
        ) as client:
            assert (await client.get("/api/jobs/old")).json()["expiresAt"] is None
            assert (await client.get("/output/jobs/old/song_1/full.mp3")).content == b"audio"
            policy = (await client.get("/api/health")).json()["retention"]
            assert policy["enabled"] and policy["dryRun"]
            assert policy["pendingDeletionJobIds"] == ["previous"]
    summaries = [
        json.loads(record.message)
        for record in caplog.records
        if '"event": "cleanup_finished"' in record.message
    ]
    assert summaries[-1]["wouldDeleteJobIds"] == ["old", "previous"]
    assert summaries[-1]["deletedJobIds"] == []
    assert summaries[-1]["result"] == "success"
    assert retention_status(app)["dryRun"] is True
    assert retention_status(app)["lastCleanupResult"] == "success"
    assert (settings.output_dir / "jobs/old/song_1/full.mp3").is_file()
    assert (settings.output_dir / ".trash/old/song_1/vocal.wav").is_file()
    assert (pending / "full.mp3").is_file()


async def test_disabled_policy_reports_pending_deletions_in_logs_and_health(tmp_path, caplog):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    pending = settings.output_dir / ".expired/previous"
    pending.mkdir(parents=True)
    (pending / "audio.mp3").write_bytes(b"keep while disabled")
    caplog.set_level(logging.INFO, logger="app.services.retention")
    async with app.router.lifespan_context(app):
        await wait_for_inventory(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://testserver"
        ) as client:
            policy = (await client.get("/api/health")).json()["retention"]
    assert policy["pendingDeletionCount"] == 1
    assert policy["pendingDeletionJobIds"] == ["previous"]
    record = next(
        json.loads(record.message)
        for record in caplog.records
        if '"event": "retention_inventory"' in record.message
    )
    assert record["pendingDeletionJobIds"] == ["previous"] and not record["enabled"]
    assert (pending / "audio.mp3").is_file()


async def test_expired_active_task_can_be_observed_and_cancelled(tmp_path):
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    old = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
    job = add_job(app, "busy", old)
    job.replace_status, job.replace_stage = "running", "replacing"
    job.replace_message = "正在替换人声"
    release = asyncio.Event()
    job.replace_task = asyncio.create_task(release.wait())
    try:
        await cleanup_expired_jobs(app, reason="test")
        await refresh_retention_inventory(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://testserver"
        ) as client:
            assert (await client.get("/api/jobs/busy")).status_code == 200
            assert (await client.get("/api/jobs")).json()["jobs"][0]["jobId"] == "busy"
            assert (await client.get("/api/health")).json()["retention"]["expiredActiveJobIds"] == [
                "busy"
            ]
            assert (await client.get("/output/jobs/busy/song_1/full.mp3")).status_code == 404
            assert (await client.post("/api/jobs/busy/replace")).status_code == 404
            assert (
                await client.patch("/api/jobs/busy", json={"status": "cancelled"})
            ).status_code == 200
            assert job.replace_cancel_requested is True
            job.replace_status = "failed"  # Already timed out, while the worker still runs.
            assert (await client.get("/api/jobs/busy")).status_code == 200
    finally:
        release.set()
        await job.replace_task
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        assert (await client.get("/api/jobs/busy")).status_code == 404


async def test_sse_waits_for_expired_active_operation_to_finish(tmp_path):
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    job = add_job(app, "busy", datetime.now(timezone.utc).isoformat())
    job.replace_status, job.replace_stage = "running", "replacing"
    job.replace_message = "正在替换人声"
    release = asyncio.Event()
    job.replace_task = asyncio.create_task(release.wait())
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/jobs/{job_id}/events"
    )
    response = await endpoint("busy", events_request(app, "/api/jobs/busy/events"))
    stream = response.body_iterator
    try:
        assert (await anext(stream)).startswith("data: ")
        job.created_at = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
        for queue in app.state.job_subscribers["busy"]:
            queue.put_nowait(None)
        assert (await anext(stream)).startswith("data: ")
        release.set()
        await job.replace_task
        job.replace_status = "failed"
        for queue in app.state.job_subscribers["busy"]:
            queue.put_nowait(None)
        final = await anext(stream)
        assert final.startswith("event: done\ndata: ")
        assert json.loads(final.split("data: ", 1)[1])["status"] == "expired"
    finally:
        release.set()
        await job.replace_task
        await stream.aclose()


@pytest.mark.parametrize("kind", ["fifo", "socket", "directory", "permission", "fstat_error"])
async def test_audio_nonregular_nodes_and_io_errors_return_404(tmp_path, monkeypatch, kind):
    import os
    import socket
    from pathlib import Path

    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    target = settings.output_dir / "audio.mp3"
    bound_socket = None
    descriptors = []
    original_open = os.open
    original_close = os.close
    original_path_open = Path.open

    def capture_open(path, flags, *args, **kwargs):
        if kind == "fifo" and path == target:
            assert flags & os.O_NONBLOCK
        descriptor = original_open(path, flags, *args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    def deny_open(*args, **kwargs):
        raise PermissionError("denied")

    def guard_path_open(path, *args, **kwargs):
        # Fail promptly if a regression reintroduces a blocking FIFO open.
        assert kind != "fifo" or path != target
        return original_path_open(path, *args, **kwargs)

    if kind == "fifo":
        os.mkfifo(target)
    elif kind == "socket":
        bound_socket = socket.socket(socket.AF_UNIX)
        bound_socket.bind(str(target))
    elif kind == "directory":
        target.mkdir()
    else:
        target.write_bytes(b"audio")
    monkeypatch.setattr(os, "open", deny_open if kind == "permission" else capture_open)
    monkeypatch.setattr(Path, "open", guard_path_open)
    if kind == "fstat_error":

        def fail_stat(descriptor):
            raise OSError("IO error")

        monkeypatch.setattr(os, "fstat", fail_stat)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://testserver"
        ) as client:
            response = await asyncio.wait_for(client.get("/output/audio.mp3"), timeout=1)
            assert response.status_code == {"permission": 403, "fstat_error": 500}.get(kind, 404)
            assert (await client.get("/api/health")).status_code == 200
        monkeypatch.undo()
        for descriptor in descriptors:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        if bound_socket is not None:
            bound_socket.close()
        # Avoid leaving a descriptor behind if an assertion fails.
        for descriptor in descriptors:
            try:
                original_close(descriptor)
            except OSError:
                pass


async def test_health_inventory_is_bounded_and_never_scans_on_request(
    tmp_path, monkeypatch, caplog
):
    from pathlib import Path

    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    for index in range(1000):
        (settings.output_dir / "jobs" / f"orphan_{index:04}").mkdir(parents=True)
    caplog.set_level(logging.INFO, logger="app.services.retention")
    await refresh_retention_inventory(app)
    record = json.loads(caplog.records[-1].message)
    assert record["orphanCount"] == 1000 and record["orphanTruncated"] is True
    assert len(record["orphanJobIds"]) == 20
    assert record["unmanagedCount"] == 1000 and len(record["unmanagedJobs"]) == 20
    assert len(caplog.records[-1].message) < 20000

    def forbidden(*args, **kwargs):
        raise AssertionError("health must not scan or parse tasks")

    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr("app.services.retention.is_expired", forbidden)
    monkeypatch.setattr("app.services.retention._scan_inventory", forbidden)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        for _ in range(3):
            response = await client.get("/api/health")
            assert response.status_code == 200 and len(response.content) < 20000
            assert response.json()["retention"]["orphanCount"] == 1000
            assert (
                response.json()["retention"]["inventoryUpdatedAt"] == record["inventoryUpdatedAt"]
            )


@pytest.mark.parametrize("prefix,directory", [("pendingDeletion", ".expired"), ("orphan", "jobs")])
@pytest.mark.parametrize("previous_snapshot", [False, True])
async def test_inventory_scan_failure_preserves_last_good_category(
    tmp_path, prefix, directory, previous_snapshot, caplog
):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    root = settings.output_dir
    (root / ".expired/pending").mkdir(parents=True)
    (root / "jobs/orphan").mkdir(parents=True)
    app.state.jobs["active"] = GenerationJob(
        job_id="active",
        prompt="rock",
        status="running",
        created_at=(datetime.now(timezone.utc) - timedelta(days=4)).isoformat(),
    )
    initial = retention_status(app)
    assert initial["inventoryStale"] is True
    assert initial[f"{prefix}Count"] is initial[f"{prefix}Truncated"] is None
    if previous_snapshot:
        await refresh_retention_inventory(app)
        good = dict(app.state.retention_inventory)
        assert good["inventoryStale"] is False and good[f"{prefix}Count"] == 1
    else:
        good = initial

    child = "pending" if directory == ".expired" else "orphan"
    (root / directory / child).rmdir()
    (root / directory).rmdir()
    (root / directory).write_text("not a directory")
    app.state.jobs["active"].status = "succeeded"
    other = "jobs/new_orphan" if directory == ".expired" else ".expired/new_pending"
    (root / other).mkdir(parents=True)
    await refresh_retention_inventory(app)
    failed = retention_status(app)
    assert failed["inventoryStale"] is True and failed[f"{prefix}Stale"] is True
    assert failed[f"{prefix}Count"] == good[f"{prefix}Count"]
    assert failed[f"{prefix}JobIds"] == good[f"{prefix}JobIds"]
    assert failed[f"{prefix}Truncated"] == good[f"{prefix}Truncated"]
    assert failed["inventoryUpdatedAt"] == good["inventoryUpdatedAt"]
    assert failed["inventoryAttemptedAt"] is not None
    assert failed["expiredActiveCount"] == 0 and failed["expiredActiveStale"] is False
    other_prefix = "orphan" if directory == ".expired" else "pendingDeletion"
    assert failed[f"{other_prefix}Count"] == 2 and failed[f"{other_prefix}Stale"] is False
    if directory == "jobs":
        assert failed["unmanagedStale"] is True
        assert failed["unmanagedJobs"] == good["unmanagedJobs"]
    assert "NotADirectoryError" in failed["inventoryError"]
    assert str(root) not in failed["inventoryError"]
    assert str(root) in caplog.text  # Full diagnostic remains in server logs.

    (root / directory).unlink()
    (root / directory).mkdir()
    await refresh_retention_inventory(app)
    recovered = retention_status(app)
    assert recovered["inventoryStale"] is False and recovered["inventoryError"] is None
    assert recovered["inventoryUpdatedAt"] == recovered["inventoryAttemptedAt"]
    assert recovered["inventoryUpdatedAt"] != good["inventoryUpdatedAt"]
    assert recovered[f"{prefix}Count"] == 0


async def test_health_metadata_error_omits_absolute_path(tmp_path, monkeypatch, caplog):
    import errno
    from pathlib import Path

    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    add_job(app, "unreadable", datetime.now(timezone.utc).isoformat())
    target = settings.output_dir / "jobs/unreadable/job.json"
    original_read = Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == target:
            raise PermissionError(errno.EACCES, "private path", str(target))
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    restarted = create_app(settings, make_orchestrator(settings))
    await refresh_retention_inventory(restarted)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(restarted), base_url="http://testserver"
    ) as client:
        policy = (await client.get("/api/health")).json()["retention"]
    reason = policy["unmanagedJobs"][0]["reason"]
    assert reason == "PermissionError: errno=13 (Permission denied)"
    assert str(target) not in json.dumps(policy) and "private path" not in reason
    assert str(target) in caplog.text


async def test_inventory_refreshes_in_background_even_when_retention_disabled(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    monkeypatch.setattr("app.services.retention.INVENTORY_REFRESH_SECONDS", 0.02)
    async with app.router.lifespan_context(app):
        await wait_for_inventory(app)
        first = app.state.retention_inventory["inventoryUpdatedAt"]
        (settings.output_dir / "jobs/new_orphan").mkdir(parents=True)

        async def updated():
            while app.state.retention_inventory["orphanCount"] != 1:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(updated(), timeout=2)
        assert app.state.retention_inventory["inventoryUpdatedAt"] != first
        assert app.state.retention_inventory["orphanJobIds"] == ["new_orphan"]
    assert not any(task.get_name() == "retention-inventory" for task in asyncio.all_tasks())


@pytest.mark.parametrize(
    "enabled,dry_run,expected",
    [
        (False, False, "disabled"),
        (True, True, "dry_run"),
        (True, False, "unmanaged"),
    ],
)
async def test_invalid_timestamp_is_explicit_and_metadata_errors_are_reported(
    tmp_path,
    enabled,
    dry_run,
    expected,
):
    settings = make_settings(
        tmp_path, song_retention_enabled=enabled, song_retention_dry_run=dry_run
    )
    app = create_app(settings, make_orchestrator(settings))
    job = add_job(app, "invalid", "bad time")
    job.result = {
        "jobId": "invalid",
        "prompt": "rock",
        "durationMinutes": 2,
        "structuredPrompt": "rock",
        "lyrics": "test",
        "fullTrack": "jobs/invalid/song_1/full.mp3",
        "stems": {},
        "stemUrls": [],
        "waveforms": {},
        "splitEnabled": False,
        "debug": {},
    }
    job.save(settings.output_dir)
    add_job(app, "mismatch", datetime.now(timezone.utc).isoformat())
    metadata = json.loads((settings.output_dir / "jobs/mismatch/job.json").read_text())
    metadata["jobId"] = "another_job"
    (settings.output_dir / "jobs/mismatch/job.json").write_text(json.dumps(metadata))
    for job_id, text in (("bad_json", "{"), ("bad_schema", '{"prompt":"private input"}')):
        root = settings.output_dir / "jobs" / job_id
        root.mkdir()
        (root / "job.json").write_text(text)
    restarted = create_app(settings, make_orchestrator(settings))
    await refresh_retention_inventory(restarted)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(restarted), base_url="http://testserver"
    ) as client:
        history = (await client.get("/api/jobs")).json()["jobs"][0]
        detail = (await client.get("/api/jobs/invalid")).json()
        for row in (history, detail):
            assert row["retentionState"] == expected and row["expiresAt"] is None
            assert row["audioAvailable"] is (expected != "unmanaged")
        policy = (await client.get("/api/health")).json()["retention"]
        assert policy["unmanagedCount"] == 4 and policy["unmanagedTruncated"] is False
        reasons = {item["jobId"]: item["reason"] for item in policy["unmanagedJobs"]}
        assert reasons["invalid"] == "invalid_created_at"
        assert reasons["mismatch"] == "ValueError: jobId does not match directory"
        assert reasons["bad_json"] == (
            "JSONDecodeError: Expecting property name enclosed in double quotes (line 1 column 2)"
        )
        assert reasons["bad_schema"] == "jobId: missing; createdAt: missing; status: missing"
        assert "private input" not in json.dumps(reasons)
        audio = await client.get("/output/jobs/invalid/song_1/full.mp3")
        assert audio.status_code == (404 if expected == "unmanaged" else 200)


@pytest.mark.parametrize("stage", ["open", "read"])
@pytest.mark.parametrize(
    "error_name,status_code",
    [
        ("ENOENT", 404),
        ("ENOTDIR", 404),
        ("ELOOP", 404),
        ("EACCES", 403),
        ("EMFILE", 503),
        ("ENFILE", 503),
        ("EIO", 500),
        ("ENOMEM", 500),
    ],
)
async def test_audio_io_errors_preserve_server_failure_status(
    tmp_path,
    monkeypatch,
    caplog,
    stage,
    error_name,
    status_code,
):
    import errno
    import os
    from unittest.mock import Mock

    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    target = settings.output_dir / "audio.mp3"
    target.write_bytes(b"audio")
    handles = []
    original_fdopen = os.fdopen
    error = OSError(getattr(errno, error_name), "test failure")

    def fail_open(*args, **kwargs):
        raise error

    def fail_read(descriptor, *args, **kwargs):
        source = original_fdopen(descriptor, *args, **kwargs)
        handles.append(source)
        wrapped = Mock(wraps=source)
        wrapped.read.side_effect = error
        return wrapped

    monkeypatch.setattr(
        os, "open" if stage == "open" else "fdopen", fail_open if stage == "open" else fail_read
    )
    caplog.set_level(logging.ERROR, logger="app.main")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        response = await client.get("/output/audio.mp3")
    assert response.status_code == status_code
    if status_code == 503:
        assert response.headers["retry-after"] == "5"
    if status_code >= 500:
        assert any(record.exc_info and record.exc_info[1] is error for record in caplog.records)
    assert all(handle.closed for handle in handles)


@pytest.mark.parametrize("operation", ["split", "replace", "mix"])
@pytest.mark.parametrize("status", ["pending", "running"])
@pytest.mark.parametrize("task_done", [False, True])
async def test_active_operation_status_preserves_expired_job(
    tmp_path, operation, status, task_done
):
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    job = add_job(app, "busy", (datetime.now(timezone.utc) - timedelta(days=4)).isoformat())
    setattr(job, f"{operation}_status", status)
    if task_done:
        task = asyncio.create_task(asyncio.sleep(0))
        await task
        setattr(job, f"{operation}_task", task)
    await cleanup_expired_jobs(app, reason="test")
    await refresh_retention_inventory(app)
    assert app.state.jobs["busy"] is job
    assert (settings.output_dir / "jobs/busy/song_1/full.mp3").is_file()
    assert retention_status(app)["expiredActiveJobIds"] == ["busy"]
    setattr(job, f"{operation}_status", "succeeded")
    await cleanup_expired_jobs(app, reason="test")
    assert "busy" not in app.state.jobs


async def test_cleanup_with_stop_already_set_preserves_candidates(tmp_path, caplog):
    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    add_job(app, "old", (datetime.now(timezone.utc) - timedelta(days=4)).isoformat())
    stop = asyncio.Event()
    stop.set()
    caplog.set_level(logging.INFO, logger="app.services.retention")
    await cleanup_expired_jobs(app, reason="test", stop=stop)
    assert "old" in app.state.jobs
    assert (settings.output_dir / "jobs/old/song_1/full.mp3").is_file()
    assert retention_status(app)["lastCleanupResult"] == "cancelled"
    summary = json.loads(caplog.records[-1].message)
    assert summary["deletedCount"] == 0


@pytest.mark.parametrize("reason", ["startup", "scheduled"])
async def test_shutdown_finishes_current_deletion_and_preserves_remaining_jobs(
    tmp_path, monkeypatch, caplog, reason
):
    from app.services.retention import retention_lifecycle, shutil

    settings = make_settings(tmp_path, song_retention_enabled=True, song_retention_dry_run=False)
    app = create_app(settings, make_orchestrator(settings))
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original = shutil.rmtree

    def slow_remove(path):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=2)
        original(path)

    monkeypatch.setattr("app.services.retention.shutil.rmtree", slow_remove)
    monkeypatch.setattr("app.services.retention.seconds_until_cleanup", lambda now: 0.02)
    lifecycle = retention_lifecycle(app)
    await lifecycle.__aenter__()
    closing = None
    try:
        if reason == "scheduled":
            await wait_for_inventory(app)
        old = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
        for job_id in ("first", "second"):
            add_job(app, job_id, old)
        await asyncio.wait_for(started.wait(), timeout=2)
        closing = asyncio.create_task(lifecycle.__aexit__(None, None, None))
        await asyncio.sleep(0)  # Let shutdown signal the monitor while deletion is blocked.
        assert not closing.done()
    finally:
        release.set()
        if closing is None:
            await lifecycle.__aexit__(None, None, None)
        else:
            await asyncio.wait_for(closing, timeout=2)
    assert "first" not in app.state.jobs and "second" in app.state.jobs
    assert not (settings.output_dir / ".expired/first").exists()
    assert (settings.output_dir / "jobs/second/song_1/full.mp3").is_file()
    policy = retention_status(app)
    assert policy["lastCleanupReason"] == reason
    assert policy["lastCleanupResult"] == "cancelled"
    assert policy["cleanupRunning"] is False and policy["remainingJobs"] == 0
    summaries = [
        json.loads(record.message)
        for record in caplog.records
        if '"event": "cleanup_finished"' in record.message
    ]
    assert summaries[-1]["deletedJobIds"] == ["first"]


@pytest.mark.parametrize("failed_step", range(5))
async def test_monitor_recovers_after_each_startup_and_scheduled_step(
    tmp_path, monkeypatch, caplog, failed_step
):
    from app.services.retention import retention_lifecycle

    settings = make_settings(tmp_path, song_retention_enabled=True)
    app = create_app(settings, make_orchestrator(settings))
    calls = []
    recovered = asyncio.Event()
    error = RuntimeError("unexpected background failure")

    def record(step):
        calls.append(step)
        if len(calls) == failed_step + 1:
            raise error
        if len(calls) >= 7:
            recovered.set()

    async def cleanup(application, *, reason, stop):
        assert isinstance(stop, asyncio.Event)
        record(reason)

    async def refresh(application):
        record("refresh")

    monkeypatch.setattr("app.services.retention.cleanup_expired_jobs", cleanup)
    monkeypatch.setattr("app.services.retention.refresh_retention_inventory", refresh)
    monkeypatch.setattr("app.services.retention.seconds_until_cleanup", lambda now: 0.01)
    async with retention_lifecycle(app):
        await asyncio.wait_for(recovered.wait(), timeout=2)
    assert calls[:7] == [
        "refresh", "startup", "refresh", "scheduled", "refresh", "scheduled", "refresh"
    ]
    assert any(record.exc_info and record.exc_info[1] is error for record in caplog.records)


@pytest.mark.parametrize("step", ["cleanup_expired_jobs", "refresh_retention_inventory"])
async def test_monitor_propagates_step_cancellation(tmp_path, monkeypatch, step):
    from app.services.retention import retention_lifecycle

    settings = make_settings(tmp_path, song_retention_enabled=True)
    app = create_app(settings, make_orchestrator(settings))
    cancelled = asyncio.Event()

    async def cancel(*args, **kwargs):
        cancelled.set()
        raise asyncio.CancelledError

    monkeypatch.setattr(f"app.services.retention.{step}", cancel)
    with pytest.raises(asyncio.CancelledError):
        async with retention_lifecycle(app):
            await asyncio.wait_for(cancelled.wait(), timeout=2)
