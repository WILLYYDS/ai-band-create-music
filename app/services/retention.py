from __future__ import annotations

import asyncio
import heapq
import json
import logging
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.errors import UnsupportedOutputLayoutError, public_error_reason
from app.services.job_files import job_song_dir

logger = logging.getLogger(__name__)
BEIJING = timezone(timedelta(hours=8))
INVENTORY_LIMIT = 20
# ponytail: inventory may lag one hour; lower the interval if monitoring needs fresher snapshots.
INVENTORY_REFRESH_SECONDS = 3600


def expires_at(job, days: int) -> datetime | None:
    try:
        created = datetime.fromisoformat(job.created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            return None
        return created + timedelta(days=days)
    except (ValueError, TypeError, AttributeError, OverflowError):
        return None


def is_expired(job, days: int, now: datetime | None = None) -> bool:
    expiry = expires_at(job, days)
    return expiry is not None and expiry <= (now or datetime.now(timezone.utc))


def retention_enforced(settings) -> bool:
    return settings.song_retention_enabled and not settings.song_retention_dry_run


def is_job_active(job) -> bool:
    return any(
        status in {"pending", "running"}
        for status in (job.status, job.split_status, job.replace_status, job.mix_status)
    ) or any(
        task is not None and not task.done()
        for task in (job.task, job.split_task, job.replace_task, job.mix_task)
    )


def retained_job(request, job_id: str, *, allow_active: bool = False):
    job = request.app.state.jobs.get(job_id)
    settings = request.app.state.settings
    if (
        retention_enforced(settings)
        and job is not None
        and is_expired(job, settings.song_retention_days)
        and not (allow_active and is_job_active(job))
    ):
        return None
    return job


def seconds_until_cleanup(now: datetime) -> float:
    local = now.astimezone(BEIJING)
    target = local.replace(hour=3, minute=0, second=0, microsecond=0)
    if target <= local:
        target += timedelta(days=1)
    return (target - local).total_seconds()


def retention_policy(settings) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "enabled": settings.song_retention_enabled,
        "dryRun": settings.song_retention_dry_run,
        "days": settings.song_retention_days,
        "cleanupTimezone": "Asia/Shanghai",
        "nextCleanupAt": (
            (now + timedelta(seconds=seconds_until_cleanup(now))).isoformat()
            if settings.song_retention_enabled
            else None
        ),
    }


def _audit(event: str, *, level: int = logging.INFO, **values) -> None:
    logger.log(
        level,
        json.dumps(
            {"time": datetime.now(timezone.utc).isoformat(), "event": event, **values},
            ensure_ascii=False,
        ),
    )


def _safe_path(root: Path, *parts: str) -> Path:
    target = root.joinpath(*parts)
    if target.is_symlink() or target.resolve() != target:
        raise UnsupportedOutputLayoutError(f"Refusing cleanup through a symlink: {target}")
    return target


def _id_summary(values, prefix: str) -> dict:
    count = 0

    def counted():
        nonlocal count
        for value in values:
            count += 1
            yield value

    sample = heapq.nsmallest(INVENTORY_LIMIT, counted())
    return {
        f"{prefix}JobIds": sample,
        f"{prefix}Count": count,
        f"{prefix}Truncated": count > INVENTORY_LIMIT,
    }


def _empty_inventory() -> dict:
    result = {
        "inventoryUpdatedAt": None,
        "inventoryAttemptedAt": None,
        "inventoryStale": True,
        "inventoryError": None,
        "unsupportedLayout": None,
        "unsupportedLayoutDirectories": [],
        "unmanagedJobs": [],
    }
    for prefix in ("pendingDeletion", "orphan", "expiredActive", "unmanaged"):
        result.update(_id_summary((), prefix))
        result[f"{prefix}Count"] = result[f"{prefix}Truncated"] = None
        result[f"{prefix}Stale"] = True
    return result


def _scan_inventory(application) -> dict:
    state = application.state
    jobs = dict(state.jobs)
    result = dict(getattr(state, "retention_inventory", None) or _empty_inventory())
    attempted_at = datetime.now(timezone.utc).isoformat()
    errors = []
    unsupported_layout = []
    layout_checked = 0
    try:
        pending_root = _safe_path(state.settings.output_dir.resolve(), ".expired")
        layout_checked += 1
        result.update(
            _id_summary(
                (path.name for path in pending_root.iterdir()) if pending_root.exists() else (),
                "pendingDeletion",
            )
        )
        result["pendingDeletionStale"] = False
    except (OSError, ValueError) as exc:
        if isinstance(exc, UnsupportedOutputLayoutError):
            unsupported_layout.append(".expired")
            layout_checked += 1
        result["pendingDeletionStale"] = True
        errors.append(f"pendingDeletion: {public_error_reason(exc)}")
        logger.exception("Unable to scan pending deletions")

    try:
        jobs_root = _safe_path(state.settings.output_dir.resolve(), "jobs")
        layout_checked += 1
        unmanaged = {
            job_id: "invalid_created_at"
            for job_id, job in jobs.items()
            if expires_at(job, state.settings.song_retention_days) is None
        }

        def orphans():
            if jobs_root.exists():
                for path in jobs_root.iterdir():
                    if path.is_dir() and path.name not in jobs:
                        unmanaged[path.name] = getattr(state, "job_metadata_errors", {}).get(
                            path.name, "metadata_missing_or_not_loaded"
                        )
                        yield path.name

        orphan_summary = _id_summary(orphans(), "orphan")
        unmanaged_summary = _id_summary(unmanaged, "unmanaged")
        result.update(orphan_summary)
        result.update(unmanaged_summary)
        result["unmanagedJobs"] = [
            {"jobId": job_id, "reason": unmanaged[job_id][:160]}
            for job_id in result["unmanagedJobIds"]
        ]
        result["orphanStale"] = result["unmanagedStale"] = False
    except (OSError, ValueError) as exc:
        if isinstance(exc, UnsupportedOutputLayoutError):
            unsupported_layout.append("jobs")
            layout_checked += 1
        result["orphanStale"] = result["unmanagedStale"] = True
        errors.append(f"jobs: {public_error_reason(exc)}")
        logger.exception("Unable to scan job inventory")

    try:
        result.update(
            _id_summary(
                (
                    job_id
                    for job_id, job in jobs.items()
                    if is_expired(job, state.settings.song_retention_days) and is_job_active(job)
                ),
                "expiredActive",
            )
        )
        result["expiredActiveStale"] = False
    except (OSError, ValueError) as exc:
        result["expiredActiveStale"] = True
        errors.append(f"expiredActive: {public_error_reason(exc)}")
        logger.exception("Unable to scan active job inventory")

    result["inventoryAttemptedAt"] = attempted_at
    result["unsupportedLayoutDirectories"] = unsupported_layout
    result["unsupportedLayout"] = (
        True if unsupported_layout else (False if layout_checked == 2 else None)
    )
    result["inventoryError"] = "; ".join(errors) or None
    result["inventoryStale"] = bool(errors)
    if not errors:
        result["inventoryUpdatedAt"] = attempted_at
    return result


async def refresh_retention_inventory(application) -> None:
    previous = getattr(application.state, "retention_inventory", None) or _empty_inventory()
    inventory = await asyncio.to_thread(_scan_inventory, application)
    application.state.retention_inventory = inventory
    if inventory["unsupportedLayout"] and (
        inventory["unsupportedLayoutDirectories"] != previous["unsupportedLayoutDirectories"]
    ):
        _audit(
            "unsupported_layout",
            level=logging.WARNING,
            directories=inventory["unsupportedLayoutDirectories"],
            message="Symbolic links are unsupported; mount volumes directly.",
        )
    _audit("retention_inventory", **retention_policy(application.state.settings), **inventory)


def retention_status(application) -> dict:
    state = application.state
    inventory = getattr(state, "retention_inventory", None) or _empty_inventory()
    return {
        **retention_policy(state.settings),
        "cleanupRunning": getattr(state, "retention_cleanup_running", False),
        "remainingJobs": getattr(state, "retention_cleanup_remaining", 0),
        **getattr(
            state,
            "retention_last_cleanup",
            {
                "lastCleanupAt": None,
                "lastCleanupReason": None,
                "lastCleanupResult": None,
                "lastCleanupError": None,
                "pendingDeletionScanFailed": None,
            },
        ),
        **inventory,
    }


def _stage_job(root: Path, job_id: str) -> Path:
    # Reuse the existing job id validation before constructing deletion paths.
    job_song_dir(root, job_id)
    pending = _safe_path(root, ".expired", job_id)
    moves = [
        (_safe_path(root, directory, job_id), _safe_path(root, ".expired", job_id, name))
        for directory, name in (("jobs", "job"), (".trash", "trash"))
    ]
    for source, target in moves:
        if source.exists() and target.exists():
            raise ValueError(f"Cleanup destination already exists: {target}")
    pending.mkdir(parents=True, exist_ok=True)
    for source, target in moves:
        if source.exists():
            source.rename(target)
    return pending


async def cleanup_expired_jobs(
    application,
    *,
    reason: str,
    now: datetime | None = None,
    stop: asyncio.Event | None = None,
) -> None:
    settings = application.state.settings
    if not settings.song_retention_enabled:
        return
    now = now or datetime.now(timezone.utc)
    deleted, failed, skipped, would_delete = [], [], [], []
    pending_scan_failed = False
    cleanup_error = None
    result = "failed"
    _audit("cleanup_started", reason=reason)
    application.state.retention_cleanup_running = True
    try:
        root = settings.output_dir.resolve()
        candidates = set()
        try:
            pending_root = _safe_path(root, ".expired")
            if pending_root.exists():
                candidates = {path.name for path in pending_root.iterdir()}
        except (OSError, ValueError) as exc:
            pending_scan_failed = True
            cleanup_error = f"pendingDeletion: {public_error_reason(exc)}"
            _audit("cleanup_failed", stage="pendingDeletionScan", error=str(exc))
        for job_id, job in list(application.state.jobs.items()):
            if expires_at(job, settings.song_retention_days) is None:
                candidates.discard(job_id)
                skipped.append(job_id)
                _audit("job_skipped", jobId=job_id, reason="invalid_created_at")
            elif is_expired(job, settings.song_retention_days, now):
                candidates.add(job_id)
        application.state.retention_cleanup_remaining = len(candidates)
        for index, job_id in enumerate(sorted(candidates)):
            if stop is not None and stop.is_set():
                result = "cancelled"
                return
            application.state.retention_cleanup_remaining = len(candidates) - index
            job = application.state.jobs.get(job_id)
            if job is not None and is_job_active(job):
                skipped.append(job_id)
                _audit("job_skipped", jobId=job_id, reason="active_task")
                continue
            if job is not None and not is_expired(job, settings.song_retention_days, now):
                skipped.append(job_id)
                _audit("job_skipped", jobId=job_id, reason="not_expired")
                continue
            try:
                if settings.song_retention_dry_run:
                    job_song_dir(root, job_id)
                    would_delete.append(job_id)
                    _audit("job_would_delete", jobId=job_id)
                    continue
                _audit("job_delete_started", jobId=job_id)
                # Rename and remove the in-memory entry before yielding: an edit cannot
                # start while the thread deletes the isolated directory.
                pending = _stage_job(root, job_id)
                application.state.jobs.pop(job_id, None)
                application.state.job_subscribers.pop(job_id, None)
                await asyncio.to_thread(shutil.rmtree, pending)
                deleted.append(job_id)
                _audit("job_deleted", jobId=job_id)
            except (OSError, ValueError) as exc:
                failed.append(job_id)
                if cleanup_error is None:
                    cleanup_error = f"job {job_id}: {public_error_reason(exc)}"
                _audit("job_delete_failed", jobId=job_id, error=str(exc))
        if pending_scan_failed or failed:
            result = "partial_failure" if deleted or would_delete else "failed"
        else:
            result = "success"
    except (OSError, ValueError) as exc:
        cleanup_error = public_error_reason(exc)
        _audit("cleanup_failed", stage="run", error=str(exc))
    except asyncio.CancelledError:
        result = "cancelled"
        cleanup_error = "CancelledError"
        raise
    finally:
        application.state.retention_cleanup_running = False
        application.state.retention_cleanup_remaining = 0
        application.state.retention_last_cleanup = {
            "lastCleanupAt": datetime.now(timezone.utc).isoformat(),
            "lastCleanupReason": reason,
            "lastCleanupResult": result,
            "lastCleanupError": cleanup_error,
            "pendingDeletionScanFailed": pending_scan_failed,
        }
        _audit(
            "cleanup_finished",
            reason=reason,
            result=result,
            error=cleanup_error,
            pendingDeletionScanFailed=pending_scan_failed,
            **_id_summary(deleted, "deleted"),
            **_id_summary(failed, "failed"),
            **_id_summary(skipped, "skipped"),
            **_id_summary(would_delete, "wouldDelete"),
        )


@asynccontextmanager
async def retention_lifecycle(application):
    log_dir = application.state.settings.output_dir / "logs"
    logger.setLevel(logging.INFO)
    handler = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_dir / "retention.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    except OSError as exc:
        logger.warning("Retention file logging unavailable; using application logs: %s", exc)
    _audit("retention_policy", **retention_status(application))
    stop = asyncio.Event()

    async def monitor():
        async def run_step(step, **kwargs):
            try:
                await step(application, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Retention background step failed: %s", step.__name__)

        await run_step(refresh_retention_inventory)
        if application.state.settings.song_retention_enabled:
            await run_step(cleanup_expired_jobs, reason="startup", stop=stop)
            await run_step(refresh_retention_inventory)
        while not stop.is_set():
            delay = seconds_until_cleanup(datetime.now(timezone.utc))
            try:
                await asyncio.wait_for(stop.wait(), min(delay, INVENTORY_REFRESH_SECONDS))
            except asyncio.TimeoutError:
                if delay <= INVENTORY_REFRESH_SECONDS:
                    await run_step(cleanup_expired_jobs, reason="scheduled", stop=stop)
                await run_step(refresh_retention_inventory)

    task = None
    try:
        name = (
            "song-retention"
            if application.state.settings.song_retention_enabled
            else "retention-inventory"
        )
        task = asyncio.create_task(monitor(), name=name)
        yield
    finally:
        stop.set()
        try:
            if task is not None:
                await task
        finally:
            if handler is not None:
                logger.removeHandler(handler)
                handler.close()
