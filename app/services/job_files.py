from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4


def job_song_dir(output_dir: Path, job_id: str, song: int = 0) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", job_id) or song < 0:
        raise ValueError("Invalid job output path")
    return output_dir / "jobs" / job_id / f"song_{song + 1}"


def read_job_diagnostics(output_dir: Path, job_id: str) -> dict[str, Any]:
    target = output_dir / "jobs" / job_id / "prompts.json"
    if not target.is_file():
        return {}
    data = json.loads(target.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def update_job_diagnostics(output_dir: Path, job_id: str, **values: Any) -> dict[str, Any]:
    data = {**read_job_diagnostics(output_dir, job_id), "jobId": job_id, **values}
    target = output_dir / "jobs" / job_id / "prompts.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"prompts.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return data


def update_provider_diagnostic(
    output_dir: Path,
    job_id: str | None,
    variation: int,
    **values: Any,
) -> dict[str, Any]:
    if not job_id:
        return values
    data = read_job_diagnostics(output_dir, job_id)
    requests = list(data.get("providerRequests") or [])
    entry = next(
        (item for item in requests if item.get("variation") == variation),
        None,
    )
    if entry is None:
        entry = {"variation": variation, "songNumber": variation + 1}
        requests.append(entry)
    entry.update(values)
    update_job_diagnostics(output_dir, job_id, providerRequests=requests)
    return entry
