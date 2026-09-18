from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.services.audio_files import output_path_from_url


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


def drop_mix_artifact(output: dict[str, Any]) -> bool:
    """摘掉一首歌的合轨成品引用与它的 mix 车道，返回是否真的摘过。

    成品文件留在磁盘上（与替换人声的旧文件同一取舍：不删、只不再被引用），下一次合轨会
    覆盖同名文件。这样 `mixStatus` 可以由结果推导成"没有成品"，客户端不必猜它是否过期。
    """
    if not isinstance(output, dict) or not isinstance(output.get("mixedTrack"), str):
        return False
    output.pop("mixedTrack", None)
    waveforms = output.get("waveforms")
    if isinstance(waveforms, dict):
        waveforms.pop("mix", None)
    return True


def capture_mix_artifact(output: dict[str, Any]) -> dict[str, Any]:
    """打包当前成品的引用与车道，供作废后按需还原（撤回删除、替换失败回滚）。"""
    if not isinstance(output, dict) or not isinstance(output.get("mixedTrack"), str):
        return {}
    captured: dict[str, Any] = {"mixTrack": output["mixedTrack"]}
    waveforms = output.get("waveforms")
    if isinstance(waveforms, dict) and isinstance(waveforms.get("mix"), list):
        captured["mixWaveform"] = waveforms["mix"]
    return captured


def restore_mix_artifact(output: dict[str, Any], captured: object) -> bool:
    """还原打包过的成品引用与车道；当前已有成品（更新的那一版）时不覆盖。"""
    if not isinstance(output, dict) or not isinstance(captured, dict):
        return False
    if "mixedTrack" in output or not isinstance(captured.get("mixTrack"), str):
        return False
    output["mixedTrack"] = captured["mixTrack"]
    waveforms = output.get("waveforms")
    if isinstance(waveforms, dict) and isinstance(captured.get("mixWaveform"), list):
        waveforms["mix"] = captured["mixWaveform"]
    return True


def reset_mix_state(job: Any) -> None:
    """复位 job 的合轨运行态。

    job 是鸭子类型（端点持有的是 GenerationJob，测试里可能是轻量替身），所以统一走
    getattr/setattr，并跳过仍在跑的合轨任务。
    """
    mix_task = getattr(job, "mix_task", None)
    if mix_task is not None and not mix_task.done():
        return
    for attribute in (
        "mix_song",
        "mix_status",
        "mix_stage",
        "mix_progress",
        "mix_message",
        "mix_error",
    ):
        setattr(job, attribute, None)


def invalidate_mix_artifact(job: Any, output: dict[str, Any]) -> None:
    """输入变了（人声被替换/撤回、分轨被删）就把合轨成品标记为不存在。

    改的是内存里的任务状态，调用方负责随后的 `job.save()` 落盘；job 的 mix 运行态一并复位，
    否则 `_reported_mix_status` 还会拿旧的 `mix_status` 报成功。
    """
    if drop_mix_artifact(output):
        reset_mix_state(job)


def stored_path_exists(output_dir: Path, value: object) -> bool:
    """记录里的路径是否仍指向一个真实文件；解析不了或读不到都算"不能确认"。

    走 `output_path_from_url` 而不是直接拼路径：记录里可能是 URL 或绝对路径，直接拼接会
    静默判错（绝对路径还会绕过根目录约束）。EACCES 之类的 IO 问题不等于文件不存在，
    这里返回 False，让调用方保持"未确认"的安全默认。
    """
    if not isinstance(value, str):
        return False
    try:
        return output_path_from_url(value, output_dir).is_file()
    except (OSError, ValueError):
        return False
