from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from app.services.audio_files import output_path_from_url


def require_audio_revision(output: dict[str, Any], value: object) -> None:
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        try:
            value = int(value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="音频版本号无效。") from exc
    if value is None:
        value = 0
    if type(value) is not int or value < 0:
        raise HTTPException(status_code=400, detail="音频版本号无效。")
    if value != output.get("audioRevision", 0):
        raise HTTPException(status_code=409, detail="歌曲版本已更新，请刷新后重试。")


def drop_editor_state(output: dict[str, Any]) -> bool:
    changed = False
    for key in ("projectRevision", "editorState", "editorFullTrack"):
        if key in output:
            output.pop(key)
            changed = True
    config = output.get("mixConfig")
    if isinstance(config, dict) and ("editorState" in config or "projectRevision" in config):
        output.pop("mixConfig")
        changed = True
    for field, key in (("playback", "editorFullTrack"), ("waveforms", "editorFull")):
        if key in (output.get(field) or {}):
            output[field].pop(key)
            changed = True
    return changed


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
    playback = output.get("playback")
    if isinstance(playback, dict):
        playback.pop("mixedTrack", None)
    waveforms = output.get("waveforms")
    if isinstance(waveforms, dict):
        waveforms.pop("mix", None)
    return True


def capture_mix_artifact(output: dict[str, Any]) -> dict[str, Any]:
    """打包当前成品的引用与车道，供作废后按需还原（撤回删除、替换失败回滚）。"""
    if not isinstance(output, dict) or not isinstance(output.get("mixedTrack"), str):
        return {}
    if (output.get("mixConfig") or {}).get("commit") and output.get("fullTrack") == output[
        "mixedTrack"
    ]:
        return {}
    captured: dict[str, Any] = {"mixTrack": output["mixedTrack"]}
    playback = output.get("playback")
    if isinstance(playback, dict) and isinstance(playback.get("mixedTrack"), str):
        captured["mixPlayback"] = playback["mixedTrack"]
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
    if isinstance(captured.get("mixPlayback"), str):
        output.setdefault("playback", {})["mixedTrack"] = captured["mixPlayback"]
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
    # Editing raw inputs changes the draft, while the committed song stays playable.
    if (output.get("mixConfig") or {}).get("commit") and output.get("fullTrack") == output.get(
        "mixedTrack"
    ):
        return
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


def normalize_mixed_song(output: dict[str, Any]) -> bool:
    """「替换后」是一首独立的歌，存在 `output["mixed"]`，有自己的成品、音轨与版本号。

    旧结果（以及无请求体的旧版导出）把成品挂在原曲的 `mixedTrack` 上：用到替换后那首时
    （`song_project`、加载、渲染）迁成单轨的 `mixed`。
    与原曲母带是同一个文件时（覆盖式提交的旧形状）只是同一首歌，直接摘掉。返回是否改过。
    """
    if not isinstance(output, dict) or not isinstance(output.get("mixedTrack"), str):
        return False
    track = output.pop("mixedTrack")
    playback = output.get("playback")
    preview = playback.pop("mixedTrack", None) if isinstance(playback, dict) else None
    waveforms = output.get("waveforms")
    waveform = waveforms.pop("mix", None) if isinstance(waveforms, dict) else None
    if track != output.get("fullTrack"):
        output["mixed"] = {
            "audioRevision": (output.get("mixed") or {}).get("audioRevision", 0),
            "fullTrack": track,
            "playback": {"fullTrack": preview} if isinstance(preview, str) else {},
            "durationSeconds": output.get("durationSeconds"),
            "stems": {},
            "stemUrls": [],
            "waveforms": {"full": waveform} if isinstance(waveform, list) else {},
            "splitEnabled": False,
            "debug": {},
        }
    return True


def song_project(output: dict[str, Any] | None, variant: str) -> dict[str, Any] | None:
    """一首候选里要编辑的那首歌：`original` 是原曲本身，`mixed` 是替换后的那首。"""
    if not isinstance(output, dict):
        return None
    if variant == "original":
        return output
    normalize_mixed_song(output)
    mixed = output.get("mixed")
    return mixed if variant == "mixed" and isinstance(mixed, dict) else None
