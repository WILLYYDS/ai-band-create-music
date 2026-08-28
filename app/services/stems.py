from __future__ import annotations

import asyncio
import importlib.util
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import anyio

from app.core.config import PROJECT_ROOT, Settings
from app.core.errors import GenerationError
from app.services.audio_files import require_readable_file

STEM_NAMES = ("vocal", "drums", "bass", "other")


@dataclass(frozen=True, slots=True)
class SplitProfile:
    model: str
    jobs: str
    shifts: str
    overlap: str
    segment: str
    mp3_bitrate: str


SPLIT_PROFILES = {
    "fast": SplitProfile("mdx_q", "2", "1", "0.10", "12", "128"),
    "balanced": SplitProfile("htdemucs", "0", "1", "0.25", "", "192"),
    "quality": SplitProfile("htdemucs", "0", "2", "0.35", "", "192"),
}
MODEL_MAX_SEGMENT_SECONDS = {"htdemucs": 7.8, "htdemucs_ft": 7.8, "htdemucs_6s": 7.8}


@dataclass(frozen=True, slots=True)
class SplitResult:
    stdout: str
    stderr: str
    duration_ms: int
    files: dict[str, str]


class StemSeparator(Protocol):
    async def split(self, input_path: Path, output_dir: Path) -> SplitResult: ...


def require_demucs_runtime(device: str = "") -> None:
    missing = [
        name
        for name in ("numpy", "torch", "demucs", "diffq")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise GenerationError(
            f"Demucs 运行依赖缺失：{', '.join(missing)}。请执行 uv sync --frozen。"
        )
    if device.lower().startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise GenerationError(
                "DEMUCS_DEVICE 配置为 cuda，但 PyTorch 无法访问 CUDA。"
                "请检查 NVIDIA 驱动及 PyTorch CUDA wheel。"
            )


def resolve_segment(model: str, configured: str, profile: SplitProfile) -> str:
    segment = (
        configured.strip()
        if configured.strip()
        else (profile.segment if model == profile.model else "")
    )
    if not segment:
        return ""
    try:
        seconds = float(segment)
    except ValueError as exc:
        raise GenerationError(f"DEMUCS_SEGMENT 必须是正数，当前值：{segment}") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise GenerationError(f"DEMUCS_SEGMENT 必须是有限正数，当前值：{segment}")
    maximum = MODEL_MAX_SEGMENT_SECONDS.get(model)
    return str(maximum) if maximum is not None and seconds > maximum else segment


def build_demucs_command(settings: Settings, input_path: Path, work_dir: Path) -> list[str]:
    profile = SPLIT_PROFILES[settings.split_profile]
    model = settings.demucs_model or profile.model
    jobs = settings.demucs_jobs or profile.jobs
    shifts = settings.demucs_shifts or profile.shifts
    overlap = settings.demucs_overlap or profile.overlap
    bitrate = settings.demucs_mp3_bitrate or profile.mp3_bitrate
    command = [
        sys.executable,
        "-m",
        "demucs",
        "--name",
        model,
        "--out",
        str(work_dir),
        "--mp3",
        "--mp3-bitrate",
        bitrate,
        "--jobs",
        jobs,
        "--shifts",
        shifts,
        "--overlap",
        overlap,
    ]
    if settings.demucs_device:
        command.extend(["--device", settings.demucs_device])
    segment = resolve_segment(model, settings.demucs_segment, profile)
    if segment:
        command.extend(["--segment", segment])
    command.append(str(input_path))
    return command


def prepare_ffmpeg_environment() -> dict[str, str]:
    environment = dict(os.environ)
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return environment
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths(weak=True)
        if shutil.which("ffmpeg") and shutil.which("ffprobe"):
            environment["PATH"] = os.environ.get("PATH", "")
            return environment
    except Exception:
        pass
    raise GenerationError("音频处理需要 ffmpeg 和 ffprobe，请安装 static-ffmpeg 或系统 ffmpeg。")


def _find_stem(work_dir: Path, stem_name: str) -> Path:
    matches = list(work_dir.rglob(f"{stem_name}.mp3"))
    if not matches:
        raise GenerationError(f"Demucs 已结束，但没有找到分轨文件：{stem_name}.mp3")
    return matches[0]


def stem_output_files(input_path: Path) -> dict[str, str]:
    base_name = re.sub(r"[^\w\-. ]+", "_", input_path.stem, flags=re.UNICODE).strip(" ._")
    base_name = base_name[:160] or "music"
    return {stem_name: f"{base_name}_{stem_name}.mp3" for stem_name in STEM_NAMES}


def _copy_stems(
    work_dir: Path,
    output_dir: Path,
    output_files: dict[str, str],
) -> None:
    demucs_names = {"vocal": "vocals", "drums": "drums", "bass": "bass", "other": "other"}
    for public_name, demucs_name in demucs_names.items():
        source = _find_stem(work_dir, demucs_name)
        target = output_dir / output_files[public_name]
        shutil.copy2(source, target)
        require_readable_file(target, "拆轨输出文件为空")


class DemucsStemSeparator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def split(self, input_path: Path, output_dir: Path) -> SplitResult:
        require_readable_file(input_path, "拆轨输入音频不可读")
        require_demucs_runtime(self._settings.demucs_device)
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = output_dir / ".demucs_work"
        await anyio.to_thread.run_sync(shutil.rmtree, work_dir, True)
        work_dir.mkdir(parents=True, exist_ok=True)
        environment = await anyio.to_thread.run_sync(prepare_ffmpeg_environment)
        torch_home = PROJECT_ROOT / ".torch-cache"
        torch_home.mkdir(parents=True, exist_ok=True)
        environment.setdefault("TORCH_HOME", str(torch_home))
        command = build_demucs_command(self._settings, input_path, work_dir)
        output_files = stem_output_files(input_path)
        started = asyncio.get_running_loop().time()
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=self._settings.split_timeout_seconds
            )
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            process.kill()
            await process.wait()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise GenerationError(
                f"Demucs 拆轨超时（{self._settings.split_timeout_seconds:g} 秒）。"
            ) from exc
        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            failure_detail = stderr or "无 stderr 输出"
            raise GenerationError(
                f"Demucs 拆轨失败，退出码：{process.returncode}，stderr={failure_detail}"
            )
        await anyio.to_thread.run_sync(_copy_stems, work_dir, output_dir, output_files)
        if not self._settings.split_keep_workdir:
            await anyio.to_thread.run_sync(shutil.rmtree, work_dir, True)
        duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
        return SplitResult(stdout, stderr, duration_ms, output_files)
