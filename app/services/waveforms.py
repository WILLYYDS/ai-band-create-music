from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np

from app.services.stems import prepare_ffmpeg_environment

logger = logging.getLogger(__name__)

# 多轨编辑器按车道满宽绘制波形包络：64 个 bin 只画出几个色块，640 个才像波形。
# 该值必须同时作为 summarize_waveform 与 extract_waveform 的默认值，否则同一次拆轨
# 里 "full" 与各分轨的 bin 数量不一致，客户端的公共 x 轴会错位。
WAVEFORM_BIN_COUNT = 640


def summarize_waveform(samples: np.ndarray, bin_count: int = WAVEFORM_BIN_COUNT) -> list[float]:
    """Reduce mono PCM samples to normalized RMS bins for the multitrack editor.

    640 bins is what the editor lanes draw at full width; at 64 the envelope reads as
    a few blobs instead of a waveform. The default is shared with
    :func:`extract_waveform` so every lane of one song carries the same bin count.
    """
    if samples.size == 0 or bin_count <= 0:
        return []
    edges = np.linspace(0, samples.size, min(bin_count, samples.size) + 1, dtype=int)
    rms = np.array(
        [
            np.sqrt(np.mean(np.square(samples[start:end], dtype=np.float64)))
            for start, end in zip(edges, edges[1:], strict=False)
        ],
        dtype=np.float64,
    )
    maximum = float(rms.max(initial=0))
    if maximum <= 0:
        return [0.0] * len(rms)
    return [round(float(value / maximum), 4) for value in rms]


async def extract_waveform(path: Path, bin_count: int = WAVEFORM_BIN_COUNT) -> list[float]:
    if path.stat().st_size < 128:
        raise RuntimeError(f"音频文件过小，无法提取波形：{path.name}")
    environment = prepare_ffmpeg_environment()
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        "8000",
        "-f",
        "f32le",
        "pipe:1",
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        process.kill()
        await process.wait()
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise RuntimeError(f"波形提取超时：{path.name}") from exc
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"波形提取失败：{path.name}，{detail or 'ffmpeg 无输出'}")
    return summarize_waveform(np.frombuffer(stdout, dtype="<f4"), bin_count)


async def extract_waveforms(paths: dict[str, Path]) -> dict[str, list[float]]:
    async def extract(name: str, path: Path) -> tuple[str, list[float]] | None:
        try:
            return name, await extract_waveform(path)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("waveform extraction skipped file=%s error=%s", path, exc)
            return None

    results = await asyncio.gather(*(extract(name, path) for name, path in paths.items()))
    return {result[0]: result[1] for result in results if result is not None and result[1]}


def merge_waveform_sets(previous: object, fresh: dict[str, list[float]]) -> dict[str, list[float]]:
    """Return the waveform map to store for a freshly split song.

    A split re-extracts and rewrites every lane, so the fresh bins replace the stored
    map rather than merging into it: jobs persisted before the 640-bin change hold a
    64-bin ``"full"`` envelope, and merging 640-bin stems into it would hand clients
    lanes of different lengths on one shared x-axis. A stale lane we could not
    re-extract is dropped instead of shipped mismatched, and the stored map is only
    kept when extraction produced nothing at all, so an ffmpeg failure cannot drop
    waveforms that were already persisted.
    """
    if fresh:
        return dict(fresh)
    return dict(previous) if isinstance(previous, dict) else {}
