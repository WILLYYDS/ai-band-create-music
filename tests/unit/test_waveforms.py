import asyncio
import inspect
from pathlib import Path

import numpy as np
import pytest

from app.services.waveforms import (
    WAVEFORM_BIN_COUNT,
    extract_waveform,
    extract_waveforms,
    summarize_waveform,
)


def test_summarize_waveform_preserves_relative_energy() -> None:
    samples = np.concatenate((np.full(100, 0.25), np.full(100, 1.0))).astype(np.float32)

    assert summarize_waveform(samples, 2) == [0.25, 1.0]
    assert summarize_waveform(np.zeros(8, dtype=np.float32), 4) == [0.0] * 4


def test_summarize_waveform_defaults_to_the_editor_bin_count() -> None:
    samples = np.ones(200_000, dtype=np.float32)

    assert WAVEFORM_BIN_COUNT == 640
    assert len(summarize_waveform(samples)) == WAVEFORM_BIN_COUNT


def test_both_entry_points_share_one_default_bin_count() -> None:
    # 同一模块内的两个默认值必须一致，否则 "full" 与分轨的 x 轴长度不同。
    assert (
        inspect.signature(summarize_waveform).parameters["bin_count"].default
        == inspect.signature(extract_waveform).parameters["bin_count"].default
        == WAVEFORM_BIN_COUNT
    )


async def test_waveform_progress_counts_processed_files_including_skipped_files(monkeypatch):
    counts = []

    async def extract(path):
        if path.name == "bad":
            raise RuntimeError("invalid audio")
        return [0.5]

    monkeypatch.setattr("app.services.waveforms.extract_waveform", extract)
    result = await extract_waveforms(
        {name: Path(name) for name in ("vocal", "bad", "bass")},
        progress=lambda finished, total: counts.append((finished, total)),
    )
    assert result == {"vocal": [0.5], "bass": [0.5]}
    assert counts == [(1, 3), (2, 3), (3, 3)]


async def test_waveform_cancellation_does_not_count_unfinished_files(monkeypatch):
    counts = []
    started = asyncio.Event()

    async def extract(_path):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("app.services.waveforms.extract_waveform", extract)
    task = asyncio.create_task(
        extract_waveforms(
            {"vocal": Path("vocal")},
            progress=lambda *count: counts.append(count),
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert counts == []
