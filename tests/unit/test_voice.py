from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.voice import (
    LIMIT_FILTER,
    MIX_FILTER,
    RVCConversionError,
    RVCEngine,
    _mix_tracks,
    _reuse_base_models,
    _run_ffmpeg,
    _source_song_name,
)
from tests.helpers import make_settings


def test_engine_maps_public_f0_parameter_names(tmp_path: Path) -> None:
    class FakeInference:
        def set_params(self, **params) -> None:
            self.params = params

        def infer_file(self, input_path: str, output_path: str) -> None:
            Path(output_path).write_bytes(b"wav")

    inference = FakeInference()
    engine = RVCEngine(make_settings(tmp_path))
    engine._inference = inference
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"wav")

    engine._convert_sync(
        input_path,
        tmp_path / "output.wav",
        f0_up_key=2,
        f0_method="rmvpe",
        index_rate=0.75,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=1.0,
        protect=0.33,
    )

    assert inference.params["f0up_key"] == 2
    assert inference.params["f0method"] == "rmvpe"
    assert "f0_up_key" not in inference.params


def test_reuses_base_models_without_copying(tmp_path: Path) -> None:
    source = tmp_path / "source"
    old_source = tmp_path / "old-source"
    package = tmp_path / "package"
    source.mkdir()
    old_source.mkdir()
    package.mkdir()
    for name in ("hubert_base.pt", "rmvpe.pt", "rmvpe.onnx"):
        (source / name).write_bytes(name.encode())
    target_dir = package / "base_model"
    target_dir.mkdir()
    old_hubert = old_source / "hubert_base.pt"
    old_hubert.write_bytes(b"old")
    (target_dir / "hubert_base.pt").symlink_to(old_hubert)

    _reuse_base_models(SimpleNamespace(__file__=package / "infer.py"), source)

    for name in ("hubert_base.pt", "rmvpe.pt", "rmvpe.onnx"):
        target = package / "base_model" / name
        assert target.is_symlink()
        assert target.resolve() == (source / name).resolve()
        assert target.read_bytes() == name.encode()


def test_rvc_output_uses_source_music_name() -> None:
    assert _source_song_name("真实歌名_vocal.mp3", "AI 生成曲目") == "真实歌名"
    assert _source_song_name("真实歌名_rvc_vocal.wav", "AI 生成曲目") == "真实歌名"


def test_mix_filter_preserves_stem_levels_and_limits_clipping() -> None:
    assert "normalize=0" in MIX_FILTER
    assert "equalizer=f=3000" in MIX_FILTER
    assert "volume=3dB[vocal]" in MIX_FILTER
    assert "volume=2.5dB" not in MIX_FILTER
    assert "level=false" in LIMIT_FILTER
    assert "limit=0.891251" in LIMIT_FILTER


async def test_ffmpeg_process_does_not_block_event_loop() -> None:
    started = time.perf_counter()
    process = asyncio.create_task(
        _run_ffmpeg(
            [sys.executable, "-c", "import time; time.sleep(0.1)"],
            dict(os.environ),
            1,
        )
    )
    await asyncio.sleep(0.02)
    assert time.perf_counter() - started < 0.08
    await process


async def test_ffmpeg_process_is_killed_on_timeout() -> None:
    with pytest.raises(RVCConversionError, match="timed out"):
        await _run_ffmpeg(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            dict(os.environ),
            0.02,
        )


async def test_mix_gain_matches_original_loudness(tmp_path: Path, monkeypatch) -> None:
    reference = tmp_path / "full.wav"
    output = tmp_path / "mix.wav"
    calls = []

    timeouts = []

    async def run_ffmpeg(command, _environment, timeout, *, capture_stdout=False):
        calls.append(command)
        timeouts.append(timeout)
        if capture_stdout:
            return (
                '{"streams":[{"sample_rate":"44100","channels":1,'
                '"codec_name":"flac","bits_per_raw_sample":"24"}]}'
            )
        target = Path(command[-1])
        if target.suffix == ".wav":
            target.write_bytes(b"RIFF-audio")
        return ""

    async def loudness(path, _environment, _timeout):
        return -10.0 if path == reference else -12.5

    monkeypatch.setattr("app.services.voice._run_ffmpeg", run_ffmpeg)
    monkeypatch.setattr("app.services.voice._integrated_loudness", loudness)

    await _mix_tracks([tmp_path / f"stem-{index}.wav" for index in range(4)], reference, output, 3)

    final_filter = calls[-1][calls[-1].index("-af") + 1]
    assert "volume=2.500dB" in final_filter
    assert calls[-1][calls[-1].index("-ar") + 1] == "44100"
    assert calls[-1][calls[-1].index("-ac") + 1] == "1"
    assert calls[-1][calls[-1].index("-c:a") + 1] == "pcm_s24le"
    assert timeouts[-1] <= timeouts[0]
