from __future__ import annotations

import os
import wave
from pathlib import Path

import pytest

from app.services.stems import DemucsStemSeparator, stem_output_files
from tests.helpers import make_settings


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_REAL_DEMUCS_TEST") != "1",
    reason="set RUN_REAL_DEMUCS_TEST=1 to download/load Demucs and run the slow smoke test",
)
async def test_real_demucs_smoke(tmp_path: Path) -> None:
    source = tmp_path / "short.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8_000)
        audio.writeframes(b"\x00\x00" * 8_000)
    settings = make_settings(
        tmp_path,
        split_profile="fast",
        split_timeout_seconds=1800,
    )
    output = tmp_path / "stems"
    await DemucsStemSeparator(settings).split(source, output)
    assert all((output / name).stat().st_size > 0 for name in stem_output_files(source).values())
