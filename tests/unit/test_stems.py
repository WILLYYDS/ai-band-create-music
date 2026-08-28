from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.errors import GenerationError
from app.services.stems import (
    SPLIT_PROFILES,
    build_demucs_command,
    require_demucs_runtime,
    resolve_segment,
    stem_output_files,
)
from tests.helpers import make_settings


def test_fast_profile_builds_expected_demucs_command(tmp_path: Path) -> None:
    command = build_demucs_command(
        make_settings(tmp_path, split_profile="fast"),
        tmp_path / "input.mp3",
        tmp_path / "work",
    )
    assert command[1:3] == ["-m", "demucs"]
    assert command[command.index("--name") + 1] == "mdx_q"
    assert command[command.index("--segment") + 1] == "12"


def test_cuda_device_is_forwarded_to_demucs(tmp_path: Path) -> None:
    command = build_demucs_command(
        make_settings(tmp_path, split_profile="fast", demucs_device="cuda"),
        tmp_path / "input.mp3",
        tmp_path / "work",
    )
    assert command[command.index("--device") + 1] == "cuda"


def test_htdemucs_segment_is_clamped() -> None:
    assert resolve_segment("htdemucs", "20", SPLIT_PROFILES["balanced"]) == "7.8"


def test_demucs_runtime_reports_missing_numpy() -> None:
    with (
        patch(
            "app.services.stems.importlib.util.find_spec",
            side_effect=[None, object(), object(), object()],
        ),
        pytest.raises(GenerationError, match="运行依赖缺失：numpy.*uv sync --frozen"),
    ):
        require_demucs_runtime()


def test_demucs_runtime_rejects_unavailable_cuda() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(GenerationError, match="PyTorch 无法访问 CUDA"),
    ):
        require_demucs_runtime("cuda")


def test_invalid_segment_is_rejected() -> None:
    with pytest.raises(GenerationError):
        resolve_segment("mdx_q", "invalid", SPLIT_PROFILES["fast"])


def test_stem_outputs_include_original_music_name() -> None:
    assert stem_output_files(Path("普通话摇滚.mp3")) == {
        "vocal": "普通话摇滚_vocal.mp3",
        "drums": "普通话摇滚_drums.mp3",
        "bass": "普通话摇滚_bass.mp3",
        "other": "普通话摇滚_other.mp3",
    }
