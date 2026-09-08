from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from tests.helpers import make_settings


def test_default_port_and_prompt_limit(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert settings.port == 8010
    assert settings.prompt_max_chars == 2000
    assert settings.llm_timeout_seconds == 120
    assert settings.llm_disable_thinking is True
    assert settings.task_backend == "inline"
    assert settings.cache_backend == "none"
    assert settings.music_provider == "minimax_music"


def test_minimax_music_defaults_and_provider(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, music_provider="minimax_music")
    assert settings.minimax_base_url == "http://127.0.0.1:8111"
    assert settings.minimax_model == "MiniMaxAI/MiniMax-Music3"
    assert settings.minimax_seed == 42
    assert settings.minimax_num_inference_steps == 30
    assert settings.minimax_timeout_seconds == 7200


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 2),
        ("bad", 2),
        ("1e400", 2),
        ("Infinity", 2),
        (0, 1),
        (1.4, 1),
        (1.5, 2),
        (4.8, 5),
        (20, 5),
    ],
)
def test_duration_matches_legacy_clamping(tmp_path: Path, value: object, expected: int) -> None:
    assert make_settings(tmp_path).normalize_duration(value) == expected


def test_invalid_duration_range_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            output_dir=tmp_path,
            mock_full_song_path=tmp_path / "mock.mp3",
            min_duration_minutes=5,
            max_duration_minutes=2,
        )
