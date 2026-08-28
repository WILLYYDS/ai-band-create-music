from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.services.voice import RVCEngine, _reuse_base_models, _source_song_name
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
