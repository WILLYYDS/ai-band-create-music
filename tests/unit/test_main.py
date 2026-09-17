from __future__ import annotations

import os
from pathlib import Path

from app.main import _rvc_model_fingerprint, _sanitize_proxy_environment
from tests.helpers import make_settings


def test_sanitize_rewrites_plain_socks_to_socks5(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7897")
    _sanitize_proxy_environment()
    assert os.environ["ALL_PROXY"] == "socks5://127.0.0.1:7897"


def test_sanitize_rewrites_uppercase_socks_scheme(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "SOCKS://127.0.0.1:7897")
    _sanitize_proxy_environment()
    assert os.environ["ALL_PROXY"] == "socks5://127.0.0.1:7897"


def test_sanitize_keeps_supported_schemes_untouched(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:7897")
    monkeypatch.setenv("HTTPS_PROXY", "https://127.0.0.1:7897")
    monkeypatch.setenv("HTTP_PROXY", "127.0.0.1:7897")  # bare host:port is fine
    _sanitize_proxy_environment()
    assert os.environ["ALL_PROXY"] == "socks5://127.0.0.1:7897"
    assert os.environ["HTTPS_PROXY"] == "https://127.0.0.1:7897"
    assert os.environ["HTTP_PROXY"] == "127.0.0.1:7897"


def test_sanitize_drops_unsupported_proxy_schemes(monkeypatch) -> None:
    monkeypatch.setenv("all_proxy", "socks4://127.0.0.1:7897")
    monkeypatch.setenv("HTTP_PROXY", "ftp://127.0.0.1:7897")
    _sanitize_proxy_environment()
    assert "all_proxy" not in os.environ
    assert "HTTP_PROXY" not in os.environ


def test_sanitize_ignores_non_proxy_variables(monkeypatch) -> None:
    monkeypatch.setenv("PROXY_URL", "socks://127.0.0.1:7897")
    _sanitize_proxy_environment()
    assert os.environ["PROXY_URL"] == "socks://127.0.0.1:7897"


def test_sanitize_is_idempotent(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:7897")
    _sanitize_proxy_environment()
    _sanitize_proxy_environment()
    assert os.environ["ALL_PROXY"] == "socks5://127.0.0.1:7897"


def test_model_fingerprint_tracks_content_not_paths(tmp_path: Path) -> None:
    """指纹用于判断缓存的人声替换结果是否还属于当前模型。

    必须是内容摘要（不能把模型绝对路径泄漏进 job.json），且模型文件被就地改写后要变。
    """
    model = tmp_path / "model.pth"
    model.write_bytes(b"first-model")
    settings = make_settings(tmp_path, rvc_model_path=model)

    fingerprint = _rvc_model_fingerprint(settings)
    assert fingerprint.startswith("v1:")
    digest = fingerprint.split(":", 1)[1]
    assert len(digest) == 64
    assert str(model) not in fingerprint
    assert _rvc_model_fingerprint(settings) == fingerprint

    model.write_bytes(b"second-model")
    assert _rvc_model_fingerprint(settings) != fingerprint


def test_model_fingerprint_covers_version_and_index(tmp_path: Path) -> None:
    model = tmp_path / "model.pth"
    model.write_bytes(b"model")
    index = tmp_path / "model.index"
    index.write_bytes(b"index")
    settings = make_settings(tmp_path, rvc_model_path=model, rvc_index_path=index)
    baseline = _rvc_model_fingerprint(settings)

    assert _rvc_model_fingerprint(
        make_settings(tmp_path, rvc_model_path=model, rvc_index_path=index, rvc_model_version="v1")
    ) != baseline
    index.write_bytes(b"changed-index")
    assert _rvc_model_fingerprint(settings) != baseline


def test_model_fingerprint_tolerates_missing_assets(tmp_path: Path) -> None:
    missing = tmp_path / "not-there.pth"
    settings = make_settings(tmp_path, rvc_model_path=missing, rvc_index_path=None)
    assert _rvc_model_fingerprint(settings).startswith("v1:")
