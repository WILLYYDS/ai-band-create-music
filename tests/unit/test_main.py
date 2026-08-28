from __future__ import annotations

import os

from app.main import _sanitize_proxy_environment


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
