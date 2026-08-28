from __future__ import annotations

from pathlib import Path

import httpx

from app.main import create_app
from tests.helpers import make_orchestrator, make_settings


class FakeVoiceEngine:
    loaded = True

    async def convert(self, input_path: Path, output_path: Path, **params) -> None:
        assert input_path.is_file()
        assert params["f0_method"] == "rmvpe"
        assert params["index_rate"] == 0.5
        output_path.write_bytes(b"RIFF-converted-wav")


async def test_voice_conversion_download_delete_and_restore(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings), FakeVoiceEngine())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        converted = await client.post(
            "/api/voice/convert",
            files={"file": ("real_song_vocal.mp3", b"ID3" + b"0" * 20_000, "audio/mpeg")},
            data={"index_rate": "0.5", "song_name": "AI 生成曲目"},
        )
        downloaded = await client.post(
            "/api/voice/result",
            data={"filename": "real_song_rvc_vocal.wav"},
        )
        direct_download = await client.get(
            "/output/rvc/real_song_rvc_vocal.wav"
        )
        deleted = await client.request(
            "DELETE",
            "/api/voice/result",
            data={"filename": "real_song_rvc_vocal.wav"},
        )
        missing = await client.post(
            "/api/voice/result",
            data={"filename": "real_song_rvc_vocal.wav"},
        )
        restored = await client.request(
            "PUT",
            "/api/voice/result",
            data={"filename": "real_song_rvc_vocal.wav"},
        )

    assert converted.status_code == 200
    assert converted.headers["x-rvc-output"] == "real_song_rvc_vocal.wav"
    assert downloaded.status_code == 200
    assert downloaded.content == b"RIFF-converted-wav"
    assert direct_download.status_code == 404
    assert deleted.status_code == 200
    assert missing.status_code == 404
    assert restored.status_code == 200
    assert (settings.rvc_result_dir / "real_song_rvc_vocal.wav").is_file()


async def test_voice_result_rejects_path_traversal(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings), FakeVoiceEngine())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/voice/result",
            data={"filename": "../secret.wav"},
        )
    assert response.status_code == 400
