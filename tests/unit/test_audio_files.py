from pathlib import Path

import httpx
import pytest

from app.core.errors import GenerationError
from app.services.audio_files import download_audio


async def test_local_provider_audio_stays_within_trusted_root(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    local_audio = trusted / "song.mp3"
    local_audio.write_bytes(b"ID3-audio")
    secret = tmp_path / "secret.mp3"
    secret.write_bytes(b"secret")

    async with httpx.AsyncClient() as client:
        allowed = await download_audio(
            client,
            "song.mp3",
            trusted / "copy.mp3",
            timeout=1,
            trusted_local_root=trusted,
        )
        assert allowed == local_audio

        for untrusted_path in (str(secret), "../secret.mp3"):
            with pytest.raises(GenerationError, match="不受信任"):
                await download_audio(
                    client,
                    untrusted_path,
                    trusted / "copy.mp3",
                    timeout=1,
                    trusted_local_root=trusted,
                )
