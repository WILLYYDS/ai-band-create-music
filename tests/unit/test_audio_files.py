import asyncio
from pathlib import Path

import httpx
import pytest

from app.core.errors import GenerationError
from app.services import audio_files
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


async def test_preview_encoders_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """分轨在编码前就归还了容量配额，编码并发必须由编码闸门自己兜住。"""
    running = 0
    peak = 0
    limit = 2
    release = asyncio.Event()

    async def blocked_encode(source: Path, _output_dir: Path) -> str | None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        if running >= limit:
            release.set()
        try:
            await release.wait()
        finally:
            running -= 1
        return f"{source.stem}.mp3"

    monkeypatch.setattr(audio_files, "PLAYBACK_ENCODE_CONCURRENCY", limit)
    # 闸门按事件循环缓存，先丢掉本循环里可能已经建好的那个，保证新限额生效。
    audio_files._encoder_gates.pop(asyncio.get_running_loop(), None)
    monkeypatch.setattr(audio_files, "_encode_playback_mp3", blocked_encode)
    sources = [tmp_path / f"stem{index}.wav" for index in range(4)]
    for source in sources:
        source.write_bytes(b"RIFF")

    previews = await asyncio.gather(
        *(audio_files.make_playback_mp3(source, tmp_path) for source in sources)
    )

    assert peak == limit
    assert previews == [f"stem{index}.mp3" for index in range(4)]
