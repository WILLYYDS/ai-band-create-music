from __future__ import annotations

import os
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import quote

import httpx

from app.core.errors import GenerationError


def detect_audio_content_type(path: Path) -> str:
    try:
        with path.open("rb") as audio:
            header = audio.read(12)
        if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
            return "audio/wav"
        if header[:3] == b"ID3" or (len(header) >= 2 and header[0] == 0xFF):
            return "audio/mpeg"
    except OSError:
        pass
    return "audio/wav" if path.suffix.lower() == ".wav" else "audio/mpeg"


def require_readable_file(path: Path, message: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise GenerationError(f"{message}: {path}")


def build_public_audio_url(base_url: str, relative_path: str | Path) -> str:
    encoded = "/".join(
        quote(part, safe="") for part in Path(relative_path).parts if part not in {"/", ""}
    )
    return f"{base_url.rstrip('/')}/output/{encoded}"


async def write_stream_atomically(
    chunks: AsyncIterator[bytes],
    target_path: Path,
    empty_message: str,
) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_name(f"{target_path.name}.{os.getpid()}.part")
    written = 0
    try:
        # Writes are small sequential chunks to a local file. Keeping the handle in
        # this coroutine avoids thread-pool cancellation deadlocks while streaming.
        with temporary_path.open("wb") as output:
            async for chunk in chunks:
                if not chunk:
                    continue
                written += len(chunk)
                output.write(chunk)
        if written == 0:
            raise GenerationError(empty_message)
        temporary_path.replace(target_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target_path


async def download_audio(
    client: httpx.AsyncClient,
    audio_url: str,
    target_path: Path,
    *,
    timeout: float,
    trusted_local_root: Path,
) -> Path:
    if not audio_url.lower().startswith(("http://", "https://")):
        trusted_local_root = trusted_local_root.resolve()
        local_path = Path(audio_url).expanduser()
        if not local_path.is_absolute():
            local_path = trusted_local_root / local_path
        local_path = local_path.resolve()
        try:
            local_path.relative_to(trusted_local_root)
        except ValueError as exc:
            raise GenerationError("音乐生成接口返回了不受信任的本地音频路径。") from exc
        require_readable_file(local_path, "音乐生成接口返回的本地音频文件不可读")
        return local_path

    try:
        async with client.stream(
            "GET", audio_url, timeout=timeout, follow_redirects=True
        ) as response:
            response.raise_for_status()
            return await write_stream_atomically(
                response.aiter_bytes(), target_path, "音乐生成音频下载结果为空。"
            )
    except httpx.HTTPError as exc:
        raise GenerationError(f"音乐生成音频下载失败：{exc}") from exc


async def ensure_file_under_root(source: Path, root: Path, target_name: str) -> Path:
    source = source.resolve()
    root = root.resolve()
    require_readable_file(source, "完整音频不可读")
    try:
        source.relative_to(root)
        return source
    except ValueError:
        target = root / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target
