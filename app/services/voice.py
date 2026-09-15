from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import Response

from app.core.config import Settings
from app.core.errors import CapacityExceededError
from app.services.audio_files import (
    output_path_from_url,
    require_readable_file,
)
from app.services.job_files import job_song_dir
from app.services.stems import prepare_ffmpeg_environment

logger = logging.getLogger(__name__)

MIX_FILTER = (
    "[0:a]equalizer=f=3000:t=q:w=1:g=2.5,volume=3dB[vocal];"
    "[vocal][1:a][2:a][3:a]"
    "amix=inputs=4:duration=longest:dropout_transition=0:normalize=0[premix]"
)
LIMIT_FILTER = (
    "volume={gain_db:.3f}dB,alimiter=limit=0.891251:attack=5:release=50:level=false:latency=true"
)


class RVCConversionError(RuntimeError):
    """Raised when the local RVC inference engine cannot convert an input file."""


class RVCEngine:
    """Lazy, concurrency-safe adapter around rvc-python."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._inference = None
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self._inference is not None

    async def convert(
        self,
        input_path: Path,
        output_path: Path,
        *,
        f0_up_key: int,
        f0_method: str,
        index_rate: float,
        filter_radius: int,
        resample_sr: int,
        rms_mix_rate: float,
        protect: float,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._convert_sync,
                input_path,
                output_path,
                f0_up_key=f0_up_key,
                f0_method=f0_method,
                index_rate=index_rate,
                filter_radius=filter_radius,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
            )

    def _load_sync(self):
        if self._inference is not None:
            return self._inference

        self._validate_assets()
        try:
            import torch
            from rvc_python import infer as rvc_infer

            if self._settings.rvc_device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; set RVC_DEVICE=cpu to use CPU inference")
            if self._settings.rvc_device.startswith("cpu"):
                torch.set_num_threads(self._settings.rvc_cpu_threads)

            # rvc-python stores its large base models beside the installed package
            # and has no directory option. Link only those assets into this
            # environment; code and mutable configs remain owned by generate.
            base_model_dir = self._settings.rvc_base_model_dir
            if base_model_dir is not None:
                _reuse_base_models(rvc_infer, base_model_dir)

            inference = rvc_infer.RVCInference(
                device=self._settings.rvc_device,
                model_path=str(self._settings.rvc_model_path),
                index_path=(
                    str(self._settings.rvc_index_path)
                    if self._settings.rvc_index_path is not None
                    else ""
                ),
                version=self._settings.rvc_model_version,
            )
            _load_hubert_safely(inference, torch)
        except Exception as exc:
            raise RVCConversionError(f"Failed to load RVC model: {exc}") from exc

        self._inference = inference
        return inference

    def _convert_sync(self, input_path: Path, output_path: Path, **params) -> None:
        inference = self._load_sync()
        try:
            rvc_params = {
                **params,
                "f0up_key": params["f0_up_key"],
                "f0method": params["f0_method"],
            }
            del rvc_params["f0_up_key"]
            del rvc_params["f0_method"]
            inference.set_params(**rvc_params)
            inference.infer_file(str(input_path), str(output_path))
        except Exception as exc:
            raise RVCConversionError(f"RVC inference failed: {exc}") from exc

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RVCConversionError("RVC inference did not produce an output file")

    def _validate_assets(self) -> None:
        if not self._settings.rvc_model_path.is_file():
            raise RuntimeError(f"RVC model not found: {self._settings.rvc_model_path}")
        index_path = self._settings.rvc_index_path
        if index_path is not None and not index_path.is_file():
            raise RuntimeError(f"RVC index not found: {index_path}")
        base_model_dir = self._settings.rvc_base_model_dir
        if base_model_dir is not None:
            missing = [
                name
                for name in ("hubert_base.pt", "rmvpe.pt", "rmvpe.onnx")
                if not (base_model_dir / name).is_file()
            ]
            if missing:
                raise RuntimeError(f"RVC base models are missing: {', '.join(missing)}")


def install_voice_api(
    application: FastAPI,
    settings: Settings,
    engine: RVCEngine | None = None,
) -> RVCEngine:
    active_engine = engine or RVCEngine(settings)
    application.state.voice_engine = active_engine

    @application.post("/api/voice/mix", response_class=Response)
    async def mix_voice(
        request: Request,
        job_id: Annotated[str, Form(min_length=1, max_length=100)],
        vocal_filename: Annotated[str, Form(min_length=1, max_length=260)],
        drums: Annotated[str, Form(min_length=1, max_length=2048)],
        bass: Annotated[str, Form(min_length=1, max_length=2048)],
        other: Annotated[str, Form(min_length=1, max_length=2048)],
        song: Annotated[int, Form(ge=0)] = 0,
    ) -> Response:
        vocal_path = _result_path(settings, vocal_filename, job_id, song)
        job_result = _job_song_result(request, job_id, song)
        output_dir = vocal_path.parent
        full_track = job_result.get("fullTrack")
        if not isinstance(full_track, str):
            raise HTTPException(status_code=409, detail="Job has no original full track")
        reference_path = _output_audio_path(settings, full_track, output_dir)
        inputs = [
            vocal_path,
            _output_audio_path(settings, drums, output_dir),
            _output_audio_path(settings, bass, output_dir),
            _output_audio_path(settings, other, output_dir),
        ]
        for input_path in [reference_path, *inputs]:
            try:
                require_readable_file(input_path, "混音输入文件不可读")
            except RuntimeError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        output_name = f"{_source_song_name(vocal_filename, 'completed')}_rvc_mix.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / output_name
        try:
            async with request.app.state.orchestrator.capacity.slot():
                with tempfile.TemporaryDirectory(prefix=".mix-", dir=output_dir) as temp_dir:
                    output_path = Path(temp_dir) / "mix.wav"
                    await _mix_tracks(
                        inputs,
                        reference_path,
                        output_path,
                        settings.rvc_mix_timeout_seconds,
                    )
                    output_path.replace(result_path)
        except CapacityExceededError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except RVCConversionError as exc:
            logger.exception("Audio mixing failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return Response(
            content=result_path.read_bytes(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": (
                    f"attachment; filename*=UTF-8''{quote(output_name, safe='')}"
                ),
                "X-Mix-Output": quote(output_name, safe=""),
            },
        )

    @application.delete("/api/voice/result")
    async def delete_result(
        request: Request,
        filename: Annotated[str, Form(min_length=1, max_length=260)],
        job_id: Annotated[str, Form(min_length=1, max_length=100)],
        song: Annotated[int, Form(ge=0)] = 0,
    ) -> dict[str, bool]:
        _job_song_result(request, job_id, song)
        result_path = _result_path(settings, filename, job_id, song)
        if not result_path.is_file():
            raise HTTPException(status_code=404, detail="Converted audio not found")
        trash_path = _trash_result_path(settings, filename, job_id, song)
        trash_path.parent.mkdir(parents=True, exist_ok=True)
        trash_path.unlink(missing_ok=True)
        result_path.replace(trash_path)
        return {"success": True}

    @application.put("/api/voice/result")
    async def restore_result(
        request: Request,
        filename: Annotated[str, Form(min_length=1, max_length=260)],
        job_id: Annotated[str, Form(min_length=1, max_length=100)],
        song: Annotated[int, Form(ge=0)] = 0,
    ) -> dict[str, bool]:
        _job_song_result(request, job_id, song)
        result_path = _result_path(settings, filename, job_id, song)
        if result_path.is_file():
            return {"success": True}
        trash_path = _trash_result_path(settings, filename, job_id, song)
        if not trash_path.is_file():
            raise HTTPException(status_code=404, detail="Deleted audio not found")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        trash_path.replace(result_path)
        return {"success": True}

    @application.post("/api/voice/convert", response_class=Response)
    async def convert_voice(
        request: Request,
        job_id: Annotated[str, Form(min_length=1, max_length=100)],
        file: Annotated[UploadFile | None, File(description="Input audio file")] = None,
        audio: Annotated[UploadFile | None, File(description="Alias of the 'file' field")] = None,
        f0_up_key: Annotated[int, Form(ge=-24, le=24)] = 0,
        f0_method: Annotated[Literal["harvest", "pm", "crepe", "rmvpe"], Form()] = "rmvpe",
        index_rate: Annotated[float, Form(ge=0.0, le=1.0)] = 0.75,
        filter_radius: Annotated[int, Form(ge=0, le=7)] = 3,
        resample_sr: Annotated[int, Form(ge=0, le=96000)] = 0,
        rms_mix_rate: Annotated[float, Form(ge=0.0, le=1.0)] = 1.0,
        protect: Annotated[float, Form(ge=0.0, le=0.5)] = 0.33,
        song_name: Annotated[str, Form(min_length=1, max_length=200)] = "converted",
        song: Annotated[int, Form(ge=0)] = 0,
    ) -> Response:
        upload = file or audio
        if upload is None:
            raise HTTPException(status_code=400, detail="Upload an audio file in 'file'")

        suffix = _safe_audio_suffix(upload.filename)
        output_name = f"{_source_song_name(upload.filename, song_name)}_rvc_vocal.wav"
        result_path = _result_path(settings, output_name, job_id, song)
        _job_song_result(request, job_id, song)
        output_dir = result_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix=".rvc-", dir=output_dir) as temp_dir:
                input_path = Path(temp_dir) / f"input{suffix}"
                output_path = Path(temp_dir) / "converted.wav"
                await _save_upload(upload, input_path, settings.rvc_max_upload_bytes)
                await _voice_engine(request).convert(
                    input_path,
                    output_path,
                    f0_up_key=f0_up_key,
                    f0_method=f0_method,
                    index_rate=index_rate,
                    filter_radius=filter_radius,
                    resample_sr=resample_sr,
                    rms_mix_rate=rms_mix_rate,
                    protect=protect,
                )
                output_path.replace(result_path)
        except HTTPException:
            raise
        except RVCConversionError as exc:
            logger.exception("Voice conversion failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        finally:
            await upload.close()

        return Response(
            content=result_path.read_bytes(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": (
                    f"attachment; filename*=UTF-8''{quote(output_name, safe='')}"
                ),
                "X-RVC-Model": settings.rvc_model_path.name,
                "X-RVC-Output": quote(output_name, safe=""),
            },
        )

    return active_engine


def _voice_engine(request: Request) -> RVCEngine:
    return request.app.state.voice_engine


def _job_song_result(request: Request, job_id: str, song: int) -> dict:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Generation job not found")
    if not job.result:
        raise HTTPException(status_code=409, detail="Generation job has no audio result")
    results = [job.result, *(job.result.get("alternatives") or [])]
    if song >= len(results) or not isinstance(results[song], dict):
        raise HTTPException(status_code=404, detail="Song not found in generation job")
    return results[song]


def _reuse_base_models(rvc_infer, source_dir: Path) -> None:
    package_dir = Path(rvc_infer.__file__).resolve().parent
    target_dir = package_dir / "base_model"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("hubert_base.pt", "rmvpe.pt", "rmvpe.onnx"):
        source = source_dir / name
        target = target_dir / name
        if target.is_symlink() and target.resolve() != source.resolve():
            target.unlink()
        if not target.exists():
            target.symlink_to(source)


def _load_hubert_safely(inference, torch) -> None:
    from fairseq.data.dictionary import Dictionary
    from rvc_python.modules.vc.utils import load_hubert

    # Torch 2.6+ defaults to weights_only=True. This checkpoint needs exactly
    # Fairseq's Dictionary class, so allowlist that class instead of disabling
    # safe loading for the whole generate process.
    with torch.serialization.safe_globals([Dictionary]):
        inference.vc.hubert_model = load_hubert(inference.config, inference.lib_dir)


async def _save_upload(upload: UploadFile, destination: Path, limit: int) -> None:
    size = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"Audio exceeds the {limit // (1024 * 1024)} MB limit",
                )
            output.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty")


def _safe_audio_suffix(filename: str | None) -> str:
    suffix = Path(filename or "input.wav").suffix.lower()
    allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio extension: {suffix or '(none)'}",
        )
    return suffix


def _safe_song_name(value: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    name = re.sub(r"\s+", " ", name)
    return name[:120].rstrip(" .") or "converted"


def _source_song_name(filename: str | None, fallback: str) -> str:
    source = Path(filename).stem if filename else fallback
    source = re.sub(r"_(?:rvc_)?(?:vocal|vocals|voice)$", "", source, flags=re.IGNORECASE)
    return _safe_song_name(source)


def _result_path(
    settings: Settings,
    filename: str,
    job_id: str,
    song: int = 0,
) -> Path:
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".wav":
        raise HTTPException(status_code=400, detail="Invalid result filename")
    try:
        root = job_song_dir(settings.output_dir, job_id, song)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid job output path") from exc
    return root / filename


def _output_audio_path(settings: Settings, value: str, expected_dir: Path) -> Path:
    root = settings.output_dir.resolve()
    try:
        target = output_path_from_url(value, root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid stem URL") from exc
    relative = target.relative_to(root)
    if (
        target.parent != expected_dir.resolve()
        or ".trash" in relative.parts
        or target.suffix.lower()
        not in {
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a",
            ".aac",
            ".webm",
        }
    ):
        raise HTTPException(status_code=400, detail="Invalid stem URL")
    return target


async def _mix_tracks(
    inputs: list[Path],
    reference_path: Path,
    output_path: Path,
    timeout_seconds: float,
) -> None:
    environment = prepare_ffmpeg_environment()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    def remaining() -> float:
        value = deadline - loop.time()
        if value <= 0:
            raise RVCConversionError(f"FFmpeg mixing timed out after {timeout_seconds:g} seconds")
        return value

    sample_rate, channels, codec = await _master_audio_format(
        reference_path, environment, remaining()
    )
    premix_path = output_path.with_name("premix.wav")
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for input_path in inputs:
        command.extend(("-i", str(input_path)))
    command.extend(
        (
            "-filter_complex",
            MIX_FILTER,
            "-map",
            "[premix]",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_f32le",
            str(premix_path),
        )
    )
    await _run_ffmpeg(command, environment, remaining())
    reference_loudness, premix_loudness = await asyncio.gather(
        _integrated_loudness(reference_path, environment, remaining()),
        _integrated_loudness(premix_path, environment, remaining()),
    )
    # ponytail: cap malformed/silent-input compensation; widen only if real mixes need it.
    gain_db = max(-12.0, min(12.0, reference_loudness - premix_loudness))
    await _run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(premix_path),
            "-af",
            LIMIT_FILTER.format(gain_db=gain_db),
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            codec,
            str(output_path),
        ],
        environment,
        remaining(),
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RVCConversionError("FFmpeg mixing did not produce an output file")


async def _master_audio_format(
    input_path: Path, environment: dict[str, str], timeout_seconds: float
) -> tuple[int, int, str]:
    output = await _run_ffmpeg(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,codec_name,bits_per_sample,bits_per_raw_sample",
            "-of",
            "json",
            str(input_path),
        ],
        environment,
        timeout_seconds,
        capture_stdout=True,
    )
    try:
        stream = json.loads(output)["streams"][0]
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        bits = int(stream.get("bits_per_raw_sample") or stream.get("bits_per_sample") or 0)
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RVCConversionError("FFprobe could not read the master audio format") from exc
    if not 8_000 <= sample_rate <= 384_000 or not 1 <= channels <= 32:
        raise RVCConversionError("Master audio format is unsupported")
    codec = stream.get("codec_name")
    supported_pcm = {"pcm_u8", "pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_f64le"}
    if codec not in supported_pcm:
        codec = {8: "pcm_u8", 24: "pcm_s24le", 32: "pcm_s32le"}.get(bits, "pcm_s16le")
    return sample_rate, channels, codec


async def _integrated_loudness(
    input_path: Path, environment: dict[str, str], timeout_seconds: float
) -> float:
    stderr = await _run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(input_path),
            "-af",
            "loudnorm=I=-14:LRA=20:TP=-1:print_format=json",
            "-f",
            "null",
            "-",
        ],
        environment,
        timeout_seconds,
    )
    matches = re.findall(r'\{\s*"input_i".*?\}', stderr, flags=re.DOTALL)
    try:
        value = float(json.loads(matches[-1])["input_i"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RVCConversionError("FFmpeg could not measure audio loudness") from exc
    if not math.isfinite(value):
        raise RVCConversionError("Audio loudness is not finite")
    return value


async def _run_ffmpeg(
    command: list[str],
    environment: dict[str, str],
    timeout_seconds: float,
    *,
    capture_stdout: bool = False,
) -> str:
    started = asyncio.get_running_loop().time()
    with tempfile.TemporaryFile() as output_stream, tempfile.TemporaryFile() as error_stream:
        try:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=output_stream if capture_stdout else subprocess.DEVNULL,
                stderr=error_stream,
            )
        except OSError as exc:
            raise RVCConversionError(f"Unable to start FFmpeg: {exc}") from exc
        try:
            while process.poll() is None:
                if asyncio.get_running_loop().time() - started >= timeout_seconds:
                    with suppress(ProcessLookupError):
                        process.kill()
                    process.wait()
                    raise RVCConversionError(
                        f"FFmpeg mixing timed out after {timeout_seconds:g} seconds"
                    )
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            with suppress(ProcessLookupError):
                process.kill()
            process.wait()
            raise
        error_stream.seek(0)
        detail = error_stream.read().decode("utf-8", errors="replace")
        output_stream.seek(0)
        output = output_stream.read().decode("utf-8", errors="replace")
    if process.returncode != 0:
        raise RVCConversionError(f"FFmpeg mixing failed: {detail.strip() or process.returncode}")
    return output if capture_stdout else detail


def _trash_result_path(
    settings: Settings,
    filename: str,
    job_id: str,
    song: int = 0,
) -> Path:
    return settings.output_dir / ".trash" / job_id / f"song_{song + 1}" / filename
