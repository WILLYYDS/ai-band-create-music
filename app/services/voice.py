from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import Response

from app.core.config import Settings

logger = logging.getLogger(__name__)


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

    @application.post("/api/voice/result", response_class=Response)
    async def download_result(
        filename: Annotated[str, Form(min_length=1, max_length=260)],
    ) -> Response:
        result_path = _result_path(settings, filename)
        if not result_path.is_file():
            raise HTTPException(status_code=404, detail="Converted audio not found")
        return Response(
            content=result_path.read_bytes(),
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename, safe='')}",
            },
        )

    @application.delete("/api/voice/result")
    async def delete_result(
        filename: Annotated[str, Form(min_length=1, max_length=260)],
    ) -> dict[str, bool]:
        result_path = _result_path(settings, filename)
        if not result_path.is_file():
            raise HTTPException(status_code=404, detail="Converted audio not found")
        trash_path = _trash_result_path(settings, filename)
        trash_path.parent.mkdir(parents=True, exist_ok=True)
        trash_path.unlink(missing_ok=True)
        result_path.replace(trash_path)
        return {"success": True}

    @application.put("/api/voice/result")
    async def restore_result(
        filename: Annotated[str, Form(min_length=1, max_length=260)],
    ) -> dict[str, bool]:
        result_path = _result_path(settings, filename)
        if result_path.is_file():
            return {"success": True}
        trash_path = _trash_result_path(settings, filename)
        if not trash_path.is_file():
            raise HTTPException(status_code=404, detail="Deleted audio not found")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        trash_path.replace(result_path)
        return {"success": True}

    @application.post("/api/voice/convert", response_class=Response)
    async def convert_voice(
        request: Request,
        file: Annotated[UploadFile | None, File(description="Input audio file")] = None,
        audio: Annotated[
            UploadFile | None, File(description="Alias of the 'file' field")
        ] = None,
        f0_up_key: Annotated[int, Form(ge=-24, le=24)] = 0,
        f0_method: Annotated[
            Literal["harvest", "pm", "crepe", "rmvpe"], Form()
        ] = "rmvpe",
        index_rate: Annotated[float, Form(ge=0.0, le=1.0)] = 0.75,
        filter_radius: Annotated[int, Form(ge=0, le=7)] = 3,
        resample_sr: Annotated[int, Form(ge=0, le=96000)] = 0,
        rms_mix_rate: Annotated[float, Form(ge=0.0, le=1.0)] = 0.25,
        protect: Annotated[float, Form(ge=0.0, le=0.5)] = 0.33,
        song_name: Annotated[str, Form(min_length=1, max_length=200)] = "converted",
    ) -> Response:
        upload = file or audio
        if upload is None:
            raise HTTPException(status_code=400, detail="Upload an audio file in 'file'")

        suffix = _safe_audio_suffix(upload.filename)
        output_name = f"{_source_song_name(upload.filename, song_name)}_rvc_vocal.wav"
        settings.rvc_result_dir.mkdir(parents=True, exist_ok=True)
        result_path = settings.rvc_result_dir / output_name
        try:
            with tempfile.TemporaryDirectory(prefix="rvc-") as temp_dir:
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
                shutil.copyfile(output_path, result_path)
                output = result_path.read_bytes()
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
            content=output,
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
    source = re.sub(r"_(?:vocal|vocals|voice)$", "", source, flags=re.IGNORECASE)
    return _safe_song_name(source)


def _result_path(settings: Settings, filename: str) -> Path:
    if Path(filename).name != filename or not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid result filename")
    return settings.rvc_result_dir / filename


def _trash_result_path(settings: Settings, filename: str) -> Path:
    return settings.rvc_result_dir / ".trash" / filename
