from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Annotated
from urllib.parse import unquote, urlsplit

from fastapi import FastAPI, Form, HTTPException, Request

from app.core.config import Settings
from app.services.audio_files import output_path_from_url
from app.services.job_files import job_song_dir

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
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 1.0,
        protect: float = 0.33,
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

    @application.delete("/api/voice/result")
    async def delete_result(
        request: Request,
        filename: Annotated[str, Form(min_length=1, max_length=260)],
        job_id: Annotated[str, Form(min_length=1, max_length=100)],
        song: Annotated[int, Form(ge=0)] = 0,
    ) -> dict[str, bool]:
        job = request.app.state.jobs.get(job_id)
        job_result = _job_song_result(request, job_id, song)
        result_path = _result_path(settings, filename, job_id, song)
        if not result_path.is_file():
            raise HTTPException(status_code=404, detail="Converted audio not found")
        trash_path = trash_result_path(settings, filename, job_id, song)
        trash_path.parent.mkdir(parents=True, exist_ok=True)
        trash_path.unlink(missing_ok=True)
        result_path.replace(trash_path)
        replaced_vocal = job_result.get("replacedVocal")
        if isinstance(replaced_vocal, str):
            try:
                advertised_path = output_path_from_url(replaced_vocal, settings.output_dir)
            except ValueError:
                advertised_path = None
            if advertised_path == result_path:
                deleted = {"url": job_result.pop("replacedVocal")}
                model = job_result.pop("_replacedVocalModel", None)
                if isinstance(model, str):
                    deleted["model"] = model
                job.deleted_replaced_vocals[str(song)] = deleted
                if job.replace_song == song and (
                    job.replace_task is None or job.replace_task.done()
                ):
                    job.replace_song = None
                    job.replace_status = None
                    job.replace_stage = None
                    job.replace_progress = None
                    job.replace_message = None
                    job.replace_error = None
                job.save(settings.output_dir)
        return {"success": True}

    @application.put("/api/voice/result")
    async def restore_result(
        request: Request,
        filename: Annotated[str, Form(min_length=1, max_length=260)],
        job_id: Annotated[str, Form(min_length=1, max_length=100)],
        song: Annotated[int, Form(ge=0)] = 0,
    ) -> dict[str, bool]:
        job = request.app.state.jobs.get(job_id)
        job_result = _job_song_result(request, job_id, song)
        result_path = _result_path(settings, filename, job_id, song)
        if not result_path.is_file():
            trash_path = trash_result_path(settings, filename, job_id, song)
            if not trash_path.is_file():
                raise HTTPException(status_code=404, detail="Deleted audio not found")
            result_path.parent.mkdir(parents=True, exist_ok=True)
            trash_path.replace(result_path)
        deleted_replaced_vocals = getattr(job, "deleted_replaced_vocals", {})
        deleted = deleted_replaced_vocals.get(str(song))
        if isinstance(deleted, dict) and isinstance(deleted.get("url"), str):
            try:
                advertised_path = output_path_from_url(deleted["url"], settings.output_dir)
            except ValueError:
                advertised_path = None
            if advertised_path == result_path:
                job_result["replacedVocal"] = deleted["url"]
                if isinstance(deleted.get("model"), str):
                    job_result["_replacedVocalModel"] = deleted["model"]
                del deleted_replaced_vocals[str(song)]
                job.save(settings.output_dir)
        return {"success": True}

    return active_engine


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


def _safe_song_name(value: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    name = re.sub(r"\s+", " ", name)
    return name[:120].rstrip(" .") or "converted"


def _source_song_name(filename: str | None, fallback: str) -> str:
    source = Path(filename).stem if filename else fallback
    source = re.sub(r"_(?:rvc_)?(?:vocal|vocals|voice)$", "", source, flags=re.IGNORECASE)
    return _safe_song_name(source)


def replacement_filename(filename: str | None, fallback: str = "converted") -> str:
    return f"{_source_song_name(filename, fallback)}_rvc_vocal.wav"


def _result_filename(value: str) -> str:
    """从 `filename` 里取出结果文件名，容忍各种 URL 形态。

    前端把 job 结果里的 `replacedVocal` 原样回传，而它可能是裸文件名、站内代理路径
    （`/api/music/output/jobs/.../x.wav`）或生成服务的绝对 URL。代理层历史上做过这层
    归一化，一旦代理不再处理（或换了调用方），后端直接 400 就会让"删除替换人声 → 撤回"
    这类操作失败。这里统一剥掉 scheme/host/目录，只保留最后一段。
    """
    parsed = urlsplit(value)
    path = unquote(parsed.path) if parsed.scheme or parsed.netloc else value
    segments = [segment for segment in path.split("/") if segment not in ("", ".")]
    # 归一化之前先拒绝穿越形状：不要靠"最后一段恰好不存在"来兜底，那样既可能 500，
    # 也失去了"明确拒绝非法输入"这条回归护栏。
    if not segments or any(
        segment == ".." or "\\" in segment or "\x00" in segment for segment in segments
    ):
        raise HTTPException(status_code=400, detail="Invalid result filename")
    name = segments[-1]
    if name != Path(name).name or Path(name).suffix.lower() != ".wav":
        raise HTTPException(status_code=400, detail="Invalid result filename")
    return name


def _result_path(
    settings: Settings,
    filename: str,
    job_id: str,
    song: int = 0,
) -> Path:
    try:
        root = job_song_dir(settings.output_dir, job_id, song)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid job output path") from exc
    return root / _result_filename(filename)


def trash_result_path(
    settings: Settings,
    filename: str,
    job_id: str,
    song: int = 0,
) -> Path:
    return (
        settings.output_dir
        / ".trash"
        / job_id
        / f"song_{song + 1}"
        / _result_filename(filename)
    )
