"""固化「result / alternatives / count」的响应契约。

前端 `availableSongIndex` 会把选中曲目夹到 `musicGenerationOutputs(result).length - 1`，
也就是 `[result, ...alternatives]` 的长度。一旦出现「带 result 但 alternatives 缺失或偏短」的帧
或响应，正在看第 2 首的用户会被静默拽回第 1 首并被写回存储。这里把该形状不可能出现这件事
固化成测试。

测试数据优先使用 output/ 里真实落盘的任务（含真实的 count=2 任务）；仓库里没有时按生成器
实际写出的字段结构重建，保证结构本身不会漂移。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.requests import Request

from app.main import GenerationJob, _job_response
from tests.helpers import make_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "output"


def fake_request(app) -> Request:
    return Request(
        {
            "type": "http",
            "app": app,
            "headers": [],
            "scheme": "http",
            "server": ("testserver", 80),
            "path": "/api/jobs/active",
            "root_path": "",
            "query_string": b"",
            "method": "GET",
        }
    )


def _real_records() -> list[tuple[str, dict]]:
    records = []
    for path in sorted((OUTPUT_DIR / "jobs").glob("*/job.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data.get("result"), dict):
            records.append((path.parent.name, data))
    return records


REAL_RECORDS = _real_records()
REAL_TWO_SONG = [item for item in REAL_RECORDS if item[1]["result"].get("count") == 2]


def _song(full_track: str) -> dict:
    """按 orchestrator 实际写出的字段结构造一首歌的输出。"""
    return {
        "fullTrack": full_track,
        "stems": {},
        "stemUrls": [],
        "waveforms": {"full": [0.25, 0.5]},
        "splitEnabled": False,
        "durationSeconds": 30.0,
        "debug": {"music": {}},
    }


def _job(job_id: str, *, status: str, count: int | None, stage: str = "completed") -> GenerationJob:
    result = None
    if count is not None:
        songs = [_song(f"jobs/{job_id}/song_{index + 1}/full_song.wav") for index in range(count)]
        result = {
            "success": True,
            "jobId": job_id,
            "prompt": "rock",
            "durationMinutes": "auto",
            "structuredPrompt": "[Genre: Rock]",
            "lyrics": "[Verse]",
            "count": count,
            "requestedCount": count,
            "provider": "minimax_music",
            "createdAt": "2026-01-01T00:00:00+00:00",
            "alternatives": songs[1:],
            **songs[0],
        }
    return GenerationJob(
        job_id=job_id,
        prompt="rock",
        status=status,
        stage=stage,
        progress=100 if status == "succeeded" else None,
        result=result,
    )


def test_real_two_song_jobs_keep_result_and_alternatives_consistent():
    """真实落盘的 count=2 任务：result 一定在，alternatives 长度恒为 count-1。"""
    if not REAL_TWO_SONG:
        pytest.skip("output/ 里没有真实的 count=2 任务")
    for job_id, data in REAL_TWO_SONG:
        result = data["result"]
        assert isinstance(result.get("alternatives"), list), job_id
        assert result["count"] == 2, job_id
        assert result["count"] == 1 + len(result["alternatives"]), job_id
        assert "count" not in result["alternatives"][0], job_id


def test_every_persisted_result_declares_alternatives_and_matching_count():
    """全部落盘任务：有 result 就一定有 alternatives 列表，且 count 与长度自洽。"""
    if not REAL_RECORDS:
        pytest.skip("output/ 里没有带 result 的任务")
    for job_id, data in REAL_RECORDS:
        result = data["result"]
        assert isinstance(result.get("alternatives"), list), job_id
        assert result["count"] == 1 + len(result["alternatives"]), job_id


def test_running_job_frames_never_carry_a_partial_result(tmp_path):
    """生成中的任务：result 恒为 null —— 前端因此不会夹取用户选择。

    这也是「带 result 但 alternatives 缺失」这类帧不可能出现的根据：job.result 只在整套
    orchestrator 生成结束后被一次性赋值，中途任何一帧都还没有 result。
    """
    settings = make_settings(tmp_path)
    job = _job("job_running", status="running", count=None, stage="generating_music")
    # 模拟 orchestrator 内部 report() 推进阶段：只改状态字段，绝不动 result
    for stage, message in (
        ("generating_music", "音乐模型正在生成第 1/2 首"),
        ("saving_audio", "正在保存第 1 首"),
        ("generating_music", "音乐模型正在生成第 2/2 首"),
    ):
        job.status, job.stage, job.progress, job.message = "running", stage, None, message
        payload = _job_response(job, fake_request(object()), settings, include_waveforms=False)
        assert payload["status"] == "running"
        assert payload["result"] is None, f"{stage} 阶段不应带 result"
        assert "count" not in payload


def test_terminal_frame_always_carries_every_song(tmp_path):
    """终态帧：result.count == 1 + len(alternatives)，两首都带得满。"""
    settings = make_settings(tmp_path)
    for count in (1, 2):
        job = _job(f"job_done_{count}", status="succeeded", count=count)
        payload = _job_response(job, fake_request(object()), settings, include_waveforms=False)
        result = payload["result"]
        assert payload["status"] == "succeeded"
        assert result["count"] == count
        assert result["count"] == 1 + len(result["alternatives"])
        assert isinstance(result["fullTrack"], str)
        for alternative in result["alternatives"]:
            assert isinstance(alternative["fullTrack"], str)


def test_successful_job_can_never_be_missing_alternatives(tmp_path):
    """唯一的「带 result」来源是 orchestrator 的成功返回，它一定会写 alternatives 键。"""
    settings = make_settings(tmp_path)
    job = _job("job_two", status="succeeded", count=2)
    assert "alternatives" in job.result

    payload = _job_response(job, fake_request(object()), settings, include_waveforms=False)
    assert "alternatives" in payload["result"]
    assert len(payload["result"]["alternatives"]) == 1

    # 记录退化形状：即使 alternatives 被清空，count 仍然说明有两首候选。
    # 也就是说 count 比 alternatives.length 更稳，供契约核对时对照。
    job.result["alternatives"] = []
    degraded = _job_response(job, fake_request(object()), settings, include_waveforms=False)
    assert degraded["result"]["count"] == 2
    assert len(degraded["result"]["alternatives"]) == 0
