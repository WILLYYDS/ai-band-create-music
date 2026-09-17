from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
from starlette.requests import Request

from app.main import GenerationJob, create_app
from tests.helpers import make_orchestrator, make_settings


class SlowVoiceEngine:
    """比 BlockingVoiceEngine 多一个"调用序号"，用来区分首跑/重跑。"""

    loaded = True

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def convert(self, input_path: Path, output_path: Path, **_params) -> None:
        self.calls += 1
        marker = f"RIFF-replaced-{self.calls}".encode()
        self.started.set()
        await self.release.wait()
        output_path.write_bytes(marker)


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver")


def events_request(app, job_id: str) -> Request:
    return Request(
        {
            "type": "http",
            "app": app,
            "headers": [],
            "scheme": "http",
            "server": ("testserver", 80),
            "path": f"/api/jobs/{job_id}/events",
            "root_path": "",
            "query_string": b"",
            "method": "GET",
        }
    )


def seed_job(app, settings, job_id: str = "job-replace") -> tuple[GenerationJob, Path]:
    song_dir = settings.output_dir / "jobs" / job_id / "song_1"
    song_dir.mkdir(parents=True, exist_ok=True)
    full_track = song_dir / "full_song.wav"
    vocal = song_dir / "demo_vocal.mp3"
    full_track.write_bytes(b"RIFF-full")
    vocal.write_bytes(b"ID3-vocal")
    job = GenerationJob(
        job_id=job_id,
        prompt="rock",
        structured_prompt="[Genre: Rock]",
        lyrics="[Verse]\ntest",
        status="succeeded",
        stage="completed",
        progress=100,
        message="音乐生成完成",
        result={
            "success": True,
            "jobId": job_id,
            "prompt": "rock",
            "durationMinutes": "auto",
            "structuredPrompt": "[Genre: Rock]",
            "lyrics": "[Verse]\ntest",
            "count": 1,
            "alternatives": [],
            "fullTrack": full_track.relative_to(settings.output_dir).as_posix(),
            "stems": {"vocal": vocal.relative_to(settings.output_dir).as_posix()},
            "stemUrls": [vocal.relative_to(settings.output_dir).as_posix()],
            "waveforms": {"vocal": [0.5]},
            "splitEnabled": True,
            "debug": {},
        },
    )
    app.state.jobs[job_id] = job
    job.save(settings.output_dir)
    return job, vocal


class BlockingVoiceEngine:
    loaded = True

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def convert(self, input_path: Path, output_path: Path, **_params) -> None:
        self.calls += 1
        assert input_path.name == "demo_vocal.mp3"
        self.started.set()
        await self.release.wait()
        output_path.write_bytes(b"RIFF-replaced")


class ReadyVoiceEngine:
    """转换在 POST 返回后立刻完成，用来模拟"客户端连上 SSE 时替换已经收尾"。"""

    loaded = True

    async def convert(self, _input_path: Path, output_path: Path, **_params) -> None:
        output_path.write_bytes(b"RIFF-replaced")


class FailingVoiceEngine:
    loaded = True

    async def convert(self, *_args, **_params) -> None:
        raise RuntimeError("secret /tmp/model failure")


async def test_replace_runs_as_job_operation_and_caches_result(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    engine = BlockingVoiceEngine()
    app = create_app(settings, orchestrator, engine)
    job, _ = seed_job(app, settings)
    initial_record = json.loads(
        (settings.output_dir / "jobs" / job.job_id / "job.json").read_text(encoding="utf-8")
    )
    assert "deletedReplacedVocals" not in initial_record

    async with client(app) as http:
        accepted = await http.post(f"/api/jobs/{job.job_id}/replace?song=0")
        await asyncio.wait_for(engine.started.wait(), 2)
        running = (await http.get(f"/api/jobs/{job.job_id}")).json()
        history_running = (await http.get("/api/jobs")).json()["jobs"][0]
        duplicate = await http.post(f"/api/jobs/{job.job_id}/replace?song=0")
        blocked_split = await http.post(f"/api/jobs/{job.job_id}/split?song=0")
        events_task = asyncio.create_task(http.get(f"/api/jobs/{job.job_id}/events"))
        await asyncio.sleep(0)
        assert not events_task.done()
        engine.release.set()
        await job.replace_task
        events = await asyncio.wait_for(events_task, 2)
        completed = (await http.get(f"/api/jobs/{job.job_id}")).json()
        completed_history = (await http.get("/api/jobs")).json()["jobs"][0]
        audio = await http.get(completed["result"]["replacedVocal"])
        cached = await http.post(f"/api/jobs/{job.job_id}/replace?song=0")

    assert accepted.status_code == duplicate.status_code == 202
    assert (running["status"], running["stage"], running["progress"]) == (
        "running",
        "replacing_vocal",
        90,
    )
    assert running["replaceStatus"] == "running"
    assert running["replaceSong"] == 0
    assert running["message"] == "RVC 正在替换人声"
    assert history_running["status"] == "succeeded"
    assert "replaceStatus" not in history_running
    assert blocked_split.status_code == 409
    assert completed["status"] == completed["replaceStatus"] == "succeeded"
    assert completed["result"]["replacedVocal"].endswith("/demo_rvc_vocal.wav")
    assert not any(key.startswith("_") for key in completed_history["result"])
    assert audio.content == b"RIFF-replaced"
    assert cached.status_code == 200
    assert engine.calls == 1
    assert events.text.startswith("data: ")
    assert "event: done" in events.text
    assert json.loads(events.text.rsplit("data: ", 1)[1])["replaceStatus"] == "succeeded"
    stored_result = json.loads(
        (settings.output_dir / "jobs" / job.job_id / "job.json").read_text(encoding="utf-8")
    )["result"]
    assert stored_result["_replacedVocalModel"].startswith("v1:")
    assert len(stored_result["_replacedVocalModel"]) == 67
    assert str(settings.rvc_model_path) not in stored_result["_replacedVocalModel"]

    settings.rvc_model_path.touch()
    restarted = create_app(settings, orchestrator, engine)
    async with client(restarted) as http:
        restored = (await http.get(f"/api/jobs/{job.job_id}")).json()
        cached_after_restart = await http.post(f"/api/jobs/{job.job_id}/replace")
        replaced_url = restored["result"]["replacedVocal"]
        deleted_vocal = await http.delete(f"/api/jobs/{job.job_id}/stems/vocal")
        after_delete = (await http.get(f"/api/jobs/{job.job_id}")).json()
        deleted_replacement = await http.get(replaced_url)
        restored_vocal = await http.put(f"/api/jobs/{job.job_id}/stems/vocal")
        restored_replacement = await http.get(replaced_url)
    assert restored["result"]["replacedVocal"].endswith("/demo_rvc_vocal.wav")
    assert cached_after_restart.status_code == 200
    assert engine.calls == 1
    assert deleted_vocal.status_code == 204
    assert "replacedVocal" not in after_delete["result"]
    assert deleted_replacement.status_code == 404
    assert restored_vocal.json()["result"]["replacedVocal"] == replaced_url
    assert restored_replacement.content == b"RIFF-replaced"


async def test_deleting_vocal_cancels_active_replace(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    engine = BlockingVoiceEngine()
    app = create_app(settings, orchestrator, engine)
    job, _ = seed_job(app, settings)

    async with client(app) as http:
        await http.post(f"/api/jobs/{job.job_id}/replace")
        await asyncio.wait_for(engine.started.wait(), 2)
        deleted = await http.delete(f"/api/jobs/{job.job_id}/stems/vocal")
        cancelled = (await http.get(f"/api/jobs/{job.job_id}")).json()
        engine.release.set()
        await job.replace_task
        completed = (await http.get(f"/api/jobs/{job.job_id}")).json()

    assert deleted.status_code == 204
    assert cancelled["replaceStatus"] == "cancelled"
    assert completed["replaceStatus"] == "cancelled"
    assert "replacedVocal" not in completed["result"]
    assert job.replace_cancel_requested is False
    replaced_file = (
        settings.output_dir / "jobs" / job.job_id / "song_1" / "demo_rvc_vocal.wav"
    )
    assert not replaced_file.exists()


async def test_deleted_replace_result_can_be_regenerated(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    engine = BlockingVoiceEngine()
    engine.release.set()
    app = create_app(settings, orchestrator, engine)
    job, _ = seed_job(app, settings)

    async with client(app) as http:
        await http.post(f"/api/jobs/{job.job_id}/replace")
        await job.replace_task
        deleted = await http.request(
            "DELETE",
            "/api/voice/result",
            data={"job_id": job.job_id, "filename": "demo_rvc_vocal.wav"},
        )
        after_delete = (await http.get(f"/api/jobs/{job.job_id}")).json()
        wrong_restore = await http.request(
            "PUT",
            "/api/voice/result",
            data={"job_id": job.job_id, "filename": "wrong_rvc_vocal.wav"},
        )
        deleted_record = json.loads(
            (settings.output_dir / "jobs" / job.job_id / "job.json").read_text(
                encoding="utf-8"
            )
        )

    restarted = create_app(settings, orchestrator, engine)
    job = restarted.state.jobs[job.job_id]
    async with client(restarted) as http:
        restored = await http.request(
            "PUT",
            "/api/voice/result",
            data={"job_id": job.job_id, "filename": "demo_rvc_vocal.wav"},
        )
        after_restore = (await http.get(f"/api/jobs/{job.job_id}")).json()
        await http.request(
            "DELETE",
            "/api/voice/result",
            data={"job_id": job.job_id, "filename": "demo_rvc_vocal.wav"},
        )
        engine.started.clear()
        engine.release.clear()
        retried = await http.post(f"/api/jobs/{job.job_id}/replace")
        retry_task = job.replace_task
        await asyncio.wait_for(engine.started.wait(), 2)
        restored_while_running = await http.request(
            "PUT",
            "/api/voice/result",
            data={"job_id": job.job_id, "filename": "demo_rvc_vocal.wav"},
        )
        running = (await http.get(f"/api/jobs/{job.job_id}")).json()
        engine.release.set()
        await retry_task
        regenerated = (await http.get(f"/api/jobs/{job.job_id}")).json()

    stored = json.loads(
        (settings.output_dir / "jobs" / job.job_id / "job.json").read_text(encoding="utf-8")
    )
    assert deleted.status_code == 200
    assert "replacedVocal" not in after_delete["result"]
    assert wrong_restore.status_code == 404
    assert deleted_record["deletedReplacedVocals"]["0"]["url"].endswith(
        "demo_rvc_vocal.wav"
    )
    assert restored.status_code == 200
    assert "replacedVocal" in after_restore["result"]
    # 恢复后结果里有 replacedVocal，replaceStatus 必须同为成功，不能是 None：
    # 前端只认这个字段，缺了会把"撤回删除后再替换"判成失败。
    assert after_restore["replaceStatus"] == "succeeded"
    assert retried.status_code == 202
    assert restored_while_running.status_code == 200
    assert running["replaceStatus"] == "running"
    assert engine.calls == 2
    assert regenerated["replaceStatus"] == "succeeded"
    assert "replacedVocal" in stored["result"]
    assert "deletedReplacedVocals" not in stored


async def test_rvc_model_change_invalidates_cached_result(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    engine = SlowVoiceEngine()
    engine.release.set()
    app = create_app(settings, make_orchestrator(settings), engine)
    job, _ = seed_job(app, settings)
    async with client(app) as http:
        await http.post(f"/api/jobs/{job.job_id}/replace")
        await job.replace_task
        old_url = (await http.get(f"/api/jobs/{job.job_id}")).json()["result"][
            "replacedVocal"
        ]

    changed_model = tmp_path / "changed-model.pth"
    changed_model.write_bytes(b"changed")
    changed_settings = make_settings(tmp_path, rvc_model_path=changed_model)
    changed_engine = SlowVoiceEngine()
    restarted = create_app(
        changed_settings, make_orchestrator(changed_settings), changed_engine
    )
    restored_job = restarted.state.jobs[job.job_id]
    async with client(restarted) as http:
        retried = await http.post(f"/api/jobs/{job.job_id}/replace")
        await asyncio.wait_for(changed_engine.started.wait(), 2)
        # 重跑期间旧产物仍在盘上（它是失败时唯一的退路），但结果里已经不再宣称它
        available_during_replace = await http.get(old_url)
        during = (await http.get(f"/api/jobs/{job.job_id}")).json()
        changed_engine.release.set()
        await restored_job.replace_task
        after = (await http.get(f"/api/jobs/{job.job_id}")).json()
        refreshed = await http.get(after["result"]["replacedVocal"])

    assert retried.status_code == 202
    assert available_during_replace.status_code == 200
    assert "replacedVocal" not in during["result"]
    assert after["replaceStatus"] == "succeeded"
    # 同名新产物原子覆盖旧文件，客户端手上的链接依旧可用且已是新内容
    assert refreshed.content == b"RIFF-replaced-1"
    assert engine.calls == changed_engine.calls == 1


async def test_replace_timeout_keeps_capacity_until_worker_finishes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, rvc_conversion_timeout_seconds=0.01)
    orchestrator = make_orchestrator(settings)
    engine = BlockingVoiceEngine()
    app = create_app(settings, orchestrator, engine)
    job, _ = seed_job(app, settings)

    async with client(app) as http:
        assert (await http.post(f"/api/jobs/{job.job_id}/replace")).status_code == 202
        await asyncio.wait_for(engine.started.wait(), 1)
        await asyncio.sleep(0.02)
        failed = (await http.get(f"/api/jobs/{job.job_id}")).json()
        assert not job.replace_task.done()
        assert orchestrator.capacity.active == 1
        # 卡住的收尾要能被运维看见：/api/health 暴露"已进入终态但线程仍占额度"的数量
        health = (await http.get("/api/health")).json()
        assert health["replacement"]["workersHoldingCapacityAfterTerminal"] == 1
        patch = await http.patch(f"/api/jobs/{job.job_id}", json={"status": "cancelled"})
        retry = await http.post(f"/api/jobs/{job.job_id}/replace")
        engine.release.set()
        await asyncio.wait_for(job.replace_task, 1)
        final = (await http.get(f"/api/jobs/{job.job_id}")).json()
        settled = (await http.get("/api/health")).json()

    assert failed["status"] == "succeeded"
    assert failed["replaceStatus"] == "failed"
    assert failed["replaceError"] == "人声替换超时，请重试。"
    assert patch.json()["replaceStatus"] == "failed"
    assert patch.json()["replaceError"] == "人声替换超时，请重试。"
    assert retry.status_code == 409
    assert "安全收尾" in retry.json()["message"]
    assert final["replaceStatus"] == "failed"
    assert final["replaceError"] == "人声替换超时，请重试。"
    assert orchestrator.capacity.active == 0
    assert settled["replacement"]["workersHoldingCapacityAfterTerminal"] == 0
    assert job.replace_cancel_requested is False
    replaced_file = (
        settings.output_dir / "jobs" / job.job_id / "song_1" / "demo_rvc_vocal.wav"
    )
    assert not replaced_file.exists()
    assert not list(replaced_file.parent.glob(".rvc-*"))


async def test_failed_rerun_keeps_previous_replaced_vocal(tmp_path: Path) -> None:
    """缓存失效后的重跑不能把上一版可用的替换人声一起删掉。

    模型换掉后缓存立即失效（结果里不再对外宣称 replacedVocal），但新产物还没生成。如果这时
    立刻删旧文件，重跑又失败（模型不可用、OOM、超时……），用户就同时失去了新旧两版。旧文件的
    覆盖是原子的（同名），所以正确做法是留到新产物落位为止。
    """
    settings = make_settings(tmp_path)
    engine = SlowVoiceEngine()
    engine.release.set()
    app = create_app(settings, make_orchestrator(settings), engine)
    job, _ = seed_job(app, settings)
    async with client(app) as http:
        await http.post(f"/api/jobs/{job.job_id}/replace")
        await job.replace_task
    replaced_file = settings.output_dir / "jobs" / job.job_id / "song_1" / "demo_rvc_vocal.wav"
    assert replaced_file.read_bytes() == b"RIFF-replaced-1"

    changed_model = tmp_path / "changed-model.pth"
    changed_model.write_bytes(b"changed")
    changed_settings = make_settings(tmp_path, rvc_model_path=changed_model)

    # 重跑失败：结果不再宣称 replacedVocal，但上一版文件必须还在
    failing = create_app(
        changed_settings, make_orchestrator(changed_settings), FailingVoiceEngine()
    )
    failed_job = failing.state.jobs[job.job_id]
    async with client(failing) as http:
        await http.post(f"/api/jobs/{job.job_id}/replace")
        await failed_job.replace_task
        failed = (await http.get(f"/api/jobs/{job.job_id}")).json()

    assert failed["replaceStatus"] == "failed"
    assert "replacedVocal" not in failed["result"]
    assert replaced_file.read_bytes() == b"RIFF-replaced-1"

    # 重跑成功：新产物同名原子覆盖，链接恢复可用
    retry_engine = SlowVoiceEngine()
    retry_engine.release.set()
    retrying = create_app(
        changed_settings, make_orchestrator(changed_settings), retry_engine
    )
    retry_job = retrying.state.jobs[job.job_id]
    async with client(retrying) as http:
        assert (await http.post(f"/api/jobs/{job.job_id}/replace")).status_code == 202
        await retry_job.replace_task
        retried = (await http.get(f"/api/jobs/{job.job_id}")).json()
        audio = await http.get(retried["result"]["replacedVocal"])

    assert retried["replaceStatus"] == "succeeded"
    assert audio.content == b"RIFF-replaced-1"
    assert replaced_file.read_bytes() == b"RIFF-replaced-1"


async def test_replace_failure_and_cancel_preserve_generation(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    failed_app = create_app(settings, orchestrator, FailingVoiceEngine())
    failed_job, _ = seed_job(failed_app, settings, "job-failed")

    async with client(failed_app) as http:
        assert (await http.post(f"/api/jobs/{failed_job.job_id}/replace")).status_code == 202
        await failed_job.replace_task
        failed = (await http.get(f"/api/jobs/{failed_job.job_id}")).json()
        history = (await http.get("/api/jobs")).json()["jobs"]

    assert failed["status"] == "succeeded"
    assert failed["replaceStatus"] == "failed"
    assert failed["replaceError"] == "人声替换失败，请检查 RVC 配置后重试。"
    assert "secret" not in json.dumps(failed)
    assert all("replaceStatus" not in item for item in history)

    engine = BlockingVoiceEngine()
    app = create_app(settings, orchestrator, engine)
    job, _ = seed_job(app, settings, "job-cancelled")
    async with client(app) as http:
        await http.post(f"/api/jobs/{job.job_id}/replace")
        await asyncio.wait_for(engine.started.wait(), 2)
        cancelled = await http.patch(f"/api/jobs/{job.job_id}", json={"status": "cancelled"})
        assert not job.replace_task.done()
        assert orchestrator.capacity.active == 1
        engine.release.set()
        await job.replace_task
        after_cancel = (await http.get(f"/api/jobs/{job.job_id}")).json()

    assert cancelled.json()["status"] == "succeeded"
    assert cancelled.json()["replaceStatus"] == "cancelled"
    assert after_cancel["replaceStatus"] == "cancelled"
    assert "replacedVocal" not in after_cancel["result"]
    assert job.status == "succeeded"
    assert job.replace_cancel_requested is False
    assert orchestrator.capacity.active == 0


async def test_replace_cancel_wakes_sse_before_worker_exits(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    engine = BlockingVoiceEngine()
    app = create_app(settings, make_orchestrator(settings), engine)
    job, _ = seed_job(app, settings)
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/jobs/{job_id}/events"
    )

    async with client(app) as http:
        await http.post(f"/api/jobs/{job.job_id}/replace")
        await asyncio.wait_for(engine.started.wait(), 1)
        response = await endpoint(job.job_id, events_request(app, job.job_id))
        assert (await anext(response.body_iterator)).startswith("data: ")
        next_frame = asyncio.create_task(anext(response.body_iterator))
        await asyncio.sleep(0)
        await http.patch(f"/api/jobs/{job.job_id}", json={"status": "cancelled"})
        done = await asyncio.wait_for(next_frame, 1)

    assert done.startswith("event: done\ndata: ")
    assert json.loads(done.split("data: ", 1)[1])["replaceStatus"] == "cancelled"
    assert not job.replace_task.done()
    engine.release.set()
    await job.replace_task


async def test_replace_rejects_missing_inputs_and_capacity(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    orchestrator = make_orchestrator(settings)
    app = create_app(settings, orchestrator, BlockingVoiceEngine())
    job, vocal = seed_job(app, settings)

    async with client(app) as http:
        assert (await http.post("/api/jobs/missing/replace")).status_code == 404
        assert (await http.post(f"/api/jobs/{job.job_id}/replace?song=1")).status_code == 404
        job.result["stems"] = {}
        no_vocal = await http.post(f"/api/jobs/{job.job_id}/replace")
        job.result["stems"] = {"vocal": vocal.relative_to(settings.output_dir).as_posix()}
        vocal.unlink()
        missing_file = await http.post(f"/api/jobs/{job.job_id}/replace")
        vocal.write_bytes(b"0" * (settings.rvc_max_upload_bytes + 1))
        too_large = await http.post(f"/api/jobs/{job.job_id}/replace")
        vocal.write_bytes(b"ID3-vocal")
        await orchestrator.capacity.acquire()
        try:
            limited = await http.post(f"/api/jobs/{job.job_id}/replace")
        finally:
            await orchestrator.capacity.release()

    assert no_vocal.status_code == missing_file.status_code == 409
    assert too_large.status_code == 413
    assert limited.status_code == 429
    assert job.replace_status is None


async def test_done_frame_reports_terminal_state_when_replace_finishes_first(
    tmp_path: Path,
) -> None:
    """done 帧必须带终态，且与 replaceStatus 自洽。

    真实时序：/replace 受理后客户端才连接 SSE（Next 代理、EventSource 重建都会这样）。
    若替换在连接建立前就跑完，旧实现会用"连接那一刻还没收尾"的 status 判定终态，
    却用"已经收尾"的状态生成帧体，推出 status="running" + replaceStatus="succeeded"
    的矛盾 done 帧；前端的替换状态机会因此判定"未返回完成状态"并卡在加载页。
    """
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings), ReadyVoiceEngine())
    job, _ = seed_job(app, settings)

    async with client(app) as http:
        assert (await http.post(f"/api/jobs/{job.job_id}/replace")).status_code == 202
        await job.replace_task
        assert job.replace_status == "succeeded"
        # 客户端此时才开始收流：第一帧就是终态
        response = await http.get(f"/api/jobs/{job.job_id}/events")

    assert "event: done" in response.text
    payload = json.loads(response.text.rsplit("data: ", 1)[1])
    assert payload["replaceStatus"] == "succeeded"
    assert payload["status"] == "succeeded"
    assert payload["stage"] == "completed"
    assert payload["progress"] == 100
    assert payload["result"]["replacedVocal"].endswith("/demo_rvc_vocal.wav")
    # 收尾后顶层 message 回落到生成任务自己的文案；替换完成的说明只在操作进行中覆盖。
    # 客户端只以 replaceStatus 判定操作终态，因此这里固化"回到生成态"而不是替换文案。
    assert payload["message"] == "音乐生成完成"




async def test_replace_status_survives_delete_and_undo(tmp_path: Path) -> None:
    """删除替换人声再撤回后，POST /replace 必须给出自洽的成功响应。

    前端把结果里的 replacedVocal 原样回传（可能是绝对 URL 或站内代理路径），撤回后立刻
    可能再点一次替换。job.replace_status 不落盘、也不会被"恢复被删除的结果"重新赋值，
    所以这里必须以结果为准补齐 replaceStatus=succeeded；否则客户端拿到
    "有 replacedVocal 但没有 replaceStatus" 的矛盾响应，替换状态机直接判失败。
    """
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings), ReadyVoiceEngine())
    job, _ = seed_job(app, settings)

    async with client(app) as http:
        assert (await http.post(f"/api/jobs/{job.job_id}/replace")).status_code == 202
        await job.replace_task
        output_url = (await http.get(f"/api/jobs/{job.job_id}")).json()["result"]["replacedVocal"]

        # 前端删除/撤回时传的就是这个 URL 字符串（不是裸文件名）
        deleted = await http.request(
            "DELETE",
            "/api/voice/result",
            data={"filename": output_url, "job_id": job.job_id, "song": "0"},
        )
        assert deleted.status_code == 200
        after_delete = (await http.get(f"/api/jobs/{job.job_id}")).json()
        assert "replacedVocal" not in after_delete["result"]

        restored = await http.request(
            "PUT",
            "/api/voice/result",
            data={"filename": output_url, "job_id": job.job_id, "song": "0"},
        )
        assert restored.status_code == 200
        after_undo = (await http.get(f"/api/jobs/{job.job_id}")).json()

        # 撤回后重新点替换：命中缓存，且状态字段必须自洽
        retried = await http.post(f"/api/jobs/{job.job_id}/replace?song=0")
        body = retried.json()

    assert after_undo["result"]["replacedVocal"] == output_url
    assert after_undo["replaceStatus"] == "succeeded"
    assert retried.status_code == 200
    assert body["replaceStatus"] == "succeeded"
    assert body["result"]["replacedVocal"] == output_url
async def test_replace_status_is_derived_after_restart(tmp_path: Path) -> None:
    """重启后 replace_status 会丢（不落盘），但结果还在，响应必须仍然自洽。"""
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings), ReadyVoiceEngine())
    job, _ = seed_job(app, settings)
    async with client(app) as http:
        await http.post(f"/api/jobs/{job.job_id}/replace")
        await job.replace_task

    restarted = create_app(settings, make_orchestrator(settings), ReadyVoiceEngine())
    restored_job = restarted.state.jobs[job.job_id]
    assert restored_job.replace_status is None
    async with client(restarted) as http:
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()
        cached = await http.post(f"/api/jobs/{job.job_id}/replace?song=0")

    assert detail["replaceStatus"] == "succeeded"
    assert cached.status_code == 200
    assert cached.json()["replaceStatus"] == "succeeded"
    assert cached.json()["result"]["replacedVocal"].endswith("/demo_rvc_vocal.wav")
