"""Run the sibling frontend's actual Next routes against Uvicorn and real FFmpeg.

这一层验证的是「前端构造的请求 → Next 代理原样透传 → 后端的覆盖语义」这条链路：
stubs 测不到的代理改写、真实 FFmpeg 的逐轨增益与新音轨文件、以及前端揭晓判据
（completionMixStateFor）是否认这一轮结果。前端源码只读，编译在临时副本里做。
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from app.main import create_app
from tests.functional.test_http_server import _live_server, _tone
from tests.helpers import make_orchestrator, make_settings
from tests.integration.test_mix_api import seed_job


def test_frontend_track_mix_over_real_http(tmp_path):
    frontend = Path(
        os.environ.get(
            "SHUJI_FRONTEND_DIR",
            str(Path(__file__).resolve().parents[3] / "SHUJI-BAND"),
        )
    )
    next_cli = frontend / "node_modules/next/dist/bin/next"
    node = shutil.which("node")
    if not next_cli.is_file() or not node:
        pytest.skip("sibling SHUJI-BAND frontend and Node are required")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("real FFmpeg tools are required")

    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    project = seed_job(app, settings, job_id="project-live", count=2)
    single = seed_job(app, settings, job_id="single-live")
    single.result.update(stems={}, stemUrls=[], splitEnabled=False)
    for job in (project, single):
        for output in (job.result, *job.result["alternatives"]):
            paths = [output["fullTrack"], output["replacedVocal"], *output["stems"].values()]
            for index, path in enumerate(paths):
                _tone(settings.output_dir / path, 110 + index * 110, seconds=2)
        job.save(settings.output_dir)
    backend_url, stop, started = _live_server(app)
    assert started

    # Compile in a temporary copy so this check cannot overwrite frontend build artifacts.
    checkout = tmp_path / "frontend"
    checkout.mkdir()
    shutil.copytree(frontend / "src", checkout / "src")
    for name in ("package.json", "tsconfig.json", "next-env.d.ts"):
        shutil.copy2(frontend / name, checkout / name)
    (checkout / "node_modules").symlink_to(frontend / "node_modules", target_is_directory=True)
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    frontend_url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "next.log"
    process = None
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [
                    node,
                    str(next_cli),
                    "dev",
                    "--webpack",
                    "--hostname",
                    "127.0.0.1",
                    "-p",
                    str(port),
                ],
                cwd=checkout,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ,
                    "MUSIC_GENERATION_URL": backend_url,
                    "NEXT_TELEMETRY_DISABLED": "1",
                },
            )
            deadline = time.monotonic() + 45
            with httpx.Client(base_url=frontend_url, timeout=20, trust_env=False) as http:
                while time.monotonic() < deadline:
                    assert process.poll() is None, log_path.read_text()
                    try:
                        if http.get("/api/music/jobs/project-live").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.1)
                else:
                    pytest.fail(f"Next did not start: {log_path.read_text()}")

            script = r"""
import assert from "node:assert/strict";
import { trackMixRequest, trackMixMatches } from "./src/lib/track-editor.ts";
import { completionMixStateFor } from "./src/lib/completion-mix.ts";

const base = process.env.FRONTEND_TEST_URL;
async function request(path, init) {
  const response = await fetch(base + path, { ...init, signal: AbortSignal.timeout(60000) });
  const text = await response.text();
  let body;
  try { body = JSON.parse(text); } catch { body = text; }
  return { response, body };
}
const selected = (song) => (result) => song ? result.alternatives[song - 1] : result;
const controls = (muted = [], solo = [], trackDb = {}) => ({
  muted: new Set(muted), solo: new Set(solo), trackDb,
});

const project = (variant) => (output) => variant === "mixed" ? output.mixed : output;

/* 「完成创作」：代理透传请求体，后端回显同一份配置，终态必须过前端的揭晓判据。
   替换前后是两首歌：回显在读输入的那首上，判据与返回值是被覆盖的那首（替换编辑器写进 mixed）。 */
async function mix(job, song, config, variant = "original") {
  const path = `/api/music/jobs/${job}`;
  const query = variant === "mixed" ? "&variant=mixed" : "";
  const accepted = await request(`${path}/mix?song=${song}${query}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(config),
  });
  assert.equal(accepted.response.status, 202, JSON.stringify(accepted.body));
  const echoed = project(variant)(selected(song)(accepted.body.result));
  assert.ok(trackMixMatches(echoed.mixConfig, config), "202 must echo the exact request");
  const target = config.editor === "replace" ? "mixed" : variant;
  const sse = await request(`${path}/events`);
  assert.equal(sse.response.status, 200);
  assert.match(sse.body, /event: done/);
  const detail = (await request(path)).body;
  assert.equal(detail.mixStatus, "succeeded", JSON.stringify(detail));
  const outputs = [detail.result, ...detail.result.alternatives];
  const output = project(target)(selected(song)(detail.result));
  assert.equal(completionMixStateFor(detail, outputs, song, target), "done",
    JSON.stringify(output));
  assert.ok(trackMixMatches(output.mixConfig, config));
  assert.equal(output.audioRevision, config.audioRevision + 1);
  assert.match(output.fullTrack, /^\/api\/music\/output\//);
  const wav = await fetch(base + output.fullTrack);
  assert.equal(wav.status, 200);
  assert.equal(Buffer.from(await wav.arrayBuffer()).subarray(0, 4).toString(), "RIFF");
  const preview = await fetch(base + output.playback.fullTrack, {
    headers: { Range: "bytes=0-2" },
  });
  assert.equal(preview.status, 206);
  assert.equal((await preview.arrayBuffer()).byteLength, 3);
  return output;
}

/* ------------------------------------------- 替换人声：产出独立的「替换后」那首歌 */
const replaceLanes = [
  { id: "replaced", source: "replaced" }, { id: "drums", source: "stem", stemId: "drums" },
];
const replacedSong = await mix("project-live", 0,
  trackMixRequest("replace", replaceLanes, controls([], [], { replaced: -3 }), 0));
assert.deepEqual(Object.keys(replacedSong.stems).sort(), ["drums", "replaced"]);
let originalSong = (await request("/api/music/jobs/project-live")).body.result;
// 原曲原样保留：版本、母带、音轨都没动，替换人声已进了替换后那首。
assert.equal(originalSong.audioRevision, 0);
assert.deepEqual(Object.keys(originalSong.stems).sort(), ["bass", "drums", "other", "vocal"]);
assert.equal(originalSong.replacedVocal ?? null, null);
assert.equal(originalSong.mixedTrack, replacedSong.fullTrack);
// 替换后那首的分轨删除/撤回走同一个代理，按它自己的版本校验。
const mixedStem = (method, revision) => request("/api/music/jobs/project-live/stems/drums", {
  method,
  headers: { "X-Song-Index": "0", "X-Song-Variant": "mixed", "X-Audio-Revision": String(revision) },
});
assert.equal((await mixedStem("DELETE", 0)).response.status, 409);
assert.equal((await mixedStem("DELETE", 1)).response.status, 204);
assert.equal((await mixedStem("PUT", 1)).response.status, 200);
// 单独编辑替换后那首：只覆盖它自己。
const editedReplaced = await mix("project-live", 0, trackMixRequest("tracks",
  [{ id: "replaced", source: "stem", stemId: "replaced" }], controls(), 1), "mixed");
assert.equal(editedReplaced.audioRevision, 2);
assert.deepEqual(Object.keys(editedReplaced.stems), ["replaced"]);
originalSong = (await request("/api/music/jobs/project-live")).body.result;
assert.equal(originalSong.audioRevision, 0);
assert.equal(originalSong.mixed.fullTrack, editedReplaced.fullTrack);

/* ---------------------------------------------------------- 分轨歌曲：solo 与音量 */
const lanes = ["vocal", "drums", "bass", "other"].map((id) => ({ id, source: "stem", stemId: id }));
const soloControls = controls(["drums"], ["vocal"], { vocal: -6, bass: -Infinity });
const soloRequest = trackMixRequest("tracks", lanes, soloControls);
// solo 单选 + mute 与 −∞ 都不进请求，只交付被保留的那一条。
assert.deepEqual(soloRequest.tracks,
  [{ id: "vocal", source: "stem", stemId: "vocal", gainDb: -6 }]);
const first = await mix("project-live", 0, soloRequest);
assert.deepEqual(Object.keys(first.stems), ["vocal"]);
assert.equal(first.splitEnabled, true);
assert.equal(first.replacedVocal ?? null, null);
assert.deepEqual(Object.keys(first.playback.stems), ["vocal"]);
// 分轨波形整组替换：请到的车道都有，被移除的旧车道一条都不剩。
assert.ok(Array.isArray(first.waveforms.full) && first.waveforms.full.length > 1);
assert.ok(Object.keys(first.stems).every((id) => Array.isArray(first.waveforms[id])));
assert.deepEqual(Object.keys(first.waveforms).filter((id) => ["bass", "other"].includes(id)), []);

// 再次编辑：保留音轨就是新的输入，控件从 0 dB 重新开始，版本必须推进。
const retained = [{ id: "vocal", source: "stem", stemId: "vocal" }];
const second = await mix("project-live", 0, trackMixRequest("tracks", retained, controls(), 1));
assert.equal(second.audioRevision, 2);
assert.notEqual(second.fullTrack, first.fullTrack);
assert.notEqual(second.stems.vocal, first.stems.vocal);

// 迟到的旧快照不能回退到上一版文件。
const stale = await request("/api/music/jobs/project-live/mix?song=0", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ ...second.mixConfig, audioRevision: 1 }),
});
assert.equal(stale.response.status, 409, JSON.stringify(stale.body));

// 全部静音/删除后没有有效输入：前端会构造出空列表，后端必须拒绝而不是生成空白成品。
const silent = trackMixRequest("tracks", retained, controls(["vocal"], [], {}), 2);
assert.deepEqual(silent.tracks, []);
const refused = await request("/api/music/jobs/project-live/mix?song=0", {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(silent),
});
assert.equal(refused.response.status, 400, JSON.stringify(refused.body));
assert.equal((await request("/api/music/jobs/project-live")).body.result.audioRevision, 2);

/* ------------------------------------------------- 第二首：派生音频与音轨的版本检查 */
const second_song = selected(1)((await request("/api/music/jobs/project-live")).body.result);
const vocalFile = second_song.replacedVocal;
assert.ok(vocalFile);
const voiceCall = (method, audioRevision) => request("/api/voice/result", {
  method, headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ filename: vocalFile, jobId: "project-live", songIndex: 1, audioRevision }),
});
assert.equal((await voiceCall("DELETE", 5)).response.status, 409);
assert.equal((await voiceCall("DELETE", -1)).response.status, 400);
assert.equal((await voiceCall("DELETE", 0)).response.status, 200);
assert.equal((await voiceCall("PUT", 0)).response.status, 200);

const stemCall = (method, stem, audioRevision) => request(
  `/api/music/jobs/project-live/stems/${stem}`,
  { method, headers: { "X-Song-Index": "1", "X-Audio-Revision": String(audioRevision) } },
);
assert.equal((await stemCall("DELETE", "drums", 1)).response.status, 409);
assert.equal((await stemCall("DELETE", "drums", 0)).response.status, 204);
assert.equal((await stemCall("PUT", "drums", 1)).response.status, 409);
assert.equal((await stemCall("PUT", "drums", 0)).response.status, 200);

// 两条车道一起交付：新音轨集合与预览都整组替换，独立替换结果被清理。
const twoLanes = ["vocal", "drums"].map((id) => ({ id, source: "stem", stemId: id }));
const third = await mix("project-live", 1,
  trackMixRequest("tracks", twoLanes, controls([], [], { vocal: -6 }), 0));
assert.deepEqual(Object.keys(third.stems).sort(), ["drums", "vocal"]);
assert.deepEqual(Object.keys(third.playback.stems).sort(), ["drums", "vocal"]);
assert.equal(third.replacedVocal ?? null, null);
// 覆盖成功后旧版本失效：删除必须带当前版本。
assert.equal((await stemCall("DELETE", "drums", 0)).response.status, 409);
assert.equal((await stemCall("DELETE", "drums", 1)).response.status, 204);

/* ------------------------------------------------------------- 单轨歌曲：完整混音 */
const fullLane = [{ id: "full", source: "full" }];
const fullFirst = await mix("single-live", 0,
  trackMixRequest("tracks", fullLane, controls([], [], { full: -6 }), 0));
assert.equal(fullFirst.splitEnabled, false);
assert.deepEqual(fullFirst.stems, {});
assert.deepEqual(fullFirst.stemUrls, []);
assert.equal(fullFirst.mixedTrack ?? null, null);
const fullSecond = await mix("single-live", 0, trackMixRequest("tracks", fullLane, controls(), 1));
assert.equal(fullSecond.audioRevision, 2);
assert.notEqual(fullSecond.fullTrack, fullFirst.fullTrack);

/* ------------------------------------- 生成记录：一个任务一条，替换前后两首分别可见 */
const history = (await request("/api/music/jobs")).body.jobs;
const rows = history.find(({ jobId }) => jobId === "project-live");
assert.ok(rows);
assert.equal(history.length, 2);
assert.equal(rows.result.count, 2);
// 记录行指向覆盖后的成品，不是分轨前的母带，也不是第二条记录。
assert.equal(rows.result.fullTrack, second.fullTrack);
assert.equal(rows.result.audioRevision, 2);
assert.equal(rows.result.alternatives[0].fullTrack, third.fullTrack);
// 原曲被覆盖了两次，替换后那首仍是它自己最后一次覆盖的版本。
assert.equal(rows.result.mixed.fullTrack, editedReplaced.fullTrack);
assert.equal(rows.result.mixed.audioRevision, 2);
assert.equal(rows.result.mixedTrack, editedReplaced.fullTrack);
console.log("Next → FastAPI → FFmpeg: proxy passthrough, solo, gains, in-place overwrite,"
  + " separate replaced song, revisions, undo failure, single-track and history passed");
"""
            result = subprocess.run(
                [node, "--experimental-strip-types", "--input-type=module", "-e", script],
                cwd=checkout,
                capture_output=True,
                text=True,
                timeout=180,
                env={**os.environ, "FRONTEND_TEST_URL": frontend_url},
            )
            assert result.returncode == 0, result.stdout + result.stderr + log_path.read_text()
            assert "passed" in result.stdout
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        stop()
