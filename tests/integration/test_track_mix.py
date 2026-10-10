"""Tracks commits: the retained tracks overwrite the song, and no editor state is stored.

文档是 `SHUJI-BAND/docs/track-mix-backend.md`。这里只钉三件事：请求体里的完整最终列表、
覆盖之后的歌曲文件与音轨集合、以及失败或过期提交不能碰到上一版成品。编辑控件（M/S、
音量、撤回栈）不进后端，所以这里没有 editorState/projectRevision 可断言。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.main import GenerationJob, create_app
from app.services.audio_files import output_path_from_url
from app.services.voice import RVCConversionError
from tests.helpers import make_orchestrator, make_settings
from tests.integration.test_mix_api import (
    MIX_WAVEFORM,
    StubReplaceEngine,
    build_app,
    client,
    mix_url,
    seed_job,
    song_result,
)

# install_stubs 同时是 fixture 参数名：显式 re-export 别名让 ruff 认为这个导入是用过的。
from tests.integration.test_mix_api import install_stubs as install_stubs


def commit_body(tracks, *, editor="tracks", audio=0):
    """前端排除删除、mute、−∞ 与非 solo 轨道之后发出的完整保留列表。"""
    return {"commit": True, "audioRevision": audio, "editor": editor, "tracks": tracks}


def track(id, *, source="stem", stem=None, gain=0):
    entry = {"id": id, "source": source, "gainDb": gain}
    if source == "stem":
        entry["stemId"] = stem or id
    return entry


def song_directory(settings, job, song=0):
    output = song_result(job, song)
    index = (output.get("songNumber") or song + 1) - 1
    return settings.output_dir / "jobs" / job.job_id / f"song_{index + 1}"


def absolute(settings, url):
    """同时接受存储里的相对路径与响应里的公开 URL。"""
    return output_path_from_url(url, settings.output_dir)


def revocable_files(settings, job):
    """回收站里当前可撤回的文件名（替换结果与分轨分两处存放）。"""
    root = settings.output_dir / ".trash" / job.job_id
    return sorted(path.name for path in root.rglob("*") if path.is_file())


def detached(output):
    return {key: value for key, value in output.items() if key != "alternatives"}


def pick(result, song=0):
    return result if song == 0 else result["alternatives"][song - 1]


async def commit(http, job, mix, song=0):
    """提交一轮并等它收尾：202 的回显与终态都必须带上本轮 mixConfig。"""
    response = await http.post(mix_url(job, song), json=mix)
    assert response.status_code == 202, response.text
    accepted = pick(response.json()["result"], song)
    assert accepted["mixConfig"] == mix
    # 受理时版本还没推进；audioRevision 也不能被序列化过滤掉。
    assert accepted["audioRevision"] == mix["audioRevision"]
    await job.mix_task
    detail = (await http.get(f"/api/jobs/{job.job_id}")).json()
    assert detail["mixStatus"] == "succeeded", detail
    output = pick(detail["result"], song)
    assert output["mixConfig"] == mix
    return output


# ------------------------------------------------------------------ the overwrite itself


@pytest.mark.parametrize("song", [0, 1])
async def test_commit_overwrites_the_song_with_the_retained_tracks(tmp_path, install_stubs, song):
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings, count=2)
    selected = song_result(job, song)
    other_before = copy.deepcopy(song_result(job, 1 - song))
    old_stems = dict(selected["stems"])
    old_replaced = selected["replacedVocal"]
    old_full = selected["fullTrack"]
    # 上一轮替换的操作结果也要一起清掉，不能留在状态里当"当前产物"。
    job.replace_song = song
    job.replace_status = "succeeded"
    mix = commit_body([track("vocal", gain=-6), track("drums")])

    async with client(app) as http:
        first = await commit(http, job, mix, song)
        listing = (await http.get("/api/jobs")).json()["jobs"]
        event = (await http.get(f"/api/jobs/{job.job_id}/events")).text
        emitted = json.loads(event.split("data: ", 1)[1])

    # 成品与音轨都指向这一轮覆盖出来的新文件。
    assert first["fullTrack"] == first["mixedTrack"]
    assert "master_v1" in Path(first["fullTrack"]).name
    assert first["audioRevision"] == 1
    assert Path(first["fullTrack"]).parent.name == f"song_{song + 1}"
    assert set(first["stems"]) == {"vocal", "drums"}
    assert first["stemUrls"] == list(first["stems"].values())
    assert first["splitEnabled"] is True
    # 只有请求里的两条音轨参与混音，增益在写文件时逐轨应用。
    assert [call[0] for call in mixer.calls] == [
        [absolute(settings, old_stems["vocal"]), absolute(settings, old_stems["drums"])]
    ]
    assert mixer.gains == [pytest.approx([10 ** (-6 / 20), 1.0])]
    assert first["waveforms"]["full"] == first["waveforms"]["mix"] == MIX_WAVEFORM
    assert set(first["waveforms"]) == {"full", "mix", "vocal", "drums"}
    # 独立替换结果已经变成普通音轨：下一次编辑不会再冒出一条旧替换车道。
    assert "replacedVocal" not in first
    assert "replacedVocal" not in first["playback"]
    # 分轨预览与波形都是整组替换，不能把旧 stems 的键合并进来（stub 生成不出可用
    # 预览时 playback.stems 为空，所以这里只能断言没有旧键漏下来）。
    assert not set(first["playback"].get("stems", {})) - {"vocal", "drums"}
    for key in ("replace_song", "replace_status", "replace_stage", "replace_progress"):
        assert getattr(job, key) is None
    # 成功后才清理无引用的旧文件：旧成品、被丢弃的音轨、旧替换产物都不再留在磁盘上。
    for url in (old_full, old_replaced, old_stems["bass"], old_stems["other"]):
        assert not absolute(settings, url).exists(), url
    assert absolute(settings, first["fullTrack"]).is_file()
    assert absolute(settings, first["stems"]["vocal"]).is_file()

    # 另一首候选不受影响（第一首的 alternatives 里本来就装着第二首）。
    assert detached(song_result(job, 1 - song)) == detached(other_before)

    # 列表、SSE 与重启都不能把 audioRevision / mixConfig 过滤掉。
    row = next(item for item in listing if item["jobId"] == job.job_id)["result"]
    row_output = row if song == 0 else row["alternatives"][song - 1]
    assert row_output["audioRevision"] == 1 and row_output["mixConfig"] == mix
    emitted_output = emitted["result"] if song == 0 else emitted["result"]["alternatives"][song - 1]
    assert emitted_output["audioRevision"] == 1 and emitted_output["mixConfig"] == mix

    relaunched = create_app(settings, make_orchestrator(settings))
    async with client(relaunched) as http:
        stored = (await http.get(f"/api/jobs/{job.job_id}")).json()["result"]
    stored_output = stored if song == 0 else stored["alternatives"][song - 1]
    assert stored_output["audioRevision"] == 1 and stored_output["mixConfig"] == mix
    assert set(stored_output["stems"]) == {"vocal", "drums"}
    assert "replacedVocal" not in stored_output


async def test_next_commit_reads_the_committed_tracks_and_starts_from_zero(tmp_path, install_stubs):
    """覆盖后的文件就是下一次编辑的输入：滑杆从 0 dB 开始，不会重复衰减。"""
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)
    original_vocal = job.result["stems"]["vocal"]
    first_mix = commit_body([track("vocal", gain=-6)])

    async with client(app) as http:
        first = await commit(http, job, first_mix)
        # 迟到的旧快照既不能提交，也不能把新文件顶回上一版。
        assert (await http.post(mix_url(job), json=first_mix)).status_code == 409
        second = await commit(http, job, commit_body([track("vocal")], audio=1))
        assert (await http.post(mix_url(job), json=first_mix)).status_code == 409

    assert second["audioRevision"] == 2
    assert second["fullTrack"] != first["fullTrack"]
    assert second["stems"]["vocal"] != first["stems"]["vocal"]
    # 第二次读的是第一次覆盖出来的 vocal（已含 −6 dB），并且只应用本轮请求的 0 dB。
    assert [call[0] for call in mixer.calls] == [
        [absolute(settings, original_vocal)],
        [absolute(settings, first["stems"]["vocal"])],
    ]
    assert mixer.gains[0] == pytest.approx([10 ** (-6 / 20)])
    assert mixer.gains[1] == pytest.approx([1.0])
    assert job.result["audioRevision"] == 2


async def test_first_split_keeps_the_committed_product_and_its_credential(
    tmp_path, install_stubs, monkeypatch
):
    """首次分离用的是同一份成品：已提交成品与其凭据都还在，只推进文件版本。"""
    install_stubs()

    async def any_waveforms(inputs, **_kwargs):
        return {name: MIX_WAVEFORM.copy() for name in inputs}

    # 合轨的 stub 只认 {"mix"}，分轨会额外带上各条车道（含 "full"）。
    monkeypatch.setattr("app.main.extract_waveforms", any_waveforms)
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)
    job.result.update(stems={}, stemUrls=[], splitEnabled=False)
    mix = commit_body([track("full", source="full", gain=-6)])

    async with client(app) as http:
        committed = await commit(http, job, mix)
        assert (await http.post(f"/api/jobs/{job.job_id}/split")).status_code == 202
        await job.split_task
        split = pick((await http.get(f"/api/jobs/{job.job_id}")).json()["result"])

    assert job.split_status == "succeeded"
    assert split["stems"] and split["splitEnabled"] is True
    assert split["audioRevision"] == 2
    # 重新分轨没有改动成品本身，所以引用与提交凭据都保留。
    assert split["fullTrack"] == split["mixedTrack"] == committed["fullTrack"]
    assert split["mixConfig"] == mix
    # 旧方案的工程状态字段仍然不会回到结果里。
    assert not {"editorState", "projectRevision", "editorFullTrack"} & set(split)
    assert "editorFull" not in split["waveforms"]


async def test_single_track_commit_keeps_the_song_unsplit(tmp_path, install_stubs):
    """source=full 只用于未分轨的单轨编辑，覆盖后仍然没有分轨。"""
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)
    job.result.update(stems={}, stemUrls=[], splitEnabled=False)
    original = job.result["fullTrack"]

    async with client(app) as http:
        first = await commit(http, job, commit_body([track("full", source="full", gain=-6)]))
        second = await commit(http, job, commit_body([track("full", source="full")], audio=1))

    assert first["splitEnabled"] is False
    assert first["stems"] == {} and first["stemUrls"] == []
    assert first["fullTrack"] == first["mixedTrack"]
    assert first["audioRevision"] == 1 and second["audioRevision"] == 2
    assert second["fullTrack"] != first["fullTrack"]
    # 第一次读原始母带，第二次读第一次覆盖出来的成品，增益都按本轮请求重算。
    assert [call[0] for call in mixer.calls] == [
        [absolute(settings, original)],
        [absolute(settings, first["fullTrack"])],
    ]
    assert mixer.gains[0] == pytest.approx([10 ** (-6 / 20)])
    assert mixer.gains[1] == pytest.approx([1.0])


# --------------------------------------------------------------- deletion, undo, revisions


async def test_track_deletion_before_a_commit_is_not_undoable_afterwards(tmp_path, install_stubs):
    """已提交的手动删除不能被撤回，而且删原人声不连带删除当前的替换结果。"""
    install_stubs()
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)
    replacement = job.result["replacedVocal"]
    mix = commit_body([track("replaced", source="replaced")], editor="replace")

    async with client(app) as http:
        deleted = await http.delete(
            f"/api/jobs/{job.job_id}/stems/vocal", headers={"X-Audio-Revision": "0"}
        )
        assert deleted.status_code == 204
        # 删原人声不是删替换结果（替换页只隐藏那条车道）。
        assert job.result["replacedVocal"] == replacement
        assert absolute(settings, replacement).is_file()
        assert revocable_files(settings, job)
        committed = await commit(http, job, mix)
        # 版本不一致的 PUT 先被挡下；版本对上时删除记录已被覆盖清空。
        assert (
            await http.put(f"/api/jobs/{job.job_id}/stems/vocal", headers={"X-Audio-Revision": "0"})
        ).status_code == 409
        assert (
            await http.put(f"/api/jobs/{job.job_id}/stems/vocal", headers={"X-Audio-Revision": "1"})
        ).status_code == 404

    assert committed["audioRevision"] == 1
    assert set(committed["stems"]) == {"replaced"}
    assert committed["splitEnabled"] is True
    assert "replacedVocal" not in committed
    # 替换产物成了普通音轨，文件保留；可撤回的删除文件随覆盖一起清掉。
    assert absolute(settings, committed["stems"]["replaced"]).is_file()
    assert not revocable_files(settings, job)
    assert "vocal" not in job.result["stems"]
    assert job.deleted_stems == {}


class RecordingReplaceEngine(StubReplaceEngine):
    """记下推理输入：用来证明"替换产物变成普通音轨"之后仍能继续替换。"""

    def __init__(self):
        self.inputs = []

    async def convert(self, input_path, output_path, **_params):
        self.inputs.append(Path(input_path))
        await super().convert(input_path, output_path, **_params)


async def test_committed_replacement_lane_can_be_replaced_again(tmp_path, install_stubs):
    """`replaced` 提交后就是普通音轨，下一次替换直接拿它当输入。"""
    install_stubs()
    settings = make_settings(tmp_path)
    engine = RecordingReplaceEngine()
    app = build_app(settings, engine)
    job = seed_job(app, settings)
    mix = commit_body([track("replaced", source="replaced")], editor="replace")

    async with client(app) as http:
        committed = await commit(http, job, mix)
        again = await http.post(f"/api/jobs/{job.job_id}/replace")
        assert again.status_code == 202, again.text
        await job.replace_task
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()

    assert job.replace_status == "succeeded"
    assert engine.inputs == [absolute(settings, committed["stems"]["replaced"])]
    # 新一轮替换产生独立的 replacedVocal，等待用户在下一次提交里决定去留。
    assert detail["result"]["replacedVocal"]
    assert detail["result"]["replacedVocal"] != committed["stems"]["replaced"]
    assert set(detail["result"]["stems"]) == {"replaced"}


async def test_replace_editor_accepts_prefixed_and_replaced_lane_ids(tmp_path, install_stubs):
    """第二轮替换提交：`original:vocal` 不再叠加前缀，`replaced` 作为普通车道也被接受。"""
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)
    # 第一轮：原人声还是原始键 vocal；提交后 stems 的键自己就成了 original:vocal / replaced。
    first_mix = commit_body(
        [track("original:vocal", stem="vocal"), track("replaced", source="replaced")],
        editor="replace",
    )
    second_mix = commit_body(
        [
            track("original:vocal", stem="original:vocal"),
            track("replaced", source="stem", stem="replaced"),
        ],
        editor="replace",
        audio=1,
    )

    async with client(app) as http:
        first = await commit(http, job, first_mix)
        assert set(first["stems"]) == {"original:vocal", "replaced"}
        second = await commit(http, job, second_mix)
        listing = (await http.get("/api/jobs")).json()["jobs"]
        event = (await http.get(f"/api/jobs/{job.job_id}/events")).text
        emitted = json.loads(event.split("data: ", 1)[1])

    assert set(second["stems"]) == {"original:vocal", "replaced"}
    assert second["audioRevision"] == 2
    # 第二轮读的是第一轮覆盖出来的两个文件：读文件用 stemId，输出键用 id。
    assert [call[0] for call in mixer.calls][1] == [
        absolute(settings, first["stems"]["original:vocal"]),
        absolute(settings, first["stems"]["replaced"]),
    ]
    # 落盘的 mixConfig 必须能被渲染：详情、列表与 SSE 都会重新校验它。
    row = next(item for item in listing if item["jobId"] == job.job_id)["result"]
    assert row["mixConfig"] == second_mix
    assert emitted["result"]["mixConfig"] == second_mix
    stored = json.loads((settings.output_dir / "jobs" / job.job_id / "job.json").read_text())
    assert stored["result"]["mixConfig"] == second_mix


async def test_restoring_a_track_does_not_revive_a_deleted_replacement(tmp_path, install_stubs):
    """撤回一条普通音轨只恢复它自己：不会顺手把刚软删除的替换结果搬回来。"""
    install_stubs()
    settings = make_settings(tmp_path)
    engine = RecordingReplaceEngine()
    app = build_app(settings, engine)
    job = seed_job(app, settings)

    async with client(app) as http:
        await commit(
            http, job, commit_body([track("replaced", source="replaced")], editor="replace")
        )
        assert (await http.post(f"/api/jobs/{job.job_id}/replace")).status_code == 202
        await job.replace_task
        standalone = job.result["replacedVocal"]
        assert standalone
        deleted = await http.request(
            "DELETE",
            "/api/voice/result",
            data={
                "job_id": job.job_id,
                "filename": Path(standalone).name,
                "audio_revision": "1",
            },
        )
        assert deleted.status_code == 200
        assert "replacedVocal" not in job.result
        assert (
            await http.delete(
                f"/api/jobs/{job.job_id}/stems/replaced", headers={"X-Audio-Revision": "1"}
            )
        ).status_code == 204
        restored = await http.put(
            f"/api/jobs/{job.job_id}/stems/replaced", headers={"X-Audio-Revision": "1"}
        )
        assert restored.status_code == 200
        restored_result = pick(restored.json()["result"])

    assert set(restored_result["stems"]) == {"replaced"}
    assert restored_result["stems"]["replaced"] not in (standalone, None)
    # 替换结果仍然留在回收站里，等它自己的 PUT；存档也还在。
    assert "replacedVocal" not in restored_result
    assert "replacedVocal" not in restored_result["playback"]
    assert "replaced" not in restored_result["waveforms"]
    assert not absolute(settings, standalone).exists()
    assert str(0) in job.deleted_replaced_vocals


async def test_legacy_bodyless_mix_survives_a_null_playback(tmp_path, install_stubs):
    """`playback` 允许是 null：老客户端那条无请求体的导出路径不能因此变成失败。"""
    install_stubs()
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)
    original_full = job.result["fullTrack"]
    job.result["playback"] = None

    async with client(app) as http:
        accepted = await http.post(mix_url(job))
        assert accepted.status_code == 202, accepted.text
        await job.mix_task
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()

    assert detail["mixStatus"] == "succeeded", detail
    assert detail["result"]["mixedTrack"]
    assert pick(detail["result"])["fullTrack"].endswith(Path(original_full).name)


async def test_derived_audio_endpoint_refuses_committed_tracks(tmp_path, install_stubs):
    """提交后的保留音轨与成品只能走 /stems 的软删除：派生音频入口必须拒绝。"""
    install_stubs()
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)

    async with client(app) as http:
        committed = await commit(http, job, commit_body([track("vocal"), track("drums")]))
        for url in (committed["fullTrack"], *committed["stems"].values()):
            response = await http.request(
                "DELETE",
                "/api/voice/result",
                data={
                    "job_id": job.job_id,
                    "filename": Path(url).name,
                    "audio_revision": "1",
                },
            )
            # 被拒绝后文件必须原地不动，否则结果里会留下一条指向回收站的死链。
            assert response.status_code == 409, (url, response.text)
            assert absolute(settings, url).is_file()

    assert job.deleted_replaced_vocals == {}
    assert set(job.result["stems"]) == {"vocal", "drums"}


@pytest.mark.parametrize(
    "mutation,status",
    [
        ("empty", 400),
        ("duplicate", 400),
        ("unknown source", 400),
        ("missing stem id", 400),
        ("mismatched id", 400),
        ("gain too high", 400),
        ("gain too low", 400),
        ("gain not a number", 400),
        ("gain boolean", 400),
        ("gain null", 400),
        ("commit false", 400),
        ("full with siblings", 400),
        ("full on a split song", 400),
        ("extra field", 400),
        ("legacy editor state", 400),
        ("stale revision", 409),
        ("future revision", 409),
        ("unknown job", 404),
        ("other song", 404),
        ("unfinished", 409),
        ("stem outside the song", 400),
        ("stem in trash", 400),
        ("missing stem file", 404),
        ("stem not audio", 400),
        ("missing stem entry", 409),
        ("missing replacement", 409),
    ],
)
async def test_invalid_commits_never_start_or_modify_the_song(
    tmp_path, install_stubs, mutation, status
):
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)
    url = mix_url(job)
    mix = commit_body([track("vocal")])
    trash = settings.output_dir / ".trash" / job.job_id / "song_1"
    trash.mkdir(parents=True, exist_ok=True)
    (trash / "drums.wav").write_bytes(b"trashed")
    (song_directory(settings, job) / "notes.txt").write_text("not audio")
    cases = {
        "empty": lambda: mix.update(tracks=[]),
        "duplicate": lambda: mix["tracks"].append(dict(mix["tracks"][0])),
        "unknown source": lambda: mix["tracks"][0].update(source="url"),
        "missing stem id": lambda: mix["tracks"][0].pop("stemId"),
        "mismatched id": lambda: mix["tracks"][0].update(id="original:vocal"),
        "gain too high": lambda: mix["tracks"][0].update(gainDb=7),
        "gain too low": lambda: mix["tracks"][0].update(gainDb=-66.1),
        "gain not a number": lambda: mix["tracks"][0].update(gainDb="NaN"),
        "gain boolean": lambda: mix["tracks"][0].update(gainDb=True),
        "gain null": lambda: mix["tracks"][0].update(gainDb=None),
        "commit false": lambda: mix.update(commit=False),
        "full with siblings": lambda: mix["tracks"].append(track("full", source="full")),
        "full on a split song": lambda: mix.update(tracks=[track("full", source="full")]),
        "extra field": lambda: mix["tracks"][0].update(muted=False),
        # 旧方案的控制状态既不是本轮契约的一部分，也不能被静默接受。
        "legacy editor state": lambda: mix.update(
            editorState={"mutedTrackIds": [], "trackDb": {}}, projectRevision=0
        ),
        "stale revision": lambda: mix.update(audioRevision=1),
        "future revision": lambda: mix.update(audioRevision=2),
        "unknown job": lambda: None,
        "other song": lambda: None,
        "unfinished": lambda: setattr(job, "status", "running"),
        "stem outside the song": lambda: job.result["stems"].update(
            vocal=f"jobs/{job.job_id}/song_1/../../../etc/passwd.wav"
        ),
        "stem in trash": lambda: job.result["stems"].update(
            vocal=(trash / "drums.wav").relative_to(settings.output_dir).as_posix()
        ),
        "missing stem file": lambda: absolute(settings, job.result["stems"]["vocal"]).unlink(),
        "stem not audio": lambda: job.result["stems"].update(
            vocal=(song_directory(settings, job) / "notes.txt")
            .relative_to(settings.output_dir)
            .as_posix()
        ),
        "missing stem entry": lambda: job.result["stems"].pop("vocal"),
        "missing replacement": lambda: (
            job.result.pop("replacedVocal"),
            mix.update(tracks=[track("replaced", source="replaced")]),
        ),
    }
    cases[mutation]()
    if mutation == "unknown job":
        url = "/api/jobs/nope/mix"
    elif mutation == "other song":
        url = mix_url(job, 2)
    before = copy.deepcopy(job.result)
    async with client(app) as http:
        response = await http.post(url, json=mix)
    assert response.status_code == status, (mutation, response.text)
    body = response.json()
    assert body["success"] is False and body["message"]
    assert "private diagnostic" not in response.text
    assert not mixer.calls
    assert job.result == before
    assert job.mix_status is None
    assert app.state.orchestrator.capacity.active == 0


@pytest.mark.parametrize("busy", ["mix", "split", "replace"])
async def test_commit_is_refused_while_another_audio_operation_runs(tmp_path, install_stubs, busy):
    """并发音频修改共用同一批输入文件，正在跑的那一轮不能被合轨抢走。"""
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)
    if busy == "mix":
        job.mix_status = "running"
    elif busy == "split":
        job.split_status = "running"
    else:
        job.replace_status = "running"
        job.replace_task = _pending_task()
    try:
        async with client(app) as http:
            response = await http.post(mix_url(job), json=commit_body([track("vocal")]))
    finally:
        if job.replace_task is not None:
            job.replace_task.cancel()
    assert response.status_code == 409
    assert not mixer.calls
    assert "mixConfig" not in job.result


def _pending_task():
    import asyncio

    return asyncio.get_running_loop().create_future()


# ------------------------------------------------------------- failures keep the product


@pytest.mark.parametrize("song", [0, 1])
@pytest.mark.parametrize("failure", ["mix", "publish", "save", "preview", "cancel"])
async def test_failed_commit_keeps_the_song_revision_and_the_revocable_files(
    tmp_path, install_stubs, monkeypatch, failure, song
):
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings, count=2)
    # 保留两条音轨，后面才能删掉其中一条再撤回。
    first_mix = commit_body([track("vocal", gain=-6), track("other")])
    stem_url = f"/api/jobs/{job.job_id}/stems/other?song={song}"

    async with client(app) as http:
        first = await commit(http, job, first_mix, song)
        # 一次成功提交之后仍然可以手动软删除，并在页面内撤回。
        assert (await http.delete(stem_url, headers={"X-Audio-Revision": "1"})).status_code == 204
        revocable = revocable_files(settings, job)
        assert revocable
        before = copy.deepcopy(song_result(job, song))
        failed_mix = commit_body([track("vocal", gain=-12)], audio=1)
        if failure == "mix":
            mixer.error = RVCConversionError("private diagnostic")
        elif failure == "publish":
            replace = Path.replace

            def fail_publish(source, target):
                if Path(target).name.startswith("master_v2"):
                    raise OSError("private diagnostic")
                return replace(source, target)

            monkeypatch.setattr(Path, "replace", fail_publish)
        elif failure == "save":
            save = GenerationJob.save

            def fail_save(current_job, directory):
                if pick(current_job.result, song).get("audioRevision") == 2:
                    raise OSError("private diagnostic")
                return save(current_job, directory)

            monkeypatch.setattr(GenerationJob, "save", fail_save)
        else:

            async def fail_preview(*_args):
                if failure == "cancel":
                    import asyncio

                    raise asyncio.CancelledError
                raise RuntimeError("private diagnostic")

            monkeypatch.setattr("app.main.make_playback_mp3", fail_preview)
        accepted = await http.post(mix_url(job, song), json=failed_mix)
        assert accepted.status_code == 202
        await job.mix_task
        # 失败必须保留可撤回文件：这一次删除既没被清理，也还能撤回。
        revocable_after_failure = revocable_files(settings, job)
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()
        audio = await http.get(first["fullTrack"])
        undone = await http.put(stem_url, headers={"X-Audio-Revision": "1"})

    selected = song_result(job, song)
    assert detail["mixStatus"] == ("cancelled" if failure == "cancel" else "failed")
    assert pick(detail["result"], song)["fullTrack"] == first["fullTrack"]
    assert audio.status_code == 200 and audio.content == b"RIFF-mixed-stub"
    assert selected["audioRevision"] == 1
    assert revocable_after_failure == revocable
    assert undone.status_code == 200
    # 删除/恢复本身不推进版本，返回的任务状态必须带着当前版本与提交凭据。
    undo_result = pick(undone.json()["result"], song)
    assert undo_result["audioRevision"] == 1
    assert undo_result["mixConfig"] == failed_mix
    assert "editorState" not in undo_result
    assert set(selected["stems"]) == set(before["stems"]) | {"other"}
    # 本轮请求作为失败凭据保留：前端据此退回原编辑页。
    assert selected["mixConfig"] == failed_mix
    assert selected["mixConfig"]["editor"] == "tracks"
    assert not list(song_directory(settings, job, song).glob("master_v2*"))
    assert app.state.orchestrator.capacity.active == 0


async def test_capacity_rejection_leaves_the_song_unchanged(tmp_path, install_stubs):
    install_stubs()
    settings = make_settings(tmp_path, max_concurrent_generations=1)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)
    before = copy.deepcopy(job.result)
    await app.state.orchestrator.capacity.acquire()
    async with client(app) as http:
        response = await http.post(mix_url(job), json=commit_body([track("vocal")]))
    assert response.status_code == 429
    assert job.result == before and job.mix_status is None
    await app.state.orchestrator.capacity.release()


async def test_first_split_failure_keeps_the_previous_song_and_undo(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)
    job.result.update(stems={}, stemUrls=[], splitEnabled=False)
    job.result["waveforms"] = {"full": [0.25] * 640}
    job.deleted_stems["0:drums"] = {"url": "old.wav", "index": 0, "waveform": None}
    job.deleted_replaced_vocals["0"] = {"url": "old-rvc.wav"}
    job.save(settings.output_dir)
    before = copy.deepcopy(job.result)
    old_stems = copy.deepcopy(job.deleted_stems)
    old_vocals = copy.deepcopy(job.deleted_replaced_vocals)

    def fail_save(_job, _output_dir):
        raise OSError("private diagnostic")

    monkeypatch.setattr(GenerationJob, "save", fail_save)
    async with client(app) as http:
        response = await http.post(f"/api/jobs/{job.job_id}/split")
        assert response.status_code == 202
        await job.split_task
        assert job.split_status == "failed"
    assert job.result == before
    assert job.deleted_stems == old_stems
    assert job.deleted_replaced_vocals == old_vocals
    stored = json.loads((settings.output_dir / "jobs" / job.job_id / "job.json").read_text())
    assert stored["result"] == before
    assert app.state.orchestrator.capacity.active == 0
