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


def project(output, variant):
    return output["mixed"] if variant == "mixed" else output


async def commit(http, job, mix, song=0, variant="original"):
    """提交一轮并等它收尾：202 的回显与终态都必须带上本轮 mixConfig。

    回显在读输入的那首歌上；返回被覆盖的那首（替换编辑器写进替换后那首）。
    """
    url = mix_url(job, song) + ("&variant=mixed" if variant == "mixed" else "")
    response = await http.post(url, json=mix)
    assert response.status_code == 202, response.text
    accepted = project(pick(response.json()["result"], song), variant)
    assert accepted["mixConfig"] == mix
    # 受理时版本还没推进；audioRevision 也不能被序列化过滤掉。
    assert accepted["audioRevision"] == mix["audioRevision"]
    await job.mix_task
    detail = (await http.get(f"/api/jobs/{job.job_id}")).json()
    assert detail["mixStatus"] == "succeeded", detail
    target = "mixed" if mix["editor"] == "replace" else variant
    output = project(pick(detail["result"], song), target)
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
    # 原曲覆盖自己：不会顺带产出一首「替换后」。
    assert "mixedTrack" not in first and "mixed" not in first
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
    assert split["fullTrack"] == committed["fullTrack"] and "mixedTrack" not in split
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
    assert "mixedTrack" not in first
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


async def test_replace_commit_creates_a_separate_song(tmp_path, install_stubs):
    """替换编辑器的提交产出「替换后」那首；原曲的音轨、版本与可撤回删除都原样保留。"""
    install_stubs()
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)
    replacement = job.result["replacedVocal"]
    original_full = job.result["fullTrack"]
    mix = commit_body(
        [track("replaced", source="replaced", gain=-3), track("drums")], editor="replace"
    )

    async with client(app) as http:
        deleted = await http.delete(
            f"/api/jobs/{job.job_id}/stems/bass", headers={"X-Audio-Revision": "0"}
        )
        assert deleted.status_code == 204
        committed = await commit(http, job, mix)
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()
        listing = (await http.get("/api/jobs")).json()["jobs"]
        # 原曲没有被覆盖：版本仍是 0，提交前的删除照样可以撤回。
        restored = await http.put(
            f"/api/jobs/{job.job_id}/stems/bass", headers={"X-Audio-Revision": "0"}
        )

    assert committed["audioRevision"] == 1
    assert set(committed["stems"]) == {"replaced", "drums"}
    assert committed["splitEnabled"] is True
    assert Path(committed["fullTrack"]).name.startswith("mixed_master_v1_")
    assert all(Path(url).name.startswith("mixed_") for url in committed["stems"].values())
    original = detail["result"]
    assert original["audioRevision"] == 0
    assert original["fullTrack"].endswith(Path(original_full).name)
    assert set(original["stems"]) == {"vocal", "drums", "other"}
    # 替换人声已经进了替换后那首，原曲上不再有未提交的替换产物。
    assert "replacedVocal" not in original and "replacedVocal" not in job.result
    assert not absolute(settings, replacement).exists()
    # 兼容字段：生成记录按 mixedTrack 列出替换后那首。
    assert original["mixedTrack"] == committed["fullTrack"]
    assert listing[0]["result"]["mixed"]["fullTrack"] == committed["fullTrack"]
    assert restored.status_code == 200
    assert "bass" in job.result["stems"] and job.deleted_stems == {}


async def test_the_two_songs_are_edited_and_overwritten_separately(tmp_path, install_stubs):
    """原曲与替换后那首各自编辑、各自覆盖，版本号与撤回记录互不影响。"""
    mixer = install_stubs()
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)

    async with client(app) as http:
        mixed = await commit(
            http,
            job,
            commit_body([track("replaced", source="replaced"), track("drums")], editor="replace"),
        )
        mixed_url = f"/api/jobs/{job.job_id}/stems/drums?variant=mixed"
        assert (await http.delete(mixed_url, headers={"X-Audio-Revision": "0"})).status_code == 409
        assert (await http.delete(mixed_url, headers={"X-Audio-Revision": "1"})).status_code == 204
        assert set(job.result["mixed"]["stems"]) == {"replaced"}
        assert set(job.result["stems"]) == {"vocal", "drums", "bass", "other"}
        assert (await http.put(mixed_url, headers={"X-Audio-Revision": "1"})).status_code == 200
        # 编辑替换后那首：读它自己的音轨，只覆盖它自己。
        edited = await commit(
            http,
            job,
            commit_body([track("replaced", gain=-6), track("drums")], audio=1),
            variant="mixed",
        )
        original_before = copy.deepcopy(
            {key: value for key, value in job.result.items() if key != "mixed"}
        )
        # 编辑原曲：只覆盖原曲，替换后那首原样保留。
        original = await commit(http, job, commit_body([track("vocal"), track("bass")]))
        detail = (await http.get(f"/api/jobs/{job.job_id}")).json()["result"]

    assert [call[0] for call in mixer.calls][1] == [
        absolute(settings, mixed["stems"]["replaced"]),
        absolute(settings, mixed["stems"]["drums"]),
    ]
    assert edited["audioRevision"] == 2 and set(edited["stems"]) == {"replaced", "drums"}
    # 两首各记自己最近一次完成创作的时间，历史列表也带着它。
    assert mixed["updatedAt"] and edited["updatedAt"] and original["updatedAt"]
    assert detail["mixed"]["updatedAt"] == edited["updatedAt"]
    assert detail["updatedAt"] == original["updatedAt"]
    assert original_before.get("audioRevision", 0) == 0 and "mixed" not in original_before
    assert original["audioRevision"] == 1 and set(original["stems"]) == {"vocal", "bass"}
    assert detail["mixed"]["fullTrack"] == edited["fullTrack"]
    assert detail["mixed"]["audioRevision"] == 2
    # 原曲覆盖后的清理不能删掉替换后那首的文件。
    for url in (edited["fullTrack"], *edited["stems"].values()):
        assert absolute(settings, url).is_file(), url
    # 替换编辑器只能基于原曲提交；替换后那首不再有独立的替换人声。
    async with client(app) as http:
        refused = await http.post(
            mix_url(job) + "&variant=mixed",
            json=commit_body([track("replaced", source="replaced")], editor="replace", audio=2),
        )
    assert refused.status_code == 400


class RecordingReplaceEngine(StubReplaceEngine):
    """记下推理输入：用来证明"替换产物变成普通音轨"之后仍能继续替换。"""

    def __init__(self):
        self.inputs = []

    async def convert(self, input_path, output_path, **_params):
        self.inputs.append(Path(input_path))
        await super().convert(input_path, output_path, **_params)


async def test_replacing_again_reads_the_original_and_overwrites_the_replaced_song(
    tmp_path, install_stubs
):
    """再次替换读的是原曲人声；再次提交覆盖的是替换后那首，版本号继续递增。"""
    install_stubs()
    settings = make_settings(tmp_path)
    engine = RecordingReplaceEngine()
    app = build_app(settings, engine)
    job = seed_job(app, settings)
    mix = commit_body([track("replaced", source="replaced")], editor="replace")

    async with client(app) as http:
        first = await commit(http, job, mix)
        again = await http.post(f"/api/jobs/{job.job_id}/replace")
        assert again.status_code == 202, again.text
        await job.replace_task
        replaced = (await http.get(f"/api/jobs/{job.job_id}")).json()["result"]
        # 替换后那首在新一轮替换期间保持不变。
        assert replaced["mixed"] == first
        second = await commit(
            http,
            job,
            commit_body(
                [track("original:vocal", stem="vocal"), track("replaced", source="replaced")],
                editor="replace",
            ),
        )

    assert engine.inputs == [absolute(settings, job.result["stems"]["vocal"])]
    assert replaced["replacedVocal"]
    assert second["audioRevision"] == 2
    assert set(second["stems"]) == {"original:vocal", "replaced"}
    assert not absolute(settings, first["fullTrack"]).exists()
    assert job.result.get("audioRevision", 0) == 0


async def test_restoring_a_track_does_not_revive_a_deleted_replacement(tmp_path, install_stubs):
    """撤回替换后那首的一条音轨只恢复它自己：不会顺手把原曲上软删除的替换结果搬回来。"""
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
                "audio_revision": "0",
            },
        )
        assert deleted.status_code == 200
        assert "replacedVocal" not in job.result
        stem_url = f"/api/jobs/{job.job_id}/stems/replaced?variant=mixed"
        assert (await http.delete(stem_url, headers={"X-Audio-Revision": "1"})).status_code == 204
        restored = await http.put(stem_url, headers={"X-Audio-Revision": "1"})
        assert restored.status_code == 200
        restored_result = pick(restored.json()["result"])

    assert set(restored_result["mixed"]["stems"]) == {"replaced"}
    assert restored_result["mixed"]["stems"]["replaced"] not in (standalone, None)
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


async def test_cleanup_failure_after_a_commit_keeps_the_mix_succeeded(
    tmp_path, install_stubs, monkeypatch
):
    """清理在提交落盘之后：它出错只能记日志，合轨仍然成功、版本照常推进。"""
    install_stubs()
    settings = make_settings(tmp_path)
    app = create_app(settings, make_orchestrator(settings))
    job = seed_job(app, settings)
    real_glob = Path.glob

    def failing_glob(self, pattern, *args, **kwargs):
        # 只有提交后的清理会扫 playtrack/；OSError 以外的异常原样冒出来。
        if self.name == "playtrack":
            raise RuntimeError("symlink loop")
        return real_glob(self, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "glob", failing_glob)
    async with client(app) as http:
        output = await commit(http, job, commit_body([track("vocal"), track("drums")]))

    assert output["audioRevision"] == 1
    assert job.mix_status == "succeeded"
