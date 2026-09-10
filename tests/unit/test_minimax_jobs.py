import asyncio
import json

import httpx
import pytest

from app.core.errors import GenerationError
from app.services.providers import MiniMaxMusicProvider
from tests.helpers import make_settings


@pytest.mark.parametrize("outcome", ["succeeded", "failed", "cancel"])
async def test_minimax_actual_progress_terminal_and_cancellation(tmp_path, outcome):
    requests, reports = [], []
    polls = 0

    def handler(request):
        nonlocal polls
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(202, json={"jobId": "remote"})
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        if request.url.path.endswith("/audio"):
            return httpx.Response(200, content=b"RIFF-test-audio")
        polls += 1
        return httpx.Response(
            200,
            json={
                "status": "running" if polls == 1 else outcome,
                "stage": "denoising" if polls == 1 else "completed",
                "step": 3 if polls == 1 else None,
                "totalSteps": 120 if polls == 1 else None,
                "error": "GPU failed" if outcome == "failed" else None,
            },
        )

    async def progress(*values):
        reports.append(values)
        if outcome == "cancel":
            raise asyncio.CancelledError

    settings = make_settings(tmp_path, music_poll_interval_seconds=0.001)
    lyrics = "[Verse]\n" + "long supplied lyrics\n" * 300
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = MiniMaxMusicProvider(settings, http)
        call = provider.generate(
            "rock",
            180,
            f"[歌词与创作内容]\n{lyrics.rstrip()}",
            progress=progress,
            job_id="local-job",
        )
        if outcome == "succeeded":
            result = await call
            assert result.audio_path.read_bytes() == b"RIFF-test-audio"
            assert requests[-1].method == "DELETE"
        else:
            with pytest.raises(asyncio.CancelledError if outcome == "cancel" else GenerationError):
                await call
            assert requests[-1].method == "DELETE"
    assert reports[0] == ("denoising", 3, 120)
    body = json.loads(requests[0].content)
    assert body["input"] == lyrics.rstrip()
    assert len(body["jobId"]) == 32
    if outcome == "succeeded":
        assert reports[-1] == ("completed", None, None)
