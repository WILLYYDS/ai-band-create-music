import json
from pathlib import Path

import httpx
import pytest

from app.core.errors import GenerationError
from app.services.prompt import (
    OpenAICompatiblePromptExpander,
    build_elevenlabs_planning_prompt,
    effective_llm_output_tokens,
    enhance_elevenlabs_composition_plan,
    extract_structured_music_tags,
    is_expanded_music_prompt,
    looks_like_chinese_music_request,
    music_prompt_tag_count,
    normalize_llm_output,
)
from tests.helpers import make_settings

EXPANDED_MANDARIN_ROCK_PROMPT = (
    "[Genre and Era: Contemporary Mandarin pop-rock with polished modern production and "
    "a live-band foundation], "
    "[Tempo and Meter: Energetic 132 BPM in 4/4 with a steady driving eighth-note pulse "
    "and controlled syncopation], "
    "[Mood: Bright, confident, uplifting, youthful, and emotionally direct without becoming "
    "overly sweet], "
    "[Harmony: Major-key center with open power-chord verses, rising pre-chorus tension, and "
    "a broad singable chorus resolution], "
    "[Instrumentation: Layered electric rhythm guitars, selective melodic lead guitar, warm "
    "electric bass, acoustic rock drums, and subtle supporting synth pads], "
    "[Drums and Bass: Punchy kick, crisp snare, energetic tom fills, bright cymbal lifts, and "
    "a tight bass line locked to the kick], "
    "[Vocal: Clear Mandarin Chinese female lead with precise consonants, natural phrasing, a "
    "confident chest voice, and restrained harmonies only in the chorus], "
    "[Arrangement: Short guitar-and-drum intro, focused verse, rising pre-chorus, wide anthemic "
    "chorus, second verse, bridge breakdown, final double chorus, and concise outro], "
    "[Dynamics: Keep verses lean and vocal-forward, expand guitars and cymbals through each "
    "transition, then reach the strongest impact in the final chorus], "
    "[Production and Mix: Clean contemporary stereo mix with centered vocals, tight low end, "
    "wide guitars, transient-rich drums, light plate reverb, and gentle bus saturation], "
    "[Negative Constraints: No muddy low mids, no buried vocals, no excessive vocal reverb, "
    "no harsh cymbals, no metal screaming, and no dense backing-vocal clutter]"
)
QWEN_FLAT_MUSIC_PROMPT = (
    "[Contemporary Mandopop Rock, 2020s Indie Pop, 128 BPM, 4/4 Time Signature, "
    "Uplifting and Energetic Mood, Crisp Female Vocals, Mandarin Lyrics, "
    "Belting and Breathiness, Punchy Kick Drum, Driving Snare, Bright Electric Guitars, "
    "Clean Synth Pads, Layered Background Harmonies, Progressive Arrangement, "
    "Verse-Chorus Structure, Wide Stereo Mix, High-Fidelity Production, Dynamic Swells, "
    "Subtle Reverb, No Distortion, No Lo-Fi, No Auto-Tune Overuse]"
)


def test_normalize_llm_output_removes_fences_and_quotes() -> None:
    assert normalize_llm_output('```text\n"[Genre: Folk]"\n```') == "[Genre: Folk]"


def test_chinese_request_detection() -> None:
    assert looks_like_chinese_music_request("普通话女声摇滚")
    assert looks_like_chinese_music_request("Mandarin lead vocal")
    assert not looks_like_chinese_music_request("instrumental dark techno")


def test_structured_tag_extraction_rejects_thinking_process_examples() -> None:
    thinking = (
        "Thinking Process:\nUse format [Genre: Acoustic Folk], [Mood: Bright].\n"
        "Still analyzing the user request."
    )
    assert extract_structured_music_tags(thinking) is None


def test_structured_tag_extraction_prefers_final_answer() -> None:
    response = (
        "Thinking Process:\nExample [Genre: Folk].\n"
        "Final Answer:\n[Genre: Mandarin Rock], [Vocal: Clear Female Lead]"
    )
    assert extract_structured_music_tags(response) == (
        "[Genre: Mandarin Rock], [Vocal: Clear Female Lead]"
    )


def test_structured_tag_extraction_accepts_final_tag_block_without_marker() -> None:
    response = (
        "Thinking Process:\nThe request calls for Mandarin rock.\n\n"
        "[Genre: Mandarin Rock], [Mood: Bright]\n"
        "[Vocal: Clear Female Lead], [Drums: Powerful Acoustic Kit]"
    )
    assert extract_structured_music_tags(response) == (
        "[Genre: Mandarin Rock], [Mood: Bright], "
        "[Vocal: Clear Female Lead], [Drums: Powerful Acoustic Kit]"
    )


def test_structured_tag_extraction_accepts_trailing_tags_after_prose() -> None:
    response = (
        "I have analyzed the request. The production tags are "
        "[Genre: Mandarin Rock], [Mood: Bright], [Vocal: Clear Female Lead]."
    )
    assert extract_structured_music_tags(response) == (
        "[Genre: Mandarin Rock], [Mood: Bright], [Vocal: Clear Female Lead]"
    )


def test_structured_tag_extraction_accepts_json_tag_list() -> None:
    response = json.dumps(
        {"tags": ["Genre: Mandarin Rock", "Mood: Bright", "Vocal: Clear Female Lead"]}
    )
    assert extract_structured_music_tags(response) == (
        "[Genre: Mandarin Rock], [Mood: Bright], [Vocal: Clear Female Lead]"
    )


def test_structured_tag_extraction_accepts_qwen_flat_music_tags() -> None:
    assert extract_structured_music_tags(QWEN_FLAT_MUSIC_PROMPT) == QWEN_FLAT_MUSIC_PROMPT
    assert is_expanded_music_prompt(QWEN_FLAT_MUSIC_PROMPT)
    assert music_prompt_tag_count(QWEN_FLAT_MUSIC_PROMPT) == 22


def test_flat_music_tags_reject_short_unexpanded_list() -> None:
    assert extract_structured_music_tags("[Mandopop Rock, Bright, Female Vocal]") is None


def test_expanded_music_prompt_requires_detail_and_category_coverage() -> None:
    concise = (
        "[Genre: Mandarin Rock], [Mood: Bright, Energetic], [Vocal: Female, Clear], "
        "[Instrumentation: Powerful Drums], [Production: Clean]"
    )
    assert not is_expanded_music_prompt(concise)
    assert is_expanded_music_prompt(EXPANDED_MANDARIN_ROCK_PROMPT)


async def test_prompt_expander_retries_insufficient_expansion(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            content = (
                "[Genre: Mandarin Rock], [Mood: Bright, Energetic], [Vocal: Female, Clear], "
                "[Instrumentation: Powerful Drums], [Production: Clean]"
            )
        else:
            content = EXPANDED_MANDARIN_ROCK_PROMPT
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    settings = make_settings(
        tmp_path,
        llm_api_key="secret",
        llm_base_url="https://llm.test/v1",
        llm_max_tokens=256,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await OpenAICompatiblePromptExpander(settings, client).expand("普通话摇滚")

    assert result == EXPANDED_MANDARIN_ROCK_PROMPT
    assert is_expanded_music_prompt(result)
    assert len(requests) == 2
    retry_body = json.loads(requests[1].content)
    assert retry_body["temperature"] == 0
    assert retry_body["max_tokens"] == 2048
    assert "Rejected draft" in retry_body["messages"][1]["content"]
    assert "[Genre: Mandarin Rock]" in retry_body["messages"][1]["content"]


async def test_qwen_disables_thinking_and_caps_output_tokens(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": EXPANDED_MANDARIN_ROCK_PROMPT}}]},
        )

    settings = make_settings(
        tmp_path,
        llm_api_key="secret",
        llm_base_url="https://llm.test/v1",
        llm_model="Qwen/Qwen3.5-9B-FP8",
        llm_max_tokens=8192,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await OpenAICompatiblePromptExpander(settings, client).expand("普通话摇滚")

    assert result == EXPANDED_MANDARIN_ROCK_PROMPT
    body = json.loads(requests[0].content)
    assert body["max_tokens"] == 4096
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_effective_output_token_budget_has_safe_ceiling() -> None:
    assert effective_llm_output_tokens(8192, strict=False) == 4096
    assert effective_llm_output_tokens(256, strict=True) == 2048


async def test_prompt_expander_reports_actionable_read_timeout(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow model", request=request)

    settings = make_settings(
        tmp_path,
        llm_api_key="secret",
        llm_base_url="https://llm.test/v1",
        llm_timeout_seconds=45,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="读取响应超过 45 秒"):
            await OpenAICompatiblePromptExpander(settings, client).expand("普通话摇滚")


@pytest.mark.parametrize("response_json", [[], "unexpected"])
async def test_prompt_expander_rejects_non_object_json(
    tmp_path: Path, response_json: object
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_json)

    settings = make_settings(
        tmp_path,
        llm_api_key="secret",
        llm_base_url="https://llm.test/v1",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="LLM response JSON must be an object"):
            await OpenAICompatiblePromptExpander(settings, client).expand("普通话摇滚")


def test_planning_prompt_adds_clear_vocal_requirements() -> None:
    prompt = build_elevenlabs_planning_prompt("[Genre: Rock]", 3, True)
    assert "Target duration: 3 minutes" in prompt
    assert "Mandarin Chinese lead vocals" in prompt


def test_composition_plan_enhancement_is_non_mutating_and_limits_lines() -> None:
    original = {
        "positive_global_styles": ["rock"],
        "sections": [
            {
                "duration_ms": 12_000,
                "lines": ["一", "二", "三", "四"],
            }
        ],
    }
    enhanced = enhance_elevenlabs_composition_plan(original, True)
    assert original["sections"][0]["lines"] == ["一", "二", "三", "四"]
    assert enhanced["sections"][0]["lines"] == ["一", "二"]
    assert "clear vocal articulation" in enhanced["positive_global_styles"]
    assert "mumbled vocals" in enhanced["negative_global_styles"]
