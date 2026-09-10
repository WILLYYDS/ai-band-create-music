import json
from pathlib import Path

import httpx
import pytest

from app.core.errors import GenerationError
from app.services.prompt import (
    LYRICS_SYSTEM_PROMPT,
    OpenAICompatiblePromptExpander,
    build_elevenlabs_planning_prompt,
    effective_llm_output_tokens,
    enhance_elevenlabs_composition_plan,
    extract_structured_music_tags,
    extract_tagged_lyrics,
    is_expanded_music_prompt,
    looks_like_chinese_music_request,
    music_prompt_tag_count,
    normalize_llm_output,
    split_generation_prompt,
    truncate_lyrics,
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


def test_structured_tag_extraction_accepts_detailed_tag_values() -> None:
    value = "detailed audible production direction " * 7
    prompt = f"[Instrumentation: {value}], [Production and Mix: {value}]"
    assert extract_structured_music_tags(prompt) == prompt


def test_structured_tag_extraction_removes_lyrics_generation_conflicts() -> None:
    prompt = (
        "[Genre: Alternative Rock], "
        "[Negative Constraints: no EDM drops, no lyrics or melody generation]"
    )
    assert extract_structured_music_tags(prompt) == (
        "[Genre: Alternative Rock], [Negative Constraints: no EDM drops]"
    )


def test_tagged_lyrics_must_preserve_every_original_line() -> None:
    lyrics = "第一句\n第二句"
    assert extract_tagged_lyrics("[Verse]\n第一句\n[Chorus]\n第二句", lyrics)
    assert extract_tagged_lyrics("[Verse]\n改写的第一句\n第二句", lyrics) is None


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
    assert retry_body["max_tokens"] == 1024
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
    assert body["max_tokens"] == 1024
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_effective_output_token_budget_has_safe_ceiling() -> None:
    assert effective_llm_output_tokens(8192, strict=False) == 4096
    assert effective_llm_output_tokens(256, strict=True) == 2048


def test_create_prompt_splits_lyrics_from_style() -> None:
    lyrics, style = split_generation_prompt(
        "[歌词与创作内容]\n第一句\n第二句\n\n[风格要求]\n梦幻流行、空灵女声"
    )
    assert lyrics == "第一句\n第二句"
    assert style == "梦幻流行、空灵女声"


def test_create_prompt_splits_crlf_lyrics_from_style() -> None:
    lyrics, style = split_generation_prompt(
        "[歌词与创作内容]\r\n第一句\r\n第二句\r\n\r\n[风格要求]\r\n梦幻流行"
    )
    assert lyrics == "第一句\r\n第二句"
    assert style == "梦幻流行"


async def test_prompt_expander_only_sends_style_and_disables_doubao_thinking(
    tmp_path: Path,
) -> None:
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
        llm_model="doubao-seed-evolving",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await OpenAICompatiblePromptExpander(settings, client).expand(
            "[歌词与创作内容]\n不会发给 LLM 的歌词\n\n[风格要求]\n梦幻流行、空灵女声"
        )

    body = json.loads(requests[0].content)
    assert body["messages"][1]["content"] == "梦幻流行、空灵女声"
    assert body["thinking"] == {"type": "disabled"}


async def test_lyrics_writer_returns_normalized_lyrics(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        content = "```text\n[Verse]\n第一句歌词\n[Chorus]\n第二句歌词\n```"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        lyrics = await OpenAICompatiblePromptExpander(settings, client).write_lyrics(
            "[Genre: Mandarin rock]", "普通话摇滚", 2
        )

    assert lyrics.startswith("[Verse]")
    body = json.loads(requests[0].content)
    assert body["temperature"] == 0.8
    assert body["messages"][0]["content"] == LYRICS_SYSTEM_PROMPT
    assert "请用简体中文" in body["messages"][1]["content"]
    assert "目标时长：严格 120 秒（约 2 分钟）" in body["messages"][1]["content"]
    assert "歌词最多 40 行" in body["messages"][1]["content"]
    assert "结尾歌词被截断" in body["messages"][1]["content"]


async def test_prepare_tags_lyrics_and_expands_style_in_one_request(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "taggedLyrics": "[Verse]\n第一句\n[Chorus]\n第二句",
                                    "styleTags": EXPANDED_MANDARIN_ROCK_PROMPT,
                                    "durationSeconds": 150,
                                    "durationPlan": {
                                        "vocalSeconds": 130,
                                        "introAndTransitionsSeconds": 10,
                                        "soloAndInstrumentalSeconds": 5,
                                        "outroSeconds": 5,
                                    },
                                }
                            )
                        }
                    }
                ]
            },
        )

    settings = make_settings(
        tmp_path,
        llm_api_key="secret",
        llm_base_url="https://llm.test/v1",
        llm_model="doubao-seed-evolving",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        prepared = await OpenAICompatiblePromptExpander(settings, client).prepare(
            "[歌词与创作内容]\n第一句\n第二句\n\n[风格要求]\n普通话摇滚"
        )

    assert prepared.structured_prompt == EXPANDED_MANDARIN_ROCK_PROMPT
    assert prepared.lyrics == "[Verse]\n第一句\n[Chorus]\n第二句"
    assert prepared.duration_seconds == 150
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["temperature"] == 0
    assert body["max_tokens"] == 4096
    assert body["thinking"] == {"type": "disabled"}
    assert body["response_format"] == {"type": "json_object"}


async def test_prepare_enables_json_mode_for_official_openai(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "taggedLyrics": ("[Verse]\n自动生成第一句\n自动生成第二句"),
                                    "styleTags": EXPANDED_MANDARIN_ROCK_PROMPT,
                                }
                            )
                        }
                    }
                ]
            },
        )

    settings = make_settings(tmp_path, llm_api_key="secret")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await OpenAICompatiblePromptExpander(settings, client).prepare("[风格要求]\n男声摇滚", 1)

    body = json.loads(requests[0].content)
    assert body["response_format"] == {"type": "json_object"}
    assert "歌词最多 20 行" in body["messages"][1]["content"]
    assert "结尾歌词被截断" in body["messages"][1]["content"]


async def test_prepare_generates_lyrics_and_style_for_auto_duration(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "taggedLyrics": (
                                        "[Verse]\n自动生成第一句\n[Chorus]\n自动生成第二句"
                                    ),
                                    "styleTags": EXPANDED_MANDARIN_ROCK_PROMPT,
                                    "durationSeconds": 155,
                                    "durationPlan": {
                                        "vocalSeconds": 135,
                                        "introAndTransitionsSeconds": 10,
                                        "soloAndInstrumentalSeconds": 5,
                                        "outroSeconds": 5,
                                    },
                                }
                            )
                        }
                    }
                ]
            },
        )

    settings = make_settings(
        tmp_path,
        llm_api_key="secret",
        llm_base_url="https://llm.test/v1",
        llm_model="doubao-seed-evolving",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        prepared = await OpenAICompatiblePromptExpander(settings, client).prepare(
            "[风格要求]\n男声、摇滚",
            None,
            job_id="prompt-job",
        )

    assert prepared.structured_prompt == EXPANDED_MANDARIN_ROCK_PROMPT
    assert prepared.lyrics.startswith("[Verse]")
    assert prepared.duration_seconds == 155
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert "同时生成原创歌词和音乐风格说明" in body["messages"][0]["content"]
    assert "目标时长：自动" in body["messages"][1]["content"]
    assert "audio_duration 只是模型可提前结束的上限" in body["messages"][1]["content"]
    diagnostics = json.loads(
        (settings.output_dir / "jobs/prompt-job/prompts.json").read_text(encoding="utf-8")
    )
    assert diagnostics["llmAttempts"][0]["request"]["body"] == body
    assert diagnostics["llmAttempts"][0]["response"]["statusCode"] == 200


async def test_prepare_retries_invalid_auto_duration_plan(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "taggedLyrics": "[Verse]\n第一句\n第二句",
                                    "styleTags": EXPANDED_MANDARIN_ROCK_PROMPT,
                                    "durationSeconds": 180,
                                    "durationPlan": {
                                        "vocalSeconds": 160,
                                        "introAndTransitionsSeconds": 10,
                                        "soloAndInstrumentalSeconds": 5,
                                        "outroSeconds": 10 if len(requests) == 1 else 5,
                                    },
                                }
                            )
                        }
                    }
                ]
            },
        )

    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        prepared = await OpenAICompatiblePromptExpander(settings, client).prepare("摇滚")

    assert prepared.duration_seconds == 180
    assert len(requests) == 2


async def test_prepare_retries_generated_lyrics_with_custom_stage_directions(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        lyrics = (
            "[Guitar Solo]\n（25 秒吉他独奏）\n第一句"
            if len(requests) == 1
            else "[Solo]\n[Verse]\n第一句"
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "taggedLyrics": lyrics,
                                    "styleTags": EXPANDED_MANDARIN_ROCK_PROMPT,
                                    "durationSeconds": 90,
                                    "durationPlan": {
                                        "vocalSeconds": 70,
                                        "introAndTransitionsSeconds": 10,
                                        "soloAndInstrumentalSeconds": 5,
                                        "outroSeconds": 5,
                                    },
                                }
                            )
                        }
                    }
                ]
            },
        )

    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        prepared = await OpenAICompatiblePromptExpander(settings, client).prepare("男声摇滚")

    assert prepared.lyrics == "[Solo]\n[Verse]\n第一句"
    assert len(requests) == 2


async def test_prepare_extracts_json_surrounded_by_commentary(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "taggedLyrics": "[Verse]\n自动生成第一句\n自动生成第二句",
            "styleTags": EXPANDED_MANDARIN_ROCK_PROMPT,
        },
        ensure_ascii=False,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": f"Here is the JSON:\n```json\n{payload}\n```\nDone."}}
                ]
            },
        )

    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        prepared = await OpenAICompatiblePromptExpander(settings, client).prepare(
            "[风格要求]\n男声摇滚",
            1,
        )

    assert prepared.structured_prompt == EXPANDED_MANDARIN_ROCK_PROMPT
    assert prepared.lyrics == "[Verse]\n自动生成第一句\n自动生成第二句"
    assert prepared.duration_seconds == 60


async def test_prepare_retries_once_after_invalid_json(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        content = "not json"
        if len(requests) == 2:
            content = json.dumps(
                {
                    "taggedLyrics": "[Verse]\n自动生成第一句\n自动生成第二句",
                    "styleTags": EXPANDED_MANDARIN_ROCK_PROMPT,
                },
                ensure_ascii=False,
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        prepared = await OpenAICompatiblePromptExpander(settings, client).prepare(
            "[风格要求]\n男声摇滚",
            1,
        )

    assert prepared.structured_prompt == EXPANDED_MANDARIN_ROCK_PROMPT
    assert len(requests) == 2
    retry_body = json.loads(requests[1].content)
    assert retry_body["messages"][-1]["content"].startswith("上次响应未满足")


async def test_prepare_retries_concise_style_and_adds_missing_verse_tag(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "lyrics": "自动生成第一句\n自动生成第二句",
                                    "style": (
                                        "[Genre: Rock]"
                                        if len(requests) == 1
                                        else EXPANDED_MANDARIN_ROCK_PROMPT
                                    ),
                                }
                            )
                        }
                    }
                ]
            },
        )

    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        prepared = await OpenAICompatiblePromptExpander(settings, client).prepare(
            "[风格要求]\n男声摇滚",
            1,
        )

    assert prepared.structured_prompt == EXPANDED_MANDARIN_ROCK_PROMPT
    assert prepared.lyrics == "[Verse]\n自动生成第一句\n自动生成第二句"
    assert len(requests) == 2


@pytest.mark.parametrize(
    "original",
    [
        "原歌词第一句\n原歌词第二句",
        "[Intro]\r\n\r\n[Verse 1]\r\n  原歌词第一句  \r\n\r\n[Chorus]\r\n原歌词第二句",
    ],
)
async def test_prepare_preserves_original_lyrics_when_model_rewrites_them(
    tmp_path: Path, original: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "taggedLyrics": "[Verse]\n被模型改写的歌词",
                                    "styleTags": EXPANDED_MANDARIN_ROCK_PROMPT,
                                    "durationSeconds": 150,
                                    "durationPlan": {
                                        "vocalSeconds": 130,
                                        "introAndTransitionsSeconds": 10,
                                        "soloAndInstrumentalSeconds": 5,
                                        "outroSeconds": 5,
                                    },
                                }
                            )
                        }
                    }
                ]
            },
        )

    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        prepared = await OpenAICompatiblePromptExpander(settings, client).prepare(
            f"[歌词与创作内容]\n{original}\n\n[风格要求]\n摇滚"
        )

    assert prepared.lyrics == (original if original.startswith("[") else f"[Verse]\n{original}")


async def test_prepare_accepts_style_tags_as_json_array(tmp_path: Path) -> None:
    tags = EXPANDED_MANDARIN_ROCK_PROMPT.split(", [")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "taggedLyrics": "[Verse]\n第一句\n第二句",
                                    "styleTags": [
                                        tag if tag.startswith("[") else f"[{tag}" for tag in tags
                                    ],
                                    "durationSeconds": 150,
                                    "durationPlan": {
                                        "vocalSeconds": 130,
                                        "introAndTransitionsSeconds": 10,
                                        "soloAndInstrumentalSeconds": 5,
                                        "outroSeconds": 5,
                                    },
                                }
                            )
                        }
                    }
                ]
            },
        )

    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        prepared = await OpenAICompatiblePromptExpander(settings, client).prepare(
            "[歌词与创作内容]\n第一句\n第二句\n\n[风格要求]\n普通话摇滚"
        )

    assert prepared.structured_prompt == EXPANDED_MANDARIN_ROCK_PROMPT


async def test_lyrics_writer_handles_partitioned_content(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "[Verse]\nfirst line"},
                                {"type": "text", "text": "\n[Chorus]\nsecond line"},
                            ]
                        }
                    }
                ]
            },
        )

    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        lyrics = await OpenAICompatiblePromptExpander(settings, client).write_lyrics(
            "[Genre: pop]", "pop", 2
        )
    assert "[Chorus]" in lyrics


async def test_lyrics_writer_rejects_empty_output(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="歌词生成失败"):
            await OpenAICompatiblePromptExpander(settings, client).write_lyrics(
                "[Genre: pop]", "pop", 2
            )


def test_truncate_lyrics_keeps_complete_lines_within_limit() -> None:
    lyrics = "first\nsecond\nthird-long-line"
    assert truncate_lyrics(lyrics, limit=12) == "first\nsecond"


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
