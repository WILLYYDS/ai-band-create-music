import json
from pathlib import Path

import httpx
import pytest

from app.core.errors import GenerationError
from app.services.prompt import (
    STRUCTURED_TAG_PATTERN,
    STYLE_TAG_MAX,
    OpenAICompatiblePromptExpander,
    extract_structured_music_tags,
    extract_tagged_lyrics,
    is_expanded_music_prompt,
    normalize_llm_output,
    normalize_lyrics_section_tags,
    split_generation_prompt,
    validate_tagged_lyrics,
)
from app.services.providers import build_elevenlabs_prompt
from tests.helpers import make_settings

EXPANDED_MANDARIN_ROCK_PROMPT = (
    "[Genre and Era: Contemporary Mandarin pop-rock with polished modern production and "
    "a live-band foundation], "
    "[Tempo and Meter: Energetic 132 BPM in 4/4 with a steady driving eighth-note pulse "
    "and controlled syncopation], "
    "[Mood: Bright, confident, uplifting, youthful, and emotionally direct], "
    "[Instrumentation: Layered electric rhythm guitars, selective melodic lead guitar, warm "
    "electric bass, acoustic rock drums, and subtle supporting synth pads], "
    "[Vocal: Clear Mandarin Chinese female lead with precise consonants, natural phrasing, a "
    "confident chest voice, and restrained harmonies only in the chorus], "
    "[Arrangement: Short guitar-and-drum intro, focused verse, rising pre-chorus, wide anthemic "
    "chorus, second verse, bridge breakdown, final double chorus, and concise outro], "
    "[Production and Mix: Clean contemporary stereo mix with centered vocals, tight low end, "
    "wide guitars, transient-rich drums, light plate reverb, and gentle bus saturation], "
    "[Negative Constraints: No muddy low mids, no buried vocals, no excessive vocal reverb, "
    "no harsh cymbals, no metal screaming, and no dense backing-vocal clutter]"
)


def generated_lyrics(line_count: int) -> str:
    return "[Verse]\n" + "\n".join(f"第 {index} 句歌词" for index in range(line_count))


ONE_MINUTE_LYRICS = generated_lyrics(10)


def test_normalize_llm_output_removes_fences_and_quotes() -> None:
    assert normalize_llm_output('```text\n"[Genre: Folk]"\n```') == "[Genre: Folk]"


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
    assert extract_structured_music_tags(prompt) == (
        f"[Instrumentation: {value.strip()}], [Production and Mix: {value.strip()}]"
    )


def test_structured_tag_extraction_removes_lyrics_generation_conflicts() -> None:
    prompt = (
        "[Genre: Alternative Rock], "
        "[Negative Constraints: no EDM drops, no lyrics or melody generation]"
    )
    assert extract_structured_music_tags(prompt) == (
        "[Genre: Alternative Rock], [Negative Constraints: no EDM drops]"
    )
    assert extract_structured_music_tags("[Negative Constraints: no vocals]") is None


def test_tagged_lyrics_must_preserve_every_original_line() -> None:
    lyrics = "第一句\n第二句"
    assert extract_tagged_lyrics("[Verse]\n第一句\n[Chorus]\n第二句", lyrics)
    assert extract_tagged_lyrics("[Verse]\n改写的第一句\n第二句", lyrics) is None


def test_lyrics_section_tags_are_normalized_without_changing_unknown_labels() -> None:
    lyrics = (
        "[verse1]\n第一句\nintro\n第二句\n[pre chorus]\n第三句\n"
        "[间奏]\nSolo\nBridge\nOutro\n间奏\n[吉他独奏]\n[Verse 1: 主唱]\n[Guitar Solo]"
    )
    assert normalize_lyrics_section_tags(lyrics) == (
        "[Verse 1]\n第一句\nintro\n第二句\n[Pre-Chorus]\n第三句\n"
        "[Instrumental]\nSolo\nBridge\nOutro\n间奏\n[吉他独奏]\n"
        "[Verse 1: 主唱]\n[Guitar Solo]"
    )


def test_tagged_lyrics_validation_preserves_supported_labels_and_rejects_unknown_ones() -> None:
    lyrics = "[Verse 1]\n第一句\n[Instrumental Break]\n[Chorus]\n第二句"
    assert validate_tagged_lyrics(lyrics) == lyrics
    with pytest.raises(ValueError, match="不支持"):
        validate_tagged_lyrics("[Verse]\n第一句\n[Final Chorus]\n第二句")
    user_lyrics = "[Verse]\n第一句\n[Guitar Solo]\n（副歌重复两次）\n第二句"
    assert validate_tagged_lyrics(user_lyrics, strict=False) == user_lyrics
    with pytest.raises(ValueError, match="没有可演唱内容"):
        validate_tagged_lyrics("[Verse]\n[Guitar Solo]\n（副歌重复两次）", strict=False)


def test_flat_music_tags_are_rejected() -> None:
    assert extract_structured_music_tags("[Mandopop Rock, Bright, Female Vocal]") is None


def test_expanded_music_prompt_requires_detail_and_category_coverage() -> None:
    concise = (
        "[Genre: Mandarin Rock], [Mood: Bright, Energetic], [Vocal: Female, Clear], "
        "[Instrumentation: Powerful Drums], [Production: Clean]"
    )
    assert not is_expanded_music_prompt(concise)
    assert is_expanded_music_prompt(EXPANDED_MANDARIN_ROCK_PROMPT)
    six_tags = (
        "[Genre and Era: Modern Mandarin rock], [Tempo Meter and Mood: 128 BPM 4/4 uplifting], "
        "[Instrumentation: Electric guitars bass and acoustic drums], "
        "[Vocal: Clear Mandarin lead], [Arrangement: Verse chorus bridge outro], "
        "[Production Mix and Negative Constraints: Wide clean mix; avoid muddy low mids]"
    )
    assert is_expanded_music_prompt(six_tags)
    extra_tags = (
        f"{EXPANDED_MANDARIN_ROCK_PROMPT}, [Energy: Explosive], [Regional Texture: Guzheng accents]"
    )
    normalized = extract_structured_music_tags(extra_tags)
    assert normalized is not None
    assert is_expanded_music_prompt(normalized)
    assert normalized.count("[") == STYLE_TAG_MAX
    assert all(tag[1:-1] in normalized for tag in STRUCTURED_TAG_PATTERN.findall(extra_tags))

    oversized_value = f"[Genre: {'x' * 801}]"
    assert extract_structured_music_tags(oversized_value) is None
    oversized_overflow = ", ".join(
        [EXPANDED_MANDARIN_ROCK_PROMPT, *(f"[Extra {index}: {'x' * 110}]" for index in range(8))]
    )
    assert extract_structured_music_tags(oversized_overflow) is None


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
    assert prepared.duration_seconds is None
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["temperature"] == 0
    assert body["max_tokens"] == 2048
    assert body["thinking"] == {"type": "disabled"}
    assert body["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("lyrics", "tagged", "expected"),
    [
        (
            "[Verse]\n第一句\n[Guitar Solo]\n（副歌重复两次）\n第二句",
            "[Verse]\n模型返回内容会被忽略",
            "[Verse]\n第一句\n[Guitar Solo]\n（副歌重复两次）\n第二句",
        ),
        (
            "我走过长街\n[间奏]\n灯火通明",
            "[Verse]\n我走过长街\n[间奏]\n灯火通明",
            "我走过长街\n[Instrumental]\n灯火通明",
        ),
        (
            "[verse1]\n第一句\nintro\n第二句",
            "[Verse]\n模型返回内容会被忽略",
            "[Verse 1]\n第一句\nintro\n第二句",
        ),
    ],
)
async def test_prepare_normalizes_known_user_labels_without_changing_unknown_ones(
    tmp_path: Path, lyrics: str, tagged: str, expected: str
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        content = json.dumps(
            {"taggedLyrics": tagged, "styleTags": EXPANDED_MANDARIN_ROCK_PROMPT},
            ensure_ascii=False,
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    user_prompt = f"[歌词与创作内容]\n{lyrics}\n\n[风格要求]\n普通话摇滚"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        prepared = await OpenAICompatiblePromptExpander(settings, client).prepare(user_prompt)

    assert prepared.lyrics == expected
    assert len(requests) == 1


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
                                    "taggedLyrics": ONE_MINUTE_LYRICS,
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
    assert "目标时长" not in body["messages"][1]["content"]


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
                                    "taggedLyrics": generated_lyrics(12),
                                    "styleTags": EXPANDED_MANDARIN_ROCK_PROMPT,
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
    assert prepared.lyrics == generated_lyrics(12)
    assert prepared.duration_seconds is None
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert "同时生成原创歌词和音乐风格说明" in body["messages"][0]["content"]
    assert "原创简体中文歌词" in body["messages"][0]["content"]
    assert "目标时长" not in body["messages"][1]["content"]
    diagnostics = json.loads(
        (settings.output_dir / "jobs/prompt-job/prompts.json").read_text(encoding="utf-8")
    )
    assert diagnostics["llmAttempts"][0]["request"]["body"] == body
    assert diagnostics["llmAttempts"][0]["response"]["statusCode"] == 200


async def test_prepare_retries_generated_lyrics_with_custom_stage_directions(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        lyrics = (
            "[Guitar Solo]\n（25 秒吉他独奏）\n第一句"
            if len(requests) == 1
            else generated_lyrics(12)
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

    assert prepared.lyrics == generated_lyrics(12)
    assert len(requests) == 2


async def test_prepare_extracts_json_surrounded_by_commentary(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "taggedLyrics": ONE_MINUTE_LYRICS,
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
    assert prepared.lyrics == ONE_MINUTE_LYRICS
    assert prepared.duration_seconds == 60


async def test_prepare_retries_once_after_invalid_json(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        content = "not json"
        if len(requests) == 2:
            content = json.dumps(
                {
                    "taggedLyrics": ONE_MINUTE_LYRICS,
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


async def test_prepare_retry_skips_empty_assistant_message(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        content = ""
        if len(requests) == 2:
            content = json.dumps(
                {
                    "taggedLyrics": ONE_MINUTE_LYRICS,
                    "styleTags": EXPANDED_MANDARIN_ROCK_PROMPT,
                },
                ensure_ascii=False,
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await OpenAICompatiblePromptExpander(settings, client).prepare("[风格要求]\n男声摇滚", 1)

    retry_body = json.loads(requests[1].content)
    assert [message["role"] for message in retry_body["messages"]] == ["system", "user", "user"]


async def test_prepare_reports_second_validation_failure(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GenerationError, match="模型未返回 JSON 对象"):
            await OpenAICompatiblePromptExpander(settings, client).prepare("[风格要求]\n男声摇滚")

    assert len(requests) == 2


async def test_prepare_rejects_missing_llm_key_before_request(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, llm_api_key=None)
    async with httpx.AsyncClient() as client:
        with pytest.raises(GenerationError, match="缺少 LLM_API_KEY"):
            await OpenAICompatiblePromptExpander(settings, client).prepare("[风格要求]\n男声摇滚")


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
                                    "lyrics": ONE_MINUTE_LYRICS.removeprefix("[Verse]\n"),
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
    assert prepared.lyrics == ONE_MINUTE_LYRICS
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


async def test_prepare_accepts_unbracketed_style_tag_array(tmp_path: Path) -> None:
    # Real Qwen3.5 response shape: JSON array items without square brackets.
    tags = [
        "Genre: Hardcore Hip-Hop",
        "Tempo: 140 BPM",
        "Mood: Aggressive, Furious, Defiant",
        "Vocals: Raw, Distorted, High-Energy Rap",
        "Instrumentation: Heavy Distorted 808 Bass, Aggressive Drums, Fuzz Guitar",
        "Arrangement: Minimalist, Fast-Paced, Chaotic Energy",
        "Production and Mix: Lo-Fi Texture, Front-Loaded Vocals, Punchy Transients",
        "Negative Constraints: No Muddy Bass, No Excessive Reverb",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        content = json.dumps({"taggedLyrics": "[Verse]\n第一句\n第二句", "styleTags": tags})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    settings = make_settings(tmp_path, llm_api_key="secret", llm_base_url="https://llm.test/v1")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        prepared = await OpenAICompatiblePromptExpander(settings, client).prepare(
            "[歌词与创作内容]\n第一句\n第二句\n\n[风格要求]\n硬核说唱", 0.5
        )

    assert prepared.structured_prompt.startswith("[Genre: Hardcore Hip-Hop]")
    assert prepared.structured_prompt.count("[") == len(tags)
    assert prepared.duration_seconds == 30


def test_elevenlabs_prompt_preserves_complete_lyrics_and_clear_vocal_requirements() -> None:
    lyrics = "[Verse]\n" + "完整歌词行\n" * 800
    prompt = build_elevenlabs_prompt("[Genre: Rock]", 3, True, lyrics)
    assert "Target duration: 3 minutes" in prompt
    assert "Mandarin Chinese lead vocals" in prompt
    assert prompt.endswith(lyrics)
