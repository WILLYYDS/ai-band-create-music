from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.core.config import Settings
from app.core.errors import GenerationError
from app.services.job_files import update_job_diagnostics

logger = logging.getLogger(__name__)

LLM_SYSTEM_PROMPT = "\n".join(
    [
        "你是一位专业音乐制作人和音乐风格 Prompt 扩写助手。",
        "将用户的简短风格要求扩写为可直接用于音乐生成的详细英文制作说明；"
        "不是逐词翻译，也不是只提取用户已经说过的词。",
        "只扩写曲风、编曲、演唱和制作参数，不生成、改写或复述歌词。",
        "不得把“不生成歌词”误写成音乐要求；Negative Constraints 禁止出现 no lyrics、"
        "no vocals 或 no melody。",
        "保留所有显式要求，并为未指定项补充协调一致的专业选择；不要引用具体艺人或受版权保护作品。",
        "输出恰好 8 个简洁的英文方括号标签，总长度约 350-900 字符。",
        "必须覆盖 Genre and Era、Tempo and Meter、Mood、Instrumentation、Vocal、"
        "Arrangement、Production and Mix、Negative Constraints。",
        "标签值要包含可听见、可执行的细节，例如 BPM、鼓组律动、乐器音色、人声语言与唱法、"
        "段落推进、空间效果、动态变化和需要避免的声音。",
        "优先使用 [Category: detailed value]，每个分类各占一个方括号。",
        "只返回标签本身，不要解释，不要寒暄，不要 Markdown，不要代码块。",
        "第一个字符必须是 [，最后一个字符必须是 ]，禁止输出思考过程。",
    ]
)

LYRICS_SYSTEM_PROMPT = "\n".join(
    [
        "你是专业音乐填词人，为指定歌曲创作可直接演唱的原创歌词。",
        "只输出歌词本身：每行一句歌词，用换行分隔；",
        "在合适位置使用 [Intro]、[Verse]、[Chorus]、[Bridge]、[Outro] 结构标签，每个标签独占一行。",
        "不解释、不寒暄、不输出代码块，不要写歌名、艺人名或音乐风格描述。",
        "歌词需贴合主题与情绪，韵脚自然、便于演唱，严禁抄袭受版权保护的作品。",
    ]
)

LYRICS_AND_STYLE_SYSTEM_PROMPT = "\n".join(
    [
        "你是专业音乐制作人与歌词结构编辑，同时完成歌词分段和音乐风格扩写。",
        "只能插入 [Intro]、[Verse]、[Pre-Chorus]、[Chorus]、[Post-Chorus]、"
        "[Bridge]、[Instrumental]、[Solo]、[Outro]，标签必须单独占一行。",
        "不得增加、删除、改写、重排或重复任何歌词。",
        "将风格要求扩写为 8-14 个详细英文音乐制作标签，格式为 [Category: value]。",
        "风格标签需覆盖曲风、速度、情绪、配器、人声、编曲、制作与混音。",
        "只返回 JSON 对象，包含字符串 taggedLyrics、字符串 styleTags 和整数 durationSeconds；"
        "不要解释或输出代码块。",
    ]
)

GENERATE_LYRICS_AND_STYLE_SYSTEM_PROMPT = "\n".join(
    [
        "你是专业音乐制作人与填词人，根据用户的创作要求同时生成原创歌词和音乐风格说明。",
        "taggedLyrics 必须是可直接演唱的原创中文歌词，使用 [Intro]、[Verse]、[Chorus]、"
        "[Bridge]、[Outro] 等结构标签，每个标签独占一行。",
        "styleTags 必须是恰好 8 个简洁的英文 [Category: value] 音乐制作标签，覆盖流派、"
        "速度、情绪、配器、人声、编曲、制作与混音、排除项。",
        "保留用户指定的人声、时长和风格；不得引用具体艺人或受版权保护作品。",
        "排除项禁止出现 no lyrics、no vocals 或 no melody。",
        "只返回 JSON 对象，包含字符串 taggedLyrics、字符串 styleTags 和整数 durationSeconds；"
        "不要解释或输出代码块。",
    ]
)

STRICT_LLM_SYSTEM_PROMPT = " ".join(
    [
        "Expand the user's request into exactly 8 concise English music-production tags, "
        "roughly 350-900 characters total.",
        "Preserve explicit requirements and make coherent professional decisions "
        "for missing details.",
        "Required categories: Genre and Era, Tempo and Meter, Mood, Instrumentation, Vocal, "
        "Arrangement, Production and Mix, and Negative Constraints.",
        "Use concrete audible directions: BPM, groove, timbre, vocal language and delivery, "
        "section progression, spatial effects, dynamics, and exclusions.",
        "Prefer one bracket per category in the form [Category: detailed value].",
        "Do not name artists or copyrighted songs.",
        "Never include no lyrics, no vocals, or no melody as a production constraint.",
        "Start with [ and end with ].",
        "Do not explain, reason, use Markdown, or repeat these instructions.",
    ]
)
EXPANDED_PROMPT_MIN_TAGS = 8
EXPANDED_PROMPT_MAX_TAGS = 14
EXPANDED_PROMPT_MIN_CHARS = 280
EXPANDED_PROMPT_MAX_CHARS = 3000
LLM_OUTPUT_TOKEN_CEILING = 4096
STRUCTURED_TAG_PATTERN = re.compile(
    r"\[\s*[A-Za-z][A-Za-z0-9 /_-]{0,39}\s*:\s*[^\[\]\r\n]{1,320}\s*\]"
)
LYRICS_SECTION_PATTERN = re.compile(
    r"\[(?:Intro|Verse(?: \d+)?|Pre-Chorus|Chorus(?: \d+)?|Post-Chorus|"
    r"Bridge(?: \d+)?|Instrumental(?: Break)?|Solo|Outro)\]",
    re.IGNORECASE,
)
FORBIDDEN_LYRICS_CONSTRAINT_PATTERN = re.compile(
    r"(?:[,;]\s*)?\bno (?:lyrics(?:\s+or\s+melody(?:\s+generation)?)?|"
    r"vocals?|melody(?:\s+generation)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    structured_prompt: str
    lyrics: str
    duration_seconds: int


def lyrics_duration_instruction(
    duration_minutes: float | None,
    *,
    minimum_seconds: int = 60,
    maximum_seconds: int = 360,
    include_json_duration: bool = True,
) -> str:
    if duration_minutes is None:
        return (
            f"目标时长：自动。请根据歌词字数、段落、BPM 和编曲自行确定 {minimum_seconds}-"
            f"{maximum_seconds} 秒内最短且足够的生成上限，并在 JSON 的 durationSeconds "
            "返回整数秒数。"
            "audio_duration 只是模型可提前结束的上限，不是必须填满的目标；请逐段估算演唱、独奏、"
            "前奏和尾奏时间，只增加约 10 秒安全余量，避免明显高估。歌词必须能在该时长结束前"
            "完整唱完；不要重复、灌水或在歌词唱完后继续演唱。"
        )
    max_lines = round(duration_minutes * 20)
    duration_seconds = round(duration_minutes * 60)
    json_instruction = (
        f"JSON 的 durationSeconds 必须为 {duration_seconds}。" if include_json_duration else ""
    )
    return (
        f"目标时长：严格 {duration_seconds} 秒（约 {duration_minutes:g} 分钟）；"
        f"{json_instruction}必须让全部歌词在目标时长内完整唱完，"
        f"并为前奏、间奏和尾奏留出时间；歌词最多 {max_lines} 行（结构标签不计），"
        "每行简短，宁可少写，也不要让结尾歌词被截断；歌词唱完后绝对不能继续演唱。"
    )


class PromptExpander(Protocol):
    async def expand(self, user_prompt: str) -> str: ...

    async def prepare(
        self,
        user_prompt: str,
        duration_minutes: int | None = None,
        *,
        job_id: str | None = None,
    ) -> PreparedPrompt: ...

    async def write_lyrics(
        self, structured_prompt: str, user_prompt: str, duration_minutes: int
    ) -> str: ...


def split_generation_prompt(user_prompt: str) -> tuple[str, str]:
    """Split the exact section format sent by the Create page."""
    prompt = user_prompt.strip()
    lyrics_marker = "[歌词与创作内容]"
    style_marker = "[风格要求]"
    if prompt.startswith(lyrics_marker):
        parts = re.split(
            r"(?:\r\n|\n|\r)\[风格要求\](?:\r\n|\n|\r)", prompt[len(lyrics_marker) :], maxsplit=1
        )
        return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
    if prompt.startswith(style_marker):
        return "", prompt[len(style_marker) :].strip()
    return "", prompt


def normalize_llm_output(raw_content: object) -> str:
    content = str(raw_content or "").strip()
    content = re.sub(r"^```[a-z]*\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```$", "", content)
    return content.strip().strip("\"'").strip()


def extract_json_object(raw_content: object) -> dict[str, Any]:
    """Extract the first JSON object from an otherwise chatty LLM response."""
    content = normalize_llm_output(raw_content)
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, ValueError):
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            parsed, _ = decoder.raw_decode(content[match.start() :])
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("模型未返回 JSON 对象")


def extract_tagged_lyrics(raw_content: object, original_lyrics: str) -> str | None:
    lines = [
        line.strip() for line in normalize_llm_output(raw_content).splitlines() if line.strip()
    ]
    lyric_lines = [line for line in lines if not LYRICS_SECTION_PATTERN.fullmatch(line)]
    original_lines = [line.strip() for line in original_lyrics.splitlines() if line.strip()]
    if lyric_lines != original_lines or len(lyric_lines) == len(lines):
        return None
    return "\n".join(lines)


def truncate_lyrics(lyrics: str, limit: int = 3500) -> str:
    """Keep complete lyric lines within the local model input budget."""
    lines: list[str] = []
    size = 0
    for line in lyrics.splitlines():
        added = len(line) + (1 if lines else 0)
        if size + added > limit:
            break
        lines.append(line)
        size += added
    return "\n".join(lines)


def extract_structured_music_tags(raw_content: object) -> str | None:
    content = normalize_llm_output(raw_content)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
    candidates = [content]
    for marker in ("Final Answer:", "Final:", "最终答案：", "最终答案:", "Output:"):
        if marker in content:
            candidates.insert(0, content.rsplit(marker, 1)[-1].strip())

    json_candidate = _extract_json_tags(content)
    if json_candidate:
        candidates.insert(0, json_candidate)

    for candidate in candidates:
        normalized = _normalize_tags_only(candidate)
        if normalized:
            return normalized
        normalized = _normalize_flat_tag_list(candidate)
        if normalized:
            return normalized

    # Reasoning models sometimes expose their analysis in `content`, followed by
    # a final tag-only block without a stable "Final Answer" marker. Walk from the
    # bottom and accept only a contiguous block whose lines contain tags and no prose.
    tag_blocks: list[list[str]] = []
    current_block: list[str] = []
    for line in content.splitlines():
        if _normalize_tags_only(line):
            current_block.append(line)
        elif current_block:
            tag_blocks.append(current_block)
            current_block = []
    if current_block:
        tag_blocks.append(current_block)
    for block in reversed(tag_blocks):
        normalized = _normalize_tags_only("\n".join(block))
        if normalized and normalized.count("[") >= 2:
            return normalized

    trailing_tags = _extract_trailing_tag_sequence(content)
    if trailing_tags:
        return trailing_tags
    for line in reversed(content.splitlines()):
        flat_tags = _normalize_flat_tag_list(line.strip())
        if flat_tags:
            return flat_tags
    return None


def _normalize_tags_only(candidate: str) -> str | None:
    tags = STRUCTURED_TAG_PATTERN.findall(candidate)
    if not tags:
        return None
    remainder = STRUCTURED_TAG_PATTERN.sub("", candidate)
    if remainder.strip(" \t\r\n,;*_`-'\""):
        return None
    normalized_tags = [
        FORBIDDEN_LYRICS_CONSTRAINT_PATTERN.sub("", re.sub(r"\s+", " ", tag)).strip()
        for tag in tags
    ]
    return ", ".join(normalized_tags)


def _normalize_flat_tag_list(candidate: str) -> str | None:
    candidate = candidate.strip()
    if not candidate.startswith("[") or not candidate.endswith("]"):
        return None
    if candidate.count("[") != 1 or candidate.count("]") != 1 or ":" in candidate:
        return None
    items = [re.sub(r"\s+", " ", item).strip() for item in candidate[1:-1].split(",")]
    if not 8 <= len(items) <= 30:
        return None
    if any(not item or len(item) > 120 for item in items):
        return None
    return f"[{', '.join(items)}]"


def _extract_json_tags(content: str) -> str | None:
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    tags = data.get("tags")
    if isinstance(tags, list):
        normalized_tags = []
        for tag in tags:
            text = str(tag).strip()
            if not text:
                continue
            normalized_tags.append(text if text.startswith("[") else f"[{text}]")
        return ", ".join(normalized_tags) or None
    pairs = [
        f"[{key}: {value}]"
        for key, value in data.items()
        if isinstance(key, str) and isinstance(value, (str, int, float))
    ]
    return ", ".join(pairs) if pairs else None


def _extract_trailing_tag_sequence(content: str) -> str | None:
    """Extract at least three adjacent tags when they are the final payload."""
    matches = list(STRUCTURED_TAG_PATTERN.finditer(content))
    if len(matches) < 3:
        return None
    trailing = matches[-1]
    if content[trailing.end() :].strip(" \t\r\n,;.*_`-'\""):
        return None

    selected = [trailing.group()]
    cursor = trailing.start()
    for match in reversed(matches[:-1]):
        separator = content[match.end() : cursor]
        if separator.strip(" \t\r\n,;*_`-'\""):
            break
        selected.append(match.group())
        cursor = match.start()
    if len(selected) < 3:
        return None
    selected.reverse()
    return _normalize_tags_only(", ".join(selected))


def is_expanded_music_prompt(structured_prompt: str) -> bool:
    tags = STRUCTURED_TAG_PATTERN.findall(structured_prompt)
    flat_prompt = _normalize_flat_tag_list(structured_prompt)
    if flat_prompt:
        return (
            EXPANDED_PROMPT_MIN_CHARS <= len(flat_prompt) <= EXPANDED_PROMPT_MAX_CHARS
            and music_prompt_tag_count(flat_prompt) >= 10
        )
    if not EXPANDED_PROMPT_MIN_TAGS <= len(tags) <= EXPANDED_PROMPT_MAX_TAGS:
        return False
    if not EXPANDED_PROMPT_MIN_CHARS <= len(structured_prompt) <= EXPANDED_PROMPT_MAX_CHARS:
        return False

    keys = [tag[1 : tag.index(":")].strip().lower() for tag in tags]
    required_category_aliases = (
        ("genre", "style"),
        ("tempo", "rhythm", "meter"),
        ("mood", "emotion"),
        ("instrument", "guitar", "drum", "bass"),
        ("vocal", "voice"),
        ("arrangement", "structure", "section"),
        ("production", "mix", "recording"),
    )
    return all(
        any(alias in key for key in keys for alias in aliases)
        for aliases in required_category_aliases
    )


def effective_llm_output_tokens(configured: int, *, strict: bool) -> int:
    requested = max(2048, configured) if strict else configured
    return min(requested, LLM_OUTPUT_TOKEN_CEILING)


def music_prompt_tag_count(structured_prompt: str) -> int:
    keyed_count = len(STRUCTURED_TAG_PATTERN.findall(structured_prompt))
    if keyed_count:
        return keyed_count
    flat_prompt = _normalize_flat_tag_list(structured_prompt)
    return len(flat_prompt[1:-1].split(",")) if flat_prompt else 0


def expansion_diagnostic(raw_content: object, structured_prompt: str | None) -> str:
    if structured_prompt is None:
        return f"无可解析标签/正文 {len(normalize_llm_output(raw_content))} 字符"
    return f"{music_prompt_tag_count(structured_prompt)} 个标签/{len(structured_prompt)} 字符"


def looks_like_chinese_music_request(*values: object) -> bool:
    text = " ".join(str(value) for value in values if value).lower()
    return bool(re.search(r"[\u3400-\u9fff]", text)) or any(
        marker in text for marker in ("mandarin", "chinese", "zhongwen", "zhong guo")
    )


def build_elevenlabs_planning_prompt(
    structured_prompt: str,
    duration_minutes: float,
    clear_chinese_vocal_mode: bool,
    lyrics: str = "",
) -> str:
    requirements = [structured_prompt, f"Target duration: {duration_minutes:g} minutes."]
    if clear_chinese_vocal_mode:
        requirements.append(
            " ".join(
                [
                    "Chinese clear vocal mode:",
                    "Use Mandarin Chinese lead vocals with clear articulation "
                    "and natural phrasing.",
                    "Keep lyric density moderate with short singable lines "
                    "and breathing space between phrases.",
                    "Make the lead vocal forward in the mix.",
                    "Use light reverb, minimal backing vocals, and avoid mumbling "
                    "or swallowed syllables.",
                ]
            )
        )
    if lyrics:
        requirements.extend(
            ["Original lyrics (preserve wording and line order):", truncate_lyrics(lyrics)]
        )
    return "\n".join(requirements)


def _append_unique(items: object, additions: list[str]) -> list[str]:
    output = list(items) if isinstance(items, list) else []
    for item in additions:
        if item not in output:
            output.append(item)
    return output


def enhance_elevenlabs_composition_plan(
    composition_plan: Any,
    clear_chinese_vocal_mode: bool,
) -> Any:
    if not clear_chinese_vocal_mode or not isinstance(composition_plan, dict):
        return composition_plan

    plan = json.loads(json.dumps(composition_plan))
    positive_styles = [
        "Mandarin Chinese vocals",
        "clear vocal articulation",
        "natural Chinese phrasing",
        "vocal-forward mix",
        "moderate lyric density",
        "short singable lyric lines",
    ]
    negative_styles = [
        "mumbled vocals",
        "swallowed syllables",
        "unclear consonants",
        "excessive reverb on vocals",
        "dense backing vocals",
        "overlapping vocal lines",
    ]
    plan["positive_global_styles"] = _append_unique(
        plan.get("positive_global_styles"), positive_styles
    )
    plan["negative_global_styles"] = _append_unique(
        plan.get("negative_global_styles"), negative_styles
    )

    sections = plan.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            section["positive_local_styles"] = _append_unique(
                section.get("positive_local_styles"), positive_styles
            )
            section["negative_local_styles"] = _append_unique(
                section.get("negative_local_styles"), negative_styles
            )
            lines = section.get("lines")
            if isinstance(lines, list) and lines:
                try:
                    duration_ms = max(1, int(section.get("duration_ms", 10_000)))
                except (TypeError, ValueError):
                    duration_ms = 10_000
                max_lines = max(1, min(4, duration_ms // 6000))
                section["lines"] = [str(line).strip() for line in lines if str(line).strip()][
                    :max_lines
                ]
    return plan


class OpenAICompatiblePromptExpander:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    async def expand(self, user_prompt: str) -> str:
        if self._settings.llm_api_key is None:
            raise GenerationError("LLM 扩写失败：缺少 LLM_API_KEY 环境变量。")

        try:
            _, style_prompt = split_generation_prompt(user_prompt)
            first_content = await self._completion(style_prompt, strict=False)
            first_structured_prompt = extract_structured_music_tags(first_content)
            if first_structured_prompt and is_expanded_music_prompt(first_structured_prompt):
                return first_structured_prompt
            self._log_insufficient_expansion(first_structured_prompt, strict=False)

            retry_content = await self._completion(
                style_prompt,
                strict=True,
                rejected_draft=first_structured_prompt,
            )
            structured_prompt = extract_structured_music_tags(retry_content)
            if structured_prompt and is_expanded_music_prompt(structured_prompt):
                return structured_prompt
            self._log_insufficient_expansion(structured_prompt, strict=True)
            raise GenerationError(
                "LLM 扩写失败：模型连续两次未返回足够详细的音乐制作 Prompt。"
                f"首轮={expansion_diagnostic(first_content, first_structured_prompt)}；"
                f"重试={expansion_diagnostic(retry_content, structured_prompt)}。"
            )
        except GenerationError:
            raise
        except httpx.ConnectTimeout as exc:
            raise GenerationError(
                "LLM 扩写失败：连接 LLM 服务超时（10 秒）。请检查 LLM_BASE_URL 和代理。"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise GenerationError(
                f"LLM 扩写失败：读取响应超过 {self._settings.llm_timeout_seconds:g} 秒。"
                "请使用非推理模型，或增大 LLM_TIMEOUT_SECONDS。"
            ) from exc
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            raise GenerationError(f"LLM 扩写失败：{_http_failure_message(exc)}") from exc

    async def prepare(
        self,
        user_prompt: str,
        duration_minutes: int | None = None,
        *,
        job_id: str | None = None,
    ) -> PreparedPrompt:
        if self._settings.llm_api_key is None:
            raise GenerationError("歌词与风格处理失败：缺少 LLM_API_KEY 环境变量。")
        lyrics, style = split_generation_prompt(user_prompt)
        generate_lyrics = not lyrics
        duration_instruction = lyrics_duration_instruction(
            duration_minutes,
            minimum_seconds=self._settings.min_duration_minutes * 60,
            maximum_seconds=self._settings.max_duration_minutes * 60,
        )
        if generate_lyrics:
            request_content = f"创作要求：\n{style or user_prompt}\n\n{duration_instruction}"
        else:
            request_content = (
                f"歌词：\n{lyrics}\n\n风格要求：\n{style or '请补充协调的音乐风格'}"
                f"\n\n{duration_instruction}"
            )
        request_body: dict[str, object] = {
            "model": self._settings.llm_model,
            "temperature": 0,
            "max_tokens": LLM_OUTPUT_TOKEN_CEILING,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        GENERATE_LYRICS_AND_STYLE_SYSTEM_PROMPT
                        if generate_lyrics
                        else LYRICS_AND_STYLE_SYSTEM_PROMPT
                    ),
                },
                {"role": "user", "content": request_content},
            ],
        }
        if self._settings.llm_disable_thinking:
            if "qwen3" in self._settings.llm_model.lower():
                request_body["chat_template_kwargs"] = {"enable_thinking": False}
            elif "doubao" in self._settings.llm_model.lower():
                request_body["thinking"] = {"type": "disabled"}
        if "doubao" in self._settings.llm_model.lower() or self._settings.llm_url.startswith(
            "https://api.openai.com/"
        ):
            request_body["response_format"] = {"type": "json_object"}
        try:
            original_messages = list(request_body["messages"])
            rejected_content = ""
            diagnostics: list[dict[str, object]] = []
            for attempt in range(2):
                if attempt:
                    request_body["messages"] = [
                        *original_messages,
                        {"role": "assistant", "content": rejected_content[:2000]},
                        {
                            "role": "user",
                            "content": (
                                "上次响应未满足歌词、时长或至少 8 个详细风格标签的要求。"
                                "请修正后只输出 JSON 对象。"
                            ),
                        },
                    ]
                try:
                    diagnostics.append(
                        {
                            "attempt": attempt + 1,
                            "request": {
                                "method": "POST",
                                "url": self._settings.llm_url,
                                "timeoutSeconds": self._settings.llm_timeout_seconds,
                                "body": json.loads(json.dumps(request_body)),
                            },
                        }
                    )
                    if job_id:
                        update_job_diagnostics(
                            self._settings.output_dir, job_id, llmAttempts=diagnostics
                        )
                    response = await self._client.post(
                        self._settings.llm_url,
                        json=request_body,
                        headers={
                            "Authorization": (
                                f"Bearer {self._settings.llm_api_key.get_secret_value()}"
                            ),
                            "Content-Type": "application/json",
                        },
                        timeout=httpx.Timeout(
                            self._settings.llm_timeout_seconds,
                            connect=min(10, self._settings.llm_timeout_seconds),
                        ),
                    )
                    try:
                        response_body: object = response.json()
                    except ValueError:
                        response_body = response.text
                    diagnostics[-1]["response"] = {
                        "statusCode": response.status_code,
                        "headers": dict(response.headers),
                        "body": response_body,
                    }
                    if job_id:
                        update_job_diagnostics(
                            self._settings.output_dir, job_id, llmAttempts=diagnostics
                        )
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if isinstance(content, list):
                        content = "\n".join(
                            str(part.get("text") or part.get("content") or "")
                            for part in content
                            if isinstance(part, dict)
                        )
                    rejected_content = str(content or "")
                    prepared = extract_json_object(content)
                    tagged_content = (
                        prepared.get("taggedLyrics")
                        or prepared.get("tagged_lyrics")
                        or prepared.get("lyrics")
                    )
                    if isinstance(tagged_content, list):
                        tagged_content = "\n".join(str(line) for line in tagged_content)
                    if generate_lyrics:
                        tagged = normalize_llm_output(tagged_content)
                        if tagged and not LYRICS_SECTION_PATTERN.search(tagged):
                            tagged = f"[Verse]\n{tagged}"
                    elif LYRICS_SECTION_PATTERN.search(lyrics):
                        tagged = lyrics
                    else:
                        tagged = extract_tagged_lyrics(tagged_content, lyrics) or (
                            f"[Verse]\n{lyrics}"
                        )
                    style_tags = (
                        prepared.get("styleTags")
                        or prepared.get("style_tags")
                        or prepared.get("structuredPrompt")
                        or prepared.get("style")
                    )
                    if isinstance(style_tags, list):
                        style_tags = ", ".join(str(tag) for tag in style_tags)
                    elif isinstance(style_tags, dict):
                        style_tags = json.dumps(style_tags, ensure_ascii=False)
                    structured = extract_structured_music_tags(style_tags) or normalize_llm_output(
                        style_tags
                    )
                    if not tagged or len(tagged) < 10 or not LYRICS_SECTION_PATTERN.search(tagged):
                        raise ValueError(
                            "模型未生成结构化歌词"
                            if generate_lyrics
                            else "模型修改了歌词或未添加段落标签"
                        )
                    if not structured or not is_expanded_music_prompt(structured):
                        raise ValueError("模型未返回至少 8 个详细风格标签")
                    if duration_minutes is None:
                        duration_seconds = prepared.get("durationSeconds")
                        if type(duration_seconds) is not int or not (
                            self._settings.min_duration_minutes * 60
                            <= duration_seconds
                            <= self._settings.max_duration_minutes * 60
                        ):
                            raise ValueError("模型未返回范围内的整数 durationSeconds")
                    else:
                        duration_seconds = duration_minutes * 60
                    return PreparedPrompt(structured, tagged, duration_seconds)
                except (ValueError, KeyError, IndexError, TypeError):
                    if attempt:
                        raise
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise GenerationError(f"歌词与风格处理失败：{_http_failure_message(exc)}") from exc

    async def write_lyrics(
        self, structured_prompt: str, user_prompt: str, duration_minutes: float
    ) -> str:
        if self._settings.llm_api_key is None:
            raise GenerationError("歌词生成失败：缺少 LLM_API_KEY 环境变量。")
        language = (
            "请用简体中文。"
            if looks_like_chinese_music_request(user_prompt, structured_prompt)
            else "Write in English."
        )
        request_body: dict[str, object] = {
            "model": self._settings.llm_model,
            "temperature": 0.8,
            "max_tokens": effective_llm_output_tokens(self._settings.llm_max_tokens, strict=False),
            "messages": [
                {"role": "system", "content": LYRICS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            "Original request:",
                            user_prompt,
                            "",
                            "Music style tags:",
                            structured_prompt[:1200],
                            "",
                            language,
                            lyrics_duration_instruction(
                                duration_minutes, include_json_duration=False
                            ),
                            "总长不超过 3500 字符。",
                        ]
                    ),
                },
            ],
        }
        if self._settings.llm_disable_thinking:
            if "qwen3" in self._settings.llm_model.lower():
                request_body["chat_template_kwargs"] = {"enable_thinking": False}
            elif "doubao" in self._settings.llm_model.lower():
                request_body["thinking"] = {"type": "disabled"}
        try:
            response = await self._client.post(
                self._settings.llm_url,
                json=request_body,
                headers={
                    "Authorization": f"Bearer {self._settings.llm_api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(
                    self._settings.llm_timeout_seconds,
                    connect=min(10, self._settings.llm_timeout_seconds),
                ),
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            message = choices[0].get("message") if isinstance(choices, list) and choices else None
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                content = "\n".join(
                    str(part.get("text") or part.get("content") or "")
                    for part in content
                    if isinstance(part, dict)
                )
            lyrics = normalize_llm_output(content)
            if len(lyrics) < 10:
                raise ValueError("模型未返回足够的歌词")
            return lyrics
        except GenerationError:
            raise
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise GenerationError(
                f"歌词生成失败：模型 {self._settings.llm_model}：{_http_failure_message(exc)}"
            ) from exc

    async def _completion(
        self,
        user_prompt: str,
        *,
        strict: bool,
        rejected_draft: str | None = None,
    ) -> object:
        request_prompt = user_prompt
        if strict:
            draft = rejected_draft[:1800] if rejected_draft else "No parseable draft was produced."
            request_prompt = "\n".join(
                [
                    "Original music request:",
                    user_prompt,
                    "",
                    "Rejected draft (too concise or incomplete):",
                    draft,
                    "",
                    "Rewrite and substantially enrich this draft. Return tags only.",
                ]
            )
        request_body: dict[str, object] = {
            "model": self._settings.llm_model,
            "temperature": 0 if strict else self._settings.llm_temperature,
            "max_tokens": min(
                1024,
                effective_llm_output_tokens(self._settings.llm_max_tokens, strict=strict),
            ),
            "messages": [
                {
                    "role": "system",
                    "content": STRICT_LLM_SYSTEM_PROMPT if strict else LLM_SYSTEM_PROMPT,
                },
                {"role": "user", "content": request_prompt},
            ],
        }
        if self._settings.llm_disable_thinking:
            if "qwen3" in self._settings.llm_model.lower():
                request_body["chat_template_kwargs"] = {"enable_thinking": False}
            elif "doubao" in self._settings.llm_model.lower():
                request_body["thinking"] = {"type": "disabled"}
        response = await self._client.post(
            self._settings.llm_url,
            json=request_body,
            headers={
                "Authorization": f"Bearer {self._settings.llm_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(
                self._settings.llm_timeout_seconds,
                connect=min(10, self._settings.llm_timeout_seconds),
            ),
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("LLM response JSON must be an object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("LLM response choices must contain an object")
        choice = choices[0]
        message = choice.get("message", {})
        if not isinstance(message, dict):
            raise ValueError("LLM response message must be an object")
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text") or part.get("content") or "")
                for part in content
                if isinstance(part, dict)
            )
        if extract_structured_music_tags(content) is None:
            reasoning_content = message.get("reasoning_content")
            logger.warning(
                "LLM response has no parseable music tags: "
                "model=%s strict=%s finish_reason=%r content_chars=%d reasoning_chars=%d",
                self._settings.llm_model,
                strict,
                choice.get("finish_reason"),
                len(str(content or "")),
                len(str(reasoning_content or "")),
            )
        return content

    def _log_insufficient_expansion(self, structured_prompt: str | None, *, strict: bool) -> None:
        if structured_prompt is None:
            return
        logger.warning(
            "LLM music prompt is valid tags but insufficiently expanded: "
            "model=%s strict=%s tags=%d chars=%d",
            self._settings.llm_model,
            strict,
            music_prompt_tag_count(structured_prompt),
            len(structured_prompt),
        )


def _http_failure_message(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        message = ""
        try:
            data = response.json()
            detail = data.get("detail") if isinstance(data, dict) else None
            if isinstance(detail, dict):
                message = str(detail.get("message", ""))
            elif detail:
                message = str(detail)
            if not message and isinstance(data, dict):
                nested_error = data.get("error")
                if isinstance(nested_error, dict):
                    message = str(nested_error.get("message", ""))
                elif nested_error:
                    message = str(nested_error)
                message = message or str(data.get("message") or data.get("msg") or "")
        except ValueError:
            message = response.text.strip()
        return f"HTTP {response.status_code}{f': {message}' if message else ''}"
    return str(error) or error.__class__.__name__
