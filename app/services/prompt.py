from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

import httpx

from app.core.config import Settings
from app.core.errors import GenerationError

logger = logging.getLogger(__name__)

LLM_SYSTEM_PROMPT = "\n".join(
    [
        "你是一位专业音乐制作人和 ElevenLabs Music Prompt 扩写助手。",
        "将用户的简短中文需求扩写为可直接用于音乐生成的详细英文制作说明；"
        "不是逐词翻译，也不是只提取用户已经说过的词。",
        "保留所有显式要求，并为未指定项补充协调一致的专业选择；不要引用具体艺人或受版权保护作品。",
        "输出 8-14 个逗号分隔的方括号标签，总长度约 350-1200 字符。",
        "必须覆盖 Genre and Era、Tempo and Meter、Mood、Instrumentation、Vocal、"
        "Arrangement、Production and Mix；建议补充 Harmony、Dynamics、Negative Constraints。",
        "标签值要包含可听见、可执行的细节，例如 BPM、鼓组律动、乐器音色、人声语言与唱法、"
        "段落推进、空间效果、动态变化和需要避免的声音。",
        "优先使用 [Category: detailed value]，每个分类各占一个方括号。",
        "只返回标签本身，不要解释，不要寒暄，不要 Markdown，不要代码块。",
        "第一个字符必须是 [，最后一个字符必须是 ]，禁止输出思考过程。",
    ]
)

STRICT_LLM_SYSTEM_PROMPT = " ".join(
    [
        "Expand the user's request into 8-14 detailed English music-production tags, "
        "roughly 350-1200 characters total.",
        "Preserve explicit requirements and make coherent professional decisions "
        "for missing details.",
        "Required categories: Genre and Era, Tempo and Meter, Mood, Instrumentation, Vocal, "
        "Arrangement, Production and Mix.",
        "Also add useful Harmony, Dynamics, and Negative Constraints when appropriate.",
        "Use concrete audible directions: BPM, groove, timbre, vocal language and delivery, "
        "section progression, spatial effects, dynamics, and exclusions.",
        "Prefer one bracket per category in the form [Category: detailed value].",
        "Do not name artists or copyrighted songs.",
        "Start with [ and end with ].",
        "Do not explain, reason, use Markdown, or repeat these instructions.",
    ]
)
EXPANDED_PROMPT_MIN_TAGS = 8
EXPANDED_PROMPT_MAX_TAGS = 14
EXPANDED_PROMPT_MIN_CHARS = 280
EXPANDED_PROMPT_MAX_CHARS = 1800
LLM_OUTPUT_TOKEN_CEILING = 4096
STRUCTURED_TAG_PATTERN = re.compile(
    r"\[\s*[A-Za-z][A-Za-z0-9 /_-]{0,39}\s*:\s*[^\[\]\r\n]{1,200}\s*\]"
)


class PromptExpander(Protocol):
    async def expand(self, user_prompt: str) -> str: ...


def normalize_llm_output(raw_content: object) -> str:
    content = str(raw_content or "").strip()
    content = re.sub(r"^```[a-z]*\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```$", "", content)
    return content.strip().strip("\"'").strip()


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
    normalized_tags = [re.sub(r"\s+", " ", tag).strip() for tag in tags]
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
    duration_minutes: int,
    clear_chinese_vocal_mode: bool,
) -> str:
    requirements = [structured_prompt, f"Target duration: {duration_minutes} minutes."]
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
            first_content = await self._completion(user_prompt, strict=False)
            first_structured_prompt = extract_structured_music_tags(first_content)
            if first_structured_prompt and is_expanded_music_prompt(first_structured_prompt):
                return first_structured_prompt
            self._log_insufficient_expansion(first_structured_prompt, strict=False)

            retry_content = await self._completion(
                user_prompt,
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
            "max_tokens": effective_llm_output_tokens(
                self._settings.llm_max_tokens,
                strict=strict,
            ),
            "messages": [
                {
                    "role": "system",
                    "content": STRICT_LLM_SYSTEM_PROMPT if strict else LLM_SYSTEM_PROMPT,
                },
                {"role": "user", "content": request_prompt},
            ],
        }
        if self._settings.llm_disable_thinking and "qwen3" in self._settings.llm_model.lower():
            request_body["chat_template_kwargs"] = {"enable_thinking": False}
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
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
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
