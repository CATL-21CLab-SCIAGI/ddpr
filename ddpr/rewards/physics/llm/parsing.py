from __future__ import annotations

import json
import re
from typing import Any

from ..schemas import CompletionResult

_JSON_FENCE = re.compile(
    r"```json\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)
_OUTER_JSON_FENCE = re.compile(
    r"\A```json\s*(.*)\s*```\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_json_object(
    content: str,
    *,
    source: str = "LLM response",
) -> dict[str, Any]:
    """Extract one JSON object from plain, fenced, or narrated model output."""

    outer_fence = _OUTER_JSON_FENCE.fullmatch(content.strip())
    if outer_fence is not None:
        data = _loads_json(outer_fence.group(1))
        if not isinstance(data, dict):
            raise TypeError(f"{source} JSON must be an object.")
        return _normalize_generated_text_controls(data)

    try:
        data = _loads_json(content)
    except json.JSONDecodeError as initial_error:
        fenced_error: json.JSONDecodeError | None = None
        found_fence = False
        for match in _JSON_FENCE.finditer(content):
            found_fence = True
            try:
                data = _loads_json(match.group(1))
            except json.JSONDecodeError as error:
                fenced_error = error
                continue
            if isinstance(data, dict):
                return _normalize_generated_text_controls(data)

        if found_fence:
            raise fenced_error or initial_error

        decoder = json.JSONDecoder()
        cleaned_content = _strip_trailing_commas(content)
        for index, character in enumerate(cleaned_content):
            if character != "{":
                continue
            try:
                data, _ = decoder.raw_decode(cleaned_content[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return _normalize_generated_text_controls(data)
        raise initial_error
    if not isinstance(data, dict):
        raise TypeError(f"{source} JSON must be an object.")
    return _normalize_generated_text_controls(data)


def _normalize_generated_text_controls(value: Any) -> Any:
    """Match upstream handling of stray control bytes in Judge JSON values."""

    if isinstance(value, str):
        return "".join(
            character
            for character in value
            if not (
                (ord(character) < 0x20 and character not in "\t\n\r")
                or ord(character) == 0x7F
            )
        )
    if isinstance(value, list):
        return [_normalize_generated_text_controls(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_generated_text_controls(item) for key, item in value.items()
        }
    return value


def _loads_json(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError as initial_error:
        cleaned = _strip_trailing_commas(content)
        if cleaned != content:
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        repaired = _escape_control_characters_in_strings(cleaned)
        if repaired != cleaned:
            return json.loads(repaired)
        raise initial_error


def _escape_control_characters_in_strings(content: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False

    for character in content:
        if not in_string:
            result.append(character)
            if character == '"':
                in_string = True
            continue

        if escaped:
            result.append(character)
            escaped = False
        elif character == "\\":
            result.append(character)
            escaped = True
        elif character == '"':
            result.append(character)
            in_string = False
        elif ord(character) < 0x20:
            result.append(json.dumps(character)[1:-1])
        else:
            result.append(character)

    return "".join(result)


def _strip_trailing_commas(content: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    length = len(content)

    for index, character in enumerate(content):
        if in_string:
            result.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
            result.append(character)
            continue

        if character == ",":
            next_index = index + 1
            while next_index < length and content[next_index].isspace():
                next_index += 1
            if next_index < length and content[next_index] in "}]":
                continue

        result.append(character)

    return "".join(result)


def normalize_result(
    value: str | CompletionResult,
) -> CompletionResult:
    if isinstance(value, CompletionResult):
        return value
    if isinstance(value, str):
        return CompletionResult(content=value)
    raise TypeError("Judge completion returned an unsupported value")
