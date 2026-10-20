from __future__ import annotations

import json
import re
from typing import Any

from ..schemas import CompletionResult

_JSON_FENCE = re.compile(
    r"^[ \t]*```(?:json)?[ \t]*$", flags=re.IGNORECASE | re.MULTILINE
)


def extract_json_object(
    content: str,
    *,
    source: str = "LLM response",
) -> dict[str, Any]:
    """Read one unambiguous JSON object, allowing fences and surrounding prose."""
    content = _JSON_FENCE.sub("", content).strip()
    try:
        data = _loads_json(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
        cleaned = _escape_control_characters_in_strings(_strip_trailing_commas(content))
        objects = []
        position = 0
        while (start := cleaned.find("{", position)) != -1:
            data, end = decoder.raw_decode(cleaned, start)
            objects.append(data)
            position = end
        if len(objects) != 1:
            raise ValueError(f"{source} must contain exactly one JSON object") from None
        data = objects[0]
    if not isinstance(data, dict):
        raise TypeError(f"{source} JSON must be an object.")
    return _normalize_generated_text_controls(data)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


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
        return json.loads(content, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as initial_error:
        cleaned = _strip_trailing_commas(content)
        if cleaned != content:
            try:
                return json.loads(cleaned, object_pairs_hook=_unique_object)
            except json.JSONDecodeError:
                pass

        repaired = _escape_control_characters_in_strings(cleaned)
        if repaired != cleaned:
            return json.loads(repaired, object_pairs_hook=_unique_object)
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
