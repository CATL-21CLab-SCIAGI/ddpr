from __future__ import annotations

from typing import Any

from ..schemas import PrimitiveGrade, unsupported_grading_args
from .common import _grade


def grade_categorical(
    actual: Any,
    expected: Any,
    args: dict[str, Any] | None = None,
) -> PrimitiveGrade:
    args = args or {}
    unsupported_args = unsupported_grading_args("categorical", args)
    if unsupported_args:
        return _grade(
            "categorical",
            "",
            False,
            f"unsupported categorical args: {unsupported_args}",
        )
    case_sensitive = bool(args.get("case_sensitive", True))
    allowed_aliases = args.get("allowed_aliases") or {}
    allowed_choices = args.get("allowed_choices") or args.get("choices")

    actual_set = _category_set(actual, case_sensitive, allowed_aliases)
    expected_set = _category_set(expected, case_sensitive, allowed_aliases)
    allowed_set = (
        _category_set(allowed_choices, case_sensitive, allowed_aliases)
        if allowed_choices is not None
        else None
    )
    if allowed_set is not None and not actual_set <= allowed_set:
        extra = sorted(actual_set - allowed_set)
        return _grade(
            "categorical",
            "",
            False,
            f"selected categories are not candidates: {extra}",
        )

    passed = actual_set == expected_set
    reason = (
        "category set matches"
        if passed
        else f"{sorted(actual_set)!r} != {sorted(expected_set)!r}"
    )
    return _grade("categorical", "", passed, reason)


def _normalize_category(value: Any, case_sensitive: bool) -> str:
    text = str(value).strip()
    return text if case_sensitive else text.casefold()


def _category_set(
    value: Any, case_sensitive: bool, allowed_aliases: dict[str, Any]
) -> set[str]:
    aliases = _alias_lookup(allowed_aliases, case_sensitive)
    if isinstance(value, set | frozenset | list | tuple):
        raw_items = value
    else:
        raw_items = [value]
    normalized = {_normalize_category(item, case_sensitive) for item in raw_items}
    return {aliases.get(item, item) for item in normalized}


def _alias_lookup(
    allowed_aliases: dict[str, Any], case_sensitive: bool
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in allowed_aliases.items():
        normalized_canonical = _normalize_category(canonical, case_sensitive)
        lookup[normalized_canonical] = normalized_canonical
        if isinstance(aliases, str):
            alias_values = [aliases]
        else:
            alias_values = list(aliases)
        for alias in alias_values:
            lookup[_normalize_category(alias, case_sensitive)] = normalized_canonical
    return lookup
