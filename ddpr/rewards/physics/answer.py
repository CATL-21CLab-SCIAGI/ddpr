"""Extract and execute submitted answers for physics grading."""

from __future__ import annotations

import ast
import inspect
import re
from collections.abc import Callable
from typing import Any

from .runtime import AnswerExtractionError
from .schemas import GradingPlan, UnexecutedAnswer
from .validation import (
    UnsupportedAnswerOperation,
    _sympy_execution_namespace,
    _validate_answer_code,
)

_PYTHON_FENCE = re.compile(r"```(?:python|py)\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def execute_answer_function(
    response: str,
    *,
    grading_plan: GradingPlan | None = None,
) -> dict[str, Any]:
    """Run the final Python ``answer`` function block and return normalized answers."""

    code = _final_answer_code(response)
    _validate_answer_code(code, grading_plan=grading_plan)
    namespace = _execute_answer_code(code)
    fn = namespace.get("answer")
    if not callable(fn):
        raise ValueError("Python answer block must define callable answer().")
    raw_answer = _call_answer_function(fn, namespace)
    return raw_answer if isinstance(raw_answer, dict) else {"answer": raw_answer}


def _final_answer_code(response: str) -> str:
    candidates = [
        block
        for block in _PYTHON_FENCE.findall(response)
        if re.search(r"\bdef\s+answer\s*\(", block)
    ]
    if not candidates:
        raise ValueError("No final Python answer function block found.")

    return candidates[-1]


def extract_gradeable_answer(
    response: str,
    *,
    grading_plan: GradingPlan,
) -> dict[str, Any]:
    """Keep unsupported mathematical answers as inert text for equivalence review."""
    try:
        return execute_answer_function(response, grading_plan=grading_plan)
    except UnsupportedAnswerOperation as exc:
        code = _final_answer_code(response)
        tree = _validate_answer_code(
            code,
            grading_plan=grading_plan,
            allow_unsupported_calls=True,
        )
        answer = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "answer"
        )
        if (
            not answer.body
            or not isinstance(answer.body[-1], ast.Return)
            or any(
                not isinstance(node, (ast.Assign, ast.Return)) for node in answer.body
            )
            or sum(isinstance(node, ast.Return) for node in ast.walk(answer)) != 1
        ):
            raise ValueError(
                "Text-only answer must have local assignments and one final return."
            ) from exc
        returned = answer.body[-1].value
        assignments = {
            target.id: node.value
            for node in answer.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        # Resolve only the return container; preserve all mathematics as source text.
        visited: set[str] = set()
        while isinstance(returned, ast.Name) and returned.id in assignments:
            if returned.id in visited:
                raise ValueError("Cyclic answer assignments cannot be read.") from exc
            visited.add(returned.id)
            returned = assignments[returned.id]
        if returned is None:
            raise ValueError("answer() must return a value.") from exc
        if isinstance(returned, ast.Dict):
            values = {}
            for key, value in zip(returned.keys, returned.values, strict=True):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    raise ValueError(
                        "Text-only answer targets must be literal strings."
                    ) from exc
                values[key.value] = value
        else:
            values = {"answer": returned}
        result: dict[str, Any] = {}
        for call in grading_plan.calls:
            if call.target not in values:
                continue
            value = values[call.target]
            if call.primitive in {"exact", "symbolic"}:
                expression = (
                    value.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                    else ast.unparse(value)
                )
                result[call.target] = UnexecutedAnswer(expression, code, str(exc))
            else:
                try:
                    result[call.target] = ast.literal_eval(value)
                except (ValueError, TypeError) as literal_exc:
                    raise ValueError(
                        "Non-symbolic answers must remain literal values."
                    ) from literal_exc
        return result


def _execute_answer_code(code: str) -> dict[str, Any]:
    import sympy as sp

    allowed_sp = _sympy_execution_namespace(sp)

    def safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sympy":
            return allowed_sp
        raise ImportError(f"Import of {name!r} is not allowed in answer blocks.")

    namespace: dict[str, Any] = {
        "__builtins__": {"__import__": safe_import},
        "sp": allowed_sp,
        "sympy": allowed_sp,
    }
    try:
        exec(code, namespace, namespace)
    except UnsupportedAnswerOperation:
        raise
    except Exception as exc:
        raise ValueError(f"Could not execute Python answer block: {exc}") from exc
    return namespace


def _call_answer_function(fn: Callable[..., Any], namespace: dict[str, Any]) -> Any:
    import sympy as sp

    signature = inspect.signature(fn)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise ValueError("answer() may not use *args or **kwargs.")
        symbol = namespace.get(name)
        if not isinstance(symbol, sp.Basic):
            symbol = sp.Symbol(name)
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(symbol)
        else:
            kwargs[name] = symbol
    try:
        return fn(*args, **kwargs)
    except UnsupportedAnswerOperation:
        raise
    except Exception as exc:
        raise ValueError(f"Could not call answer(): {exc}") from exc


def extract_answer_locally(
    response: str,
    grading_plan: GradingPlan,
    extract_fn: Callable[..., dict[str, Any]] = extract_gradeable_answer,
) -> dict[str, Any]:
    try:
        return extract_fn(response, grading_plan=grading_plan)
    except ValueError as exc:
        raise AnswerExtractionError(str(exc)) from exc
