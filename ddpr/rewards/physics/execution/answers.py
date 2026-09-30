from __future__ import annotations

import ast
import inspect
import math
import re
from typing import Any

from ..rules.expressions import (
    _ALLOWED_SYMPY_CALLS,
    _FORBIDDEN_AST_NODES,
    _SYMPY_CONSTANT_NAMES,
    _UNSAFE_CALL_NAMES,
    UnsupportedAnswerOperation,
    _string_call_requires_review,
    _sympy_execution_namespace,
)
from ..schemas import GradingPlan, UnexecutedAnswer

_PYTHON_FENCE = re.compile(r"```(?:python|py)\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def extract_answer_function(
    response: str,
    *,
    grading_plan: GradingPlan | None = None,
) -> dict[str, Any]:
    """Run the final Python ``answer`` function block and return normalized answers."""

    code = _final_answer_code(response)
    _validate_answer_code(code, grading_plan=grading_plan)
    namespace = _execute_answer_code(code)
    answer_fn = namespace.get("answer")
    if not callable(answer_fn):
        raise ValueError("Python answer block must define callable answer().")
    raw_answer = _call_answer_function(answer_fn, namespace)
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
        return extract_answer_function(response, grading_plan=grading_plan)
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


def _validate_answer_code(
    code: str,
    *,
    grading_plan: GradingPlan | None = None,
    allow_unsupported_calls: bool = False,
) -> ast.Module:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Python answer block has invalid syntax: {exc}") from exc

    answer_defs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "answer"
    ]
    if len(answer_defs) != 1:
        raise ValueError(
            "Python answer block must define exactly one answer() function."
        )
    if answer_defs[0].args.vararg or answer_defs[0].args.kwarg:
        raise ValueError("answer() may not use *args or **kwargs.")
    if grading_plan is not None:
        _validate_floating_literal_returns(tree, grading_plan)

    unsupported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_AST_NODES):
            raise ValueError(
                f"Python answer block uses unsupported syntax: {type(node).__name__}"
            )
        if isinstance(node, ast.Import):
            _validate_import(node)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Attribute):
                raise ValueError(
                    "Python answer block may only call SymPy functions or expression methods."
                )
            if node.func.attr in _UNSAFE_CALL_NAMES:
                raise ValueError(f"Python answer block may not use {node.func.attr!r}.")
            if not _is_allowed_call(node.func) or _string_call_requires_review(node):
                unsupported.add(ast.unparse(node.func))
        if isinstance(node, ast.Attribute):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                raise ValueError(
                    "Python answer block may not modify object attributes."
                )
            if node.attr.startswith("_"):
                raise ValueError(
                    "Python answer block may not access private attributes."
                )
            if (
                not isinstance(node.value, ast.Name)
                or node.value.id not in {"sp", "sympy"}
                or node.attr not in _ALLOWED_SYMPY_CALLS | _SYMPY_CONSTANT_NAMES
            ):
                unsupported.add(ast.unparse(node))
        if isinstance(node, ast.Name) and node.id in _UNSAFE_CALL_NAMES:
            raise ValueError(f"Python answer block may not use {node.id!r}.")
    if unsupported and not allow_unsupported_calls:
        raise UnsupportedAnswerOperation(
            "Local execution does not support: " + ", ".join(sorted(unsupported))
        )
    return tree


def _validate_floating_literal_returns(
    tree: ast.Module,
    grading_plan: GradingPlan,
) -> None:
    floating_targets = {
        call.target
        for call in grading_plan.calls
        if call.primitive == "floating_numeric"
    }
    if not floating_targets:
        return

    answer_def = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "answer"
    )
    returns = [node for node in ast.walk(answer_def) if isinstance(node, ast.Return)]
    if len(returns) != 1 or returns[0].value is None:
        raise ValueError(
            "floating_numeric answers must use exactly one return statement "
            "with direct finite Python float literals."
        )

    returned = returns[0].value
    values_by_target: dict[str, ast.expr] = {}
    if isinstance(returned, ast.Dict):
        for key, value in zip(returned.keys, returned.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                values_by_target[key.value] = value
    elif floating_targets == {"answer"}:
        values_by_target["answer"] = returned

    for target in sorted(floating_targets):
        value = values_by_target.get(target)
        if value is None:
            continue
        if not _is_direct_finite_float_literal(value):
            raise ValueError(
                f"floating_numeric target {target!r} must be returned as a "
                "direct finite Python float literal."
            )


def _is_direct_finite_float_literal(node: ast.expr) -> bool:
    value_node = node
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        value_node = node.operand
    return (
        isinstance(value_node, ast.Constant)
        and type(value_node.value) is float
        and math.isfinite(value_node.value)
    )


def _validate_import(node: ast.Import) -> None:
    for alias in node.names:
        if alias.name != "sympy" or alias.asname not in {None, "sp"}:
            raise ValueError("Python answer block may only import sympy as sp.")


def _is_allowed_call(func: ast.expr) -> bool:
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id in {"sp", "sympy"} and func.attr in _ALLOWED_SYMPY_CALLS
    return False


def _execute_answer_code(code: str) -> dict[str, Any]:
    try:
        import sympy as sp
    except ImportError as exc:  # pragma: no cover - dependency is in project metadata.
        raise ValueError(f"sympy is not installed: {exc}") from exc

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


def _call_answer_function(answer_fn: Any, namespace: dict[str, Any]) -> Any:
    try:
        import sympy as sp
    except ImportError as exc:  # pragma: no cover - dependency is in project metadata.
        raise ValueError(f"sympy is not installed: {exc}") from exc

    signature = inspect.signature(answer_fn)
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
        return answer_fn(*args, **kwargs)
    except UnsupportedAnswerOperation:
        raise
    except Exception as exc:
        raise ValueError(f"Could not call answer(): {exc}") from exc
