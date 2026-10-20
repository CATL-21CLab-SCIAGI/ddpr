"""Syntax checks and restricted SymPy operations for local evaluation."""

from __future__ import annotations

import ast
import math
import re
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

from .schemas import GradingPlan

_ALLOWED_SYMPY_CALLS = {
    "Abs",
    "Eq",
    "Float",
    "Integer",
    "KroneckerDelta",
    "Pow",
    "Piecewise",
    "Rational",
    "Symbol",
    "acos",
    "asin",
    "atan",
    "ceiling",
    "cos",
    "cosh",
    "cot",
    "csc",
    "exp",
    "factorial",
    "floor",
    "gamma",
    "harmonic",
    "hyper",
    "ln",
    "log",
    "polylog",
    "polygamma",
    "sec",
    "simplify",
    "sin",
    "sinh",
    "sqrt",
    "symbols",
    "zeta",
    "tan",
    "tanh",
    "acot",
    "acsc",
    "asec",
    "atan2",
    "sinc",
    "coth",
    "csch",
    "sech",
    "acosh",
    "asinh",
    "atanh",
    "acoth",
    "acsch",
    "asech",
    "root",
    "real_root",
    "cbrt",
    "conjugate",
    "re",
    "im",
    "arg",
    "sign",
    "erf",
    "erfc",
    "erfi",
    "erfinv",
    "erfcinv",
    "besselj",
    "bessely",
    "besseli",
    "besselk",
    "hankel1",
    "hankel2",
    "airyai",
    "airybi",
    "airyaiprime",
    "airybiprime",
    "elliptic_k",
    "elliptic_e",
    "elliptic_f",
    "elliptic_pi",
    "beta",
    "digamma",
    "lowergamma",
    "uppergamma",
    "loggamma",
    "LambertW",
    "legendre",
    "assoc_legendre",
    "hermite",
    "laguerre",
    "assoc_laguerre",
    "chebyshevt",
    "chebyshevu",
    "jacobi",
    "gegenbauer",
    "binomial",
    "expand",
    "factor",
    "cancel",
    "together",
    "collect",
    "trigsimp",
}


class UnsupportedAnswerOperation(ValueError):
    """A readable answer uses operations outside the local execution policy."""


_FORBIDDEN_AST_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.DictComp,
    ast.For,
    ast.GeneratorExp,
    ast.Global,
    ast.ImportFrom,
    ast.Lambda,
    ast.ListComp,
    ast.Nonlocal,
    ast.Raise,
    ast.SetComp,
    ast.Try,
    ast.While,
    ast.With,
)


_UNSAFE_CALL_NAMES = {
    "eval",
    "exec",
    "open",
    "__import__",
    "compile",
    "sympify",
    "parse_expr",
    "lambdify",
    "autowrap",
    "ufuncify",
}


def _numeric_constructor_text(name: str, value: str) -> bool:
    if name not in {"Integer", "Rational", "Float"}:
        return False
    try:
        Fraction(value)
    except (ValueError, ZeroDivisionError, OverflowError):
        return False
    return True


def _string_call_requires_review(node: ast.Call) -> bool:
    name = (
        node.func.attr
        if isinstance(node.func, ast.Attribute)
        else getattr(node.func, "id", "")
    )
    if name in {"Symbol", "symbols"}:
        return False
    return any(
        isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and not _numeric_constructor_text(name, value.value)
        for value in [*node.args, *(keyword.value for keyword in node.keywords)]
    )


def _sympy_execution_namespace(sp: Any) -> SimpleNamespace:
    def contains_text(value: Any, name: str) -> bool:
        if isinstance(value, str):
            return not _numeric_constructor_text(name, value)
        if isinstance(value, dict):
            return any(
                contains_text(item, name) for item in (*value.keys(), *value.values())
            )
        if isinstance(value, (tuple, list, set, frozenset)):
            return any(contains_text(item, name) for item in value)
        return False

    def checked(name: str) -> Any:
        fn = getattr(sp, name)

        def call(*args: Any, **kwargs: Any) -> Any:
            # Many mathematical APIs sympify strings internally, including aliases
            # held in local variables that cannot be checked as literal AST arguments.
            if name not in {"Symbol", "symbols"} and (
                contains_text(args, name)
                or any(contains_text(value, name) for value in kwargs.values())
            ):
                raise UnsupportedAnswerOperation(
                    f"String arguments to sp.{name} require text-only comparison."
                )
            return fn(*args, **kwargs)

        return call

    return SimpleNamespace(
        **{name: checked(name) for name in _ALLOWED_SYMPY_CALLS},
        **{name: getattr(sp, name) for name in _SYMPY_CONSTANT_NAMES},
    )


def _validate_expression_call_names(value: str) -> None:
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", value))
    if any(name.startswith("_") or name in _UNSAFE_CALL_NAMES for name in names):
        raise ValueError(
            "expression may not use private names or code execution operations"
        )
    try:
        tree = ast.parse(value, mode="eval")
    except SyntaxError:
        tree = None  # SymPy also accepts implicit multiplication syntax.
        if "'" in value or '"' in value:
            raise UnsupportedAnswerOperation(
                "Quoted implicit expressions require text-only comparison."
            )
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, _FORBIDDEN_AST_NODES) or isinstance(
                node, ast.NamedExpr
            ):
                raise ValueError("expression contains unsupported syntax")
            if isinstance(node, ast.Call) and _string_call_requires_review(node):
                raise UnsupportedAnswerOperation(
                    "String arguments require text-only comparison."
                )
            if isinstance(node, ast.Attribute):
                if not isinstance(node.value, ast.Name) or node.value.id not in {
                    "sp",
                    "sympy",
                }:
                    raise UnsupportedAnswerOperation(
                        "Expression methods require text-only comparison."
                    )
    called_names = {
        match.group(1) for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", value)
    }
    unsupported = sorted(called_names - _ALLOWED_SYMPY_CALLS)
    if unsupported:
        raise UnsupportedAnswerOperation(
            "expression contains unsupported or undefined function calls: "
            f"{unsupported}"
        )


_SYMPY_CONSTANT_NAMES = {
    "Catalan",
    "E",
    "EulerGamma",
    "GoldenRatio",
    "I",
    "nan",
    "oo",
    "pi",
    "zoo",
}


def _validate_code(tree: ast.Module, *, allow_unsupported_calls: bool = False) -> None:
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


def _validate_import(node: ast.Import) -> None:
    for alias in node.names:
        if alias.name != "sympy" or alias.asname not in {None, "sp"}:
            raise ValueError("Python answer block may only import sympy as sp.")


def _is_allowed_call(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id in {"sp", "sympy"} and node.attr in _ALLOWED_SYMPY_CALLS
    return False


def _validate_floating_literal_returns(
    node: ast.FunctionDef,
    grading_plan: GradingPlan,
) -> None:
    floating_targets = {
        call.target
        for call in grading_plan.calls
        if call.primitive == "floating_numeric"
    }
    if not floating_targets:
        return

    returns = [node for node in ast.walk(node) if isinstance(node, ast.Return)]
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
        _validate_floating_literal_returns(answer_defs[0], grading_plan)

    _validate_code(tree, allow_unsupported_calls=allow_unsupported_calls)
    return tree
