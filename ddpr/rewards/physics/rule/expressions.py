from __future__ import annotations

import ast
import re
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

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


def allowed_sympy_calls() -> tuple[str, ...]:
    """Internal execution vocabulary, not a public mathematical answer contract."""
    return tuple(sorted(_ALLOWED_SYMPY_CALLS))


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
        function = getattr(sp, name)

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
            return function(*args, **kwargs)

        return call

    return SimpleNamespace(
        **{name: checked(name) for name in _ALLOWED_SYMPY_CALLS},
        **{name: getattr(sp, name) for name in _SYMPY_CONSTANT_NAMES},
    )


def _parse_expression(
    value: Any,
    sp: Any,
    *,
    local_dict: dict[str, Any] | None = None,
) -> Any:
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("expression values may not be booleans or Python floats")
    if isinstance(value, sp.Expr):
        return value
    if isinstance(value, int):
        return sp.Integer(value)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            "expression value must be an integer, SymPy expression, or non-empty string"
        )
    _validate_expression_call_names(value)
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    expression_locals = dict(local_dict or {})
    for name in _bare_expression_identifiers(value):
        expression_locals.setdefault(name, sp.Symbol(name))
    allowed_sp = _sympy_execution_namespace(sp)
    expression = parse_expr(
        value,
        local_dict=expression_locals,
        global_dict={
            "__builtins__": {},
            **{
                name: getattr(allowed_sp, name)
                for name in _ALLOWED_SYMPY_CALLS | _SYMPY_CONSTANT_NAMES
            },
            "sp": allowed_sp,
            "sympy": allowed_sp,
        },
        transformations=transformations,
    )
    if not isinstance(expression, sp.Expr):
        raise TypeError("expression value must be a scalar SymPy expression")
    return expression


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


def _bare_expression_identifiers(value: str) -> set[str]:
    names: set[str] = set()
    for match in re.finditer(r"\b[A-Za-z_]\w*\b", value):
        name = match.group(0)
        before = value[: match.start()].rstrip()
        after = value[match.end() :].lstrip()
        if name in _SYMPY_CONSTANT_NAMES:
            continue
        if (before and before[-1] == ".") or (after and after[0] in ".("):
            continue
        names.add(name)
    return names


def _validate_closed_form_expression(
    expression: Any,
    sp: Any,
    *,
    label: str,
) -> None:
    from sympy.core.function import AppliedUndef

    unsupported = (
        sp.Integral,
        sp.Derivative,
        sp.Limit,
        sp.Sum,
        sp.Product,
    )
    present = [kind.__name__ for kind in unsupported if expression.has(kind)]
    if present:
        raise ValueError(
            f"{label} expression contains unevaluated operations: {present}"
        )
    undefined = sorted(str(item.func) for item in expression.atoms(AppliedUndef))
    if undefined:
        raise ValueError(
            f"{label} expression contains undefined functions or operators: {undefined}"
        )
    if expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        raise ValueError(f"{label} expression must be finite")


def _validate_exact_expression(
    expression: Any,
    sp: Any,
    *,
    label: str,
) -> None:
    if expression.free_symbols:
        names = sorted(str(symbol) for symbol in expression.free_symbols)
        raise ValueError(f"{label} exact value contains free variables: {names}")
    if expression.atoms(sp.Float):
        raise ValueError(f"{label} exact value contains floating-point terms")
