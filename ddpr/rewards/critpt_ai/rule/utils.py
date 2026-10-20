"""Mathematical parsing and shared rule-result helpers."""

from __future__ import annotations

import re
from typing import Any

from ..schemas import PrimitiveGrade, PrimitiveName, UnexecutedAnswer
from ..validation import (
    _ALLOWED_SYMPY_CALLS,
    _SYMPY_CONSTANT_NAMES,
    UnsupportedAnswerOperation,
    _sympy_execution_namespace,
    _validate_expression_call_names,
)


def _grade(
    primitive: PrimitiveName,
    passed: bool,
    reason: str,
    *,
    equivalence_candidate: bool = False,
) -> PrimitiveGrade:
    return PrimitiveGrade(
        primitive=primitive,
        target="",
        passed=passed,
        score=1.0 if passed else 0.0,
        equivalence_candidate=equivalence_candidate,
        reason=reason,
    )


def _text_only_grade(primitive: PrimitiveName, actual: Any) -> PrimitiveGrade | None:
    reason = actual.reason if isinstance(actual, UnexecutedAnswer) else None
    if isinstance(actual, str):
        try:
            _validate_expression_call_names(actual)
        except UnsupportedAnswerOperation as exc:
            reason = str(exc)
        except ValueError:
            return None
    if reason is None:
        return None
    return PrimitiveGrade(
        primitive=primitive,
        target="",
        passed=False,
        score=0.0,
        reason="Local evaluation skipped: " + reason,
        equivalence_candidate=True,
        deterministic_skip_reason=reason,
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

    if isinstance(value, bool | float):
        raise TypeError("expression values may not be booleans or Python floats")
    if isinstance(value, sp.Expr):
        symbols = local_dict or {}
        return value.xreplace(
            {s: symbols.get(str(s), sp.Symbol(str(s))) for s in value.free_symbols}
        )
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
    symbols = local_dict or {}
    return expression.xreplace(
        {s: symbols.get(str(s), sp.Symbol(str(s))) for s in expression.free_symbols}
    )


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
