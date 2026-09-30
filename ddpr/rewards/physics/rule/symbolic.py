from __future__ import annotations

from itertools import product
from typing import Any

from ..schemas import PrimitiveGrade, unsupported_grading_args
from .utils import (
    _grade,
    _parse_expression,
    _text_only_grade,
    _validate_closed_form_expression,
)


def grade_symbolic(
    answer: Any,
    target: Any,
    **kwargs: Any,
) -> PrimitiveGrade:
    unsupported_args = unsupported_grading_args("symbolic", kwargs)
    if unsupported_args:
        raise ValueError(f"unsupported symbolic args: {unsupported_args}")
    if not isinstance(target, str) or not target.strip():
        raise ValueError(
            "symbolic expected value must be a non-empty expression string"
        )
    import sympy as sp

    variables = kwargs.get("variables")
    if variables is not None and (
        not isinstance(variables, list | tuple)
        or any(
            not isinstance(name, str) or not name.isidentifier() for name in variables
        )
    ):
        raise ValueError("symbolic variables must be valid Python identifiers")

    declared_names = list(variables or [])
    raw_assumptions = kwargs.get("variable_assumptions", {})
    if not isinstance(raw_assumptions, dict):
        raise TypeError("symbolic variable_assumptions must be an object")

    allowed_assumptions = {
        "even",
        "finite",
        "integer",
        "negative",
        "nonnegative",
        "nonpositive",
        "nonzero",
        "odd",
        "positive",
        "real",
    }
    normalized_assumptions: dict[str, dict[str, bool]] = {}
    for name, assumptions in raw_assumptions.items():
        if name not in declared_names:
            raise ValueError(
                f"symbolic assumptions reference undeclared variable {name!r}"
            )
        if not isinstance(assumptions, dict) or any(
            key not in allowed_assumptions or type(value) is not bool
            for key, value in assumptions.items()
        ):
            raise ValueError(f"invalid symbolic assumptions for variable {name!r}")
        normalized_assumptions[name] = dict(assumptions)

    domain_constraints = kwargs.get("domain_constraints", [])
    if not isinstance(domain_constraints, list) or any(
        not isinstance(constraint, str) or not constraint.strip()
        for constraint in domain_constraints
    ):
        raise ValueError(
            "symbolic domain_constraints must be a list of non-empty strings"
        )

    symbols = {
        name: sp.Symbol(name, **normalized_assumptions.get(name, {}))
        for name in declared_names
    }
    reference_fallback = _text_only_grade("symbolic", target)
    if reference_fallback is not None:
        return _text_only_grade("symbolic", answer) or reference_fallback
    expected_expr = _parse_expression(target, sp, local_dict=symbols)
    _validate_closed_form_expression(expected_expr, sp, label="expected")
    if not expected_expr.free_symbols:
        raise ValueError("expected symbolic expression must contain a free variable")
    if declared_names:
        missing = sorted(
            {str(s) for s in expected_expr.free_symbols} - set(declared_names)
        )
        if missing:
            raise ValueError(f"symbolic variables omit expected symbols: {missing}")
    simplified_expected = sp.simplify(expected_expr)
    if not simplified_expected.free_symbols:
        raise ValueError("expected symbolic expression simplifies to a constant")
    fallback = _text_only_grade("symbolic", answer)
    if fallback is not None:
        return fallback
    try:
        actual_expr = _parse_expression(answer, sp, local_dict=symbols)
        _validate_closed_form_expression(actual_expr, sp, label="actual")
    except (SyntaxError, TypeError, ValueError) as exc:
        return _grade("symbolic", False, f"could not parse symbolic expression: {exc}")

    if not actual_expr.free_symbols:
        return _grade(
            "symbolic",
            False,
            "actual symbolic expression must contain a free variable",
        )
    # The reference has already been checked for disguised constants.
    if actual_expr == expected_expr:
        return _grade(
            "symbolic",
            True,
            "symbolic expressions are structurally identical",
            equivalence_candidate=False,
        )

    # SymPy represents Li_s(z) both as polylog(s, z) and, for integer s, as
    # hypergeometric forms such as z*hyper((1,1,1),(2,2),z).  Normalizing that
    # recognized function identity is much cheaper and more predictable than
    # asking general simplify to discover it inside a large expression.
    if actual_expr.has(sp.polylog, sp.hyper) or expected_expr.has(sp.polylog, sp.hyper):
        try:
            expanded_actual = sp.expand_func(actual_expr)
            expanded_expected = sp.expand_func(expected_expr)
        except Exception:
            pass
        else:
            if expanded_actual == expanded_expected:
                return _grade(
                    "symbolic",
                    True,
                    "symbolic special-function forms are structurally equivalent",
                    equivalence_candidate=False,
                )

    simplified_actual = sp.simplify(actual_expr)
    if not simplified_actual.free_symbols:
        return _grade(
            "symbolic",
            False,
            "actual symbolic expression simplifies to a value with no free variables",
        )
    comparison_actual = _canonicalize_piecewise_conditions(actual_expr, sp)
    comparison_expected = _canonicalize_piecewise_conditions(expected_expr, sp)
    if bool(kwargs.get("equivalent_up_to_constant_factor", False)):
        symbols = list(
            comparison_actual.free_symbols | comparison_expected.free_symbols
        )
        passed = _symbolic_constant_factor_equal(
            comparison_actual,
            comparison_expected,
            symbols,
            sp,
        )
    else:
        difference = sp.simplify(comparison_actual - comparison_expected)
        passed = difference == 0
        if not passed and comparison_expected != 0:
            passed = sp.simplify(comparison_actual / comparison_expected) == 1
        if not passed:
            passed = _symbolic_integer_parity_equal(
                comparison_actual,
                comparison_expected,
                sp,
            )

    reason = (
        "symbolic expressions are equivalent"
        if passed
        else "symbolic expressions differ"
    )
    return _grade(
        "symbolic",
        bool(passed),
        reason,
        equivalence_candidate=not bool(passed),
    )


def _symbolic_constant_factor_equal(
    answer: Any,
    target: Any,
    symbols: list[Any],
    sp: Any,
) -> bool:
    if target == 0:
        return sp.simplify(answer) == 0
    ratio = sp.simplify(answer / target)
    return ratio != 0 and not ratio.has(*symbols)


_MAX_PARITY_PROOF_VARIABLES = 6


def _symbolic_integer_parity_equal(
    answer: Any,
    target: Any,
    sp: Any,
) -> bool:
    integer_symbols = sorted(
        (answer.free_symbols | target.free_symbols),
        key=str,
    )
    integer_symbols = [
        symbol for symbol in integer_symbols if symbol.is_integer is True
    ]
    if not integer_symbols or len(integer_symbols) > _MAX_PARITY_PROOF_VARIABLES:
        return False

    quotient_symbols = {
        symbol: sp.Dummy(
            f"{symbol.name}_parity_quotient",
            integer=True,
        )
        for symbol in integer_symbols
    }
    residue_options = [
        (0,) if symbol.is_even is True else (1,) if symbol.is_odd is True else (0, 1)
        for symbol in integer_symbols
    ]
    for residues in product(*residue_options):
        replacements = {
            symbol: 2 * quotient_symbols[symbol] + residue
            for symbol, residue in zip(integer_symbols, residues, strict=True)
        }
        if sp.simplify((answer - target).xreplace(replacements)) != 0:
            return False
    return True


def _canonicalize_piecewise_conditions(expression: Any, sp: Any) -> Any:
    if not isinstance(expression, sp.Piecewise):
        return expression
    return sp.Piecewise(
        *[
            (
                _canonicalize_piecewise_conditions(branch, sp),
                _canonicalize_boolean_condition(condition, sp),
            )
            for branch, condition in expression.args
        ],
        evaluate=False,
    )


def _canonicalize_boolean_condition(condition: Any, sp: Any) -> Any:
    if condition in (True, False, sp.true, sp.false):
        return condition
    if condition.func in (sp.And, sp.Or):
        return condition.func(
            *[
                _canonicalize_boolean_condition(argument, sp)
                for argument in condition.args
            ]
        )
    if condition.func == sp.Not:
        return sp.Not(_canonicalize_boolean_condition(condition.args[0], sp))

    ordered_relations = (
        sp.StrictLessThan,
        sp.LessThan,
        sp.StrictGreaterThan,
        sp.GreaterThan,
    )
    if not isinstance(condition, ordered_relations):
        return condition

    lhs, rhs = condition.lhs, condition.rhs
    if (
        lhs.is_nonnegative is True
        and rhs.is_nonnegative is True
        and (lhs.has(sp.Abs) or rhs.has(sp.Abs))
    ):
        lhs = sp.simplify(lhs**2)
        rhs = sp.simplify(rhs**2)

    if isinstance(condition, (sp.StrictLessThan, sp.LessThan)):
        difference = lhs - rhs
    else:
        difference = rhs - lhs
    factored = sp.factor_terms(sp.simplify(difference))
    coefficient, remainder = factored.as_coeff_Mul()
    if coefficient.is_positive:
        normalized_difference = remainder
    elif coefficient.is_negative:
        normalized_difference = -remainder
    else:
        normalized_difference = factored

    relation = (
        sp.StrictLessThan
        if isinstance(condition, (sp.StrictLessThan, sp.StrictGreaterThan))
        else sp.LessThan
    )
    return relation(sp.simplify(normalized_difference), sp.Integer(0))
