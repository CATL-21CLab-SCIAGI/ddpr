from __future__ import annotations

from itertools import product
from typing import Any

from ..schemas import PrimitiveGrade, unsupported_grading_args
from .common import _grade, _text_only_grade
from .expressions import _parse_expression, _validate_closed_form_expression


def grade_symbolic(
    actual: Any,
    expected: Any,
    args: dict[str, Any] | None = None,
) -> PrimitiveGrade:
    args = args or {}
    unsupported_args = unsupported_grading_args("symbolic", args)
    if unsupported_args:
        return _grade(
            "symbolic",
            "",
            False,
            f"unsupported symbolic args: {unsupported_args}",
        )
    if not isinstance(expected, str) or not expected.strip():
        return _grade(
            "symbolic",
            "",
            False,
            "symbolic expected value must be a non-empty expression string",
        )
    fallback = _text_only_grade("symbolic", actual)
    if fallback is not None:
        return fallback
    try:
        import sympy as sp
    except ImportError as exc:  # pragma: no cover - dependency is in project metadata.
        return _grade("symbolic", "", False, f"sympy is not installed: {exc}")

    variables = args.get("variables")
    if variables is not None and (
        not isinstance(variables, list | tuple)
        or any(
            not isinstance(name, str) or not name.isidentifier() for name in variables
        )
    ):
        return _grade(
            "symbolic",
            "",
            False,
            "symbolic variables must be valid Python identifiers",
        )

    declared_names = list(variables or [])
    raw_assumptions = args.get("variable_assumptions", {})
    if not isinstance(raw_assumptions, dict):
        return _grade(
            "symbolic",
            "",
            False,
            "symbolic variable_assumptions must be an object",
        )

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
            return _grade(
                "symbolic",
                "",
                False,
                f"symbolic assumptions reference undeclared variable {name!r}",
            )
        if not isinstance(assumptions, dict) or any(
            key not in allowed_assumptions or type(value) is not bool
            for key, value in assumptions.items()
        ):
            return _grade(
                "symbolic",
                "",
                False,
                f"invalid symbolic assumptions for variable {name!r}",
            )
        normalized_assumptions[name] = dict(assumptions)

    domain_constraints = args.get("domain_constraints", [])
    if not isinstance(domain_constraints, list) or any(
        not isinstance(constraint, str) or not constraint.strip()
        for constraint in domain_constraints
    ):
        return _grade(
            "symbolic",
            "",
            False,
            "symbolic domain_constraints must be a list of non-empty strings",
        )

    try:
        initial_symbols = {
            name: sp.Symbol(name, **normalized_assumptions.get(name, {}))
            for name in declared_names
        }
        actual_expr = _parse_expression(actual, sp, local_dict=initial_symbols)
        actual_expr = actual_expr.xreplace(
            {
                symbol: initial_symbols[str(symbol)]
                for symbol in actual_expr.free_symbols
                if str(symbol) in initial_symbols
            }
        )
        actual_symbols = {str(symbol): symbol for symbol in actual_expr.free_symbols}
        all_names = set(declared_names) | set(actual_symbols)
        local_dict = {
            name: actual_symbols.get(name, initial_symbols.get(name, sp.Symbol(name)))
            for name in all_names
        }
        expected_expr = _parse_expression(expected, sp, local_dict=local_dict)
        _validate_closed_form_expression(actual_expr, sp, label="actual")
        _validate_closed_form_expression(expected_expr, sp, label="expected")
    except (SyntaxError, TypeError, ValueError) as exc:
        return _grade(
            "symbolic",
            "",
            False,
            f"could not parse symbolic expression: {exc}",
        )
    except Exception as exc:
        return _grade(
            "symbolic",
            "",
            False,
            f"could not parse symbolic expression: {exc}",
        )

    if not actual_expr.free_symbols:
        return _grade(
            "symbolic",
            "",
            False,
            "actual symbolic expression must contain a free variable",
        )
    if not expected_expr.free_symbols:
        return _grade(
            "symbolic",
            "",
            False,
            "expected symbolic expression must contain a free variable",
        )

    if declared_names:
        expected_names = {str(symbol) for symbol in expected_expr.free_symbols}
        missing = sorted(expected_names - set(declared_names))
        if missing:
            return _grade(
                "symbolic",
                "",
                False,
                f"symbolic variables omit expected symbols: {missing}",
            )

    # SymPy expressions use structural equality for ``==``.  This common path
    # must run before any simplify call: simplifying an already identical but
    # pathological expression can consume unbounded CPU and memory.
    if actual_expr == expected_expr:
        if not _has_observable_symbolic_dependence(actual_expr, sp):
            return _grade(
                "symbolic",
                "",
                False,
                (
                    "actual symbolic expression simplifies to a value with "
                    "no free variables"
                ),
            )
        return _grade(
            "symbolic",
            "",
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
                if not _has_observable_symbolic_dependence(
                    expanded_actual,
                    sp,
                ):
                    return _grade(
                        "symbolic",
                        "",
                        False,
                        (
                            "actual symbolic expression simplifies to a value "
                            "with no free variables"
                        ),
                    )
                return _grade(
                    "symbolic",
                    "",
                    True,
                    "symbolic special-function forms are structurally equivalent",
                    equivalence_candidate=False,
                )

    try:
        simplified_actual = sp.simplify(actual_expr)
        simplified_expected = sp.simplify(expected_expr)
    except Exception as exc:
        return _grade(
            "symbolic",
            "",
            False,
            f"could not simplify symbolic expression: {exc}",
        )
    if not simplified_actual.free_symbols:
        return _grade(
            "symbolic",
            "",
            False,
            "actual symbolic expression simplifies to a value with no free variables",
        )
    if not simplified_expected.free_symbols:
        return _grade(
            "symbolic",
            "",
            False,
            "expected symbolic expression simplifies to a value with no free variables",
        )
    try:
        comparison_actual = _canonicalize_piecewise_conditions(actual_expr, sp)
        comparison_expected = _canonicalize_piecewise_conditions(expected_expr, sp)
        if bool(args.get("equivalent_up_to_constant_factor", False)):
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
    except Exception as exc:
        return _grade(
            "symbolic",
            "",
            False,
            f"could not compare symbolic expressions: {exc}",
        )

    reason = (
        "symbolic expressions are equivalent"
        if passed
        else "symbolic expressions differ"
    )
    return _grade(
        "symbolic",
        "",
        bool(passed),
        reason,
        equivalence_candidate=not bool(passed),
    )


def _has_observable_symbolic_dependence(expression: Any, sp: Any) -> bool:
    """Cheaply reject disguised constants without invoking ``simplify``.

    The structural-identity fast path exists specifically to avoid expensive
    symbolic normalization.  A few deterministic rational substitutions keep
    the prior guard against answers such as ``sin(x)**2 + cos(x)**2`` while
    preserving that no simplify call occurs on this path.  If numeric
    evaluation is unsupported, retain the expression rather than introducing
    a false rejection; the enclosing deterministic audit is separately
    isolated behind a hard resource deadline.
    """

    symbols = sorted(expression.free_symbols, key=lambda item: str(item))
    if not symbols:
        return False
    if expression.has(sp.floor, sp.ceiling, sp.Piecewise):
        return True
    samples: list[Any] = []
    # Non-integer rational points avoid aliasing periodic expressions such as
    # tan(pi*x), which vanishes at every integer despite depending on x.
    sample_points = ((2, 7), (5, 11), (11, 17))
    for numerator, denominator in sample_points:
        substitutions = {
            symbol: sp.Rational(numerator + index, denominator + 2 * index)
            for index, symbol in enumerate(symbols)
        }
        try:
            value = sp.N(expression.subs(substitutions), 30)
            if value.free_symbols or value.has(sp.nan, sp.zoo):
                return True
            samples.append(complex(value))
        except (TypeError, ValueError, OverflowError):
            return True
    baseline = samples[0]
    return any(
        abs(value - baseline) > 1e-12 * max(1.0, abs(value), abs(baseline))
        for value in samples[1:]
    )


def _symbolic_constant_factor_equal(
    actual: Any,
    expected: Any,
    symbols: list[Any],
    sp: Any,
) -> bool:
    if expected == 0:
        return sp.simplify(actual) == 0
    ratio = sp.simplify(actual / expected)
    return ratio != 0 and not ratio.has(*symbols)


_MAX_PARITY_PROOF_VARIABLES = 6


def _symbolic_integer_parity_equal(
    actual: Any,
    expected: Any,
    sp: Any,
) -> bool:
    integer_symbols = sorted(
        (actual.free_symbols | expected.free_symbols),
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
        if sp.simplify((actual - expected).xreplace(replacements)) != 0:
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
