from __future__ import annotations

import ast
import inspect
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from types import SimpleNamespace
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

PrimitiveName = Literal["floating_numeric", "exact", "symbolic", "categorical"]

SUPPORTED_GRADING_ARGS: dict[PrimitiveName, frozenset[str]] = {
    "floating_numeric": frozenset({"abs_tol", "rel_tol"}),
    "exact": frozenset(),
    "symbolic": frozenset(
        {
            "domain_constraints",
            "variables",
            "variable_assumptions",
            "equivalent_up_to_constant_factor",
        }
    ),
    "categorical": frozenset(
        {
            "allowed_aliases",
            "allowed_choices",
            "case_sensitive",
            "choices",
        }
    ),
}


def unsupported_grading_args(
    primitive: PrimitiveName,
    args: dict[str, Any],
) -> list[str]:
    """Return grading-call argument names the primitive does not implement."""

    return sorted(set(args) - SUPPORTED_GRADING_ARGS[primitive])


class PrecisionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["significant_digits", "decimal_places"]
    digits: int = Field(ge=0, le=15)

    @model_validator(mode="after")
    def validate_digits(self) -> Self:
        if self.kind == "significant_digits" and self.digits < 1:
            raise ValueError("significant_digits precision must use at least one digit")
        return self


class GradingCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primitive: PrimitiveName
    target: str
    expected: Any
    args: dict[str, Any] = Field(default_factory=dict)
    precision: PrecisionSpec | None = None
    weight: float = 1.0
    required: bool = True

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        unsupported_args = unsupported_grading_args(self.primitive, self.args)
        if unsupported_args:
            raise ValueError(
                f"{self.primitive} grading call has unsupported args: "
                + ", ".join(unsupported_args)
            )
        if self.primitive == "floating_numeric" and self.precision is None:
            raise ValueError("floating_numeric grading calls require precision")
        if self.primitive != "floating_numeric" and self.precision is not None:
            raise ValueError(
                "precision is only valid for floating_numeric grading calls"
            )
        return self

    @model_serializer(mode="wrap")
    def serialize_call(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        data = handler(self)
        if self.precision is None:
            data.pop("precision", None)
        return data


class GradingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calls: list[GradingCall] = Field(min_length=1)
    composition: Literal["all_required", "weighted"] = "all_required"


class PrimitiveGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primitive: PrimitiveName
    target: str
    passed: bool
    score: float
    reason: str = ""
    equivalence_candidate: bool = False
    deterministic_skip_reason: str | None = None


class ExactNumericComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    status: Literal["match", "different", "unavailable"]
    significant_digits: int = Field(default=16, ge=1, le=50)
    actual_decimal: str | None = None
    expected_decimal: str | None = None
    relative_difference: str | None = None
    reason: str


class GradingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    score: float
    calls: list[PrimitiveGrade]


_PYTHON_FENCE = re.compile(r"```(?:python|py)\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
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


@dataclass(frozen=True)
class UnexecutedAnswer:
    expression: str
    answer_code: str
    reason: str

    def __str__(self) -> str:
        return self.expression


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


def grade_floating_numeric(
    actual: Any,
    expected: Any,
    args: dict[str, Any] | None = None,
) -> PrimitiveGrade:
    args = args or {}
    unsupported_args = unsupported_grading_args("floating_numeric", args)
    if unsupported_args:
        return _grade(
            "floating_numeric",
            "",
            False,
            f"unsupported floating_numeric args: {unsupported_args}",
        )
    if not ({"abs_tol", "rel_tol"} & args.keys()):
        return _grade(
            "floating_numeric",
            "",
            False,
            "floating_numeric grading requires abs_tol or rel_tol",
        )
    raw_abs_tol = args.get("abs_tol", 0.0)
    raw_rel_tol = args.get("rel_tol", 0.0)
    if any(
        isinstance(value, bool) or not isinstance(value, int | float)
        for value in (raw_abs_tol, raw_rel_tol)
    ):
        return _grade(
            "floating_numeric",
            "",
            False,
            "floating tolerances must be JSON numbers",
        )
    abs_tol = float(raw_abs_tol)
    rel_tol = float(raw_rel_tol)
    if (
        not math.isfinite(abs_tol)
        or not math.isfinite(rel_tol)
        or abs_tol < 0
        or rel_tol < 0
    ):
        return _grade(
            "floating_numeric",
            "",
            False,
            "floating tolerances must be finite and non-negative",
        )
    if type(actual) is not float or not math.isfinite(actual):
        return _grade(
            "floating_numeric",
            "",
            False,
            "actual value must be a finite Python float",
        )
    if type(expected) is not float or not math.isfinite(expected):
        return _grade(
            "floating_numeric",
            "",
            False,
            "expected value must be a finite Python float",
        )

    passed = math.isclose(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol)
    reason = "floating values match" if passed else "floating values differ"
    return _grade("floating_numeric", "", passed, reason)


def grade_exact(
    actual: Any,
    expected: Any,
    args: dict[str, Any] | None = None,
) -> PrimitiveGrade:
    args = args or {}
    unsupported_args = unsupported_grading_args("exact", args)
    if unsupported_args:
        return _grade(
            "exact",
            "",
            False,
            f"unsupported exact args: {unsupported_args}",
        )
    if not isinstance(expected, str) or not expected.strip():
        return _grade(
            "exact",
            "",
            False,
            "exact expected value must be a non-empty expression string",
        )
    fallback = _text_only_grade("exact", actual)
    if fallback is not None:
        return fallback
    try:
        import sympy as sp

        actual_expr = _parse_expression(actual, sp)
        expected_expr = _parse_expression(expected, sp)
        _validate_closed_form_expression(actual_expr, sp, label="actual")
        _validate_closed_form_expression(expected_expr, sp, label="expected")
        _validate_exact_expression(actual_expr, sp, label="actual")
        _validate_exact_expression(expected_expr, sp, label="expected")
        difference = sp.simplify(actual_expr - expected_expr)
        passed = difference == 0
        if not passed:
            passed = difference.equals(0) is True
    except (TypeError, ValueError) as exc:
        return _grade("exact", "", False, str(exc))
    except Exception as exc:
        return _grade("exact", "", False, f"could not compare exact expressions: {exc}")

    reason = "exact values are equivalent" if passed else "exact values differ"
    return _grade(
        "exact",
        "",
        bool(passed),
        reason,
        equivalence_candidate=not bool(passed),
    )


def compare_exact_numerically(
    actual: Any,
    expected: Any,
    *,
    target: str,
    significant_digits: int = 16,
) -> ExactNumericComparison:
    """Return supporting numerical evidence for two validated exact constants."""

    if not 1 <= significant_digits <= 50:
        raise ValueError("significant_digits must be between 1 and 50")
    try:
        import sympy as sp

        actual_expr = _parse_expression(actual, sp)
        expected_expr = _parse_expression(expected, sp)
        _validate_closed_form_expression(actual_expr, sp, label="actual")
        _validate_closed_form_expression(expected_expr, sp, label="expected")
        _validate_exact_expression(actual_expr, sp, label="actual")
        _validate_exact_expression(expected_expr, sp, label="expected")

        working_digits = max(32, significant_digits + 16)
        actual_numeric = sp.N(actual_expr, working_digits)
        expected_numeric = sp.N(expected_expr, working_digits)
        if (
            actual_numeric.is_real is not True
            or expected_numeric.is_real is not True
            or actual_numeric.is_finite is not True
            or expected_numeric.is_finite is not True
        ):
            raise ValueError("numerical cross-check requires finite real constants")

        absolute_actual = sp.Abs(actual_numeric)
        absolute_expected = sp.Abs(expected_numeric)
        scale = (
            absolute_actual
            if absolute_actual >= absolute_expected
            else absolute_expected
        )
        difference = sp.Abs(actual_numeric - expected_numeric)
        relative_difference = (
            sp.Integer(0) if scale == 0 else sp.N(difference / scale, working_digits)
        )
        threshold = sp.Float(5, working_digits) * sp.Pow(
            10,
            -significant_digits,
        )
        matched = bool(relative_difference <= threshold)
        return ExactNumericComparison(
            target=target,
            status="match" if matched else "different",
            significant_digits=significant_digits,
            actual_decimal=str(sp.N(actual_numeric, significant_digits + 2)),
            expected_decimal=str(sp.N(expected_numeric, significant_digits + 2)),
            relative_difference=str(sp.N(relative_difference, significant_digits + 2)),
            reason=(
                f"values agree to {significant_digits} significant digits"
                if matched
                else f"values differ within a {significant_digits}-digit check"
            ),
        )
    except Exception as exc:
        return ExactNumericComparison(
            target=target,
            status="unavailable",
            significant_digits=significant_digits,
            reason=f"numeric exact-value comparison unavailable: {exc}",
        )


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


def grade_with_plan(
    answer_block: dict[str, Any],
    grading_plan: GradingPlan,
) -> GradingResult:
    grades: list[PrimitiveGrade] = []
    for call in grading_plan.calls:
        if call.target not in answer_block:
            grades.append(
                PrimitiveGrade(
                    primitive=call.primitive,
                    target=call.target,
                    passed=False,
                    score=0.0,
                    reason="target missing from answer block",
                )
            )
            continue
        actual = answer_block[call.target]
        if call.primitive == "floating_numeric":
            grade = grade_floating_numeric(actual, call.expected, call.args)
        elif call.primitive == "exact":
            grade = grade_exact(actual, call.expected, call.args)
        elif call.primitive == "symbolic":
            grade = grade_symbolic(actual, call.expected, call.args)
        else:
            grade = grade_categorical(actual, call.expected, call.args)
        grades.append(grade.model_copy(update={"target": call.target}))

    return compose_grades(grading_plan, grades)


def compose_grades(
    grading_plan: GradingPlan,
    grades: list[PrimitiveGrade],
) -> GradingResult:
    if len(grades) != len(grading_plan.calls):
        raise ValueError("grades must align with grading_plan calls")
    required_failed = any(
        call.required and not grade.passed
        for call, grade in zip(grading_plan.calls, grades, strict=True)
    )
    total_weight = sum(max(call.weight, 0.0) for call in grading_plan.calls)
    if total_weight <= 0:
        score = 1.0 if all(grade.passed for grade in grades) else 0.0
    else:
        score = (
            sum(
                max(call.weight, 0.0) * grade.score
                for call, grade in zip(grading_plan.calls, grades, strict=True)
            )
            / total_weight
        )
    passed = (not required_failed) and (
        score >= 1.0 if grading_plan.composition == "all_required" else score > 0
    )
    return GradingResult(passed=passed, score=score, calls=grades)


def _grade(
    primitive: PrimitiveName,
    target: str,
    passed: bool,
    reason: str,
    *,
    equivalence_candidate: bool = False,
) -> PrimitiveGrade:
    return PrimitiveGrade(
        primitive=primitive,
        target=target,
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
