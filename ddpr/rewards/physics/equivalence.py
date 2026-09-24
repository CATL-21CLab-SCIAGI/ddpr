"""Optional fallback for Auto exact/symbolic expressions, not a new answer type."""

from ._core.equivalence import (
    EquivalenceJudge,
    EquivalenceJudgeConfig,
    ExpressionEquivalenceJudge,
)

__all__ = ["EquivalenceJudge", "EquivalenceJudgeConfig", "ExpressionEquivalenceJudge"]
