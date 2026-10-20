"""Optional CritPt-AI rewards, installed with ddpr[critpt-ai]."""

from ddsr_bench.benchmarks.critpt.evaluation.ai.grader import grade_response

from .reward import CritptAiReward, CritptAiRewardError, load_problem

__all__ = ["CritptAiReward", "CritptAiRewardError", "grade_response", "load_problem"]
