"""Framework-independent contract for response verifiers."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class Reward(ABC):
    """Score one generated response; concrete physics verifiers are not bundled."""

    @abstractmethod
    def __call__(
        self,
        *,
        response: str,
        sample_id: str,
        reference_completion: str,
        metadata: Mapping[str, Any],
    ) -> float:
        """Return a finite score, with larger values indicating better responses.

        The reference is an unverified teacher completion. Implementations define
        the scale, verification evidence and handling of invalid responses.
        Verifier failures must not silently become correctness scores.
        """
        raise NotImplementedError
