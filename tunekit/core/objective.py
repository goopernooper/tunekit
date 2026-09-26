from __future__ import annotations

from enum import Enum
from typing import Any, Protocol


class ObjectiveDirection(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class Objective(Protocol):
    """Protocol that any user-provided objective function must satisfy.

    The function receives a hyperparameter dict and returns either a float
    (the score) or a tuple of (score, extra_metrics_dict).
    """

    def __call__(self, params: dict[str, Any]) -> float | tuple[float, dict[str, Any]]:
        ...
