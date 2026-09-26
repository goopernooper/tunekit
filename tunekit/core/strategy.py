from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tunekit.core.search_space import SearchSpace
from tunekit.core.trial import Trial


class SearchStrategy(ABC):
    """Base class for all search strategies.

    Subclasses implement `suggest` to propose the next set of hyperparameters
    and `update` to incorporate the result of a completed trial.
    """

    def __init__(self, search_space: SearchSpace, seed: int | None = None) -> None:
        self.search_space = search_space
        self.seed = seed

    @abstractmethod
    def suggest(self) -> dict[str, Any]:
        """Return the next hyperparameter configuration to evaluate."""
        ...

    def update(self, trial: Trial) -> None:
        """Incorporate a completed trial's result. Override for model-based strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...
