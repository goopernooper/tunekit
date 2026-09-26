from __future__ import annotations

from abc import ABC, abstractmethod

from tunekit.core.trial import Trial


class ExperimentTracker(ABC):
    """Abstract interface for persisting trial results."""

    @abstractmethod
    def log_trial(self, trial: Trial) -> None:
        ...

    @abstractmethod
    def get_all_trials(self) -> list[Trial]:
        ...

    @abstractmethod
    def get_best_trial(self, direction: str = "minimize") -> Trial | None:
        ...
