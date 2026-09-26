from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

from tunekit.core.objective import ObjectiveDirection
from tunekit.core.strategy import SearchStrategy
from tunekit.core.trial import Trial, TrialStatus
from tunekit.tracking.tracker import ExperimentTracker

logger = logging.getLogger(__name__)

ObjectiveFn = Callable[[Dict[str, Any]], Union[float, Tuple[float, Dict[str, Any]]]]
EarlyStopFn = Callable[["Optimizer"], bool]


def no_improvement_stopping(patience: int) -> EarlyStopFn:
    """Stop after `patience` consecutive trials with no improvement."""
    def check(optimizer: Optimizer) -> bool:
        completed = [t for t in optimizer.trials if t.status == TrialStatus.COMPLETED]
        if len(completed) < patience + 1:
            return False
        recent = completed[-patience:]
        best_before = optimizer._best_score(completed[:-patience])
        if optimizer.direction == ObjectiveDirection.MINIMIZE:
            return all(t.score >= best_before for t in recent)
        return all(t.score <= best_before for t in recent)
    return check


class Optimizer:
    def __init__(
        self,
        strategy: SearchStrategy,
        objective: ObjectiveFn,
        direction: ObjectiveDirection = ObjectiveDirection.MINIMIZE,
        tracker: ExperimentTracker | None = None,
        early_stop: EarlyStopFn | None = None,
    ) -> None:
        self.strategy = strategy
        self.objective = objective
        self.direction = direction
        self.tracker = tracker
        self.early_stop = early_stop
        self.trials: list[Trial] = []

    def _best_score(self, trials: list[Trial]) -> float:
        scores = [t.score for t in trials if t.status == TrialStatus.COMPLETED]
        if not scores:
            return float("inf") if self.direction == ObjectiveDirection.MINIMIZE else float("-inf")
        return min(scores) if self.direction == ObjectiveDirection.MINIMIZE else max(scores)

    def run(self, n_trials: int) -> Trial:
        """Run up to `n_trials` evaluations and return the best trial."""
        for i in range(n_trials):
            params = self.strategy.suggest()
            trial = Trial(params=params)
            trial.start()

            try:
                result = self.objective(params)
                if isinstance(result, tuple):
                    score, metrics = result
                else:
                    score, metrics = result, {}
                trial.complete(score, metrics)
            except Exception as exc:
                trial.fail(str(exc))
                logger.warning("Trial %s failed: %s", trial.trial_id, exc)

            self.strategy.update(trial)
            self.trials.append(trial)

            if self.tracker:
                self.tracker.log_trial(trial)

            if trial.status == TrialStatus.COMPLETED:
                logger.info(
                    "Trial %d/%d [%s] score=%.6f params=%s",
                    i + 1, n_trials, trial.trial_id, trial.score, trial.params,
                )

            if self.early_stop and self.early_stop(self):
                logger.info("Early stopping triggered at trial %d/%d", i + 1, n_trials)
                break

        return self.best_trial

    @property
    def best_trial(self) -> Trial:
        completed = [t for t in self.trials if t.status == TrialStatus.COMPLETED]
        if not completed:
            raise RuntimeError("No completed trials")
        if self.direction == ObjectiveDirection.MINIMIZE:
            return min(completed, key=lambda t: t.score)
        return max(completed, key=lambda t: t.score)
