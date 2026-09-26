from __future__ import annotations

import math
import random
from typing import Any, Callable

from tunekit.core.search_space import SearchSpace
from tunekit.core.strategy import SearchStrategy
from tunekit.core.trial import Trial, TrialStatus

ReportFn = Callable[[dict[str, Any], int], float]


class Bracket:
    """A single Hyperband bracket running successive halving."""

    def __init__(
        self,
        configs: list[dict[str, Any]],
        max_resource: int,
        eta: int,
        n_rungs: int,
    ) -> None:
        self.configs = configs
        self.max_resource = max_resource
        self.eta = eta
        self.n_rungs = n_rungs
        self.current_rung = 0
        self._scores: list[tuple[dict[str, Any], float]] = []
        self.completed = False

    @property
    def current_resource(self) -> int:
        r = self.max_resource * self.eta ** (self.current_rung - self.n_rungs + 1)
        return max(1, int(r))

    @property
    def n_configs(self) -> int:
        return len(self.configs)

    def report(self, config: dict[str, Any], score: float) -> None:
        self._scores.append((config, score))

    def advance(self) -> None:
        """Promote the top 1/eta configs to the next rung."""
        self._scores.sort(key=lambda x: x[1])
        n_keep = max(1, len(self._scores) // self.eta)
        self.configs = [cfg for cfg, _ in self._scores[:n_keep]]
        self._scores = []
        self.current_rung += 1
        if self.current_rung >= self.n_rungs or len(self.configs) == 0:
            self.completed = True


class Hyperband(SearchStrategy):
    """Hyperband: adaptive resource allocation via successive halving brackets.

    The user provides a `report_fn(params, resource) -> score` instead of a
    simple objective. The `resource` argument controls training budget
    (epochs, data fraction, etc.).
    """

    def __init__(
        self,
        search_space: SearchSpace,
        report_fn: ReportFn,
        max_resource: int = 81,
        eta: int = 3,
        minimize: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(search_space, seed)
        self._rng = random.Random(seed)
        self.report_fn = report_fn
        self.max_resource = max_resource
        self.eta = eta
        self.minimize = minimize
        self._brackets: list[Bracket] = []
        self._all_trials: list[Trial] = []

    def _build_brackets(self) -> list[Bracket]:
        s_max = int(math.log(self.max_resource) / math.log(self.eta))
        brackets = []
        for s in range(s_max, -1, -1):
            n = int(math.ceil((s_max + 1) / (s + 1)) * self.eta**s)
            configs = [self.search_space.sample(self._rng) for _ in range(n)]
            brackets.append(Bracket(
                configs=configs,
                max_resource=self.max_resource,
                eta=self.eta,
                n_rungs=s + 1,
            ))
        return brackets

    def run_all(self) -> Trial:
        """Execute all Hyperband brackets and return the best trial."""
        self._brackets = self._build_brackets()
        for bracket in self._brackets:
            self._run_bracket(bracket)
        return self.best_trial

    def _run_bracket(self, bracket: Bracket) -> None:
        while not bracket.completed:
            resource = bracket.current_resource
            for config in bracket.configs:
                trial = Trial(params=config)
                trial.start()
                try:
                    score = self.report_fn(config, resource)
                    sort_score = -score if not self.minimize else score
                    bracket.report(config, sort_score)
                    trial.complete(score, {"resource": resource})
                except Exception as exc:
                    trial.fail(str(exc))
                self._all_trials.append(trial)
            bracket.advance()

    @property
    def best_trial(self) -> Trial:
        completed = [t for t in self._all_trials if t.status == TrialStatus.COMPLETED]
        if not completed:
            raise RuntimeError("No completed trials")
        if self.minimize:
            return min(completed, key=lambda t: t.score)
        return max(completed, key=lambda t: t.score)

    @property
    def all_trials(self) -> list[Trial]:
        return list(self._all_trials)

    def suggest(self) -> dict[str, Any]:
        return self.search_space.sample(self._rng)

    @property
    def name(self) -> str:
        return "hyperband"
