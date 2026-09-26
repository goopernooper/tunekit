from __future__ import annotations

import itertools
from typing import Any

from tunekit.core.search_space import (
    Categorical,
    IntUniform,
    LogUniform,
    SearchSpace,
    Uniform,
)
from tunekit.core.strategy import SearchStrategy


class GridSearch(SearchStrategy):
    """Exhaustive grid search over a discretized search space.

    Continuous parameters are split into `resolution` evenly-spaced points.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        resolution: int = 10,
        seed: int | None = None,
    ) -> None:
        super().__init__(search_space, seed)
        self.resolution = resolution
        self._grid = self._build_grid()
        self._index = 0

    def _build_grid(self) -> list[dict[str, Any]]:
        axes: list[list[Any]] = []
        for param in self.search_space.parameters:
            if isinstance(param, Categorical):
                axes.append(param.choices)
            elif isinstance(param, IntUniform):
                step = max(1, (param.high - param.low) // (self.resolution - 1))
                axes.append(list(range(param.low, param.high + 1, step)))
            elif isinstance(param, LogUniform):
                import math
                log_low = math.log(param.low)
                log_high = math.log(param.high)
                axes.append([
                    math.exp(log_low + i * (log_high - log_low) / (self.resolution - 1))
                    for i in range(self.resolution)
                ])
            elif isinstance(param, Uniform):
                axes.append([
                    param.low + i * (param.high - param.low) / (self.resolution - 1)
                    for i in range(self.resolution)
                ])
            else:
                raise TypeError(f"Unsupported parameter type: {type(param)}")

        names = self.search_space.dimension_names
        return [dict(zip(names, combo)) for combo in itertools.product(*axes)]

    def suggest(self) -> dict[str, Any]:
        if self._index >= len(self._grid):
            self._index = 0
        config = self._grid[self._index]
        self._index += 1
        return config

    @property
    def total_combinations(self) -> int:
        return len(self._grid)

    @property
    def name(self) -> str:
        return "grid_search"
