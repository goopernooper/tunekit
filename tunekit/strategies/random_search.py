from __future__ import annotations

import random
from typing import Any

from tunekit.core.search_space import SearchSpace
from tunekit.core.strategy import SearchStrategy


class RandomSearch(SearchStrategy):
    def __init__(self, search_space: SearchSpace, seed: int | None = None) -> None:
        super().__init__(search_space, seed)
        self._rng = random.Random(seed)

    def suggest(self) -> dict[str, Any]:
        return self.search_space.sample(self._rng)

    @property
    def name(self) -> str:
        return "random_search"
