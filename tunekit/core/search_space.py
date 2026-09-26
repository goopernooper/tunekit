from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class HyperParameter(ABC):
    """Base class for a single hyperparameter dimension."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def sample(self, rng: random.Random) -> Any:
        ...

    @abstractmethod
    def contains(self, value: Any) -> bool:
        ...


class Categorical(HyperParameter):
    def __init__(self, name: str, choices: list[Any]) -> None:
        super().__init__(name)
        if not choices:
            raise ValueError("Categorical requires at least one choice")
        self.choices = choices

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.choices)

    def contains(self, value: Any) -> bool:
        return value in self.choices


class Uniform(HyperParameter):
    def __init__(self, name: str, low: float, high: float) -> None:
        super().__init__(name)
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")
        self.low = low
        self.high = high

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)

    def contains(self, value: Any) -> bool:
        return isinstance(value, (int, float)) and self.low <= value <= self.high


class LogUniform(HyperParameter):
    """Samples uniformly in log space — useful for learning rates, regularization."""

    def __init__(self, name: str, low: float, high: float) -> None:
        super().__init__(name)
        if low <= 0 or high <= 0:
            raise ValueError("LogUniform bounds must be positive")
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")
        self.low = low
        self.high = high

    def sample(self, rng: random.Random) -> float:
        log_low = math.log(self.low)
        log_high = math.log(self.high)
        return math.exp(rng.uniform(log_low, log_high))

    def contains(self, value: Any) -> bool:
        return isinstance(value, (int, float)) and self.low <= value <= self.high


class IntUniform(HyperParameter):
    def __init__(self, name: str, low: int, high: int) -> None:
        super().__init__(name)
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")
        self.low = low
        self.high = high

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.low, self.high)

    def contains(self, value: Any) -> bool:
        return isinstance(value, int) and self.low <= value <= self.high


@dataclass
class SearchSpace:
    """Defines the full hyperparameter search space as a collection of dimensions."""

    parameters: list[HyperParameter] = field(default_factory=list)

    def add(self, param: HyperParameter) -> SearchSpace:
        self.parameters.append(param)
        return self

    def sample(self, rng: random.Random | None = None) -> dict[str, Any]:
        rng = rng or random.Random()
        return {p.name: p.sample(rng) for p in self.parameters}

    @property
    def dimension_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    def __len__(self) -> int:
        return len(self.parameters)
