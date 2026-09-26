from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrialStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass
class Trial:
    params: dict[str, Any]
    trial_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: TrialStatus = TrialStatus.PENDING
    score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None

    def start(self) -> None:
        self.status = TrialStatus.RUNNING
        self.start_time = time.time()

    def complete(self, score: float, metrics: dict[str, Any] | None = None) -> None:
        self.status = TrialStatus.COMPLETED
        self.score = score
        self.end_time = time.time()
        if metrics:
            self.metrics.update(metrics)

    def fail(self, error: str | None = None) -> None:
        self.status = TrialStatus.FAILED
        self.end_time = time.time()
        if error:
            self.metrics["error"] = error

    def prune(self) -> None:
        self.status = TrialStatus.PRUNED
        self.end_time = time.time()

    @property
    def duration(self) -> float | None:
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time
