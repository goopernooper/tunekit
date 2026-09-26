from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tunekit.core.trial import Trial, TrialStatus
from tunekit.tracking.tracker import ExperimentTracker


class SQLiteTracker(ExperimentTracker):
    def __init__(self, db_path: str | Path = "experiments.db", experiment_name: str = "default") -> None:
        self.db_path = str(db_path)
        self.experiment_name = experiment_name
        self._conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                experiment TEXT NOT NULL,
                params TEXT NOT NULL,
                score REAL,
                status TEXT NOT NULL,
                metrics TEXT,
                start_time REAL,
                end_time REAL
            )
        """)
        self._conn.commit()

    def log_trial(self, trial: Trial) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO trials
               (trial_id, experiment, params, score, status, metrics, start_time, end_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trial.trial_id,
                self.experiment_name,
                json.dumps(trial.params),
                trial.score,
                trial.status.value,
                json.dumps(trial.metrics),
                trial.start_time,
                trial.end_time,
            ),
        )
        self._conn.commit()

    def get_all_trials(self) -> list[Trial]:
        cursor = self._conn.execute(
            "SELECT trial_id, params, score, status, metrics, start_time, end_time "
            "FROM trials WHERE experiment = ? ORDER BY start_time",
            (self.experiment_name,),
        )
        trials = []
        for row in cursor.fetchall():
            trial = Trial(
                params=json.loads(row[1]),
                trial_id=row[0],
                score=row[2],
                status=TrialStatus(row[3]),
                metrics=json.loads(row[4]) if row[4] else {},
                start_time=row[5],
                end_time=row[6],
            )
            trials.append(trial)
        return trials

    def get_best_trial(self, direction: str = "minimize") -> Trial | None:
        trials = [t for t in self.get_all_trials() if t.status == TrialStatus.COMPLETED]
        if not trials:
            return None
        if direction == "minimize":
            return min(trials, key=lambda t: t.score)
        return max(trials, key=lambda t: t.score)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteTracker:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
