import tempfile
from pathlib import Path

from tunekit.core.objective import ObjectiveDirection
from tunekit.core.optimizer import Optimizer, no_improvement_stopping
from tunekit.core.search_space import SearchSpace, Uniform
from tunekit.strategies.random_search import RandomSearch
from tunekit.tracking.sqlite_tracker import SQLiteTracker


def quadratic(params: dict) -> float:
    x = params["x"]
    y = params["y"]
    return (x - 3) ** 2 + (y + 2) ** 2


class TestOptimizer:
    def test_minimize(self):
        space = SearchSpace()
        space.add(Uniform("x", -10, 10))
        space.add(Uniform("y", -10, 10))

        strategy = RandomSearch(space, seed=42)
        optimizer = Optimizer(strategy, quadratic, ObjectiveDirection.MINIMIZE)
        best = optimizer.run(n_trials=200)

        assert best.score < 5.0
        assert len(optimizer.trials) == 200

    def test_maximize(self):
        space = SearchSpace()
        space.add(Uniform("x", -10, 10))
        space.add(Uniform("y", -10, 10))

        def neg_quadratic(params):
            return -quadratic(params)

        strategy = RandomSearch(space, seed=42)
        optimizer = Optimizer(strategy, neg_quadratic, ObjectiveDirection.MAXIMIZE)
        best = optimizer.run(n_trials=50)
        assert best.score is not None

    def test_with_sqlite_tracker(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            space = SearchSpace()
            space.add(Uniform("x", -5, 5))

            strategy = RandomSearch(space, seed=0)
            with SQLiteTracker(db_path, experiment_name="test_exp") as tracker:
                optimizer = Optimizer(
                    strategy, lambda p: p["x"] ** 2, ObjectiveDirection.MINIMIZE, tracker=tracker
                )
                optimizer.run(n_trials=20)
                saved = tracker.get_all_trials()

            assert len(saved) == 20
            assert all(t.score is not None for t in saved)

    def test_objective_returning_tuple(self):
        space = SearchSpace()
        space.add(Uniform("x", 0, 10))

        def objective_with_metrics(params):
            score = params["x"] ** 2
            return score, {"raw_x": params["x"]}

        strategy = RandomSearch(space, seed=1)
        optimizer = Optimizer(strategy, objective_with_metrics, ObjectiveDirection.MINIMIZE)
        best = optimizer.run(n_trials=10)
        assert "raw_x" in best.metrics

    def test_failed_trial_does_not_crash(self):
        space = SearchSpace()
        space.add(Uniform("x", -1, 1))

        def bad_objective(params):
            raise ValueError("boom")

        strategy = RandomSearch(space, seed=0)
        optimizer = Optimizer(strategy, bad_objective, ObjectiveDirection.MINIMIZE)
        try:
            optimizer.run(n_trials=5)
        except RuntimeError:
            pass
        assert all(t.status.value == "failed" for t in optimizer.trials)

    def test_early_stopping(self):
        space = SearchSpace()
        space.add(Uniform("x", 0, 0.001))

        strategy = RandomSearch(space, seed=42)
        optimizer = Optimizer(
            strategy,
            lambda p: p["x"] ** 2,
            ObjectiveDirection.MINIMIZE,
            early_stop=no_improvement_stopping(patience=10),
        )
        best = optimizer.run(n_trials=1000)
        assert len(optimizer.trials) < 1000

    def test_no_early_stop_when_improving(self):
        space = SearchSpace()
        space.add(Uniform("x", -10, 10))

        strategy = RandomSearch(space, seed=42)
        optimizer = Optimizer(
            strategy,
            lambda p: p["x"] ** 2,
            ObjectiveDirection.MINIMIZE,
            early_stop=no_improvement_stopping(patience=200),
        )
        optimizer.run(n_trials=200)
        assert len(optimizer.trials) == 200
