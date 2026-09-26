from tunekit.core.objective import ObjectiveDirection
from tunekit.core.optimizer import Optimizer
from tunekit.core.search_space import Categorical, IntUniform, SearchSpace, Uniform
from tunekit.strategies.grid_search import GridSearch


class TestGridSearch:
    def test_total_combinations(self):
        space = SearchSpace()
        space.add(Uniform("x", 0, 1))
        space.add(Categorical("opt", ["a", "b"]))
        gs = GridSearch(space, resolution=5)
        assert gs.total_combinations == 5 * 2

    def test_all_configs_unique(self):
        space = SearchSpace()
        space.add(Uniform("x", 0, 1))
        space.add(IntUniform("n", 1, 3))
        gs = GridSearch(space, resolution=3)
        configs = [tuple(sorted(gs.suggest().items())) for _ in range(gs.total_combinations)]
        assert len(set(configs)) == gs.total_combinations

    def test_wraps_around(self):
        space = SearchSpace()
        space.add(Categorical("a", ["x", "y"]))
        gs = GridSearch(space, resolution=5)
        for _ in range(gs.total_combinations):
            gs.suggest()
        first_after_wrap = gs.suggest()
        assert first_after_wrap is not None

    def test_grid_search_finds_minimum(self):
        space = SearchSpace()
        space.add(Uniform("x", -5, 5))

        strategy = GridSearch(space, resolution=100)
        optimizer = Optimizer(strategy, lambda p: p["x"] ** 2, ObjectiveDirection.MINIMIZE)
        best = optimizer.run(n_trials=100)
        assert best.score < 0.1

    def test_grid_search_with_log_uniform(self):
        from tunekit.core.search_space import LogUniform
        space = SearchSpace()
        space.add(LogUniform("lr", 1e-4, 1e-1))
        gs = GridSearch(space, resolution=5)
        configs = [gs.suggest() for _ in range(5)]
        assert all(1e-4 * 0.99 <= c["lr"] <= 1e-1 * 1.01 for c in configs)
