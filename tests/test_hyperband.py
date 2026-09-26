import math

from tunekit.core.search_space import SearchSpace, Uniform
from tunekit.strategies.hyperband import Bracket, Hyperband


class TestBracket:
    def test_current_resource_increases(self):
        configs = [{"x": i} for i in range(27)]
        bracket = Bracket(configs, max_resource=81, eta=3, n_rungs=4)
        resources = []
        while not bracket.completed:
            resources.append(bracket.current_resource)
            for c in bracket.configs:
                bracket.report(c, c["x"])
            bracket.advance()
        assert resources == [3, 9, 27, 81]

    def test_successive_halving_reduces_configs(self):
        configs = [{"x": i} for i in range(27)]
        bracket = Bracket(configs, max_resource=81, eta=3, n_rungs=4)
        sizes = []
        while not bracket.completed:
            sizes.append(bracket.n_configs)
            for c in bracket.configs:
                bracket.report(c, c["x"])
            bracket.advance()
        assert sizes == [27, 9, 3, 1]

    def test_keeps_best_configs(self):
        configs = [{"x": float(i)} for i in range(9)]
        bracket = Bracket(configs, max_resource=9, eta=3, n_rungs=2)
        for c in bracket.configs:
            bracket.report(c, c["x"])
        bracket.advance()
        surviving_x = sorted(c["x"] for c in bracket.configs)
        assert surviving_x == [0.0, 1.0, 2.0]


class TestHyperband:
    def test_finds_minimum_of_quadratic(self):
        space = SearchSpace()
        space.add(Uniform("x", -5, 5))
        space.add(Uniform("y", -5, 5))

        def report_fn(params, resource):
            noise = 1.0 / resource
            return (params["x"] - 1) ** 2 + (params["y"] + 1) ** 2 + noise

        hb = Hyperband(space, report_fn, max_resource=27, eta=3, seed=42)
        best = hb.run_all()
        assert best.score < 5.0

    def test_maximize_direction(self):
        space = SearchSpace()
        space.add(Uniform("x", 0, 10))

        def report_fn(params, resource):
            return -(params["x"] - 5) ** 2

        hb = Hyperband(space, report_fn, max_resource=9, eta=3, minimize=False, seed=0)
        best = hb.run_all()
        assert best.score > -10.0

    def test_all_trials_recorded(self):
        space = SearchSpace()
        space.add(Uniform("x", 0, 1))

        hb = Hyperband(space, lambda p, r: p["x"], max_resource=9, eta=3, seed=0)
        hb.run_all()
        assert len(hb.all_trials) > 0

    def test_bracket_count(self):
        space = SearchSpace()
        space.add(Uniform("x", 0, 1))
        hb = Hyperband(space, lambda p, r: p["x"], max_resource=27, eta=3, seed=0)
        brackets = hb._build_brackets()
        s_max = int(math.log(27) / math.log(3))
        assert len(brackets) == s_max + 1

    def test_trials_have_resource_metadata(self):
        space = SearchSpace()
        space.add(Uniform("x", 0, 1))

        hb = Hyperband(space, lambda p, r: p["x"], max_resource=9, eta=3, seed=42)
        hb.run_all()
        for trial in hb.all_trials:
            assert "resource" in trial.metrics
