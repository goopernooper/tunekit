import numpy as np
import pytest

from tunekit.core.objective import ObjectiveDirection
from tunekit.core.optimizer import Optimizer
from tunekit.core.search_space import Categorical, SearchSpace, Uniform
from tunekit.strategies.bayesian import (
    BayesianOptimization,
    GaussianProcessRegressor,
    expected_improvement,
)


class TestGaussianProcess:
    def test_fit_predict_shape(self):
        gp = GaussianProcessRegressor()
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 4.0, 9.0])
        gp.fit(X, y)
        mu, var = gp.predict(np.array([[1.5], [2.5]]))
        assert mu.shape == (2,)
        assert var.shape == (2,)

    def test_interpolates_training_points(self):
        gp = GaussianProcessRegressor(noise=1e-8)
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 4.0])
        gp.fit(X, y)
        mu, var = gp.predict(X)
        np.testing.assert_allclose(mu, y, atol=0.01)
        assert all(v < 0.01 for v in var)

    def test_high_variance_far_from_data(self):
        gp = GaussianProcessRegressor(length_scale=0.5, noise=1e-8)
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        gp.fit(X, y)
        _, var_near = gp.predict(np.array([[0.5]]))
        _, var_far = gp.predict(np.array([[10.0]]))
        assert var_far[0] > var_near[0]


class TestExpectedImprovement:
    def test_ei_positive_for_improvement(self):
        mu = np.array([0.5, 1.5, 2.0])
        var = np.array([0.1, 0.1, 0.1])
        ei = expected_improvement(mu, var, best_score=1.0, minimize=True)
        assert ei[0] > ei[2]

    def test_ei_zero_variance(self):
        mu = np.array([0.5])
        var = np.array([0.0])
        ei = expected_improvement(mu, var, best_score=1.0, minimize=True)
        assert ei[0] == 0.0


class TestBayesianOptimization:
    def test_initial_phase_is_random(self):
        space = SearchSpace()
        space.add(Uniform("x", 0, 10))
        bo = BayesianOptimization(space, seed=42, n_initial=5)
        configs = [bo.suggest() for _ in range(5)]
        assert len(configs) == 5
        assert all("x" in c for c in configs)

    def test_optimizes_quadratic(self):
        space = SearchSpace()
        space.add(Uniform("x", -5, 5))
        space.add(Uniform("y", -5, 5))

        def quadratic(params):
            return (params["x"] - 1) ** 2 + (params["y"] + 1) ** 2

        strategy = BayesianOptimization(space, seed=42, n_initial=10, n_candidates=500)
        optimizer = Optimizer(strategy, quadratic, ObjectiveDirection.MINIMIZE)
        best = optimizer.run(n_trials=40)
        assert best.score < 2.0

    def test_handles_categorical(self):
        space = SearchSpace()
        space.add(Uniform("x", 0, 5))
        space.add(Categorical("mode", ["a", "b", "c"]))

        def obj(params):
            base = (params["x"] - 2) ** 2
            if params["mode"] == "b":
                base *= 0.5
            return base

        strategy = BayesianOptimization(space, seed=0, n_initial=5)
        optimizer = Optimizer(strategy, obj, ObjectiveDirection.MINIMIZE)
        best = optimizer.run(n_trials=30)
        assert best.score < 3.0

    def test_maximize_direction(self):
        space = SearchSpace()
        space.add(Uniform("x", 0, 10))

        strategy = BayesianOptimization(space, seed=42, n_initial=5, minimize=False)
        optimizer = Optimizer(
            strategy, lambda p: -(p["x"] - 5) ** 2, ObjectiveDirection.MAXIMIZE
        )
        best = optimizer.run(n_trials=30)
        assert best.score > -5.0
