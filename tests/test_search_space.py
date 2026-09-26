import math
import random

import pytest

from tunekit.core.search_space import (
    Categorical,
    IntUniform,
    LogUniform,
    SearchSpace,
    Uniform,
)


class TestCategorical:
    def test_sample_returns_valid_choice(self):
        param = Categorical("optimizer", ["sgd", "adam", "rmsprop"])
        rng = random.Random(42)
        for _ in range(50):
            assert param.sample(rng) in ["sgd", "adam", "rmsprop"]

    def test_contains(self):
        param = Categorical("act", ["relu", "tanh"])
        assert param.contains("relu")
        assert not param.contains("sigmoid")

    def test_empty_choices_raises(self):
        with pytest.raises(ValueError):
            Categorical("x", [])


class TestUniform:
    def test_sample_within_bounds(self):
        param = Uniform("lr", 0.001, 0.1)
        rng = random.Random(42)
        for _ in range(100):
            val = param.sample(rng)
            assert 0.001 <= val <= 0.1

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            Uniform("x", 5.0, 5.0)


class TestLogUniform:
    def test_sample_within_bounds(self):
        param = LogUniform("lr", 1e-5, 1e-1)
        rng = random.Random(42)
        for _ in range(100):
            val = param.sample(rng)
            assert 1e-5 <= val <= 1e-1

    def test_log_distribution(self):
        param = LogUniform("lr", 1e-4, 1e-1)
        rng = random.Random(42)
        samples = [math.log10(param.sample(rng)) for _ in range(10000)]
        mean_log = sum(samples) / len(samples)
        assert -3.5 < mean_log < -1.5

    def test_negative_bounds_raises(self):
        with pytest.raises(ValueError):
            LogUniform("x", -1, 1)


class TestIntUniform:
    def test_sample_is_int(self):
        param = IntUniform("layers", 1, 10)
        rng = random.Random(42)
        for _ in range(50):
            val = param.sample(rng)
            assert isinstance(val, int)
            assert 1 <= val <= 10


class TestSearchSpace:
    def test_sample_all_keys(self):
        space = SearchSpace()
        space.add(Uniform("lr", 0.001, 0.1))
        space.add(IntUniform("layers", 1, 5))
        space.add(Categorical("opt", ["adam", "sgd"]))

        sample = space.sample(random.Random(42))
        assert set(sample.keys()) == {"lr", "layers", "opt"}

    def test_len(self):
        space = SearchSpace()
        space.add(Uniform("a", 0, 1))
        space.add(Uniform("b", 0, 1))
        assert len(space) == 2

    def test_fluent_add(self):
        space = SearchSpace()
        result = space.add(Uniform("x", 0, 1))
        assert result is space
