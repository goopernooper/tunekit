"""Benchmark: compare all search strategies on standard test functions."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from tunekit import (
    BayesianOptimization,
    GridSearch,
    ObjectiveDirection,
    Optimizer,
    RandomSearch,
    SearchSpace,
    Uniform,
)
from tunekit.strategies.hyperband import Hyperband


@dataclass
class BenchmarkResult:
    strategy: str
    function: str
    best_score: float
    n_trials: int
    elapsed: float


# --- Test Functions ---

def sphere(params: dict[str, Any]) -> float:
    return sum(v ** 2 for v in params.values())


def rosenbrock(params: dict[str, Any]) -> float:
    vals = list(params.values())
    return sum(
        100 * (vals[i + 1] - vals[i] ** 2) ** 2 + (1 - vals[i]) ** 2
        for i in range(len(vals) - 1)
    )


def rastrigin(params: dict[str, Any]) -> float:
    vals = list(params.values())
    n = len(vals)
    return 10 * n + sum(v ** 2 - 10 * math.cos(2 * math.pi * v) for v in vals)


def ackley(params: dict[str, Any]) -> float:
    vals = list(params.values())
    n = len(vals)
    sum_sq = sum(v ** 2 for v in vals) / n
    sum_cos = sum(math.cos(2 * math.pi * v) for v in vals) / n
    return -20 * math.exp(-0.2 * math.sqrt(sum_sq)) - math.exp(sum_cos) + 20 + math.e


TEST_FUNCTIONS: list[tuple[str, Callable, float, float, float]] = [
    ("Sphere",     sphere,     -5.0, 5.0, 0.0),
    ("Rosenbrock", rosenbrock, -5.0, 5.0, 0.0),
    ("Rastrigin",  rastrigin,  -5.12, 5.12, 0.0),
    ("Ackley",     ackley,     -5.0, 5.0, 0.0),
]

DIMENSIONS = 3
N_TRIALS = 100
SEED = 42


def make_space(low: float, high: float) -> SearchSpace:
    space = SearchSpace()
    for i in range(DIMENSIONS):
        space.add(Uniform(f"x{i}", low, high))
    return space


def run_benchmark() -> list[BenchmarkResult]:
    results = []

    for func_name, func, low, high, optimum in TEST_FUNCTIONS:
        space = make_space(low, high)
        strategies: list[tuple[str, Any]] = [
            ("Random Search", RandomSearch(space, seed=SEED)),
            ("Grid Search", GridSearch(space, resolution=5)),
            ("Bayesian Opt", BayesianOptimization(space, seed=SEED, n_initial=10)),
        ]

        for strat_name, strategy in strategies:
            start = time.time()
            optimizer = Optimizer(strategy, func, ObjectiveDirection.MINIMIZE)
            best = optimizer.run(n_trials=N_TRIALS)
            elapsed = time.time() - start

            results.append(BenchmarkResult(
                strategy=strat_name,
                function=func_name,
                best_score=best.score,
                n_trials=N_TRIALS,
                elapsed=elapsed,
            ))

        # Hyperband with resource-aware wrapper
        def make_report_fn(fn: Callable) -> Callable:
            def report_fn(params: dict, resource: int) -> float:
                noise = 1.0 / resource
                return fn(params) + noise
            return report_fn

        start = time.time()
        hb = Hyperband(space, make_report_fn(func), max_resource=27, eta=3, seed=SEED)
        best = hb.run_all()
        elapsed = time.time() - start

        results.append(BenchmarkResult(
            strategy="Hyperband",
            function=func_name,
            best_score=best.score,
            n_trials=len(hb.all_trials),
            elapsed=elapsed,
        ))

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    header = f"{'Function':<14} {'Strategy':<16} {'Best Score':>12} {'Trials':>8} {'Time (s)':>10}"
    print("\n" + "=" * len(header))
    print("tunekit benchmark")
    print(f"{DIMENSIONS}D test functions, {N_TRIALS} trials per strategy (seed={SEED})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    current_func = None
    for r in results:
        if r.function != current_func:
            if current_func is not None:
                print("-" * len(header))
            current_func = r.function
        print(f"{r.function:<14} {r.strategy:<16} {r.best_score:>12.6f} {r.n_trials:>8} {r.elapsed:>10.3f}")

    print("=" * len(header))

    # Summary: best strategy per function
    print("\nBest strategy per function:")
    funcs = list(dict.fromkeys(r.function for r in results))
    for func in funcs:
        func_results = [r for r in results if r.function == func]
        winner = min(func_results, key=lambda r: r.best_score)
        print(f"  {func:<14} -> {winner.strategy} ({winner.best_score:.6f})")
    print()


if __name__ == "__main__":
    results = run_benchmark()
    print_results(results)
