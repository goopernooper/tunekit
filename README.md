# tunekit

A lightweight hyperparameter optimization library built from scratch in Python. Plug in any model, define a search space, pick a strategy, and automatically run, track, and compare experiments — no Optuna or Ray Tune under the hood.

## Features

- **Search Strategies** — Random Search, Grid Search, Bayesian Optimization (Gaussian Process + Expected Improvement), and Hyperband (successive halving with bracket management)
- **Flexible Search Spaces** — Uniform, LogUniform, IntUniform, and Categorical hyperparameter types
- **Experiment Tracking** — SQLite-backed tracker that logs every trial with params, scores, metrics, and timing
- **Pluggable Architecture** — implement `SearchStrategy` to add your own algorithms; any callable works as an objective

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from tunekit import (
    SearchSpace, LogUniform, Uniform, Categorical,
    RandomSearch, Optimizer, ObjectiveDirection, SQLiteTracker,
)

# Define your search space
space = SearchSpace()
space.add(LogUniform("learning_rate", 1e-5, 1e-1))
space.add(Uniform("dropout", 0.0, 0.9))
space.add(Categorical("optimizer", ["sgd", "adam", "rmsprop"]))

# Define your objective (any callable that takes params and returns a score)
def train_model(params):
    # ... train your model here ...
    return validation_loss

# Run optimization
strategy = RandomSearch(space, seed=42)
with SQLiteTracker("experiments.db", experiment_name="my_experiment") as tracker:
    opt = Optimizer(
        strategy=strategy,
        objective=train_model,
        direction=ObjectiveDirection.MINIMIZE,
        tracker=tracker,
    )
    best = opt.run(n_trials=100)

print(f"Best score: {best.score}")
print(f"Best params: {best.params}")
```

## Strategies

### Random Search

Samples uniformly from the search space. Good baseline.

```python
from tunekit import RandomSearch
strategy = RandomSearch(space, seed=42)
```

### Grid Search

Exhaustive search over a discretized grid. Set `resolution` to control how many points per continuous dimension.

```python
from tunekit import GridSearch
strategy = GridSearch(space, resolution=10)
print(f"Total combinations: {strategy.total_combinations}")
```

### Bayesian Optimization

Gaussian Process surrogate with Expected Improvement acquisition function. Normalizes inputs and targets, uses Cholesky decomposition for numerical stability.

```python
from tunekit import BayesianOptimization
strategy = BayesianOptimization(
    space,
    seed=42,
    n_initial=10,       # random trials before GP kicks in
    n_candidates=1000,  # candidate points to evaluate EI over
)
```

### Hyperband

Adaptive resource allocation via successive halving. Requires a `report_fn(params, resource) -> score` where `resource` controls training budget (epochs, data fraction, etc.).

```python
from tunekit import Hyperband

def report_fn(params, resource):
    # resource = number of epochs to train
    model = build_model(params)
    model.fit(X_train, y_train, epochs=resource)
    return model.evaluate(X_val, y_val)

hb = Hyperband(
    space,
    report_fn=report_fn,
    max_resource=81,  # max epochs
    eta=3,            # reduction factor
    seed=42,
)
best = hb.run_all()
```

## Early Stopping

Stop optimization early when no improvement is found:

```python
from tunekit import Optimizer, RandomSearch, ObjectiveDirection, no_improvement_stopping

optimizer = Optimizer(
    strategy=RandomSearch(space, seed=42),
    objective=train_model,
    direction=ObjectiveDirection.MINIMIZE,
    early_stop=no_improvement_stopping(patience=20),
)
best = optimizer.run(n_trials=500)  # will stop before 500 if plateau detected
```

## Experiment Dashboard

Visualize results with the built-in FastAPI + React dashboard:

```bash
pip install -e ".[dashboard]"
python -m tunekit.dashboard --db experiments.db
```

Opens a browser UI with optimization progress charts, stats cards, and a trial history table.

## Experiment Tracking

All trials are logged to SQLite with full metadata:

```python
with SQLiteTracker("experiments.db", experiment_name="v1") as tracker:
    optimizer = Optimizer(strategy, objective, tracker=tracker)
    optimizer.run(n_trials=50)

    # Query results
    all_trials = tracker.get_all_trials()
    best = tracker.get_best_trial(direction="minimize")
```

## Custom Strategies

Extend `SearchStrategy` to implement your own algorithm:

```python
from tunekit import SearchStrategy, SearchSpace, Trial

class MyStrategy(SearchStrategy):
    def __init__(self, search_space: SearchSpace, seed=None):
        super().__init__(search_space, seed)

    def suggest(self) -> dict:
        # Return the next hyperparameter config to evaluate
        return self.search_space.sample()

    def update(self, trial: Trial) -> None:
        # Incorporate trial results (for model-based strategies)
        pass

    @property
    def name(self) -> str:
        return "my_strategy"
```

## Benchmarks

Comparison on standard 3D test functions (100 trials, seed=42):

| Function   | Random Search | Grid Search | Bayesian Opt | Hyperband |
|------------|---------------|-------------|--------------|-----------|
| Sphere     | 1.168         | **0.000**   | 0.006        | 5.390     |
| Rosenbrock | 28.995        | **2.000**   | 15.049       | 1008.580  |
| Rastrigin  | 14.567        | **0.000**   | 17.823       | 33.887    |
| Ackley     | 3.875         | **0.000**   | 1.030        | 6.841     |

Grid Search dominates in low dimensions where it can cover the space. Bayesian Optimization scales better to higher dimensions where Grid Search becomes intractable. Hyperband's advantage is with expensive objectives where early stopping saves compute.

Run benchmarks yourself:

```bash
python benchmarks/benchmark.py
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Project Structure

```
tunekit/
├── tunekit/
│   ├── core/
│   │   ├── search_space.py    # HyperParameter types + SearchSpace
│   │   ├── trial.py           # Trial dataclass + TrialStatus
│   │   ├── strategy.py        # SearchStrategy ABC
│   │   ├── objective.py       # Objective protocol + direction enum
│   │   └── optimizer.py       # Main orchestrator + early stopping
│   ├── strategies/
│   │   ├── random_search.py   # Random Search
│   │   ├── grid_search.py     # Grid Search
│   │   ├── bayesian.py        # Bayesian Optimization (GP + EI)
│   │   └── hyperband.py       # Hyperband (successive halving)
│   ├── dashboard/
│   │   ├── app.py             # FastAPI app factory
│   │   ├── routes.py          # REST API endpoints
│   │   └── static/index.html  # React dashboard
│   └── tracking/
│       ├── tracker.py         # ExperimentTracker ABC
│       └── sqlite_tracker.py  # SQLite implementation
├── benchmarks/
│   └── benchmark.py           # Strategy comparison script
├── tests/                     # 52 tests
├── examples/
│   └── quickstart.py
└── pyproject.toml
```

## License

MIT
