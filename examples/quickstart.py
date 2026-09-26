"""Quickstart: optimize a simple function with Random Search + SQLite tracking."""

from tunekit import (
    Categorical,
    LogUniform,
    ObjectiveDirection,
    Optimizer,
    RandomSearch,
    SearchSpace,
    SQLiteTracker,
    Uniform,
)


def train_model(params: dict) -> tuple[float, dict]:
    lr = params["learning_rate"]
    dropout = params["dropout"]
    optimizer_name = params["optimizer"]

    fake_loss = (lr - 0.003) ** 2 + (dropout - 0.3) ** 2
    if optimizer_name == "adam":
        fake_loss *= 0.8
    return fake_loss, {"lr": lr, "optimizer": optimizer_name}


def main():
    space = SearchSpace()
    space.add(LogUniform("learning_rate", 1e-5, 1e-1))
    space.add(Uniform("dropout", 0.0, 0.9))
    space.add(Categorical("optimizer", ["sgd", "adam", "rmsprop"]))

    strategy = RandomSearch(space, seed=42)

    with SQLiteTracker("experiments.db", experiment_name="quickstart") as tracker:
        opt = Optimizer(
            strategy=strategy,
            objective=train_model,
            direction=ObjectiveDirection.MINIMIZE,
            tracker=tracker,
        )
        best = opt.run(n_trials=50)

    print(f"\nBest trial: {best.trial_id}")
    print(f"  Score: {best.score:.6f}")
    print(f"  Params: {best.params}")
    print(f"  Duration: {best.duration:.4f}s")


if __name__ == "__main__":
    main()
