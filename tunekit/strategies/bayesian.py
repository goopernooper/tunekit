from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from tunekit.core.search_space import SearchSpace
from tunekit.core.strategy import SearchStrategy
from tunekit.core.trial import Trial, TrialStatus


class GaussianProcessRegressor:
    """Minimal GP with RBF kernel for Bayesian optimization surrogate."""

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-4) -> None:
        self.length_scale = length_scale
        self.noise = noise
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self._L: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._X_mean: np.ndarray | None = None
        self._X_std: np.ndarray | None = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diffs = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        sq_dist = np.sum(diffs**2, axis=2)
        return np.exp(-0.5 * sq_dist / (self.length_scale**2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0) + 1e-8
        self.X_train = (X - self._X_mean) / self._X_std
        self.y_train = y
        K = self._rbf_kernel(self.X_train, self.X_train)
        jitter = self.noise
        for _ in range(5):
            try:
                self._L = np.linalg.cholesky(K + jitter * np.eye(len(X)))
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        else:
            self._L = np.linalg.cholesky(K + 1.0 * np.eye(len(X)))
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))
        if not np.all(np.isfinite(self._alpha)):
            K_reg = K + 0.1 * np.eye(len(X))
            self._L = np.linalg.cholesky(K_reg)
            self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, variance) predictions at X."""
        X_norm = (X - self._X_mean) / self._X_std
        K_star = self._rbf_kernel(X_norm, self.X_train)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            mu = K_star @ self._alpha
            v = np.linalg.solve(self._L, K_star.T)
        var = 1.0 - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-10)
        mu = np.nan_to_num(mu, nan=0.0)
        return mu, var


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def expected_improvement(
    mu: np.ndarray, var: np.ndarray, best_score: float, minimize: bool = True
) -> np.ndarray:
    sigma = np.sqrt(var)
    ei = np.zeros_like(mu)
    mask = sigma > 1e-10

    if minimize:
        improvement = best_score - mu[mask]
    else:
        improvement = mu[mask] - best_score

    Z = improvement / sigma[mask]
    ei[mask] = improvement * np.array([_normal_cdf(z) for z in Z]) + \
               sigma[mask] * np.array([_normal_pdf(z) for z in Z])
    return ei


class BayesianOptimization(SearchStrategy):
    """Bayesian optimization with a Gaussian Process surrogate and Expected Improvement."""

    def __init__(
        self,
        search_space: SearchSpace,
        seed: int | None = None,
        n_initial: int = 5,
        n_candidates: int = 1000,
        minimize: bool = True,
        length_scale: float = 1.0,
    ) -> None:
        super().__init__(search_space, seed)
        self._rng = random.Random(seed)
        self.n_initial = n_initial
        self.n_candidates = n_candidates
        self.minimize = minimize
        self.gp = GaussianProcessRegressor(length_scale=length_scale)
        self._completed_trials: list[Trial] = []

    def _trial_to_vector(self, params: dict[str, Any]) -> list[float]:
        """Encode a param dict as a numeric vector for the GP."""
        vec = []
        for p in self.search_space.parameters:
            val = params[p.name]
            if hasattr(p, "choices"):
                vec.append(float(p.choices.index(val)))
            else:
                vec.append(float(val))
        return vec

    def suggest(self) -> dict[str, Any]:
        if len(self._completed_trials) < self.n_initial:
            return self.search_space.sample(self._rng)

        X = np.array([self._trial_to_vector(t.params) for t in self._completed_trials])
        y = np.array([t.score for t in self._completed_trials])

        y_mean, y_std = y.mean(), y.std() + 1e-8
        y_norm = (y - y_mean) / y_std

        self.gp.fit(X, y_norm)

        candidate_params = [
            self.search_space.sample(self._rng)
            for _ in range(self.n_candidates)
        ]
        candidates = np.array([self._trial_to_vector(p) for p in candidate_params])

        mu, var = self.gp.predict(candidates)

        if self.minimize:
            best_norm = (min(t.score for t in self._completed_trials) - y_mean) / y_std
        else:
            best_norm = (max(t.score for t in self._completed_trials) - y_mean) / y_std

        ei = expected_improvement(mu, var, best_norm, self.minimize)
        best_idx = int(np.argmax(ei))
        return candidate_params[best_idx]

    def update(self, trial: Trial) -> None:
        if trial.status == TrialStatus.COMPLETED:
            self._completed_trials.append(trial)

    @property
    def name(self) -> str:
        return "bayesian_optimization"
