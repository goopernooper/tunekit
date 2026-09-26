import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tunekit import (
    ObjectiveDirection,
    Optimizer,
    RandomSearch,
    SQLiteTracker,
    SearchSpace,
    Uniform,
)
from tunekit.dashboard.app import create_app


@pytest.fixture
def populated_db(tmp_path):
    db_path = tmp_path / "test.db"
    space = SearchSpace()
    space.add(Uniform("x", -5, 5))
    strategy = RandomSearch(space, seed=42)
    with SQLiteTracker(str(db_path), experiment_name="exp1") as tracker:
        opt = Optimizer(strategy, lambda p: p["x"] ** 2, ObjectiveDirection.MINIMIZE, tracker=tracker)
        opt.run(n_trials=10)
    return db_path


@pytest.fixture
def client(populated_db):
    app = create_app(db_path=str(populated_db))
    return TestClient(app)


class TestDashboardAPI:
    def test_list_experiments(self, client):
        resp = client.get("/api/experiments")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "exp1"

    def test_get_trials(self, client):
        resp = client.get("/api/experiments/exp1/trials")
        assert resp.status_code == 200
        trials = resp.json()
        assert len(trials) == 10
        assert all("params" in t and "score" in t for t in trials)

    def test_get_trials_sorted_by_score(self, client):
        resp = client.get("/api/experiments/exp1/trials?sort_by=score&order=asc")
        assert resp.status_code == 200
        trials = resp.json()
        scores = [t["score"] for t in trials]
        assert scores == sorted(scores)

    def test_get_best_trial(self, client):
        resp = client.get("/api/experiments/exp1/best?direction=minimize")
        assert resp.status_code == 200
        best = resp.json()
        assert best["score"] is not None

        all_resp = client.get("/api/experiments/exp1/trials")
        all_scores = [t["score"] for t in all_resp.json()]
        assert best["score"] == min(all_scores)

    def test_get_stats(self, client):
        resp = client.get("/api/experiments/exp1/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["total"] == 10
        assert stats["completed"] == 10
        assert stats["failed"] == 0

    def test_nonexistent_experiment(self, client):
        resp = client.get("/api/experiments/nope/best")
        assert resp.status_code == 404

    def test_serves_index(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "tunekit dashboard" in resp.text
