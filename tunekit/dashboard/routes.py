from __future__ import annotations

import json
import sqlite3
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Query


def create_router(db_path: str) -> APIRouter:
    router = APIRouter()

    def _get_conn() -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @router.get("/experiments")
    def list_experiments() -> list[dict[str, Any]]:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT DISTINCT experiment FROM trials ORDER BY experiment"
            ).fetchall()
            return [{"name": row["experiment"]} for row in rows]
        finally:
            conn.close()

    @router.get("/experiments/{experiment_name}/trials")
    def get_trials(
        experiment_name: str,
        sort_by: str = Query("start_time", pattern="^(start_time|score|trial_id)$"),
        order: str = Query("asc", pattern="^(asc|desc)$"),
    ) -> list[dict[str, Any]]:
        conn = _get_conn()
        try:
            rows = conn.execute(
                f"SELECT * FROM trials WHERE experiment = ? ORDER BY {sort_by} {order}",
                (experiment_name,),
            ).fetchall()
            return [_row_to_dict(row) for row in rows]
        finally:
            conn.close()

    @router.get("/experiments/{experiment_name}/best")
    def get_best_trial(
        experiment_name: str,
        direction: str = Query("minimize", pattern="^(minimize|maximize)$"),
    ) -> dict[str, Any]:
        conn = _get_conn()
        try:
            order = "ASC" if direction == "minimize" else "DESC"
            row = conn.execute(
                f"SELECT * FROM trials WHERE experiment = ? AND status = 'completed' "
                f"ORDER BY score {order} LIMIT 1",
                (experiment_name,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="No completed trials found")
            return _row_to_dict(row)
        finally:
            conn.close()

    @router.get("/experiments/{experiment_name}/stats")
    def get_experiment_stats(experiment_name: str) -> dict[str, Any]:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed, "
                "SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed, "
                "SUM(CASE WHEN status = 'pruned' THEN 1 ELSE 0 END) as pruned, "
                "MIN(score) as min_score, MAX(score) as max_score, AVG(score) as avg_score "
                "FROM trials WHERE experiment = ?",
                (experiment_name,),
            ).fetchone()
            if row is None or row["total"] == 0:
                raise HTTPException(status_code=404, detail="Experiment not found")
            return dict(row)
        finally:
            conn.close()

    return router


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if d.get("params"):
        d["params"] = json.loads(d["params"])
    if d.get("metrics"):
        d["metrics"] = json.loads(d["metrics"])
    return d
