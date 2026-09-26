"""Run the dashboard: python -m tunekit.dashboard --db experiments.db"""
from __future__ import annotations

import argparse

import uvicorn

from tunekit.dashboard.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="tunekit dashboard")
    parser.add_argument("--db", default="experiments.db", help="Path to SQLite database")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(db_path=args.db)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
