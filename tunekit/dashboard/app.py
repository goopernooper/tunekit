from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tunekit.dashboard.routes import create_router

STATIC_DIR = Path(__file__).parent / "static"


def create_app(db_path: str = "experiments.db") -> FastAPI:
    app = FastAPI(title="tunekit dashboard", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(create_router(db_path), prefix="/api")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def serve_index():
            return FileResponse(STATIC_DIR / "index.html")

    return app
