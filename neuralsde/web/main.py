"""
FastAPI server for Neural SDE forecasts.

Run from repo root (neuralsde/):
  pip install -r requirements.txt -r web/requirements-web.txt
  uvicorn web.main:app --reload --host 127.0.0.1 --port 8000

Remove this ``web/`` package anytime without affecting ``neural_sde_forecast.py`` CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from neural_sde_forecast import run_forecast

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Neural SDE Forecast", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ForecastBody(BaseModel):
    ticker: str = Field(default="SPY", min_length=1, max_length=16)
    num_epochs: int = Field(default=300, ge=10, le=800)
    num_paths: int = Field(default=100, ge=10, le=500)


@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/forecast")
async def api_forecast(body: ForecastBody) -> dict:
    try:
        return await run_in_threadpool(
            run_forecast,
            body.ticker.strip(),
            num_epochs=body.num_epochs,
            num_paths=body.num_paths,
            verbose=False,
            save_plots=False,
            show_plots=False,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
