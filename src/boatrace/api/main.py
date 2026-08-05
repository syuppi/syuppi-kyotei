"""FastAPI アプリケーション."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from boatrace.api.routes import accuracy, collect, predictions, search, venues
from boatrace.config import get_settings
from boatrace.jobs.result_refresh import (
    last_refresh,
    missing_result_stats,
    start_background_refresh,
    stop_background_refresh,
)
from boatrace.logging_setup import setup_logging

setup_logging()
settings = get_settings()

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 10分ごとに当日・前日の着順欠損を自動補完（手動修正待ちを減らす）
    start_background_refresh(interval_sec=600)
    try:
        yield
    finally:
        stop_background_refresh()


app = FastAPI(title=settings.api.title, version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

app.include_router(predictions.router, prefix="/api", tags=["predictions"])
app.include_router(collect.router, prefix="/api", tags=["collect"])
app.include_router(venues.router, prefix="/api", tags=["venues"])
app.include_router(accuracy.router, prefix="/api", tags=["accuracy"])
app.include_router(search.router, prefix="/api", tags=["search"])


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/venues", response_class=HTMLResponse)
def venues_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "venues.html")


@app.get("/accuracy", response_class=HTMLResponse)
def accuracy_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "accuracy.html")


@app.get("/search", response_class=HTMLResponse)
def search_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "search.html")


@app.get("/health")
def health() -> dict:
    stats = missing_result_stats(days=2)
    return {
        "status": "ok",
        "missing_results": stats,
        "last_result_refresh": last_refresh() or None,
    }
