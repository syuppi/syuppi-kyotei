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
from boatrace.jobs.history_warm import start_history_warm, stop_history_warm, warm_status
from boatrace.jobs.preclose_predict import (
    last_preclose_run,
    preclose_status,
    start_background_preclose_predict,
    stop_background_preclose_predict,
)
from boatrace.jobs.result_refresh import (
    last_refresh,
    missing_result_stats,
    start_background_refresh,
    stop_background_refresh,
)
from boatrace.logging_setup import get_logger, setup_logging

setup_logging()
settings = get_settings()
logger = get_logger(__name__)

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        from boatrace.jobs.today_bootstrap import ensure_today_cards

        boot = ensure_today_cards()
        logger.info(
            "today_cards_bootstrap",
            date=boot.get("date"),
            skipped=boot.get("skipped"),
            race_count=boot.get("race_count"),
            venue_count=boot.get("venue_count"),
            ok=boot.get("ok"),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("today_cards_bootstrap_failed", error=str(e))

    start_background_refresh(interval_sec=settings.jobs.result_refresh_interval_sec)
    start_background_preclose_predict(
        interval_sec=settings.jobs.preclose_predict_interval_sec,
        lead_minutes=settings.jobs.preclose_lead_minutes,
    )
    start_history_warm()
    try:
        yield
    finally:
        stop_history_warm()
        stop_background_preclose_predict()
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
    from boatrace.jobs.today_bootstrap import count_cards_for_day
    from boatrace.timeutil import japan_today

    stats = missing_result_stats(days=2)
    today = japan_today()
    race_count, venue_count, venues = count_cards_for_day(today)
    return {
        "status": "ok",
        "predict_only": bool(settings.runtime.predict_only),
        "japan_today": today.isoformat(),
        "today_races": {
            "race_count": race_count,
            "venue_count": venue_count,
            "venues": venues,
        },
        "missing_results": stats,
        "last_result_refresh": last_refresh() or None,
        "preclose_predict": preclose_status(),
        "last_preclose_predict": last_preclose_run() or None,
        "history_warm": warm_status(),
    }
