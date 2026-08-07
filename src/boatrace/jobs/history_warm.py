"""Render等の空DB起動時に、特徴用の直近履歴を裏で取得する."""

from __future__ import annotations

import threading
from datetime import date, timedelta
from typing import Any

from boatrace.config import get_settings
from boatrace.logging_setup import get_logger

logger = get_logger(__name__)

_stop = threading.Event()
_thread: threading.Thread | None = None
_status: dict[str, Any] = {"state": "idle", "days_done": [], "error": None}


def warm_status() -> dict[str, Any]:
    return dict(_status)


def warm_lookback(days: int | None = None) -> dict[str, Any]:
    """OpenAPIで直近N日を取得（オッズ・潮汐は省略して高速化）."""
    from boatrace.collectors.official import OfficialCollector
    from boatrace.collectors.openapi import OpenApiCollector

    n = int(days if days is not None else get_settings().runtime.warm_lookback_days)
    n = max(0, min(n, 30))
    _status.update({"state": "running", "days_done": [], "error": None, "target_days": n})
    if n <= 0:
        _status["state"] = "skipped"
        return warm_status()

    today = date.today()
    openapi = OpenApiCollector()
    official = OfficialCollector()
    done: list[str] = []
    for i in range(1, n + 1):
        if _stop.is_set():
            break
        d = today - timedelta(days=i)
        try:
            openapi.collect(d, venue_ids=None)
            try:
                official.collect_missing_results(d, venue_ids=None)
            except Exception:  # noqa: BLE001
                pass
            done.append(d.isoformat())
            _status["days_done"] = list(done)
            logger.info("history_warm_day_done", date=d.isoformat(), i=i, n=n)
        except Exception as e:  # noqa: BLE001
            logger.warning("history_warm_day_failed", date=d.isoformat(), error=str(e))
            _status["error"] = str(e)
    _status["state"] = "done"
    return warm_status()


def _loop() -> None:
    try:
        warm_lookback()
    except Exception as e:  # noqa: BLE001
        _status["state"] = "error"
        _status["error"] = str(e)
        logger.exception("history_warm_failed", error=str(e))


def start_history_warm() -> None:
    """predict_only 時、起動後に1回だけ暖機."""
    global _thread
    settings = get_settings()
    if not settings.runtime.predict_only:
        return
    if settings.runtime.warm_lookback_days <= 0:
        return
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="history-warm", daemon=True)
    _thread.start()
    logger.info(
        "history_warm_started",
        days=settings.runtime.warm_lookback_days,
    )


def stop_history_warm() -> None:
    _stop.set()
