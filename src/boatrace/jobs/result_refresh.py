"""着順の定期補完（手動修正ループを減らす）."""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func

from boatrace.collectors.official import OfficialCollector
from boatrace.db.models import RaceCard, RaceResult
from boatrace.db.session import session_scope
from boatrace.logging_setup import get_logger

logger = get_logger(__name__)

_stop = threading.Event()
_thread: threading.Thread | None = None
_last_run: dict[str, Any] = {"at": None, "days": []}


def missing_result_stats(days: int = 2) -> dict[str, Any]:
    """直近N日の着順欠損状況."""
    today = date.today()
    out: list[dict[str, Any]] = []
    with session_scope() as s:
        for i in range(days):
            d = today - timedelta(days=i)
            cards = s.query(func.count(RaceCard.id)).filter(RaceCard.race_date == d).scalar() or 0
            ok = (
                s.query(func.count(RaceCard.id))
                .join(RaceResult, RaceResult.race_card_id == RaceCard.id)
                .filter(RaceCard.race_date == d, RaceResult.rank1_waku.isnot(None))
                .scalar()
            ) or 0
            out.append(
                {
                    "date": d.isoformat(),
                    "cards": int(cards),
                    "with_result": int(ok),
                    "missing": int(max(0, cards - ok)),
                }
            )
    return {"days": out, "missing_total": sum(x["missing"] for x in out)}


def refresh_recent_missing(days: int = 2) -> dict[str, Any]:
    """本日〜直近の未取得着順だけ公式HTMLで埋める."""
    collector = OfficialCollector()
    today = date.today()
    day_results = []
    for i in range(days):
        d = today - timedelta(days=i)
        try:
            summary = collector.collect_missing_results(d, workers=8)
        except Exception as e:  # noqa: BLE001
            summary = {"date": d.isoformat(), "error": str(e)}
            logger.warning("result_refresh_failed", date=d.isoformat(), error=str(e))
        day_results.append(summary)
    payload = {
        "at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "days": day_results,
        "stats": missing_result_stats(days=days),
    }
    _last_run.clear()
    _last_run.update(payload)
    logger.info(
        "result_refresh_done",
        updated=sum(int(x.get("updated") or 0) for x in day_results if isinstance(x, dict)),
        missing=payload["stats"]["missing_total"],
    )
    return payload


def last_refresh() -> dict[str, Any]:
    return dict(_last_run)


def _loop(interval_sec: int) -> None:
    # 起動直後に1回、その後は定期実行
    while not _stop.is_set():
        try:
            refresh_recent_missing(days=2)
        except Exception as e:  # noqa: BLE001
            logger.exception("result_refresh_loop_error", error=str(e))
        _stop.wait(interval_sec)


def start_background_refresh(interval_sec: int = 600) -> None:
    """APIプロセス内で着順補完を常駐させる."""
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(
        target=_loop,
        args=(max(60, int(interval_sec)),),
        name="result-refresh",
        daemon=True,
    )
    _thread.start()
    logger.info("result_refresh_started", interval_sec=interval_sec)


def stop_background_refresh() -> None:
    _stop.set()
