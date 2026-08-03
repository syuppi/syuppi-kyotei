"""コレクター一括実行."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from boatrace.collectors.official import OfficialCollector
from boatrace.collectors.tide import TideCollector
from boatrace.config import get_settings
from boatrace.logging_setup import get_logger

logger = get_logger(__name__)


def collect_daily(
    race_date: date | None = None,
    venue_ids: list[str] | None = None,
    include_lookback: bool = True,
) -> dict[str, Any]:
    """当日（＋直近N日）の公式データと潮位を収集."""
    settings = get_settings()
    target = race_date or date.today()
    official = OfficialCollector()
    tide = TideCollector()

    results: dict[str, Any] = {"date": target.isoformat(), "days": []}

    dates = [target]
    if include_lookback:
        for i in range(1, settings.collect.lookback_days + 1):
            dates.append(target - timedelta(days=i))

    for d in dates:
        logger.info("collect_start", date=d.isoformat())
        try:
            o = official.collect(d, venue_ids=venue_ids)
        except Exception as e:  # noqa: BLE001
            logger.exception("official_collect_error", date=d.isoformat(), error=str(e))
            o = {"error": str(e)}
        try:
            t = tide.collect(d, venue_ids=venue_ids)
        except Exception as e:  # noqa: BLE001
            logger.exception("tide_collect_error", date=d.isoformat(), error=str(e))
            t = {"error": str(e)}
        results["days"].append({"date": d.isoformat(), "official": o, "tide": t})
        logger.info("collect_done", date=d.isoformat(), official=o, tide=t)

    return results
