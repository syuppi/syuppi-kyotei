"""コレクター一括実行."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from boatrace.collectors.official import OfficialCollector
from boatrace.collectors.openapi import OpenApiCollector
from boatrace.collectors.tide import TideCollector
from boatrace.config import get_settings
from boatrace.logging_setup import get_logger

logger = get_logger(__name__)


def collect_daily(
    race_date: date | None = None,
    venue_ids: list[str] | None = None,
    include_lookback: bool = True,
    prefer_openapi: bool = True,
) -> dict[str, Any]:
    """
    当日（＋直近N日）データを収集。
    既定では Boatrace Open API(JSON) を優先し、失敗時は公式HTMLへフォールバック。
    """
    settings = get_settings()
    target = race_date or date.today()
    openapi = OpenApiCollector()
    official = OfficialCollector()
    tide = TideCollector()

    results: dict[str, Any] = {"date": target.isoformat(), "days": []}

    dates = [target]
    if include_lookback:
        for i in range(1, settings.collect.lookback_days + 1):
            dates.append(target - timedelta(days=i))

    for d in dates:
        logger.info("collect_start", date=d.isoformat())
        day: dict[str, Any] = {"date": d.isoformat()}

        used = None
        if prefer_openapi:
            try:
                day["openapi"] = openapi.collect(d, venue_ids=venue_ids)
                used = "openapi"
            except Exception as e:  # noqa: BLE001
                logger.warning("openapi_collect_failed", date=d.isoformat(), error=str(e))
                day["openapi_error"] = str(e)

        if used is None:
            try:
                day["official"] = official.collect(d, venue_ids=venue_ids)
                used = "official"
            except Exception as e:  # noqa: BLE001
                logger.exception("official_collect_error", date=d.isoformat(), error=str(e))
                day["official_error"] = str(e)

        try:
            day["tide"] = tide.collect(d, venue_ids=venue_ids)
        except Exception as e:  # noqa: BLE001
            logger.exception("tide_collect_error", date=d.isoformat(), error=str(e))
            day["tide_error"] = str(e)

        day["primary_source"] = used
        results["days"].append(day)
        logger.info("collect_done", date=d.isoformat(), primary=used, day=day)

    return results
