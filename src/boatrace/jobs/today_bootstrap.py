"""当日出走表の確実な投入（Renderスリープ復帰・空DB対策）."""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import func

from boatrace.logging_setup import get_logger
from boatrace.timeutil import japan_today

logger = get_logger(__name__)


def count_cards_for_day(target: date) -> tuple[int, int, list[str]]:
    from boatrace.db.models import RaceCard
    from boatrace.db.session import session_scope

    with session_scope() as session:
        rows = (
            session.query(RaceCard.venue_id, func.count(RaceCard.id))
            .filter(RaceCard.race_date == target)
            .group_by(RaceCard.venue_id)
            .all()
        )
    venues = sorted(vid for vid, _ in rows)
    race_count = int(sum(n for _, n in rows))
    return race_count, len(venues), venues


def ensure_day_cards(
    target: date | None = None,
    *,
    force: bool = False,
    venue_ids: list[str] | None = None,
) -> dict[str, Any]:
    from boatrace.collectors.openapi import OpenApiCollector

    day = target or japan_today()
    race_count, venue_count, venues = count_cards_for_day(day)
    out: dict[str, Any] = {
        "date": day.isoformat(),
        "japan_today": japan_today().isoformat(),
        "force": force,
        "before": {"race_count": race_count, "venue_count": venue_count, "venues": venues},
    }
    if race_count > 0 and not force:
        out["skipped"] = True
        out["race_count"] = race_count
        out["venue_count"] = venue_count
        out["venues"] = venues
        out["ok"] = True
        return out

    summary: dict[str, Any] = {}
    try:
        summary["openapi"] = OpenApiCollector().collect(day, venue_ids=venue_ids)
        logger.info("today_bootstrap_openapi_ok", date=day.isoformat(), summary=summary["openapi"])
    except Exception as e:  # noqa: BLE001
        summary["openapi_error"] = str(e)
        logger.warning("today_bootstrap_openapi_failed", date=day.isoformat(), error=str(e))
        try:
            from boatrace.collectors.official import OfficialCollector

            summary["official"] = OfficialCollector().collect(day, venue_ids=venue_ids)
        except Exception as e2:  # noqa: BLE001
            summary["official_error"] = str(e2)

    race_count, venue_count, venues = count_cards_for_day(day)
    out.update(
        {
            "skipped": False,
            "collected": summary,
            "race_count": race_count,
            "venue_count": venue_count,
            "venues": venues,
            "ok": race_count > 0,
        }
    )
    return out


def ensure_today_cards(*, force: bool = False) -> dict[str, Any]:
    return ensure_day_cards(japan_today(), force=force)
