"""RaceEntry の前走情報を履歴から埋める."""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy.orm import Session, joinedload

from boatrace.db.models import RaceCard
from boatrace.features.history_index import get_history_index
from boatrace.logging_setup import get_logger

logger = get_logger(__name__)


def backfill_previous_for_day(
    session: Session,
    race_date: date,
    venue_ids: list[str] | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    """指定日の entry.previous_* を HistoryIndex から補完."""
    hist = get_history_index(session)
    q = (
        session.query(RaceCard)
        .options(joinedload(RaceCard.entries))
        .filter(RaceCard.race_date == race_date)
    )
    if venue_ids:
        q = q.filter(RaceCard.venue_id.in_(venue_ids))
    cards = q.all()
    scanned = updated = skipped = 0
    for card in cards:
        for entry in card.entries:
            scanned += 1
            if (
                not overwrite
                and entry.previous_rank is not None
                and entry.previous_st is not None
            ):
                skipped += 1
                continue
            if not entry.racer_id:
                continue
            prev = hist.last_start(
                entry.racer_id, card.race_date, before_race_no=card.race_no
            )
            if prev is None:
                continue
            entry.previous_rank = int(prev.rank)
            entry.previous_course = int(prev.course)
            entry.previous_st = float(prev.st) if prev.st is not None else None
            updated += 1
    session.flush()
    logger.info(
        "backfill_previous_day",
        date=race_date.isoformat(),
        scanned=scanned,
        updated=updated,
        skipped=skipped,
    )
    return {"scanned": scanned, "updated": updated, "skipped": skipped}
