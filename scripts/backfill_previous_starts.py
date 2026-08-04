"""RaceEntry.previous_* を履歴から埋める."""

from __future__ import annotations

from boatrace.db.models import RaceCard, RaceEntry
from boatrace.db.session import session_scope
from boatrace.features.history_index import get_history_index, reset_history_index
from boatrace.logging_setup import get_logger
from sqlalchemy.orm import joinedload

logger = get_logger(__name__)


def backfill_previous_starts(overwrite: bool = False) -> dict[str, int]:
    reset_history_index()
    updated = scanned = skipped = 0
    with session_scope() as s:
        hist = get_history_index(s)
        cards = (
            s.query(RaceCard)
            .options(joinedload(RaceCard.entries))
            .order_by(RaceCard.race_date, RaceCard.venue_id, RaceCard.race_no)
            .all()
        )
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
                prev = hist.last_start(
                    entry.racer_id, card.race_date, before_race_no=card.race_no
                )
                if prev is None:
                    continue
                entry.previous_rank = int(prev.rank)
                entry.previous_course = int(prev.course)
                entry.previous_st = float(prev.st) if prev.st is not None else None
                updated += 1
        s.flush()
    logger.info("backfill_previous_done", scanned=scanned, updated=updated, skipped=skipped)
    return {"scanned": scanned, "updated": updated, "skipped": skipped}


if __name__ == "__main__":
    print(backfill_previous_starts(overwrite=True))
