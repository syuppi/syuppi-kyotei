"""API routes: search."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, joinedload

from boatrace.db.models import RaceCard, RaceEntry, Racer
from boatrace.db.session import get_db

router = APIRouter()


@router.get("/search")
def search(
    venue_id: Optional[str] = None,
    racer_id: Optional[str] = None,
    waku: Optional[int] = Query(None, ge=1, le=6),
    min_wind: Optional[float] = None,
    near_high_tide: Optional[bool] = None,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict:
    q = (
        db.query(RaceEntry)
        .join(RaceCard)
        .options(
            joinedload(RaceEntry.race_card).joinedload(RaceCard.venue),
            joinedload(RaceEntry.race_card).joinedload(RaceCard.weather),
            joinedload(RaceEntry.race_card).joinedload(RaceCard.tide),
            joinedload(RaceEntry.race_card).joinedload(RaceCard.result),
            joinedload(RaceEntry.racer),
        )
    )
    if venue_id:
        q = q.filter(RaceCard.venue_id == venue_id)
    if racer_id:
        q = q.filter(RaceEntry.racer_id == racer_id)
    if waku:
        q = q.filter(RaceEntry.waku == waku)
    if min_wind is not None:
        from boatrace.db.models import WeatherSnapshot

        q = q.join(WeatherSnapshot, WeatherSnapshot.race_card_id == RaceCard.id).filter(
            WeatherSnapshot.wind_speed >= min_wind
        )
    if near_high_tide is not None:
        from boatrace.db.models import TideSnapshot

        q = q.join(TideSnapshot, TideSnapshot.race_card_id == RaceCard.id).filter(
            TideSnapshot.near_high_tide == near_high_tide
        )

    rows = q.order_by(RaceCard.race_date.desc(), RaceCard.race_no).limit(limit).all()
    items = []
    for e in rows:
        card = e.race_card
        rank = None
        if card.result and card.result.entry_results:
            for er in card.result.entry_results:
                if er.get("waku") == e.waku:
                    rank = er.get("rank")
        items.append(
            {
                "venue_id": card.venue_id,
                "venue_name": card.venue.name if card.venue else None,
                "race_date": card.race_date.isoformat(),
                "race_no": card.race_no,
                "waku": e.waku,
                "racer_id": e.racer_id,
                "racer_name": e.racer.name if e.racer else None,
                "local_win_rate": e.local_win_rate,
                "exhibition_time": e.exhibition_time,
                "wind_speed": card.weather.wind_speed if card.weather else None,
                "wind_direction": card.weather.wind_direction if card.weather else None,
                "near_high_tide": card.tide.near_high_tide if card.tide else None,
                "result_rank": rank,
            }
        )
    return {"count": len(items), "items": items}


@router.get("/racers/{racer_id}")
def racer_detail(racer_id: str, db: Session = Depends(get_db)) -> dict:
    racer = db.get(Racer, racer_id)
    if not racer:
        return {"error": "not_found"}
    return {"id": racer.id, "name": racer.name, "branch": racer.branch, "grade": racer.grade}
