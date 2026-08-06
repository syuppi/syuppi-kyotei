"""API routes: venues."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from boatrace.db.models import RaceCard, Venue, VenueBias, VenueCourseStats
from boatrace.db.session import get_db

router = APIRouter()


@router.get("/venues")
def list_venues(
    day: Optional[str] = Query(None, description="YYYY-MM-DD。指定時はその日開催場のみ"),
    db: Session = Depends(get_db),
) -> dict:
    """場一覧。day 指定時は RaceCard がある開催場だけ返す。"""
    if day:
        target = datetime.strptime(day, "%Y-%m-%d").date()
        counts = dict(
            db.query(RaceCard.venue_id, func.count(RaceCard.id))
            .filter(RaceCard.race_date == target)
            .group_by(RaceCard.venue_id)
            .all()
        )
        if not counts:
            return {"day": target.isoformat(), "items": [], "active_only": True}
        rows = (
            db.query(Venue)
            .filter(Venue.id.in_(list(counts.keys())))
            .order_by(Venue.id)
            .all()
        )
        return {
            "day": target.isoformat(),
            "active_only": True,
            "items": [
                {
                    "id": v.id,
                    "name": v.name,
                    "prefecture": v.prefecture,
                    "tide_sensitive": v.tide_sensitive,
                    "water_type": v.water_type,
                    "tide_station": v.tide_station,
                    "typical_in_advantage": v.typical_in_advantage,
                    "race_count": int(counts.get(v.id) or 0),
                }
                for v in rows
            ],
        }

    rows = db.query(Venue).order_by(Venue.id).all()
    return {
        "active_only": False,
        "items": [
            {
                "id": v.id,
                "name": v.name,
                "prefecture": v.prefecture,
                "tide_sensitive": v.tide_sensitive,
                "water_type": v.water_type,
                "tide_station": v.tide_station,
                "typical_in_advantage": v.typical_in_advantage,
            }
            for v in rows
        ],
    }


@router.get("/venues/{venue_id}/trends")
def venue_trends(venue_id: str, db: Session = Depends(get_db)) -> dict:
    venue = db.get(Venue, venue_id)
    if not venue:
        return {"error": "not_found"}
    stats = (
        db.query(VenueCourseStats)
        .filter_by(venue_id=venue_id)
        .order_by(VenueCourseStats.condition_key, VenueCourseStats.course)
        .all()
    )
    biases = db.query(VenueBias).filter_by(venue_id=venue_id).all()
    return {
        "venue": {
            "id": venue.id,
            "name": venue.name,
            "tide_sensitive": venue.tide_sensitive,
            "typical_in_advantage": venue.typical_in_advantage,
        },
        "course_stats": [
            {
                "course": s.course,
                "condition_key": s.condition_key,
                "starts": s.starts,
                "win_rate": s.win_rate,
                "quinella_rate": s.quinella_rate,
                "trio_rate": s.trio_rate,
            }
            for s in stats
        ],
        "biases": [
            {
                "feature_key": b.feature_key,
                "coefficient": b.coefficient,
                "sample_count": b.sample_count,
            }
            for b in biases
        ],
    }
