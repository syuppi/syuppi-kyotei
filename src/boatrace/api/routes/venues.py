"""API routes: venues."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from boatrace.db.models import Venue, VenueBias, VenueCourseStats
from boatrace.db.session import get_db

router = APIRouter()


@router.get("/venues")
def list_venues(db: Session = Depends(get_db)) -> dict:
    rows = db.query(Venue).order_by(Venue.id).all()
    return {
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
        ]
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
