"""API routes: accuracy."""

from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from boatrace.db.models import AccuracyDaily
from boatrace.db.session import get_db
from boatrace.learning.service import LearningService

router = APIRouter()


@router.get("/accuracy/summary")
def accuracy_summary(days: int = Query(30, ge=1, le=365), db: Session = Depends(get_db)) -> dict:
    svc = LearningService(db)
    return svc.accuracy_summary(days=days)


@router.get("/accuracy/daily")
def accuracy_daily(
    days: int = Query(14, ge=1, le=90),
    venue_id: str | None = None,
    db: Session = Depends(get_db),
) -> dict:
    since = date.today() - timedelta(days=days)
    q = db.query(AccuracyDaily).filter(AccuracyDaily.stat_date >= since)
    if venue_id:
        q = q.filter(AccuracyDaily.venue_id == venue_id)
    else:
        q = q.filter(AccuracyDaily.venue_id.is_(None))
    rows = q.order_by(AccuracyDaily.stat_date.desc(), AccuracyDaily.slice_key).all()
    return {
        "items": [
            {
                "stat_date": r.stat_date.isoformat(),
                "venue_id": r.venue_id,
                "slice_key": r.slice_key,
                "model_name": r.model_name,
                "n_races": r.n_races,
                "win_rate": r.win_rate,
                "quinella_rate": r.quinella_rate,
                "trio_rate": r.trio_rate,
                "trifecta_rate": getattr(r, "trifecta_rate", 0.0) or 0.0,
            }
            for r in rows
        ]
    }
