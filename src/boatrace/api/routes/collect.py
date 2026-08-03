"""API routes: collect / prepare day."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from boatrace.collectors.pipeline import prepare_day
from boatrace.db.session import get_db
from boatrace.prediction.service import PredictionService

router = APIRouter()


@router.post("/day/prepare")
def prepare_and_predict(
    day: Optional[str] = Query(None, description="YYYY-MM-DD"),
    venue_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> dict:
    """
    指定日の出走表・オッズを取得してから予想する。
    日付変更時にフロントから呼ばれる。
    """
    target = date.today() if not day else datetime.strptime(day, "%Y-%m-%d").date()
    venues = [venue_id] if venue_id else None
    collect_summary = prepare_day(race_date=target, venue_ids=venues)
    svc = PredictionService(db, model="lgbm")
    items = svc.predict_day(target, venue_id=venue_id, persist=True)
    return {
        "date": target.isoformat(),
        "collected": collect_summary,
        "count": len(items),
        "items": items,
    }
