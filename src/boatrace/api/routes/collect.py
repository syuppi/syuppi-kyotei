"""API routes: collect / prepare day."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, joinedload

from boatrace.collectors.pipeline import prepare_day
from boatrace.db.models import PredictHistory, RaceCard
from boatrace.db.session import get_db
from boatrace.prediction.api_items import prediction_item_from_db
from boatrace.prediction.service import PredictionService

router = APIRouter()


def _items_from_db(db: Session, target: date, venue_id: str | None) -> list[dict]:
    q = (
        db.query(RaceCard)
        .options(joinedload(RaceCard.venue), joinedload(RaceCard.result))
        .filter(RaceCard.race_date == target)
    )
    if venue_id:
        q = q.filter(RaceCard.venue_id == venue_id)
    cards = q.order_by(RaceCard.venue_id, RaceCard.race_no).all()
    items = []
    for card in cards:
        pred = (
            db.query(PredictHistory)
            .filter(PredictHistory.race_card_id == card.id)
            .order_by(PredictHistory.predicted_at.desc())
            .first()
        )
        if not pred:
            continue
        items.append(prediction_item_from_db(card, pred))
    return items


@router.post("/day/prepare")
def prepare_and_predict(
    day: Optional[str] = Query(None, description="YYYY-MM-DD"),
    venue_id: Optional[str] = Query(None),
    fast: bool = Query(True, description="Trueなら潮汐・公式オッズを省略して高速化"),
    force: bool = Query(False, description="既存予想があっても再取得・再予想する"),
    db: Session = Depends(get_db),
) -> dict:
    """
    指定日の出走表・オッズを取得してから予想する。
    Cloudflare 524 回避のため既定は fast=True。
    """
    target = date.today() if not day else datetime.strptime(day, "%Y-%m-%d").date()
    venues = [venue_id] if venue_id else None

    existing = _items_from_db(db, target, venue_id)
    if existing and not force:
        # 予想キャッシュがあっても、着順未取得分は補完する
        try:
            from boatrace.collectors.official import OfficialCollector

            filled = OfficialCollector().collect_missing_results(
                target, venue_ids=venues
            )
        except Exception as e:  # noqa: BLE001
            filled = {"error": str(e)}
        items = _items_from_db(db, target, venue_id)
        return {
            "date": target.isoformat(),
            "collected": {"skipped": True, "reason": "already_prepared", "missing_results": filled},
            "count": len(items),
            "items": items,
            "fast": fast,
            "cached": True,
        }

    collected = prepare_day(race_date=target, venue_ids=venues, fast=fast)
    svc = PredictionService(db, model="lgbm")
    svc.predict_day(target, venue_id=venue_id, persist=True)
    items = _items_from_db(db, target, venue_id)
    return {
        "date": target.isoformat(),
        "collected": collected,
        "count": len(items),
        "items": items,
        "fast": fast,
        "cached": False,
    }


@router.post("/results/refresh")
def refresh_missing_results(
    days: int = Query(2, ge=1, le=7, description="直近何日分の欠損着順を埋めるか"),
) -> dict:
    """着順欠損だけを公式HTMLから補完（全件再取得しない高速パス）."""
    from boatrace.jobs.result_refresh import refresh_recent_missing

    return refresh_recent_missing(days=days)
