"""API routes: collect / prepare day."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from boatrace.collectors.pipeline import prepare_day
from boatrace.db.models import PredictHistory, RaceCard
from boatrace.db.session import get_db
from boatrace.prediction.api_items import prediction_item_from_db
from boatrace.prediction.service import PredictionService
from boatrace.timeutil import japan_today

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


@router.post("/day/predict")
def predict_day_races(
    day: Optional[str] = Query(None, description="YYYY-MM-DD"),
    venue_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> dict:
    """出走表取得済みのレースを予想する（再取得なし）。"""
    target = japan_today() if not day else datetime.strptime(day, "%Y-%m-%d").date()
    venues = [venue_id] if venue_id else None

    race_q = db.query(RaceCard.id).filter(RaceCard.race_date == target)
    if venues:
        race_q = race_q.filter(RaceCard.venue_id.in_(venues))
    if race_q.limit(1).first() is None:
        return {
            "date": target.isoformat(),
            "error": "no_race_cards",
            "message": "出走表がありません。先に出走表を取得してください。",
            "count": 0,
            "confident_count": 0,
        }

    svc = PredictionService(db, model="lgbm")
    svc.predict_day(target, venue_id=venue_id, persist=True)
    items = _items_from_db(db, target, venue_id)
    confident_count = sum(
        1 for item in items if item.get("is_confident") or (item.get("confidence") or {}).get("is_confident")
    )
    out = {
        "date": target.isoformat(),
        "venue_id": venue_id,
        "count": len(items),
        "confident_count": confident_count,
    }
    if venue_id:
        out["items"] = items
    return out


@router.post("/day/fetch-cards")
def fetch_day_cards(
    day: Optional[str] = Query(None, description="YYYY-MM-DD"),
    venue_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> dict:
    """出走表だけ取得（予想なし）。開催場一覧を埋める用。"""
    from boatrace.jobs.today_bootstrap import ensure_day_cards

    target = japan_today() if not day else datetime.strptime(day, "%Y-%m-%d").date()
    venues = [venue_id] if venue_id else None
    boot = ensure_day_cards(target, force=True, venue_ids=venues)
    db.expire_all()

    counts = dict(
        db.query(RaceCard.venue_id, func.count(RaceCard.id))
        .filter(RaceCard.race_date == target)
        .group_by(RaceCard.venue_id)
        .all()
    )
    return {
        "date": target.isoformat(),
        "japan_today": japan_today().isoformat(),
        "venue_count": len(counts),
        "race_count": int(sum(counts.values())),
        "venues": sorted(counts.keys()),
        "collected": boot.get("collected") or boot,
    }


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
    target = japan_today() if not day else datetime.strptime(day, "%Y-%m-%d").date()
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
