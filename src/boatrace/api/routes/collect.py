"""API routes: collect / prepare day."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, joinedload

from boatrace.collectors.pipeline import prepare_day
from boatrace.db.models import PredictHistory, RaceCard
from boatrace.db.session import get_db
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
        snap = pred.feature_snapshot or {}
        tickets = snap.get("tickets") or {}
        items.append(
            {
                "race_card_id": card.id,
                "venue_id": card.venue_id,
                "venue_name": card.venue.name if card.venue else card.venue_id,
                "race_date": card.race_date.isoformat(),
                "race_no": card.race_no,
                "race_title": card.race_title,
                "status": card.status,
                "model_name": pred.model_name,
                "rankings": pred.rankings,
                "win_probs": pred.win_probs,
                "quinella_probs": pred.quinella_probs,
                "trio_probs": pred.trio_probs,
                "candidates_win": pred.candidates_win,
                "candidates_quinella": pred.candidates_quinella,
                "candidates_trio": pred.candidates_trio,
                "upset_candidates": pred.upset_candidates,
                "has_upset": pred.has_upset,
                "reasons": pred.reasons,
                "tickets": tickets,
                "exhibition": snap.get("exhibition"),
                "scenarios": (snap.get("scenarios") or {}).get("comments") or [],
                "scenario_detail": snap.get("scenarios"),
                "ev_reasons": snap.get("ev_reasons") or [],
                "has_odds": bool(snap.get("has_odds")),
                "sanrentan": [
                    t.get("combo") for t in tickets.get("sanrentan", []) if t.get("combo")
                ]
                or snap.get("sanrentan")
                or ([pred.rankings[:3]] if pred.rankings else []),
                "sanrenpuku": [
                    t.get("combo") for t in tickets.get("sanrenpuku", []) if t.get("combo")
                ]
                or snap.get("sanrenpuku")
                or ([sorted(pred.candidates_trio[:3])] if pred.candidates_trio else []),
                "result": {
                    "rank1": card.result.rank1_waku if card.result else None,
                    "rank2": card.result.rank2_waku if card.result else None,
                    "rank3": card.result.rank3_waku if card.result else None,
                },
            }
        )
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
    items = svc.predict_day(target, venue_id=venue_id, persist=True)
    return {
        "date": target.isoformat(),
        "collected": collected,
        "count": len(items),
        "items": items,
        "fast": fast,
        "cached": False,
    }
