"""API routes: predictions."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, joinedload

from boatrace.db.models import PredictHistory, RaceCard
from boatrace.db.session import get_db
from boatrace.prediction.service import PredictionService

router = APIRouter()


@router.get("/predictions/today")
def predictions_today(
    day: Optional[str] = Query(None, description="YYYY-MM-DD"),
    venue_id: Optional[str] = None,
    refresh: bool = False,
    db: Session = Depends(get_db),
) -> dict:
    target = date.today() if not day else datetime.strptime(day, "%Y-%m-%d").date()
    if refresh:
        svc = PredictionService(db)
        items = svc.predict_day(target, venue_id=venue_id, persist=True)
        return {"date": target.isoformat(), "count": len(items), "items": items}

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
                "tickets": (pred.feature_snapshot or {}).get("tickets") or {},
                "exhibition": (pred.feature_snapshot or {}).get("exhibition"),
                "scenarios": ((pred.feature_snapshot or {}).get("scenarios") or {}).get(
                    "comments"
                )
                or [],
                "scenario_detail": (pred.feature_snapshot or {}).get("scenarios"),
                "sanrentan": [
                    t.get("combo")
                    for t in ((pred.feature_snapshot or {}).get("tickets") or {}).get(
                        "sanrentan", []
                    )
                ]
                or (pred.feature_snapshot or {}).get("sanrentan")
                or ([pred.rankings[:3]] if pred.rankings else []),
                "sanrenpuku": [
                    t.get("combo")
                    for t in ((pred.feature_snapshot or {}).get("tickets") or {}).get(
                        "sanrenpuku", []
                    )
                ]
                or (pred.feature_snapshot or {}).get("sanrenpuku")
                or ([sorted(pred.candidates_trio[:3])] if pred.candidates_trio else []),
                "result": {
                    "rank1": card.result.rank1_waku if card.result else None,
                    "rank2": card.result.rank2_waku if card.result else None,
                    "rank3": card.result.rank3_waku if card.result else None,
                },
            }
        )
    return {"date": target.isoformat(), "count": len(items), "items": items}


@router.get("/predictions/{race_card_id}")
def prediction_detail(race_card_id: int, db: Session = Depends(get_db)) -> dict:
    card = (
        db.query(RaceCard)
        .options(
            joinedload(RaceCard.entries),
            joinedload(RaceCard.venue),
            joinedload(RaceCard.weather),
            joinedload(RaceCard.tide),
            joinedload(RaceCard.result),
        )
        .filter(RaceCard.id == race_card_id)
        .one_or_none()
    )
    if not card:
        return {"error": "not_found"}
    pred = (
        db.query(PredictHistory)
        .filter(PredictHistory.race_card_id == race_card_id)
        .order_by(PredictHistory.predicted_at.desc())
        .first()
    )
    return {
        "card": {
            "id": card.id,
            "venue_id": card.venue_id,
            "venue_name": card.venue.name if card.venue else None,
            "race_date": card.race_date.isoformat(),
            "race_no": card.race_no,
            "is_fixed_entry": card.is_fixed_entry,
            "weather": {
                "temperature": card.weather.temperature if card.weather else None,
                "weather": card.weather.weather if card.weather else None,
                "wind_speed": card.weather.wind_speed if card.weather else None,
                "wind_direction": card.weather.wind_direction if card.weather else None,
                "water_temperature": card.weather.water_temperature if card.weather else None,
                "wave_height": card.weather.wave_height if card.weather else None,
            },
            "tide": {
                "level_cm": card.tide.tide_level_cm if card.tide else None,
                "delta_cm": card.tide.tide_delta_cm if card.tide else None,
                "near_high": card.tide.near_high_tide if card.tide else None,
                "source": card.tide.source if card.tide else None,
            },
            "entries": [
                {
                    "waku": e.waku,
                    "racer_id": e.racer_id,
                    "racer_name": e.racer.name if e.racer else None,
                    "local_win_rate": e.local_win_rate,
                    "national_win_rate": e.national_win_rate,
                    "avg_st": e.avg_st,
                    "exhibition_time": e.exhibition_time,
                    "exhibition_st": e.exhibition_st,
                    "estimated_course": e.estimated_course,
                    "tilt": e.tilt,
                    "weight_adjustment": e.weight_adjustment,
                    "motor_quinella_rate": e.motor_quinella_rate,
                    "boat_quinella_rate": e.boat_quinella_rate,
                }
                for e in sorted(card.entries, key=lambda x: x.waku)
            ],
        },
        "prediction": {
            "model_name": pred.model_name if pred else None,
            "rankings": pred.rankings if pred else None,
            "win_probs": pred.win_probs if pred else None,
            "candidates_win": pred.candidates_win if pred else None,
            "candidates_quinella": pred.candidates_quinella if pred else None,
            "candidates_trio": pred.candidates_trio if pred else None,
            "upset_candidates": pred.upset_candidates if pred else None,
            "has_upset": pred.has_upset if pred else None,
            "reasons": pred.reasons if pred else None,
            "exhibition": (pred.feature_snapshot or {}).get("exhibition") if pred else None,
            "scenarios": ((pred.feature_snapshot or {}).get("scenarios") or {}).get(
                "comments"
            )
            if pred
            else None,
            "scenario_detail": (pred.feature_snapshot or {}).get("scenarios") if pred else None,
            "tickets": (pred.feature_snapshot or {}).get("tickets") if pred else None,
        },
        "result": {
            "rank1": card.result.rank1_waku if card.result else None,
            "rank2": card.result.rank2_waku if card.result else None,
            "rank3": card.result.rank3_waku if card.result else None,
            "kimarite": card.result.kimarite if card.result else None,
        },
    }
