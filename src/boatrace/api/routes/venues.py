"""API routes: venues."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from boatrace.db.models import PredictHistory, RaceCard, Venue, VenueBias, VenueCourseStats
from boatrace.db.session import get_db
from boatrace.timeutil import japan_today

router = APIRouter()


def _confident_counts_by_venue(db: Session, target: date) -> dict[str, int]:
    """その日の場別・自信ありレース数（PredictHistory.feature_snapshot から）."""
    rows = (
        db.query(RaceCard, PredictHistory)
        .join(PredictHistory, PredictHistory.race_card_id == RaceCard.id)
        .filter(RaceCard.race_date == target)
        .all()
    )
    # 同一カードに複数履歴がある場合は最新を優先
    latest: dict[int, PredictHistory] = {}
    card_venue: dict[int, str] = {}
    for card, pred in rows:
        card_venue[card.id] = card.venue_id
        prev = latest.get(card.id)
        if prev is None or (pred.predicted_at or datetime.min) >= (prev.predicted_at or datetime.min):
            latest[card.id] = pred
    out: dict[str, int] = {}
    for race_id, pred in latest.items():
        snap = pred.feature_snapshot or {}
        conf = snap.get("confidence") or {}
        if conf.get("is_confident"):
            vid = card_venue.get(race_id)
            if vid:
                out[vid] = out.get(vid, 0) + 1
    return out


@router.get("/venues")
def list_venues(
    day: Optional[str] = Query(None, description="YYYY-MM-DD。指定時はその日開催場のみ"),
    confident_only: bool = Query(False, description="自信ありレースがある場だけ"),
    autofetch: bool = Query(True, description="カード0件なら出走表を自動取得"),
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
        fetched = None
        if autofetch and not counts and not confident_only:
            from boatrace.jobs.today_bootstrap import ensure_day_cards

            fetched = ensure_day_cards(target, force=False)
            db.expire_all()
            counts = dict(
                db.query(RaceCard.venue_id, func.count(RaceCard.id))
                .filter(RaceCard.race_date == target)
                .group_by(RaceCard.venue_id)
                .all()
            )
        if not counts:
            return {
                "day": target.isoformat(),
                "items": [],
                "active_only": True,
                "confident_total": 0,
                "autofetched": fetched,
                "japan_today": japan_today().isoformat(),
            }
        conf_counts = _confident_counts_by_venue(db, target)
        venue_ids = list(counts.keys())
        if confident_only:
            venue_ids = [vid for vid in venue_ids if conf_counts.get(vid, 0) > 0]
        rows = (
            db.query(Venue)
            .filter(Venue.id.in_(venue_ids))
            .order_by(Venue.id)
            .all()
        ) if venue_ids else []
        items = [
            {
                "id": v.id,
                "name": v.name,
                "prefecture": v.prefecture,
                "tide_sensitive": v.tide_sensitive,
                "water_type": v.water_type,
                "tide_station": v.tide_station,
                "typical_in_advantage": v.typical_in_advantage,
                "race_count": int(counts.get(v.id) or 0),
                "confident_count": int(conf_counts.get(v.id) or 0),
            }
            for v in rows
        ]
        return {
            "day": target.isoformat(),
            "active_only": True,
            "confident_only": confident_only,
            "confident_total": int(sum(conf_counts.values())),
            "items": items,
            "autofetched": fetched,
            "japan_today": japan_today().isoformat(),
        }

    rows = db.query(Venue).order_by(Venue.id).all()
    return {
        "active_only": False,
        "japan_today": japan_today().isoformat(),
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
