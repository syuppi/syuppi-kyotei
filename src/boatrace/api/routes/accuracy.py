"""API routes: accuracy."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from boatrace.db.models import AccuracyDaily
from boatrace.db.session import get_db
from boatrace.learning.service import LearningService

router = APIRouter()


@router.get("/accuracy/summary")
def accuracy_summary(days: int = Query(30, ge=1, le=365), db: Session = Depends(get_db)) -> dict:
    from boatrace.prediction.offline_eval import get_offline_accuracy_summary

    svc = LearningService(db)
    out = svc.accuracy_summary(days=days)
    reference = get_offline_accuracy_summary()
    out["offline_reference"] = reference
    if int(out.get("n_races") or 0) <= 0:
        out["source"] = "offline_eval_report"
        out["display_note"] = reference.get("note")
        out["trio_rate"] = reference.get("trio_rate", out.get("trio_rate"))
        out["trifecta_rate"] = reference.get("trifecta_rate", out.get("trifecta_rate"))
        out["win_rate"] = reference.get("win_rate", out.get("win_rate"))
        out["reference_n_races"] = reference.get("n_races")
    return out


@router.get("/accuracy/offline-reference")
def accuracy_offline_reference() -> dict:
    """GitHub 同梱のオフライン解析レポート（参考精度）."""
    from boatrace.prediction.offline_eval import (
        get_baseline_ticket_rank_stats,
        get_holdout_reference,
        get_offline_accuracy_summary,
        report_status,
    )

    return {
        "status": report_status(),
        "summary": get_offline_accuracy_summary(),
        "holdout": get_holdout_reference(),
        "ticket_ranks": get_baseline_ticket_rank_stats(),
    }


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


@router.get("/accuracy/ticket-ranks")
def accuracy_ticket_ranks(
    days: int = Query(14, ge=1, le=90),
    db: Session = Depends(get_db),
) -> dict:
    """候補1〜5番手ごとの過去的中率（UIガイド用）."""
    from boatrace.prediction.integrity_audit import audit_predictions
    from boatrace.prediction.ticket_rank_stats import get_ticket_rank_stats

    out = get_ticket_rank_stats(db, days=days)
    audit = audit_predictions(db, days=days)
    pre = audit.get("ticket_ranks", {}).get("pre_close_only") or {}
    pre_n = int((pre.get("sanrenpuku") or {}).get("n_races") or 0)
    risk = audit.get("published_risk") or {}

    out["integrity"] = {
        "audit": audit,
        "pre_close_ticket_ranks": pre if pre_n >= 80 else None,
        "use_baseline_for_display": bool(risk.get("live_stats_unreliable")) or pre_n < 80,
    }
    if out["integrity"]["use_baseline_for_display"]:
        out["display"] = out.get("baseline") or out.get("display")
        out["display_note"] = (
            "締切前の保存予想が不足しているため、GitHub同梱のオフライン検証ベースラインを表示しています。"
            "ライブ集計は結果後の再予想を含む可能性があります。"
        )
    else:
        live = out.get("live") or out.get("display")
        if live:
            live = dict(live)
            live["label"] = "締切前保存予想（監査済み）"
            out["display"] = live
            out["display_note"] = "締切前の予想のみを集計しています。"
    return out


@router.get("/accuracy/integrity-audit")
def accuracy_integrity_audit(
    days: int = Query(14, ge=1, le=90),
    day: Optional[str] = Query(None, description="YYYY-MM-DD"),
    db: Session = Depends(get_db),
) -> dict:
    """公表予想・的中率が結果後再予想を含むか監査し、締切前のみの精度を返す."""
    from datetime import datetime

    from boatrace.prediction.integrity_audit import audit_predictions

    target = datetime.strptime(day, "%Y-%m-%d").date() if day else None
    return audit_predictions(db, days=days, day=target)
