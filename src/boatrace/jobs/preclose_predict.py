"""締切前の自動予想（結果後再予想による偽的中を防ぐ）."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import joinedload

from boatrace.config import get_settings
from boatrace.db.models import PredictHistory, RaceCard
from boatrace.db.session import session_scope
from boatrace.logging_setup import get_logger
from boatrace.prediction.timing import classify_prediction_timing
from boatrace.timeutil import JST, japan_now, japan_today

logger = get_logger(__name__)

_stop = threading.Event()
_thread: threading.Thread | None = None
_last_run: dict[str, Any] = {"at": None}


def _deadline_jst(card: RaceCard) -> datetime | None:
    raw = card.deadline_at
    if raw is None:
        return None
    if raw.tzinfo is None:
        return raw.replace(tzinfo=JST)
    return raw.astimezone(JST)


def _is_preclose_prediction(card: RaceCard, pred: PredictHistory) -> bool:
    """予想時刻が締切前か（未確定レースでも判定可能）."""
    from boatrace.prediction.timing import _DEADLINE_GRACE, _as_jst

    predicted_at = pred.predicted_at
    deadline = _deadline_jst(card)
    if predicted_at is None or deadline is None:
        return False
    pred_jst = _as_jst(predicted_at)
    if pred_jst is None:
        return False
    return pred_jst <= deadline + _DEADLINE_GRACE


def _needs_preclose_predict(card: RaceCard, pred: PredictHistory | None, *, lead_minutes: int) -> bool:
    if card.status == "finished":
        return False
    if card.result is not None and card.result.rank1_waku is not None:
        return False
    if len(card.entries) < 6:
        return False

    deadline = _deadline_jst(card)
    if deadline is None:
        return False

    now = japan_now()
    if now >= deadline:
        return False

    trigger_at = deadline - timedelta(minutes=lead_minutes)
    if now < trigger_at:
        return False

    if pred is None:
        return True

    if _is_preclose_prediction(card, pred):
        return False
    return True


def run_preclose_predict(
    *,
    lead_minutes: int | None = None,
    race_date=None,
) -> dict[str, Any]:
    """締切 lead_minutes 前〜締切までの未予想/再予想が必要なレースを予想する."""
    from boatrace.prediction.confidence import enforce_daily_confidence_cap
    from boatrace.prediction.service import PredictionService

    settings = get_settings()
    lead = int(
        lead_minutes
        if lead_minutes is not None
        else getattr(settings.jobs, "preclose_lead_minutes", 45)
        or 45
    )
    target = race_date or japan_today()
    now = japan_now()

    predicted = skipped = errors = 0
    race_ids: list[int] = []

    with session_scope() as session:
        cards = (
            session.query(RaceCard)
            .options(joinedload(RaceCard.entries), joinedload(RaceCard.result))
            .filter(RaceCard.race_date == target)
            .order_by(RaceCard.venue_id, RaceCard.race_no)
            .all()
        )

        svc = PredictionService(session, model="lgbm")
        for card in cards:
            pred = (
                session.query(PredictHistory)
                .filter(PredictHistory.race_card_id == card.id)
                .order_by(PredictHistory.predicted_at.desc())
                .first()
            )
            if not _needs_preclose_predict(card, pred, lead_minutes=lead):
                skipped += 1
                continue
            try:
                svc.predict_race(card.id, persist=True)
                predicted += 1
                race_ids.append(card.id)
            except Exception as e:  # noqa: BLE001
                errors += 1
                logger.warning(
                    "preclose_predict_failed",
                    race_card_id=card.id,
                    venue_id=card.venue_id,
                    race_no=card.race_no,
                    error=str(e),
                )

        if predicted and settings.prediction.confidence_enabled:
            try:
                enforce_daily_confidence_cap(session, target)
            except Exception as e:  # noqa: BLE001
                logger.warning("preclose_confidence_cap_failed", error=str(e))

        session.commit()

    payload = {
        "at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "date": target.isoformat(),
        "japan_now": now.isoformat(),
        "lead_minutes": lead,
        "predicted": predicted,
        "skipped": skipped,
        "errors": errors,
        "race_card_ids": race_ids,
    }
    _last_run.clear()
    _last_run.update(payload)
    logger.info(
        "preclose_predict_done",
        predicted=predicted,
        skipped=skipped,
        errors=errors,
    )
    return payload


def last_preclose_run() -> dict[str, Any]:
    return dict(_last_run)


def preclose_status(*, lead_minutes: int | None = None) -> dict[str, Any]:
    """本日レースの締切前予想カバー状況."""
    settings = get_settings()
    lead = int(
        lead_minutes
        if lead_minutes is not None
        else getattr(settings.jobs, "preclose_lead_minutes", 45)
        or 45
    )
    today = japan_today()
    now = japan_now()
    pending = pre_close = post_close = no_pred = 0

    with session_scope() as session:
        cards = (
            session.query(RaceCard)
            .options(joinedload(RaceCard.result))
            .filter(RaceCard.race_date == today)
            .all()
        )
        for card in cards:
            if card.result is not None and card.result.rank1_waku is not None:
                continue
            pred = (
                session.query(PredictHistory)
                .filter(PredictHistory.race_card_id == card.id)
                .order_by(PredictHistory.predicted_at.desc())
                .first()
            )
            if pred is None:
                no_pred += 1
                continue
            if _is_preclose_prediction(card, pred):
                pre_close += 1
                continue
            status = classify_prediction_timing(card, pred).get("status")
            if status == "post_close":
                post_close += 1
            else:
                pending += 1

    return {
        "date": today.isoformat(),
        "japan_now": now.isoformat(),
        "lead_minutes": lead,
        "pre_close": pre_close,
        "post_close": post_close,
        "no_prediction": no_pred,
        "pending": pending,
        "last_run": last_preclose_run() or None,
    }


def _loop(interval_sec: int, lead_minutes: int) -> None:
    while not _stop.is_set():
        try:
            run_preclose_predict(lead_minutes=lead_minutes)
        except Exception as e:  # noqa: BLE001
            logger.exception("preclose_predict_loop_error", error=str(e))
        _stop.wait(interval_sec)


def start_background_preclose_predict(
    interval_sec: int | None = None,
    lead_minutes: int | None = None,
) -> None:
    """APIプロセス内で締切前予想を常駐させる."""
    global _thread
    if _thread and _thread.is_alive():
        return
    settings = get_settings()
    interval = int(
        interval_sec
        if interval_sec is not None
        else getattr(settings.jobs, "preclose_predict_interval_sec", 300)
        or 300
    )
    lead = int(
        lead_minutes
        if lead_minutes is not None
        else getattr(settings.jobs, "preclose_lead_minutes", 45)
        or 45
    )
    _stop.clear()
    _thread = threading.Thread(
        target=_loop,
        args=(max(60, interval), lead),
        name="preclose-predict",
        daemon=True,
    )
    _thread.start()
    logger.info("preclose_predict_started", interval_sec=interval, lead_minutes=lead)


def stop_background_preclose_predict() -> None:
    _stop.set()
