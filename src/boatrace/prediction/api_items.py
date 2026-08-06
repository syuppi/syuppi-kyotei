"""予想一覧API用の共通シリアライズ."""

from __future__ import annotations

from typing import Any

from boatrace.db.models import PredictHistory, RaceCard
from boatrace.prediction.review import review_from_db_row


def prediction_item_from_db(card: RaceCard, pred: PredictHistory) -> dict[str, Any]:
    snap = pred.feature_snapshot or {}
    tickets = snap.get("tickets") or {}
    result_block = {
        "rank1": card.result.rank1_waku if card.result else None,
        "rank2": card.result.rank2_waku if card.result else None,
        "rank3": card.result.rank3_waku if card.result else None,
        "kimarite": card.result.kimarite if card.result else None,
        "entry_results": card.result.entry_results if card.result else None,
    }
    review = review_from_db_row(
        pred_tickets=tickets,
        feature_snapshot=snap,
        rankings=pred.rankings,
        win_probs=pred.win_probs,
        upset_candidates=pred.upset_candidates,
        race_result=card.result,
    )
    return {
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
        "race_thesis": snap.get("race_thesis") or "",
        "ticket_reasons": snap.get("ticket_reasons") or {},
        "styles": snap.get("styles") or {},
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
        "result": result_block,
        "review": review,
    }
