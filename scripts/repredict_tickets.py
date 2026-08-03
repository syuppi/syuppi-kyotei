#!/usr/bin/env python3
"""本日を複数候補チケットで再予想し、カバー的中率を表示."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.config import get_settings
from boatrace.db.models import PredictHistory, RaceCard
from boatrace.db.session import session_scope
from boatrace.learning.service import LearningService
from boatrace.logging_setup import setup_logging
from boatrace.prediction.service import PredictionService


def eval_multi(session, model_name: str = "lgbm_v1") -> dict:
    today = date.today()
    n = win = trio = tf = win1 = trio1 = tf1 = 0
    for c in session.query(RaceCard).filter_by(race_date=today).all():
        p = (
            session.query(PredictHistory)
            .filter_by(race_card_id=c.id, model_name=model_name)
            .order_by(PredictHistory.predicted_at.desc())
            .first()
        )
        if not p or not c.result or not c.result.rank1_waku:
            continue
        n += 1
        tickets = (p.feature_snapshot or {}).get("tickets") or {}
        wins = [t["combo"][0] for t in tickets.get("win", []) if t.get("combo")]
        if not wins:
            wins = list(p.candidates_win or [])[:3]
        sps = [set(t["combo"]) for t in tickets.get("sanrenpuku", []) if t.get("combo")]
        sts = [list(t["combo"]) for t in tickets.get("sanrentan", []) if t.get("combo")]
        true3 = {c.result.rank1_waku, c.result.rank2_waku, c.result.rank3_waku}
        true_ord = [c.result.rank1_waku, c.result.rank2_waku, c.result.rank3_waku]

        win += int(c.result.rank1_waku in wins)
        win1 += int(bool(wins) and wins[0] == c.result.rank1_waku)
        if None not in true3:
            trio += int(any(s == true3 for s in sps))
            trio1 += int(bool(sps) and sps[0] == true3)
        if c.result.rank2_waku and c.result.rank3_waku:
            tf += int(any(s == true_ord for s in sts))
            tf1 += int(bool(sts) and sts[0] == true_ord)
    return {
        "n": n,
        "win_top1": win1 / n if n else 0,
        "win_top3": win / n if n else 0,
        "trio_top1": trio1 / n if n else 0,
        "trio_top3": trio / n if n else 0,
        "trifecta_top1": tf1 / n if n else 0,
        "trifecta_top3": tf / n if n else 0,
    }


def main() -> None:
    setup_logging()
    get_settings.cache_clear()
    today = date.today()
    with session_scope() as session:
        ids = [c.id for c in session.query(RaceCard).filter_by(race_date=today).all()]
        if ids:
            session.query(PredictHistory).filter(
                PredictHistory.race_card_id.in_(ids)
            ).delete(synchronize_session=False)
            session.flush()
        items = PredictionService(session, model="lgbm").predict_day(today, persist=True)
        stats = eval_multi(session)
        LearningService(session).evaluate_accuracy(today)
        print("predicted:", len(items))
        print("coverage:", stats)
        # sample
        shown = 0
        for c in (
            session.query(RaceCard)
            .filter_by(race_date=today)
            .order_by(RaceCard.venue_id, RaceCard.race_no)
        ):
            p = (
                session.query(PredictHistory)
                .filter_by(race_card_id=c.id, model_name="lgbm_v1")
                .first()
            )
            if not p:
                continue
            tickets = (p.feature_snapshot or {}).get("tickets") or {}
            def fmt(kind):
                rows = tickets.get(kind, [])
                return " | ".join(
                    f"{t['label']}({t['prob']*100:.1f}%/{t['stake_share']*100:.0f}%)"
                    for t in rows
                )
            print(f"{c.venue_id} {c.race_no}R win=[{fmt('win')}]")
            print(f"  3連複=[{fmt('sanrenpuku')}]")
            print(f"  3連単=[{fmt('sanrentan')}]")
            shown += 1
            if shown >= 5:
                break


if __name__ == "__main__":
    main()
