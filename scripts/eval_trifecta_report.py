#!/usr/bin/env python3
"""3連単の本命1点/上位3点カバーと信頼度フィルタ比較レポート."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.db.models import PredictHistory, RaceCard
from boatrace.db.session import session_scope
from boatrace.logging_setup import setup_logging
from boatrace.prediction.service import PredictionService


def _eval_cards(session, cards: list[RaceCard], model_name: str = "lgbm_v1") -> dict:
    n = win1 = win3 = trio1 = trio3 = tf1 = tf3 = 0
    n_conf = tf1_c = tf3_c = 0
    n_low = 0
    cand_sum = 0
    for c in cards:
        p = (
            session.query(PredictHistory)
            .filter_by(race_card_id=c.id, model_name=model_name)
            .order_by(PredictHistory.predicted_at.desc())
            .first()
        )
        if not p or not c.result or not c.result.rank1_waku:
            continue
        snap = p.feature_snapshot or {}
        tickets = snap.get("tickets") or {}
        wins = [t["combo"][0] for t in tickets.get("win", []) if t.get("combo")]
        if not wins:
            wins = list(p.candidates_win or [])[:3]
        sps = [set(t["combo"]) for t in tickets.get("sanrenpuku", []) if t.get("combo")]
        sts = [list(t["combo"]) for t in tickets.get("sanrentan", []) if t.get("combo")]
        true3 = {c.result.rank1_waku, c.result.rank2_waku, c.result.rank3_waku}
        true_ord = [c.result.rank1_waku, c.result.rank2_waku, c.result.rank3_waku]
        low = bool(snap.get("low_confidence"))
        n += 1
        cand_sum += len(sts) if sts else 0
        win1 += int(bool(wins) and wins[0] == c.result.rank1_waku)
        win3 += int(c.result.rank1_waku in wins)
        if None not in true3:
            trio1 += int(bool(sps) and sps[0] == true3)
            trio3 += int(any(s == true3 for s in sps))
        if c.result.rank2_waku and c.result.rank3_waku:
            hit1 = bool(sts) and sts[0] == true_ord
            hit3 = any(s == true_ord for s in sts)
            tf1 += int(hit1)
            tf3 += int(hit3)
            if low:
                n_low += 1
            else:
                n_conf += 1
                tf1_c += int(hit1)
                tf3_c += int(hit3)

    def rate(x: int, d: int) -> float:
        return x / d if d else 0.0

    return {
        "n": n,
        "win_top1": rate(win1, n),
        "win_top3": rate(win3, n),
        "trio_top1": rate(trio1, n),
        "trio_top3": rate(trio3, n),
        "trifecta_top1": rate(tf1, n),
        "trifecta_top3": rate(tf3, n),
        "avg_trifecta_candidates": (cand_sum / n) if n else 0.0,
        "low_confidence_races": n_low,
        "confident_n": n_conf,
        "trifecta_top1_confident": rate(tf1_c, n_conf),
        "trifecta_top3_confident": rate(tf3_c, n_conf),
    }


def main() -> None:
    setup_logging()
    today = date.today()
    report: dict = {
        "as_of": today.isoformat(),
        "before_reference": {
            "note": "改善前（約90日学習・本日144R）",
            "trifecta_top1": 0.076,
            "trifecta_top3": 0.181,
            "win_top1": 0.563,
            "win_top3": 0.840,
            "trio_top1": 0.222,
            "trio_top3": 0.521,
        },
    }

    with session_scope() as session:
        # 検証7日 + 本日を再予想
        days = [today - timedelta(days=i) for i in range(7, -1, -1)]
        for d in days:
            ids = [c.id for c in session.query(RaceCard).filter_by(race_date=d).all()]
            if not ids:
                continue
            session.query(PredictHistory).filter(
                PredictHistory.race_card_id.in_(ids),
                PredictHistory.model_name == "lgbm_v1",
            ).delete(synchronize_session=False)
            session.flush()
            PredictionService(session, model="lgbm").predict_day(d, persist=True)

        today_cards = session.query(RaceCard).filter_by(race_date=today).all()
        holdout_cards: list[RaceCard] = []
        for d in [today - timedelta(days=i) for i in range(1, 8)]:
            holdout_cards.extend(session.query(RaceCard).filter_by(race_date=d).all())

        report["today"] = _eval_cards(session, today_cards)
        report["holdout_7d"] = _eval_cards(session, holdout_cards)
        print(json.dumps(report, ensure_ascii=False, indent=2))

        out = ROOT / "data" / "models" / "trifecta_eval_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()
