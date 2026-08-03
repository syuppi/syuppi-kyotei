#!/usr/bin/env python3
"""LightGBM学習 → 本日をMLで再予想 → ルールベースと比較."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.db.migrate import migrate
from boatrace.db.models import PredictHistory, RaceCard
from boatrace.db.session import session_scope
from boatrace.learning.service import LearningService
from boatrace.logging_setup import setup_logging
from boatrace.models.train_lgbm import DEFAULT_MODEL_PATH, retrain_all_before_today
from boatrace.prediction.service import PredictionService


def eval_today(session, model_name_filter: str | None = None) -> dict:
    today = date.today()
    hit = total = q_hit = t_hit = tf_hit = 0
    for c in session.query(RaceCard).filter_by(race_date=today).all():
        q = session.query(PredictHistory).filter_by(race_card_id=c.id)
        if model_name_filter:
            q = q.filter(PredictHistory.model_name == model_name_filter)
        p = q.order_by(PredictHistory.predicted_at.desc()).first()
        if not p or not c.result or not c.result.rank1_waku or not p.rankings:
            continue
        total += 1
        hit += int(p.rankings[0] == c.result.rank1_waku)
        true2 = {c.result.rank1_waku, c.result.rank2_waku}
        true3 = {c.result.rank1_waku, c.result.rank2_waku, c.result.rank3_waku}
        q_hit += int(None not in true2 and true2 == set(p.candidates_quinella[:2]))
        t_hit += int(None not in true3 and true3 == set(p.candidates_trio[:3]))
        tf_hit += int(
            c.result.rank2_waku is not None
            and c.result.rank3_waku is not None
            and p.rankings[0] == c.result.rank1_waku
            and p.rankings[1] == c.result.rank2_waku
            and p.rankings[2] == c.result.rank3_waku
        )
    return {
        "n": total,
        "win_rate": hit / total if total else 0.0,
        "quinella_rate": q_hit / total if total else 0.0,
        "trio_rate": t_hit / total if total else 0.0,
        "trifecta_rate": tf_hit / total if total else 0.0,
        "hit": hit,
        "trio_hit": t_hit,
        "trifecta_hit": tf_hit,
    }


def main() -> None:
    setup_logging()
    migrate()
    today = date.today()

    with session_scope() as session:
        print("=== train LightGBM (win/top2/top3 + combinations) ===")
        result = retrain_all_before_today(session, days=90)
        print("model:", result.model_path)
        print("metrics:", result.metrics)
        print("top features:", list(result.feature_importance.items())[:10])

        scoring_acc = eval_today(session, model_name_filter="scoring_v1")
        print("scoring_today:", scoring_acc)

        ids = [c.id for c in session.query(RaceCard).filter_by(race_date=today).all()]
        if ids:
            session.query(PredictHistory).filter(
                PredictHistory.race_card_id.in_(ids)
            ).delete(synchronize_session=False)
            session.flush()

        ml_svc = PredictionService(session, model="lgbm")
        items = ml_svc.predict_day(today, persist=True)
        ml_acc = eval_today(session, model_name_filter="lgbm_v1")
        print("predicted_today:", len(items))
        print("lgbm_today:", ml_acc)
        print("model_exists:", DEFAULT_MODEL_PATH.exists())

        learn = LearningService(session)
        acc = learn.evaluate_accuracy(today)
        print("accuracy_daily_all:", next((s for s in acc["slices"] if s["slice_key"] == "all" and s["venue_id"] is None), None))

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
            if not p or not c.result:
                continue
            snap = p.feature_snapshot or {}
            sanrentan = (snap.get("sanrentan") or [p.rankings[:3]])[0]
            sanrenpuku = (snap.get("sanrenpuku") or [sorted(p.candidates_trio[:3])])[0]
            true3 = [c.result.rank1_waku, c.result.rank2_waku, c.result.rank3_waku]
            print(
                f"{c.venue_id} {c.race_no}R "
                f"3連単={'-'.join(map(str, sanrentan))} "
                f"3連複={'-'.join(map(str, sorted(sanrenpuku)))} "
                f"res={'-'.join(map(str, true3))} "
                f"tf={list(sanrentan)==true3} "
                f"tr={set(sanrenpuku)==set(true3)}"
            )
            shown += 1
            if shown >= 12:
                break


if __name__ == "__main__":
    main()
