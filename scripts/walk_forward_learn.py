#!/usr/bin/env python3
"""
過去データをウォークフォワードで学習する。
各日: その日より前の情報だけで予測 → 結果で場クセ/重み更新。
最後に本日を再予想する。
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.db.models import PredictHistory, RaceCard, RaceEntry
from boatrace.db.session import session_scope
from boatrace.learning.service import LearningService
from boatrace.logging_setup import get_logger, setup_logging
from boatrace.prediction.service import PredictionService

logger = get_logger(__name__)


def walk_forward(days: int = 90) -> dict:
    setup_logging()
    today = date.today()
    start = today - timedelta(days=days)

    predicted_days = 0
    with session_scope() as session:
        pred = PredictionService(session)
        learn = LearningService(session)

        d = start
        while d < today:
            # その日のカードがあるときだけ
            n = session.query(RaceCard).filter(RaceCard.race_date == d).count()
            if n:
                # 既存予測を消して作り直し（当時知識のみで再評価）
                ids = [
                    c.id
                    for c in session.query(RaceCard).filter(RaceCard.race_date == d).all()
                ]
                if ids:
                    session.query(PredictHistory).filter(
                        PredictHistory.race_card_id.in_(ids)
                    ).delete(synchronize_session=False)
                    session.flush()
                items = pred.predict_day(d, persist=True)
                learn.learn_day(d)
                predicted_days += 1
                if predicted_days % 10 == 0:
                    logger.info(
                        "walk_forward_progress",
                        date=d.isoformat(),
                        races=len(items),
                        days_done=predicted_days,
                    )
            d += timedelta(days=1)

        # 本日再予想
        today_ids = [
            c.id for c in session.query(RaceCard).filter(RaceCard.race_date == today).all()
        ]
        if today_ids:
            session.query(PredictHistory).filter(
                PredictHistory.race_card_id.in_(today_ids)
            ).delete(synchronize_session=False)
            session.flush()
        today_items = pred.predict_day(today, persist=True)
        # 結果集計のみ（予測は維持）
        learn.evaluate_accuracy(today)
        learn.update_venue_course_stats(today)
        # bias/weights は当日結果で更新してよいが、予測は書き換えない
        learn.update_venue_bias(today)
        learn.tune_weights(today)

        hit = total = 0
        samples = []
        for c in (
            session.query(RaceCard)
            .filter(RaceCard.race_date == today)
            .order_by(RaceCard.venue_id, RaceCard.race_no)
            .all()
        ):
            p = (
                session.query(PredictHistory)
                .filter_by(race_card_id=c.id)
                .order_by(PredictHistory.predicted_at.desc())
                .first()
            )
            if not p or not c.result or not c.result.rank1_waku or not p.rankings:
                continue
            total += 1
            ok = p.rankings[0] == c.result.rank1_waku
            hit += int(ok)
            if len(samples) < 10:
                e1 = (
                    session.query(RaceEntry)
                    .filter_by(race_card_id=c.id, waku=1)
                    .one_or_none()
                )
                samples.append(
                    {
                        "venue": c.venue_id,
                        "rno": c.race_no,
                        "racer1": e1.racer.name if e1 and e1.racer else None,
                        "pred": p.rankings[:3],
                        "result": [
                            c.result.rank1_waku,
                            c.result.rank2_waku,
                            c.result.rank3_waku,
                        ],
                        "hit1": ok,
                    }
                )

        acc = learn.accuracy_summary(days=min(days, 90))

    return {
        "walk_forward_days": predicted_days,
        "predicted_today": len(today_items),
        "today_hit1": {"hit": hit, "total": total, "rate": hit / total if total else 0.0},
        "accuracy_window": acc,
        "samples": samples,
    }


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    out = walk_forward(days=days)
    print("=== WALK-FORWARD LEARN + REPREDICT ===")
    for k, v in out.items():
        if k != "samples":
            print(f"{k}: {v}")
    print("samples:")
    for s in out["samples"]:
        print(s)
