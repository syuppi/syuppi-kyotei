#!/usr/bin/env python3
"""過去N日分の実データを取得→時系列学習→本日を再予想する."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.cli.app import init_db
from boatrace.collectors.openapi import OpenApiCollector
from boatrace.collectors.tide import TideCollector
from boatrace.config import get_settings
from boatrace.db.models import PredictHistory, RaceCard, RaceEntry, RaceResult
from boatrace.db.session import session_scope
from boatrace.learning.service import LearningService
from boatrace.logging_setup import get_logger, setup_logging
from boatrace.prediction.service import PredictionService

logger = get_logger(__name__)


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def backfill(days: int = 90, include_today: bool = True) -> dict:
    setup_logging()
    init_db()
    today = date.today()
    start = today - timedelta(days=days)
    end_hist = today - timedelta(days=1)

    openapi = OpenApiCollector()
    tide = TideCollector()

    collect_stats = {"days_ok": 0, "days_fail": 0, "cards": 0, "entries": 0, "results": 0}

    # 1) 過去日を古い順に収集
    for d in daterange(start, end_hist):
        try:
            summary = openapi.collect(d)
            tide.collect(d, estimate_only=True)
            collect_stats["days_ok"] += 1
            collect_stats["cards"] += summary.get("cards", 0)
            collect_stats["entries"] += summary.get("entries", 0)
            collect_stats["results"] += summary.get("results", 0)
            logger.info("backfill_day_ok", date=d.isoformat(), summary=summary)
        except Exception as e:  # noqa: BLE001
            collect_stats["days_fail"] += 1
            logger.warning("backfill_day_fail", date=d.isoformat(), error=str(e))

    if include_today:
        try:
            summary = openapi.collect(today)
            tide.collect(today, estimate_only=True)
            collect_stats["days_ok"] += 1
            collect_stats["cards"] += summary.get("cards", 0)
            collect_stats["entries"] += summary.get("entries", 0)
            collect_stats["results"] += summary.get("results", 0)
            logger.info("backfill_today_ok", summary=summary)
        except Exception as e:  # noqa: BLE001
            collect_stats["days_fail"] += 1
            logger.warning("backfill_today_fail", error=str(e))

    # 2) 時系列学習（古い日→新しい日）。当日は「予測後」に回す
    learn_stats = {"days": 0}
    with session_scope() as session:
        learn = LearningService(session)
        for d in daterange(start, end_hist):
            out = learn.learn_day(d)
            learn_stats["days"] += 1
            if learn_stats["days"] % 10 == 0:
                logger.info("learn_progress", date=d.isoformat(), out_keys=list(out.keys()))

    # 3) 本日の予測を作り直し（学習済みの過去だけを特徴に使う）
    with session_scope() as session:
        today_ids = [
            c.id
            for c in session.query(RaceCard).filter(RaceCard.race_date == today).all()
        ]
        if today_ids:
            session.query(PredictHistory).filter(
                PredictHistory.race_card_id.in_(today_ids)
            ).delete(synchronize_session=False)
            session.flush()

        pred = PredictionService(session)
        items = pred.predict_day(today, persist=True)

        # 4) 本日の結果がある分だけ学習・精度集計（予測は上書きしない）
        learn = LearningService(session)
        acc_today = learn.learn_day(today)
        acc_summary = learn.accuracy_summary(days=min(days, 90))

        # 本日の正直な的中
        hit = total = 0
        samples = []
        cards = (
            session.query(RaceCard)
            .filter(RaceCard.race_date == today)
            .order_by(RaceCard.venue_id, RaceCard.race_no)
            .all()
        )
        for c in cards:
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
            if len(samples) < 8:
                e1 = (
                    session.query(RaceEntry)
                    .filter_by(race_card_id=c.id, waku=1)
                    .one_or_none()
                )
                samples.append(
                    {
                        "venue": c.venue_id,
                        "rno": c.race_no,
                        "title": c.race_title,
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

        # DB規模
        n_cards = session.query(RaceCard).filter(RaceCard.race_date >= start).count()
        n_finished = (
            session.query(RaceCard)
            .filter(RaceCard.race_date >= start, RaceCard.status == "finished")
            .count()
        )

    result = {
        "period": {"start": start.isoformat(), "end": today.isoformat(), "days": days},
        "collect": collect_stats,
        "learn_days": learn_stats["days"],
        "db": {"cards": n_cards, "finished": n_finished},
        "predicted_today": len(items),
        "today_hit1": {"hit": hit, "total": total, "rate": (hit / total if total else 0.0)},
        "accuracy_window": acc_summary,
        "samples": samples,
        "learn_today": {
            "course_stats_updated": acc_today.get("course_stats_updated"),
            "bias_updated": acc_today.get("bias_updated"),
        },
    }
    return result


if __name__ == "__main__":
    days = 90
    if len(sys.argv) > 1:
        days = int(sys.argv[1])
    out = backfill(days=days)
    print("=== BACKFILL + LEARN + REPREDICT ===")
    for k, v in out.items():
        if k != "samples":
            print(f"{k}: {v}")
    print("samples:")
    for s in out["samples"]:
        print(s)
