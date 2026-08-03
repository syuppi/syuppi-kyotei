#!/usr/bin/env python3
"""実データ収集 → 予測（デモ上書きをクリアしてから実行）."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import or_  # noqa: F401  # kept for future filters

from boatrace.cli.app import init_db
from boatrace.collectors.pipeline import collect_daily
from boatrace.db.models import (
    AccuracyDaily,
    PredictHistory,
    RaceCard,
    RaceEntry,
    RaceResult,
    TideSnapshot,
    VenueBias,
    VenueCourseStats,
    WeatherSnapshot,
)
from boatrace.db.session import session_scope
from boatrace.learning.service import LearningService
from boatrace.logging_setup import setup_logging, get_logger
from boatrace.prediction.service import PredictionService

logger = get_logger(__name__)


def clear_demo_and_old_predictions(days: int = 8) -> None:
    """直近カードと予測を消して実データ用に空ける."""
    since = date.today() - timedelta(days=days)
    with session_scope() as session:
        demo_ids = [
            c.id
            for c in session.query(RaceCard).filter(RaceCard.race_date >= since).all()
        ]
        if not demo_ids:
            return
        session.query(PredictHistory).filter(PredictHistory.race_card_id.in_(demo_ids)).delete(
            synchronize_session=False
        )
        session.query(RaceResult).filter(RaceResult.race_card_id.in_(demo_ids)).delete(
            synchronize_session=False
        )
        session.query(WeatherSnapshot).filter(WeatherSnapshot.race_card_id.in_(demo_ids)).delete(
            synchronize_session=False
        )
        session.query(TideSnapshot).filter(TideSnapshot.race_card_id.in_(demo_ids)).delete(
            synchronize_session=False
        )
        session.query(RaceEntry).filter(RaceEntry.race_card_id.in_(demo_ids)).delete(
            synchronize_session=False
        )
        session.query(RaceCard).filter(RaceCard.id.in_(demo_ids)).delete(synchronize_session=False)
        session.query(AccuracyDaily).filter(AccuracyDaily.stat_date >= since).delete(
            synchronize_session=False
        )
        session.query(VenueCourseStats).delete()
        session.query(VenueBias).delete()
        logger.info("cleared_recent_cards", n=len(demo_ids))


def main() -> None:
    setup_logging()
    init_db()
    clear_demo_and_old_predictions()
    summary = collect_daily(include_lookback=True, prefer_openapi=True)
    print("collect:", summary)

    with session_scope() as session:
        pred = PredictionService(session)
        # 当日予測（結果があっても特徴は結果を見ない。評価用に保存）
        today_items = pred.predict_day(date.today(), persist=True)
        print(f"predicted today: {len(today_items)}")

        # 過去日は結果学習
        learn = LearningService(session)
        for i in range(1, 8):
            d = date.today() - timedelta(days=i)
            learn.learn_day(d)
        # 当日も結果がある分は学習（予測は既に保存済み）
        learn.learn_day(date.today())
        print("accuracy:", learn.accuracy_summary(30))

        # 実データサンプル表示
        from boatrace.db.models import RaceCard, RaceEntry

        cards = (
            session.query(RaceCard)
            .filter(RaceCard.race_date == date.today())
            .order_by(RaceCard.venue_id, RaceCard.race_no)
            .limit(3)
            .all()
        )
        for c in cards:
            e1 = (
                session.query(RaceEntry)
                .filter_by(race_card_id=c.id, waku=1)
                .one_or_none()
            )
            print(
                f"sample {c.venue_id} {c.race_no}R title={c.race_title!r} "
                f"racer={e1.racer_id if e1 else None} "
                f"name={e1.racer.name if e1 and e1.racer else None} "
                f"ex={e1.exhibition_time if e1 else None} "
                f"status={c.status} source={(c.raw_payload or {}).get('source')}"
            )


if __name__ == "__main__":
    main()
