#!/usr/bin/env python3
"""デモ用の疑似レースデータを投入し、予測→学習まで通す."""

from __future__ import annotations

import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.cli.app import init_db
from boatrace.config import get_venues
from boatrace.db.models import (
    RaceCard,
    RaceEntry,
    RaceResult,
    Racer,
    TideSnapshot,
    WeatherSnapshot,
)
from boatrace.db.session import session_scope
from boatrace.learning.service import LearningService
from boatrace.logging_setup import setup_logging
from boatrace.prediction.service import PredictionService

# コース別ベース勝率
COURSE_WIN = [0.55, 0.14, 0.11, 0.10, 0.06, 0.04]


def _mk_racer_id(n: int) -> str:
    return f"{n:04d}"


def seed(days: int = 5, venues_n: int = 4, races_per_day: int = 6) -> None:
    setup_logging()
    init_db()
    rng = random.Random(42)
    venues = get_venues()[:venues_n]
    today = date.today()

    with session_scope() as session:
        # 選手プール
        for i in range(1, 61):
            rid = _mk_racer_id(1000 + i)
            if session.get(Racer, rid) is None:
                session.add(Racer(id=rid, name=f"選手{rid}", grade=rng.choice(["A1", "A2", "B1", "B2"])))

        for d_offset in range(days):
            race_date = today - timedelta(days=d_offset)
            for venue in venues:
                for rno in range(1, races_per_day + 1):
                    card = (
                        session.query(RaceCard)
                        .filter_by(venue_id=venue.id, race_date=race_date, race_no=rno)
                        .one_or_none()
                    )
                    if card is None:
                        card = RaceCard(
                            venue_id=venue.id,
                            race_date=race_date,
                            race_no=rno,
                            race_title=f"デモ {rno}R",
                            is_fixed_entry=rng.random() < 0.1,
                            status="scheduled",
                            fetched_at=datetime.utcnow(),
                        )
                        session.add(card)
                        session.flush()

                    wind_speed = round(rng.uniform(0, 7), 1)
                    wind_dir = rng.choice(["追い", "向かい", "横", "追いに切れ", "向かいに切れ"])
                    weather = session.query(WeatherSnapshot).filter_by(race_card_id=card.id).one_or_none()
                    if weather is None:
                        weather = WeatherSnapshot(race_card_id=card.id)
                        session.add(weather)
                    weather.temperature = round(rng.uniform(10, 32), 1)
                    weather.weather = rng.choice(["晴", "曇", "雨"])
                    weather.wind_speed = wind_speed
                    weather.wind_direction = wind_dir
                    weather.water_temperature = round(rng.uniform(12, 28), 1)
                    weather.wave_height = round(rng.uniform(0, 8), 0)
                    weather.fetched_at = datetime.utcnow()

                    near_high = rng.random() < (0.35 if venue.tide_sensitive else 0.1)
                    tide = session.query(TideSnapshot).filter_by(race_card_id=card.id).one_or_none()
                    if tide is None:
                        tide = TideSnapshot(race_card_id=card.id)
                        session.add(tide)
                    tide.station_name = venue.tide_station or venue.name
                    tide.tide_level_cm = round(rng.uniform(40, 220), 1)
                    tide.tide_delta_cm = round(rng.uniform(-25, 25), 1)
                    tide.near_high_tide = near_high
                    tide.near_low_tide = (not near_high) and rng.random() < 0.2
                    tide.source = "demo"
                    tide.fetched_at = datetime.utcnow()

                    # 出走
                    racers = rng.sample(range(1, 61), 6)
                    ex_times = sorted([round(rng.uniform(6.60, 7.05), 2) for _ in range(6)])
                    rng.shuffle(ex_times)

                    strengths = []
                    for waku in range(1, 7):
                        rid = _mk_racer_id(1000 + racers[waku - 1])
                        local = round(rng.uniform(3.5, 8.2), 2)
                        national = round(local + rng.uniform(-1.0, 1.0), 2)
                        motor_q = round(rng.uniform(25, 55), 1)
                        boat_q = round(rng.uniform(25, 50), 1)
                        avg_st = round(rng.uniform(0.12, 0.22), 2)
                        entry = (
                            session.query(RaceEntry)
                            .filter_by(race_card_id=card.id, waku=waku)
                            .one_or_none()
                        )
                        if entry is None:
                            entry = RaceEntry(race_card_id=card.id, waku=waku, racer_id=rid)
                            session.add(entry)
                        entry.racer_id = rid
                        entry.local_win_rate = local
                        entry.national_win_rate = national
                        entry.local_quinella_rate = round(local * 5.5, 1)
                        entry.national_quinella_rate = round(national * 5.2, 1)
                        entry.local_trio_rate = round(local * 7.5, 1)
                        entry.national_trio_rate = round(national * 7.0, 1)
                        entry.motor_no = rng.randint(1, 60)
                        entry.motor_quinella_rate = motor_q
                        entry.boat_no = rng.randint(1, 60)
                        entry.boat_quinella_rate = boat_q
                        entry.avg_st = avg_st
                        entry.exhibition_time = ex_times[waku - 1]
                        entry.tilt = rng.choice([-0.5, 0.0, 0.5, 1.0])
                        entry.previous_rank = rng.randint(1, 6)
                        entry.estimated_course = waku

                        # 生成用の強さ
                        s = (
                            COURSE_WIN[waku - 1] * 2.2
                            + local / 10 * 1.2
                            + motor_q / 100 * 0.8
                            + max(0, 1.0 - (ex_times[waku - 1] - min(ex_times)) / 0.2) * 0.7
                        )
                        if "向かい" in wind_dir and wind_speed >= 3 and waku >= 4:
                            s *= 1.25
                        if near_high and venue.tide_sensitive and waku == 1:
                            s *= 0.82
                        strengths.append(s)

                    # 過去日は結果を確定
                    if d_offset > 0 or True:
                        # 今日も含めて結果を作り、学習デモを可能に（今日は予測後に学習）
                        weights = strengths[:]
                        order = sorted(range(1, 7), key=lambda i: weights[i - 1] + rng.random() * 0.15, reverse=True)
                        entry_results = []
                        for rank, waku in enumerate(order, start=1):
                            entry_results.append(
                                {"waku": waku, "rank": rank, "st": round(rng.uniform(0.08, 0.22), 2), "course": waku}
                            )
                        # 今日以外は finished にする。今日は予測用に結果を後で入れるため一旦保留
                        if d_offset >= 1:
                            rr = session.query(RaceResult).filter_by(race_card_id=card.id).one_or_none()
                            if rr is None:
                                rr = RaceResult(race_card_id=card.id)
                                session.add(rr)
                            rr.rank1_waku = order[0]
                            rr.rank2_waku = order[1]
                            rr.rank3_waku = order[2]
                            rr.kimarite = rng.choice(["逃げ", "差し", "まくり", "まくり差し"])
                            rr.entry_results = entry_results
                            card.status = "finished"
                        else:
                            # 今日分は結果も保存（検証用）だが status は finished にして学習可能に
                            rr = session.query(RaceResult).filter_by(race_card_id=card.id).one_or_none()
                            if rr is None:
                                rr = RaceResult(race_card_id=card.id)
                                session.add(rr)
                            rr.rank1_waku = order[0]
                            rr.rank2_waku = order[1]
                            rr.rank3_waku = order[2]
                            rr.kimarite = rng.choice(["逃げ", "差し", "まくり", "まくり差し"])
                            rr.entry_results = entry_results
                            card.status = "finished"

    # 予測 → 学習
    with session_scope() as session:
        pred = PredictionService(session)
        for d_offset in range(days):
            race_date = today - timedelta(days=d_offset)
            pred.predict_day(race_date, persist=True)

        learn = LearningService(session)
        for d_offset in range(days):
            race_date = today - timedelta(days=d_offset)
            learn.learn_day(race_date)

        summary = learn.accuracy_summary(days=30)
        print("seed complete")
        print("accuracy:", summary)


if __name__ == "__main__":
    seed()
