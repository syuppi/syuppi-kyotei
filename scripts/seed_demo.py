#!/usr/bin/env python3
"""デモ用の疑似レースデータを投入し、予測→（その後）結果→学習の正しい順序で通す."""

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
    AccuracyDaily,
    PredictHistory,
    RaceCard,
    RaceEntry,
    RaceResult,
    Racer,
    TideSnapshot,
    VenueBias,
    VenueCourseStats,
    WeatherSnapshot,
)
from boatrace.db.session import session_scope
from boatrace.learning.service import LearningService
from boatrace.logging_setup import setup_logging
from boatrace.prediction.service import PredictionService

COURSE_WIN = [0.55, 0.14, 0.11, 0.10, 0.06, 0.04]


def _mk_racer_id(n: int) -> str:
    return f"{n:04d}"


def _reset_demo_tables() -> None:
    """デモ再投入のため予測・結果・カード関連をクリア（会場マスタは残す）."""
    with session_scope() as session:
        for model in (
            AccuracyDaily,
            PredictHistory,
            VenueBias,
            VenueCourseStats,
            RaceResult,
            WeatherSnapshot,
            TideSnapshot,
            RaceEntry,
            RaceCard,
        ):
            session.query(model).delete()


def _simulate_finish(rng: random.Random, strengths: list[float], noise: float = 0.55) -> list[int]:
    """特徴量由来の強さに大きめノイズを足して着順を作る（予測と同一にならないようにする）."""
    scored = [
        strengths[i - 1] + rng.gauss(0, noise) + rng.random() * noise
        for i in range(1, 7)
    ]
    return sorted(range(1, 7), key=lambda i: scored[i - 1], reverse=True)


def _attach_result(session, card: RaceCard, order: list[int], rng: random.Random) -> None:
    entry_results = [
        {
            "waku": waku,
            "rank": rank,
            "st": round(rng.uniform(0.08, 0.22), 2),
            "course": waku,
        }
        for rank, waku in enumerate(order, start=1)
    ]
    rr = session.query(RaceResult).filter_by(race_card_id=card.id).one_or_none()
    if rr is None:
        rr = RaceResult(race_card_id=card.id)
        session.add(rr)
    rr.rank1_waku = order[0]
    rr.rank2_waku = order[1]
    rr.rank3_waku = order[2]
    rr.kimarite = rng.choice(["逃げ", "差し", "まくり", "まくり差し"])
    rr.entry_results = entry_results
    rr.fetched_at = datetime.utcnow()
    card.status = "finished"
    card.updated_at = datetime.utcnow()


def seed(days: int = 5, venues_n: int = 4, races_per_day: int = 6) -> None:
    setup_logging()
    init_db()
    _reset_demo_tables()

    rng = random.Random(42)
    venues = get_venues()[:venues_n]
    today = date.today()
    pending_today: list[tuple[int, list[float]]] = []  # (card_id, strengths)

    with session_scope() as session:
        for i in range(1, 61):
            rid = _mk_racer_id(1000 + i)
            if session.get(Racer, rid) is None:
                session.add(
                    Racer(
                        id=rid,
                        name=f"選手{rid}",
                        grade=rng.choice(["A1", "A2", "B1", "B2"]),
                    )
                )

        for d_offset in range(days):
            race_date = today - timedelta(days=d_offset)
            for venue in venues:
                for rno in range(1, races_per_day + 1):
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
                    weather = WeatherSnapshot(
                        race_card_id=card.id,
                        temperature=round(rng.uniform(10, 32), 1),
                        weather=rng.choice(["晴", "曇", "雨"]),
                        wind_speed=wind_speed,
                        wind_direction=wind_dir,
                        water_temperature=round(rng.uniform(12, 28), 1),
                        wave_height=round(rng.uniform(0, 8), 0),
                        fetched_at=datetime.utcnow(),
                    )
                    session.add(weather)

                    near_high = rng.random() < (0.35 if venue.tide_sensitive else 0.1)
                    session.add(
                        TideSnapshot(
                            race_card_id=card.id,
                            station_name=venue.tide_station or venue.name,
                            tide_level_cm=round(rng.uniform(40, 220), 1),
                            tide_delta_cm=round(rng.uniform(-25, 25), 1),
                            near_high_tide=near_high,
                            near_low_tide=(not near_high) and rng.random() < 0.2,
                            source="demo",
                            fetched_at=datetime.utcnow(),
                        )
                    )

                    racers = rng.sample(range(1, 61), 6)
                    ex_times = sorted(round(rng.uniform(6.60, 7.05), 2) for _ in range(6))
                    rng.shuffle(ex_times)

                    strengths: list[float] = []
                    for waku in range(1, 7):
                        rid = _mk_racer_id(1000 + racers[waku - 1])
                        local = round(rng.uniform(3.5, 8.2), 2)
                        national = round(local + rng.uniform(-1.0, 1.0), 2)
                        motor_q = round(rng.uniform(25, 55), 1)
                        boat_q = round(rng.uniform(25, 50), 1)
                        avg_st = round(rng.uniform(0.12, 0.22), 2)
                        session.add(
                            RaceEntry(
                                race_card_id=card.id,
                                waku=waku,
                                racer_id=rid,
                                local_win_rate=local,
                                national_win_rate=national,
                                local_quinella_rate=round(local * 5.5, 1),
                                national_quinella_rate=round(national * 5.2, 1),
                                local_trio_rate=round(local * 7.5, 1),
                                national_trio_rate=round(national * 7.0, 1),
                                motor_no=rng.randint(1, 60),
                                motor_quinella_rate=motor_q,
                                boat_no=rng.randint(1, 60),
                                boat_quinella_rate=boat_q,
                                avg_st=avg_st,
                                exhibition_time=ex_times[waku - 1],
                                tilt=rng.choice([-0.5, 0.0, 0.5, 1.0]),
                                previous_rank=rng.randint(1, 6),
                                estimated_course=waku,
                            )
                        )
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

                    if d_offset >= 1:
                        # 過去日: 大きめノイズで結果を先に確定（学習用履歴）
                        order = _simulate_finish(rng, strengths, noise=0.55)
                        _attach_result(session, card, order, rng)
                    else:
                        # 今日: 予測時点では結果なし。強さは後で結果生成に使う
                        pending_today.append((card.id, strengths))

    # 1) まず予測（今日は結果なし / 過去は既に結果ありだが特徴は as_of で過去のみ参照）
    with session_scope() as session:
        pred = PredictionService(session)
        for d_offset in range(days):
            race_date = today - timedelta(days=d_offset)
            outs = pred.predict_day(race_date, persist=True)
            print(f"predicted {race_date}: {len(outs)} races")

    # 2) 今日の結果を「予測の後」に付与（検証用）。予測とは独立ノイズ
    with session_scope() as session:
        for card_id, strengths in pending_today:
            card = session.get(RaceCard, card_id)
            if card is None:
                continue
            order = _simulate_finish(rng, strengths, noise=0.70)
            _attach_result(session, card, order, rng)

    # 3) 結果を学習（翌日以降の予測用）。当日予測の書き換えはしない
    with session_scope() as session:
        learn = LearningService(session)
        for d_offset in range(days):
            race_date = today - timedelta(days=d_offset)
            learn.learn_day(race_date)
        summary = learn.accuracy_summary(days=30)
        print("seed complete")
        print("accuracy:", summary)

        # 正直な突合サンプル
        hit = total = 0
        for card_id, _ in pending_today:
            card = session.get(RaceCard, card_id)
            pred_row = (
                session.query(PredictHistory)
                .filter_by(race_card_id=card_id)
                .order_by(PredictHistory.predicted_at.desc())
                .first()
            )
            if card and card.result and pred_row and pred_row.rankings:
                total += 1
                hit += int(pred_row.rankings[0] == card.result.rank1_waku)
        if total:
            print(f"today honest hit1: {hit}/{total} = {hit/total:.3f}")


if __name__ == "__main__":
    seed()
