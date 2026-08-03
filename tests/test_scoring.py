"""スコアリングと特徴量の単体テスト."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from boatrace.config import get_settings
from boatrace.db.models import (
    Base,
    RaceCard,
    RaceEntry,
    Racer,
    TideSnapshot,
    Venue,
    WeatherSnapshot,
)
from boatrace.db.session import get_engine, session_scope
from boatrace.features.builder import FeatureBuilder
from boatrace.models.scoring import ScoringPredictor, softmax
from boatrace.prediction.service import PredictionService


@pytest.fixture(autouse=True)
def _setup_db(tmp_path, monkeypatch):
    global _race_seq
    _race_seq = 0
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("BOATRACE_DATABASE_URL", f"sqlite:///{db_path}")
    # settings cache clear
    from boatrace import config

    config.get_settings.cache_clear()
    # force re-engine
    import boatrace.db.session as sess

    sess._engine = None
    sess._SessionLocal = None

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    with session_scope() as session:
        session.add(
            Venue(
                id="24",
                name="大村",
                tide_sensitive=False,
                water_type="lentic",
                typical_in_advantage=0.58,
            )
        )
        session.add(
            Venue(
                id="03",
                name="江戸川",
                tide_sensitive=True,
                water_type="river",
                tide_station="東京",
                typical_in_advantage=0.42,
            )
        )
        for i in range(1, 7):
            session.add(Racer(id=f"{i:04d}", name=f"R{i}"))
    yield
    config.get_settings.cache_clear()
    sess._engine = None
    sess._SessionLocal = None


_race_seq = 0


def _add_race(session, venue_id: str = "24", wind_dir: str = "追い", wind_speed: float = 1.0, near_high: bool = False):
    global _race_seq
    _race_seq += 1
    card = RaceCard(
        venue_id=venue_id,
        race_date=date.today(),
        race_no=_race_seq,
        race_title="test",
        status="scheduled",
        fetched_at=datetime.utcnow(),
    )
    session.add(card)
    session.flush()
    for waku in range(1, 7):
        session.add(
            RaceEntry(
                race_card_id=card.id,
                waku=waku,
                racer_id=f"{waku:04d}",
                local_win_rate=7.0 - waku * 0.4,
                national_win_rate=6.5 - waku * 0.3,
                motor_quinella_rate=40 - waku,
                boat_quinella_rate=35 - waku * 0.5,
                avg_st=0.14 + waku * 0.01,
                exhibition_time=6.70 + waku * 0.02,
                estimated_course=waku,
                previous_rank=waku,
            )
        )
    session.add(
        WeatherSnapshot(
            race_card_id=card.id,
            temperature=20,
            weather="晴",
            wind_speed=wind_speed,
            wind_direction=wind_dir,
            water_temperature=18,
            wave_height=2,
        )
    )
    session.add(
        TideSnapshot(
            race_card_id=card.id,
            tide_level_cm=150 if near_high else 80,
            tide_delta_cm=5,
            near_high_tide=near_high,
            source="test",
        )
    )
    session.flush()
    return card.id


def test_softmax_sums_to_one():
    probs = softmax([1.0, 2.0, 3.0], temperature=0.85)
    assert abs(sum(probs) - 1.0) < 1e-6


def test_predict_produces_six_probs():
    with session_scope() as session:
        race_id = _add_race(session)
        svc = PredictionService(session)
        result = svc.predict_race(race_id, persist=True)
        assert len(result.win_probs) == 6
        assert abs(sum(result.win_probs.values()) - 1.0) < 1e-6
        assert result.rankings[0] in range(1, 7)
        assert result.reasons[result.rankings[0]]


def test_headwind_boosts_outer_relative():
    with session_scope() as session:
        calm_id = _add_race(session, wind_dir="追い", wind_speed=1.0)
        head_id = _add_race(session, wind_dir="向かい", wind_speed=5.0)

        builder = FeatureBuilder(session)
        calm = builder.build(calm_id)
        head = builder.build(head_id)
        calm_outer = next(b for b in calm.boats if b.waku == 4).values["wind_course_bias"]
        head_outer = next(b for b in head.boats if b.waku == 4).values["wind_course_bias"]
        assert head_outer >= calm_outer


def test_high_tide_penalizes_innner_on_sensitive_venue():
    with session_scope() as session:
        normal_id = _add_race(session, venue_id="03", near_high=False)
        high_id = _add_race(session, venue_id="03", near_high=True)
        builder = FeatureBuilder(session)
        normal = builder.build(normal_id)
        high = builder.build(high_id)
        n1 = next(b for b in normal.boats if b.waku == 1).values["tide_adjustment"]
        h1 = next(b for b in high.boats if b.waku == 1).values["tide_adjustment"]
        assert h1 < n1
