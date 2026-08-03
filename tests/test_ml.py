"""LightGBM予測の基本テスト."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest

from boatrace.db.models import Base, RaceCard, RaceEntry, Racer, Venue, WeatherSnapshot, TideSnapshot
from boatrace.db.session import get_engine, session_scope
from boatrace.models.ml_model import MLPredictor
from boatrace.models.scoring import ScoringPredictor
from boatrace.features.builder import FeatureBuilder


@pytest.fixture(autouse=True)
def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("BOATRACE_DATABASE_URL", f"sqlite:///{db_path}")
    from boatrace import config
    import boatrace.db.session as sess

    config.get_settings.cache_clear()
    sess._engine = None
    sess._SessionLocal = None
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    with session_scope() as session:
        session.add(Venue(id="24", name="大村", typical_in_advantage=0.58))
        for i in range(1, 7):
            session.add(Racer(id=f"{i:04d}", name=f"R{i}"))
    yield
    config.get_settings.cache_clear()
    sess._engine = None
    sess._SessionLocal = None


def test_ml_fallback_without_model():
    with session_scope() as session:
        card = RaceCard(
            venue_id="24",
            race_date=date.today(),
            race_no=1,
            race_title="t",
            status="scheduled",
            fetched_at=datetime.utcnow(),
        )
        session.add(card)
        session.flush()
        for w in range(1, 7):
            session.add(
                RaceEntry(
                    race_card_id=card.id,
                    waku=w,
                    racer_id=f"{w:04d}",
                    local_win_rate=7 - w * 0.3,
                    national_win_rate=6.5 - w * 0.2,
                    motor_quinella_rate=40,
                    boat_quinella_rate=35,
                    avg_st=0.15 + w * 0.01,
                    exhibition_time=6.7 + w * 0.02,
                    estimated_course=w,
                )
            )
        session.add(
            WeatherSnapshot(
                race_card_id=card.id,
                wind_speed=2,
                wind_direction="追い",
                wave_height=1,
                weather="晴",
                temperature=20,
                water_temperature=18,
            )
        )
        session.add(TideSnapshot(race_card_id=card.id, near_high_tide=False, source="test"))
        session.flush()
        feats = FeatureBuilder(session).build(card.id)
        pred = MLPredictor(session=session, model_path="/tmp/no_such_model.joblib")
        result = pred.predict(feats)
        assert result.model_name.endswith("fallback")
        assert abs(sum(result.win_probs.values()) - 1.0) < 1e-6
