"""場共起事前分布の単体テスト."""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from boatrace.db.models import Base, RaceCard, RaceResult, Venue
from boatrace.models.venue_cooccur import build_venue_trio_prior


def test_venue_trio_prior_respects_as_of_and_smoothing():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as s:
        s.add(
            Venue(
                id="01",
                name="テスト",
                name_kana="",
                prefecture="",
                tide_sensitive=False,
            )
        )
        s.flush()
        d0 = date(2026, 6, 1)
        for i in range(40):
            card = RaceCard(
                venue_id="01",
                race_date=d0 + timedelta(days=i // 12),
                race_no=(i % 12) + 1,
                status="finished",
            )
            s.add(card)
            s.flush()
            if i < 30:
                r1, r2, r3 = 1, 2, 3
            else:
                r1, r2, r3 = 4, 5, 6
            s.add(
                RaceResult(
                    race_card_id=card.id,
                    rank1_waku=r1,
                    rank2_waku=r2,
                    rank3_waku=r3,
                    entry_results=[],
                )
            )
        s.commit()

        prior = build_venue_trio_prior(s, "01", as_of_date=d0 + timedelta(days=4), min_samples=10)
        assert prior
        assert frozenset([1, 2, 3]) in prior
        prior_empty = build_venue_trio_prior(s, "01", as_of_date=d0, min_samples=30)
        assert prior_empty == {}
