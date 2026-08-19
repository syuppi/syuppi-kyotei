"""自信あり日次上限と締切前予想ジョブ."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from boatrace.prediction.confidence import enforce_daily_confidence_cap
from boatrace.timeutil import JST


class _Card:
    def __init__(self, *, card_id=1, venue_id="01", race_no=1, score=0.8, is_conf=False):
        self.id = card_id
        self.venue_id = venue_id
        self.race_no = race_no
        self.race_date = datetime(2026, 8, 19).date()


class _Pred:
    def __init__(self, card: _Card, score: float, is_conf: bool):
        self.id = card.id
        self.race_card_id = card.id
        self.feature_snapshot = {
            "confidence": {
                "score": score,
                "is_confident": is_conf,
                "label": "自信あり" if is_conf else "普通",
                "reasons": [],
            },
            "tickets": {"sanrenpuku": [], "sanrentan": []},
            "tickets_before_confidence_focus": {"sanrenpuku": [{"combo": [1, 2, 3]}]},
        }


class _Session:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *args):
        return self

    def join(self, *args):
        return self

    def filter(self, *args):
        return self

    def all(self):
        return self._rows

    def flush(self):
        pass


def test_enforce_daily_confidence_cap_keeps_top_n(monkeypatch):
    from boatrace.prediction import confidence as conf_mod

    monkeypatch.setattr(
        conf_mod,
        "get_confidence_model",
        lambda reload=False: type("M", (), {"threshold": 0.70})(),
    )
    monkeypatch.setattr(
        conf_mod,
        "get_settings",
        lambda: type(
            "S",
            (),
            {"prediction": type("P", (), {"confidence_max_per_day": 2, "confidence_threshold": 0.75})()},
        )(),
    )

    c1 = _Card(card_id=1, score=0.9)
    c2 = _Card(card_id=2, score=0.85)
    c3 = _Card(card_id=3, score=0.80)
    p1, p2, p3 = _Pred(c1, 0.90, True), _Pred(c2, 0.85, True), _Pred(c3, 0.80, True)
    session = _Session([(c1, p1), (c2, p2), (c3, p3)])

    out = enforce_daily_confidence_cap(session, c1.race_date, max_n=2)

    assert out["confident_after"] == 2
    assert p1.feature_snapshot["confidence"]["is_confident"] is True
    assert p2.feature_snapshot["confidence"]["is_confident"] is True
    assert p3.feature_snapshot["confidence"]["is_confident"] is False
    assert p3.feature_snapshot["confidence"].get("daily_cap_excluded")


def test_preclose_needs_predict_in_window():
    from boatrace.jobs.preclose_predict import _needs_preclose_predict

    class Entry:
        pass

    class Card:
        status = "scheduled"
        result = None
        entries = [Entry()] * 6
        deadline_at = datetime(2026, 8, 19, 15, 30, 0, tzinfo=JST)

    card = Card()
    # 15:00 JST = 45 min before deadline → should need predict
    with patch("boatrace.jobs.preclose_predict.japan_now") as mock_now:
        mock_now.return_value = datetime(2026, 8, 19, 15, 0, 0, tzinfo=JST)
        assert _needs_preclose_predict(card, None, lead_minutes=45) is True

    with patch("boatrace.jobs.preclose_predict.japan_now") as mock_now:
        mock_now.return_value = datetime(2026, 8, 19, 10, 0, 0, tzinfo=JST)
        assert _needs_preclose_predict(card, None, lead_minutes=45) is False


def test_preclose_skips_existing_pre_close_prediction():
    from boatrace.jobs.preclose_predict import _needs_preclose_predict

    class Entry:
        pass

    class Result:
        rank1_waku = None

    class Card:
        status = "scheduled"
        result = Result()
        entries = [Entry()] * 6
        deadline_at = datetime(2026, 8, 19, 15, 30, 0, tzinfo=JST)

    class Pred:
        predicted_at = datetime(2026, 8, 19, 6, 0, 0)  # 15:00 JST

    card = Card()
    pred = Pred()
    with patch("boatrace.jobs.preclose_predict.japan_now") as mock_now:
        mock_now.return_value = datetime(2026, 8, 19, 15, 0, 0, tzinfo=JST)
        assert _needs_preclose_predict(card, pred, lead_minutes=45) is False
