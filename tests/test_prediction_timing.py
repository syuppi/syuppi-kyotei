"""予想時刻と的中表示の整合性."""

from __future__ import annotations

from datetime import datetime

from boatrace.prediction.timing import classify_prediction_timing, is_hit_verifiable
from boatrace.timeutil import JST


class _Result:
    rank1_waku = 1


class _Card:
    def __init__(self, *, deadline=None, status="scheduled", has_result=False):
        self.deadline_at = deadline
        self.status = status
        self.result = _Result() if has_result else None


class _Pred:
    def __init__(self, predicted_at):
        self.predicted_at = predicted_at


def test_pre_close_prediction_is_verifiable():
    card = _Card(
        deadline=datetime(2026, 8, 19, 15, 30, 0, tzinfo=JST),
        status="finished",
        has_result=True,
    )
    pred = _Pred(datetime(2026, 8, 19, 15, 0, 0))  # UTC naive → 00:00 JST same day... wait

    # Use explicit UTC: 06:00 UTC = 15:00 JST, before 15:30 deadline
    pred = _Pred(datetime(2026, 8, 19, 6, 0, 0))
    info = classify_prediction_timing(card, pred)
    assert info["status"] == "pre_close"
    assert is_hit_verifiable(card, pred)


def test_post_close_prediction_is_not_verifiable():
    card = _Card(
        deadline=datetime(2026, 8, 19, 15, 30, 0, tzinfo=JST),
        status="finished",
        has_result=True,
    )
    # 07:00 UTC = 16:00 JST, after deadline
    pred = _Pred(datetime(2026, 8, 19, 7, 0, 0))
    info = classify_prediction_timing(card, pred)
    assert info["status"] == "post_close"
    assert not is_hit_verifiable(card, pred)


def test_finished_without_deadline_not_verifiable():
    card = _Card(status="finished", has_result=True, deadline=None)
    pred = _Pred(datetime(2026, 8, 19, 1, 0, 0))
    info = classify_prediction_timing(card, pred)
    assert info["status"] == "no_deadline"
    assert not is_hit_verifiable(card, pred)


def test_pending_race_is_verifiable():
    card = _Card(
        deadline=datetime(2026, 8, 19, 15, 30, 0, tzinfo=JST),
        status="scheduled",
        has_result=False,
    )
    pred = _Pred(datetime(2026, 8, 19, 7, 0, 0))
    info = classify_prediction_timing(card, pred)
    assert info["status"] == "pending"
    assert is_hit_verifiable(card, pred)
