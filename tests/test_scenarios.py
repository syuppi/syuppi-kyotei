"""試走シナリオの単体テスト."""

from __future__ import annotations

from boatrace.features.builder import BoatFeatures, RaceFeatures
from boatrace.features.exhibition import apply_exhibition_overrides, exhibition_status
from boatrace.models.scoring import ScoringPredictor
from boatrace.prediction.scenarios import build_exhibition_scenarios


def _features(with_exhibition: bool = True) -> RaceFeatures:
    boats = []
    for w in range(1, 7):
        raw = {
            "exhibition_time": (6.70 + w * 0.02) if with_exhibition else None,
            "exhibition_st": (0.04 + w * 0.01) if with_exhibition else None,
            "course": w,
            "local_win_rate": 7.0 - w * 0.3,
            "national_win_rate": 6.5 - w * 0.2,
            "motor_quinella_rate": 40 - w,
            "boat_quinella_rate": 35,
            "avg_st": 0.15 + w * 0.01,
        }
        values = {
            "venue_course_win_rate": 0.55 if w == 1 else 0.1,
            "local_win_rate": 0.6 - w * 0.05,
            "exhibition_advantage": 0.5,
            "exhibition_st_advantage": 0.5,
            "motor_quinella_rate": 0.4,
            "national_win_rate": 0.5,
            "grade_strength": 0.6,
            "recent_form": 0.5,
            "boat_quinella_rate": 0.35,
            "st_advantage": 0.5,
            "tide_adjustment": 0.5,
            "wind_course_bias": 0.55 if w == 1 else 0.45,
            "same_day_course_form": 0.5,
        }
        boats.append(BoatFeatures(waku=w, racer_id=f"{w:04d}", values=values, raw=raw))
    return RaceFeatures(
        race_card_id=1,
        venue_id="01",
        race_no=1,
        boats=boats,
        env={"tide_sensitive": False, "near_high_tide": False, "wind_bucket": "calm"},
    )


def test_exhibition_status_complete():
    st = exhibition_status(_features(True))
    assert st["complete"] is True
    assert st["phase"] == "試走反映済"
    assert st["best_time_waku"] == 1
    assert st["best_st_waku"] == 1


def test_override_changes_advantage():
    base = _features(True)
    muted = apply_exhibition_overrides(base, {3: {"exhibition_st": 0.01}})
    b3 = next(b for b in muted.boats if b.waku == 3)
    assert b3.values["exhibition_st_advantage"] >= 0.99


def test_scenarios_before_exhibition():
    feats = _features(False)
    pred = ScoringPredictor(weights={k: 0.1 for k in feats.boats[0].values})
    # normalize-ish weights
    from boatrace.features.builder import DEFAULT_WEIGHTS

    pred = ScoringPredictor(weights=dict(DEFAULT_WEIGHTS))
    out = build_exhibition_scenarios(pred, feats, max_scenarios=6)
    assert out["status"]["phase"] == "試走前"
    assert out["scenarios"]
    assert any("スタート展示最速" in s["comment"] for s in out["scenarios"])


def test_scenarios_after_exhibition():
    feats = apply_exhibition_overrides(_features(True), {})
    from boatrace.features.builder import DEFAULT_WEIGHTS

    pred = ScoringPredictor(weights=dict(DEFAULT_WEIGHTS))
    out = build_exhibition_scenarios(pred, feats, max_scenarios=6)
    assert out["status"]["phase"] == "試走反映済"
    assert out["comments"]
