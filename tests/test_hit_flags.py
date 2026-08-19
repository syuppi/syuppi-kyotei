"""的中判定: 3連複/3連単のみを賭け対象の的中とする."""

from __future__ import annotations

from boatrace.prediction.review import compute_hit_flags


def test_win_only_is_not_combo_hit():
    hits = compute_hit_flags(
        {"rank1": 1, "rank2": 2, "rank3": 3},
        {
            "win": [{"combo": [1], "label": "1"}],
            "sanrenpuku": [{"combo": [2, 3, 4], "label": "2-3-4"}],
            "sanrentan": [{"combo": [2, 3, 4], "label": "2-3-4"}],
        },
    )
    assert hits["hit_win"] is True
    assert hits["hit_trio"] is False
    assert hits["hit_tf"] is False
    assert hits["any_hit"] is False
    assert hits["combo_hit"] is False


def test_trio_counts_as_hit():
    hits = compute_hit_flags(
        {"rank1": 1, "rank2": 2, "rank3": 3},
        {
            "win": [{"combo": [4], "label": "4"}],
            "sanrenpuku": [{"combo": [1, 2, 3], "label": "1-2-3"}],
            "sanrentan": [{"combo": [4, 5, 6], "label": "4-5-6"}],
        },
    )
    assert hits["hit_win"] is False
    assert hits["hit_trio"] is True
    assert hits["any_hit"] is True
