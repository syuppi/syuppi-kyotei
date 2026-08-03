"""オッズ期待値ユーティリティのテスト."""

from __future__ import annotations

from boatrace.models.odds_ev import (
    enrich_tickets_with_ev,
    lookup_trifecta_odds,
    lookup_trio_odds,
    lookup_win_odds,
    top_ev_reasons,
)


def test_lookup_nested_trifecta_and_win():
    odds = {
        "win": {"1": 2.5, "5": 8.0},
        "trifecta": {"1": {"5": {"2": 34.5}}},
        "trio": {"1": {"2": {"5": 12.0}}},
    }
    assert lookup_win_odds(odds, 1) == 2.5
    assert lookup_trifecta_odds(odds, [1, 5, 2]) == 34.5
    assert lookup_trio_odds(odds, [5, 1, 2]) == 12.0


def test_enrich_tickets_sets_ev_and_reason():
    odds = {"win": {"1": 5.0}, "trifecta": {"1": {"2": {"3": 40.0}}}}
    tickets = {
        "win": [{"rank": 1, "combo": [1], "label": "1", "prob": 0.25, "stake_share": 1.0}],
        "sanrentan": [
            {"rank": 1, "combo": [1, 2, 3], "label": "1-2-3", "prob": 0.04, "stake_share": 1.0}
        ],
        "sanrenpuku": [],
    }
    out = enrich_tickets_with_ev(tickets, odds)
    assert out["win"][0]["odds"] == 5.0
    assert abs(out["win"][0]["ev"] - 1.25) < 1e-6
    assert "割安" in (out["win"][0]["ev_reason"] or "")
    assert out["sanrentan"][0]["ev"] == 1.6
    reasons = top_ev_reasons(out, limit=2)
    assert reasons
