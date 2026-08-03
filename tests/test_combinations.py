"""組み合わせ確率の単体テスト."""

from __future__ import annotations

from boatrace.models.combinations import (
    blend_place_strengths,
    build_combination_bundle,
    plackett_luce_ordered,
    unordered_trio_probs,
)


def test_plackett_luce_sums_to_one():
    strengths = {1: 0.4, 2: 0.2, 3: 0.15, 4: 0.12, 5: 0.08, 6: 0.05}
    ordered = plackett_luce_ordered(strengths)
    assert len(ordered) == 120
    assert abs(sum(p for _, p in ordered) - 1.0) < 1e-6
    assert ordered[0][0][0] == 1


def test_unordered_aggregates_perms():
    strengths = {1: 0.5, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.05, 6: 0.02}
    ordered = plackett_luce_ordered(strengths)
    unordered = unordered_trio_probs(ordered)
    assert abs(sum(p for _, p in unordered) - 1.0) < 1e-6
    assert len(unordered) == 20


def test_bundle_returns_tickets():
    win = {1: 0.4, 2: 0.2, 3: 0.15, 4: 0.1, 5: 0.1, 6: 0.05}
    top3 = {1: 0.8, 2: 0.6, 3: 0.55, 4: 0.4, 5: 0.35, 6: 0.2}
    bundle = build_combination_bundle(win, top2_probs=None, top3_probs=top3)
    assert len(bundle["rankings"]) == 6
    assert len(bundle["candidates_trio"]) >= 3
    assert len(bundle["sanrentan"][0]) == 3
    assert len(bundle["sanrenpuku"][0]) == 3
    tickets = bundle["tickets"]
    assert 2 <= len(tickets["win"]) <= 3
    assert 2 <= len(tickets["sanrenpuku"]) <= 3
    assert 2 <= len(tickets["sanrentan"]) <= 3
    assert abs(sum(t["stake_share"] for t in tickets["win"]) - 1.0) < 1e-6
    assert tickets["win"][0]["prob"] >= tickets["win"][-1]["prob"]
    strengths = blend_place_strengths(win, top3_probs=top3)
    assert abs(sum(strengths.values()) - 1.0) < 1e-6
