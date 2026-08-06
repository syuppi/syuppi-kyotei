"""試走（展示）シナリオによる期待度変化コメント."""

from __future__ import annotations

from typing import Any

from boatrace.features.builder import RaceFeatures
from boatrace.features.exhibition import (
    apply_exhibition_overrides,
    exhibition_status,
)
from boatrace.models.base import BasePredictor


def _rankings(win_probs: dict[int, float]) -> list[int]:
    return sorted(win_probs.keys(), key=lambda w: win_probs[w], reverse=True)


def _rank_of(rankings: list[int], waku: int) -> int:
    try:
        return rankings.index(waku) + 1
    except ValueError:
        return 99


def _fmt_delta_rank(before: int, after: int) -> str:
    if after < before:
        return f"{before}位→{after}位（↑{before - after}）"
    if after > before:
        return f"{before}位→{after}位（↓{after - before}）"
    return f"{before}位のまま"


def _run_scenario(
    predictor: BasePredictor,
    features: RaceFeatures,
    overrides: dict[int, dict[str, Any]],
) -> dict[int, float]:
    mutated = apply_exhibition_overrides(features, overrides)
    result = predictor.predict(mutated)
    return result.win_probs


def build_exhibition_scenarios(
    predictor: BasePredictor,
    features: RaceFeatures,
    *,
    baseline_win_probs: dict[int, float] | None = None,
    max_scenarios: int = 8,
) -> dict[str, Any]:
    """
    試走結果の仮定ごとに期待度（1着確率順位）がどう変わるかを返す。

    試走前: 主要仮定のコメント
    試走後: 現状反映メモ + さらに良く/悪くなった場合の感度
    """
    status = exhibition_status(features)
    base_probs = baseline_win_probs or predictor.predict(features).win_probs
    base_rankings = _rankings(base_probs)
    favorite = base_rankings[0]
    contenders = base_rankings[:3]
    outers = [w for w in base_rankings if w >= 4][:2]

    known_times = [
        float(b.raw["exhibition_time"])
        for b in features.boats
        if b.raw.get("exhibition_time") is not None
    ]
    known_sts = [
        float(b.raw["exhibition_st"])
        for b in features.boats
        if b.raw.get("exhibition_st") is not None
    ]
    best_time = min(known_times) if known_times else 6.70
    worst_time = max(known_times) if known_times else best_time + 0.12
    best_st = min(known_sts) if known_sts else 0.05
    slow_st = max(known_sts) if known_sts else 0.20

    scenarios: list[dict[str, Any]] = []

    def add(
        key: str,
        title: str,
        overrides: dict[int, dict[str, Any]],
        focus: int,
    ) -> None:
        if len(scenarios) >= max_scenarios:
            return
        new_probs = _run_scenario(predictor, features, overrides)
        new_rankings = _rankings(new_probs)
        before_r = _rank_of(base_rankings, focus)
        after_r = _rank_of(new_rankings, focus)
        before_p = float(base_probs.get(focus, 0.0))
        after_p = float(new_probs.get(focus, 0.0))
        delta_pt = (after_p - before_p) * 100.0
        changed = after_r != before_r or abs(delta_pt) >= 0.8
        comment = (
            f"{title} → {focus}号艇は{_fmt_delta_rank(before_r, after_r)}"
            f"、1着確率 {before_p*100:.1f}%→{after_p*100:.1f}%"
            f"（{delta_pt:+.1f}pt）"
        )
        if not changed:
            comment = f"{title} → 期待度はほぼ変わらない（{focus}号艇 {before_r}位のまま）"
        scenarios.append(
            {
                "key": key,
                "title": title,
                "focus_waku": focus,
                "comment": comment,
                "before_rank": before_r,
                "after_rank": after_r,
                "before_prob": before_p,
                "after_prob": after_p,
                "delta_pt": delta_pt,
                "after_rankings": new_rankings[:3],
                "changed": changed,
            }
        )

    # 現状メモ
    status_comments: list[str] = []
    if status["complete"]:
        status_comments.append(
            f"試走反映済: 展示タイム最速={status['best_time_waku']}号艇 / "
            f"スタート展示最速={status['best_st_waku']}号艇"
        )
        # 既に反映されているが、さらに伸びた/遅れた場合の感度
        for w in contenders:
            add(
                f"st_even_faster_{w}",
                f"もし{w}号艇のスタート展示がさらに最速（{best_st - 0.02:.2f}）なら",
                {w: {"exhibition_st": max(0.01, best_st - 0.02)}},
                w,
            )
            add(
                f"time_best_{w}",
                f"もし{w}号艇の展示タイムが最速（{best_time:.2f}）なら",
                {w: {"exhibition_time": best_time}},
                w,
            )
        for w in outers:
            add(
                f"outer_st_best_{w}",
                f"もし{w}号艇がスタート展示最速なら",
                {w: {"exhibition_st": max(0.01, best_st - 0.01)}},
                w,
            )
            add(
                f"outer_time_best_{w}",
                f"もし{w}号艇の展示タイムが最速なら",
                {w: {"exhibition_time": best_time}},
                w,
            )
        add(
            f"fav_time_slow_{favorite}",
            f"もし本命{favorite}号艇の展示が鈍足（+0.10秒）なら",
            {favorite: {"exhibition_time": best_time + 0.10}},
            favorite,
        )
        add(
            f"fav_st_slow_{favorite}",
            f"もし本命{favorite}号艇のスタート展示が遅い（{slow_st:.2f}）なら",
            {favorite: {"exhibition_st": slow_st}},
            favorite,
        )
    else:
        status_comments.append(
            "試走前（または未取得）: 以下は試走結果が出たときの期待度変化シナリオ"
        )
        targets = list(dict.fromkeys(contenders + outers))[:4]
        for w in targets:
            add(
                f"hypo_st_best_{w}",
                f"もし{w}号艇がスタート展示最速なら",
                {w: {"exhibition_st": 0.03}},
                w,
            )
            add(
                f"hypo_time_best_{w}",
                f"もし{w}号艇の展示タイムが最速なら",
                {w: {"exhibition_time": 6.68}},
                w,
            )
        add(
            f"hypo_fav_slow_{favorite}",
            f"もし本命{favorite}号艇の展示が鈍足なら",
            {favorite: {"exhibition_time": 6.90, "exhibition_st": 0.22}},
            favorite,
        )
        # インの展示が悪いケース
        if 1 in base_probs:
            add(
                "hypo_in_bad",
                "もし1号艇のスタート展示が遅く展示タイムも平凡なら",
                {1: {"exhibition_st": 0.20, "exhibition_time": 6.88}},
                1,
            )

    # 変化が大きい順に並べ替え（現状メモは別枠）
    scenarios.sort(key=lambda s: (-abs(s["delta_pt"]), s["after_rank"]))
    scenarios = scenarios[:max_scenarios]

    # 本命遅れシナリオから受益艇（順位を上げた艇）を抽出
    delay_keys = {
        f"fav_st_slow_{favorite}",
        f"fav_time_slow_{favorite}",
        f"hypo_fav_slow_{favorite}",
        "hypo_in_bad",
    }
    delay_gainers: list[dict[str, Any]] = []
    for sc in scenarios:
        if sc.get("key") not in delay_keys:
            continue
        after = sc.get("after_rankings") or []
        for w in after:
            if w == favorite:
                continue
            before_r = _rank_of(base_rankings, w)
            after_r = _rank_of(after, w)
            if after_r < before_r:
                delay_gainers.append(
                    {
                        "waku": w,
                        "before_rank": before_r,
                        "after_rank": after_r,
                        "via": sc.get("key"),
                    }
                )
    # 順位上昇幅でユニーク化
    seen: set[int] = set()
    delay_beneficiaries: list[int] = []
    for g in sorted(delay_gainers, key=lambda x: (x["after_rank"] - x["before_rank"], x["after_rank"])):
        if g["waku"] in seen:
            continue
        seen.add(g["waku"])
        delay_beneficiaries.append(int(g["waku"]))

    return {
        "status": status,
        "status_comments": status_comments,
        "baseline_rankings": base_rankings[:3],
        "scenarios": scenarios,
        "comments": status_comments + [s["comment"] for s in scenarios],
        "delay_beneficiaries": delay_beneficiaries,
        "favorite": favorite,
    }
