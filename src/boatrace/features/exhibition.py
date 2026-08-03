"""展示・試走の特徴再計算とシナリオコメント."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from boatrace.features.builder import RaceFeatures


def exhibition_status(features: RaceFeatures) -> dict[str, Any]:
    """試走データの揃い具合を要約する."""
    boats = features.boats
    times = [(b.waku, b.raw.get("exhibition_time")) for b in boats]
    sts = [(b.waku, b.raw.get("exhibition_st")) for b in boats]
    courses = [(b.waku, b.raw.get("course")) for b in boats]
    n_time = sum(1 for _, v in times if v is not None)
    n_st = sum(1 for _, v in sts if v is not None)
    complete = n_time >= 6 and n_st >= 6
    best_time_waku = None
    best_st_waku = None
    if any(v is not None for _, v in times):
        best_time_waku = min(
            ((w, v) for w, v in times if v is not None),
            key=lambda x: x[1],
        )[0]
    if any(v is not None for _, v in sts):
        best_st_waku = min(
            ((w, v) for w, v in sts if v is not None),
            key=lambda x: x[1],
        )[0]
    return {
        "complete": complete,
        "n_exhibition_time": n_time,
        "n_exhibition_st": n_st,
        "best_time_waku": best_time_waku,
        "best_st_waku": best_st_waku,
        "phase": "試走反映済" if complete else ("試走一部取得" if n_time or n_st else "試走前"),
        "entries": [
            {
                "waku": b.waku,
                "exhibition_time": b.raw.get("exhibition_time"),
                "exhibition_st": b.raw.get("exhibition_st"),
                "course": b.raw.get("course"),
                "tilt": b.raw.get("tilt"),
                "weight_adjustment": b.raw.get("weight_adjustment"),
            }
            for b in boats
        ],
    }


def recompute_exhibition_advantages(features: RaceFeatures) -> RaceFeatures:
    """raw の展示タイム/ST から優位特徴をレース内で再計算する."""
    feats = deepcopy(features)
    times = [b.raw.get("exhibition_time") for b in feats.boats]
    sts = [b.raw.get("exhibition_st") for b in feats.boats]
    known_times = [t for t in times if t is not None]
    known_sts = [s for s in sts if s is not None]
    best_time = min(known_times) if known_times else None
    best_st = min(known_sts) if known_sts else None

    for boat in feats.boats:
        t = boat.raw.get("exhibition_time")
        s = boat.raw.get("exhibition_st")
        if t is None or best_time is None:
            boat.values["exhibition_advantage"] = 0.5
            if "exhibition_advantage" not in boat.missing:
                boat.missing.append("exhibition_advantage")
        else:
            gap = float(t) - float(best_time)
            boat.values["exhibition_advantage"] = max(0.0, 1.0 - gap / 0.15)
            boat.missing = [m for m in boat.missing if m != "exhibition_advantage"]

        if s is None or best_st is None:
            boat.values["exhibition_st_advantage"] = 0.5
            if "exhibition_st_advantage" not in boat.missing:
                boat.missing.append("exhibition_st_advantage")
        else:
            gap = float(s) - float(best_st)
            boat.values["exhibition_st_advantage"] = max(0.0, 1.0 - gap / 0.12)
            boat.missing = [m for m in boat.missing if m != "exhibition_st_advantage"]
    return feats


def apply_exhibition_overrides(
    features: RaceFeatures,
    overrides: dict[int, dict[str, Any]],
) -> RaceFeatures:
    """特定艇の展示値を上書きして再計算する."""
    feats = deepcopy(features)
    for boat in feats.boats:
        ov = overrides.get(boat.waku)
        if not ov:
            continue
        for key in ("exhibition_time", "exhibition_st", "course", "tilt", "weight_adjustment"):
            if key in ov and ov[key] is not None:
                boat.raw[key] = ov[key]
        if "course" in ov and ov["course"] is not None:
            # コース変更時の風・潮特徴は簡易に触らない（展示シナリオ中心）
            pass
    return recompute_exhibition_advantages(feats)


def _rankings_from_probs(win_probs: dict[int, float]) -> list[int]:
    return sorted(win_probs.keys(), key=lambda w: win_probs[w], reverse=True)


def _rank_of(rankings: list[int], waku: int) -> int:
    try:
        return rankings.index(waku) + 1
    except ValueError:
        return 99
