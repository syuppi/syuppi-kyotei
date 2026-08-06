"""展示前（試走未反映）モードの特徴・重み調整.

展示が揃う前でも、級別・モーター・選手コース・場傾向で
3連系的中を落とさないための中立化と重み再配分。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from boatrace.features.builder import RaceFeatures
from boatrace.features.exhibition import exhibition_status


# 展示未反映時に厚くする実力系（スコアリング）
PRE_EX_BOOST_KEYS: dict[str, float] = {
    "racer_course_win": 1.55,
    "local_win_rate": 1.40,
    "national_win_rate": 1.35,
    "grade_strength": 1.45,
    "motor_quinella_rate": 1.50,
    "boat_quinella_rate": 1.25,
    "recent_form": 1.30,
    "st_advantage": 1.20,
    "venue_course_win_rate": 1.15,
}

PRE_EX_ZERO_KEYS = {
    "exhibition_advantage",
    "exhibition_st_advantage",
}


def is_exhibition_complete(features: RaceFeatures) -> bool:
    return bool(exhibition_status(features).get("complete"))


def neutralize_exhibition_channels(features: RaceFeatures) -> RaceFeatures:
    """展示未反映時: 全艇の展示系を中立化し、枠リーク順位を作らない."""
    feats = deepcopy(features)
    for boat in feats.boats:
        boat.raw["exhibition_time"] = None
        boat.raw["exhibition_st"] = None
        boat.raw["ex_time_gap"] = 0.0
        boat.raw["ex_st_gap_vs_best"] = 0.0
        boat.raw["st_gap_vs_inner"] = 0.0
        boat.values["exhibition_advantage"] = 0.5
        boat.values["exhibition_st_advantage"] = 0.5
        for key in ("exhibition_advantage", "exhibition_st_advantage"):
            if key not in boat.missing:
                boat.missing.append(key)

    # ST差に依存しない飛びリスク（選手・モーター・場・気象のみ）
    env = dict(feats.env or {})
    fly = 0.18
    by_waku = {b.waku: b for b in feats.boats}
    b1 = by_waku.get(1)
    if b1 is not None:
        mq1 = float(b1.values.get("motor_quinella_rate") or 0.3)
        if mq1 < 0.28:
            fly += 0.10
        if float(b1.raw.get("racer_course_win") or b1.values.get("racer_course_win") or 0.5) < 0.45:
            fly += 0.10
        if int(b1.raw.get("f_count") or 0) >= 1:
            fly += 0.08
        if float(b1.values.get("grade_strength") or 0.4) < 0.5:
            fly += 0.06
    if float(env.get("venue_in_win_rate") or 0.55) < 0.50:
        fly += 0.08
    if float(env.get("venue_makuri_rate") or 0) + float(env.get("venue_sashi_rate") or 0) > 0.35:
        fly += 0.06
    wind = float(env.get("wind_speed") or 0)
    wave = float(env.get("wave_height") or 0)
    if wind >= 5:
        fly += 0.06
    if wave >= 5:
        fly += 0.08
    wb = str(env.get("wind_bucket") or "")
    if wb in {"head", "head_light", "cross"}:
        fly += 0.05
    env["course1_fly_risk"] = float(max(0.08, min(0.85, fly)))
    env["exhibition_complete"] = False
    env["pre_exhibition_mode"] = True
    feats.env = env
    return feats


def pre_exhibition_scoring_weights(weights: dict[str, float]) -> dict[str, float]:
    """展示寄与をゼロにし、実力系へ再配分."""
    out = dict(weights)
    freed = 0.0
    for key in PRE_EX_ZERO_KEYS:
        freed += float(out.get(key, 0.0))
        out[key] = 0.0
    boost_keys = [k for k in PRE_EX_BOOST_KEYS if k in out]
    if not boost_keys or freed <= 0:
        return out
    # 相対ブースト後に、ゼロ化した分を上乗せ
    for k in boost_keys:
        mult = PRE_EX_BOOST_KEYS[k]
        base = float(out[k])
        out[k] = base * mult
    # 余りを実力キーへ比例配分
    boost_sum = sum(float(out[k]) for k in boost_keys) or 1.0
    for k in boost_keys:
        out[k] = float(out[k]) + freed * (float(out[k]) / boost_sum)
    return out


def prepare_features_for_predict(features: RaceFeatures) -> tuple[RaceFeatures, dict[str, Any]]:
    """推論直前: 展示完了ならそのまま、未完了なら中立化モード."""
    status = exhibition_status(features)
    if status.get("complete"):
        env = dict(features.env or {})
        env["exhibition_complete"] = True
        env["pre_exhibition_mode"] = False
        features.env = env
        return features, {"pre_exhibition_mode": False, "exhibition": status}
    prepared = neutralize_exhibition_channels(features)
    return prepared, {"pre_exhibition_mode": True, "exhibition": status}
