"""コース事前の適用（特徴量リークではなく事後校正）。"""

from __future__ import annotations

import math

from boatrace.features.builder import GLOBAL_COURSE_WIN_PRIOR

# 本命選定の既定（的中を常時1号艇に近づけつつ、確信度の高い非1のみ許可）
COURSE_PRIOR_BETA = 0.30
FAVORITE_STRENGTH_EDGE = 0.04
FAVORITE_FLY_THRESHOLD = 0.58
ML_PROB_SHARPEN_GAMMA = 1.0


def _adjusted_course_priors(
    fly_risk: float = 0.0,
    venue_in_win: float | None = None,
) -> dict[int, float]:
    """1号艇飛びリスク・場イン勝率で全国事前を補正."""
    priors = {int(k): float(v) for k, v in GLOBAL_COURSE_WIN_PRIOR.items()}
    if venue_in_win is not None:
        delta = float(venue_in_win) - 0.55
        priors[1] = max(0.28, min(0.65, priors[1] + delta * 0.6))
    fr = max(0.0, min(1.0, float(fly_risk or 0.0)))
    if fr > 0.25:
        move = min(0.28, (fr - 0.25) * 0.55)
        priors[1] = max(0.22, priors[1] - move)
        priors[2] = priors.get(2, 0.14) + move * 0.38
        priors[3] = priors.get(3, 0.13) + move * 0.36
        priors[4] = priors.get(4, 0.10) + move * 0.26
    s = sum(priors.values()) or 1.0
    return {w: v / s for w, v in priors.items()}


def apply_course_log_prior(
    probs: dict[int, float],
    beta: float = COURSE_PRIOR_BETA,
    sharpen_gamma: float = ML_PROB_SHARPEN_GAMMA,
    fly_risk: float = 0.0,
    venue_in_win: float | None = None,
) -> dict[int, float]:
    """選手力ベース確率に全国コース事前を対数空間で混合して正規化."""
    if not probs:
        return {}
    if sharpen_gamma and sharpen_gamma != 1.0:
        sharpened = {w: max(float(p), 1e-12) ** float(sharpen_gamma) for w, p in probs.items()}
        s0 = sum(sharpened.values()) or 1.0
        probs = {w: v / s0 for w, v in sharpened.items()}
    else:
        s0 = sum(float(v) for v in probs.values()) or 1.0
        probs = {w: float(v) / s0 for w, v in probs.items()}
    if beta <= 0:
        return probs
    priors = _adjusted_course_priors(fly_risk=fly_risk, venue_in_win=venue_in_win)
    scores: dict[int, float] = {}
    for w, p in probs.items():
        prior = float(priors.get(int(w), 1.0 / 6.0))
        scores[w] = math.log(max(float(p), 1e-12)) + beta * math.log(max(prior, 1e-12))
    m = max(scores.values())
    exps = {w: math.exp(v - m) for w, v in scores.items()}
    s = sum(exps.values()) or 1.0
    return {w: float(v) / s for w, v in exps.items()}


def select_favorite_probs(
    strength_probs: dict[int, float],
    *,
    beta: float = COURSE_PRIOR_BETA,
    fly_risk: float = 0.0,
    venue_in_win: float | None = None,
    strength_edge: float = FAVORITE_STRENGTH_EDGE,
) -> dict[int, float]:
    """本命選定: コース事前混合 + 非1号艇本命の実力ゲート.

    非1本命は次のときだけ許可:
    - 選手力が1号より strength_edge 以上、または
    - 飛びリスクが高く外寄りの候補が1号以上
    """
    strength = apply_course_log_prior(
        strength_probs, beta=0.0, fly_risk=0.0, venue_in_win=None
    )
    calibrated = apply_course_log_prior(
        strength_probs,
        beta=beta,
        fly_risk=fly_risk,
        venue_in_win=venue_in_win,
    )
    fav = max(calibrated, key=calibrated.get)
    if fav == 1:
        return calibrated

    s1 = float(strength.get(1, 0.0))
    sf = float(strength.get(fav, 0.0))
    fr = float(fly_risk or 0.0)
    allow = False
    if sf >= s1 + strength_edge:
        allow = True
    if fr >= FAVORITE_FLY_THRESHOLD and fav in {2, 3, 4} and sf >= s1:
        allow = True
    if venue_in_win is not None and float(venue_in_win) < 0.48 and sf >= s1 + strength_edge * 0.5:
        allow = True
    if allow:
        return calibrated

    out = dict(calibrated)
    others_max = max((v for w, v in out.items() if w != 1), default=0.0)
    out[1] = others_max + 0.01
    s = sum(out.values()) or 1.0
    return {w: float(v) / s for w, v in out.items()}
