"""コース事前の適用（特徴量リークではなく事後校正）。"""

from __future__ import annotations

import math

from boatrace.features.builder import GLOBAL_COURSE_WIN_PRIOR

# log(p_ml) + beta * log(prior)。
# v6モデルの勝率差は小さいため beta を強くすると本命が1号艇に潰れる。
# 7/28-8/3: beta=0.08 で本命1号艇率≈70%、的中≈53%。
COURSE_PRIOR_BETA = 0.08
# 1.0=尖らせない（尖らせると1号艇率が実測より下がりすぎる）
ML_PROB_SHARPEN_GAMMA = 1.0


def _adjusted_course_priors(
    fly_risk: float = 0.0,
    venue_in_win: float | None = None,
) -> dict[int, float]:
    """1号艇飛びリスク・場イン勝率で全国事前を補正."""
    priors = {int(k): float(v) for k, v in GLOBAL_COURSE_WIN_PRIOR.items()}
    # 場のインが弱いほど1コース事前を落とす
    if venue_in_win is not None:
        delta = float(venue_in_win) - 0.55
        priors[1] = max(0.28, min(0.65, priors[1] + delta * 0.6))
    # 飛びリスクで1→2/3/4へ質量移動
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
    """選手力ベース確率に全国コース事前を対数空間で混合して正規化.

    2段階:
      1) 選手力 probs（ML）
      2) 補正済みコース事前を beta で混合 → 本命選定
    """
    if not probs:
        return {}
    # レース内で尖らせ、微小差が事前に飲まれないようにする
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
