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


def apply_course_log_prior(
    probs: dict[int, float],
    beta: float = COURSE_PRIOR_BETA,
    sharpen_gamma: float = ML_PROB_SHARPEN_GAMMA,
) -> dict[int, float]:
    """選手力ベース確率に全国コース事前を対数空間で混合して正規化."""
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
    scores: dict[int, float] = {}
    for w, p in probs.items():
        prior = float(GLOBAL_COURSE_WIN_PRIOR.get(int(w), 1.0 / 6.0))
        scores[w] = math.log(max(float(p), 1e-12)) + beta * math.log(max(prior, 1e-12))
    m = max(scores.values())
    exps = {w: math.exp(v - m) for w, v in scores.items()}
    s = sum(exps.values()) or 1.0
    return {w: float(v) / s for w, v in exps.items()}
