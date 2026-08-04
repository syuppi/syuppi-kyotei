"""コース事前の適用（特徴量リークではなく事後校正）。"""

from __future__ import annotations

import math

from boatrace.features.builder import GLOBAL_COURSE_WIN_PRIOR

# log(p_ml) + beta * log(prior) の beta。
# オフライン調整: beta=0.4 で本命1号艇率 ≈ 60-65%、的中は常時1号艇に近い水準。
COURSE_PRIOR_BETA = 0.40


def apply_course_log_prior(
    probs: dict[int, float],
    beta: float = COURSE_PRIOR_BETA,
) -> dict[int, float]:
    """選手力ベース確率に全国コース事前を対数空間で混合して正規化."""
    if beta <= 0 or not probs:
        s = sum(probs.values()) or 1.0
        return {w: float(v) / s for w, v in probs.items()}
    scores: dict[int, float] = {}
    for w, p in probs.items():
        prior = float(GLOBAL_COURSE_WIN_PRIOR.get(int(w), 1.0 / 6.0))
        scores[w] = math.log(max(float(p), 1e-12)) + beta * math.log(max(prior, 1e-12))
    m = max(scores.values())
    exps = {w: math.exp(v - m) for w, v in scores.items()}
    s = sum(exps.values()) or 1.0
    return {w: float(v) / s for w, v in exps.items()}
