"""的中しやすいレースの信頼度スコア（メタモデル）.

場傾向・選手・天候・展示・モデル出力のレース単位特徴から、
3連複カバー的中確率を推定し、「自信あり」予想を選別する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np

from boatrace.config import ROOT_DIR, get_settings
from boatrace.features.builder import RaceFeatures
from boatrace.logging_setup import get_logger
from boatrace.models.base import PredictionResult

logger = get_logger(__name__)

DEFAULT_CONFIDENCE_PATH = ROOT_DIR / "data" / "models" / "race_confidence_v1.joblib"

CONFIDENCE_FEATURES: list[str] = [
    # モデル出力
    "win_margin",
    "fav_win_prob",
    "win_entropy",
    "top_trifecta_prob",
    "top_trio_ticket_prob",
    "trifecta_mass_top3",
    "course1_fly_risk",
    "fav_is_course1",
    "fav_top3_prob",
    "second_win_prob",
    "top3_mass",
    "low_confidence_flag",
    "has_upset_flag",
    # 1) インが強い水面
    "venue_in_win_rate",
    "venue_nige_rate",
    "venue_makuri_rate",
    "venue_sashi_rate",
    "venue_kado_strength",
    "typical_in_advantage",
    "in_fav_alignment",  # イン優勢×本命1号
    # 2) モーター差
    "fav_motor_q",
    "motor_spread",  # 上位と下位の2連対差
    "motor_top2_mean",
    "motor_clear_gap",  # 明確なモーター差フラグ的スコア
    # 3) 選手実力差
    "fav_local_win",
    "fav_national_win",
    "fav_racer_course_win",
    "fav_grade_strength",
    "field_nat_std",
    "field_grade_mean",
    "field_grade_std",
    "a1_on_course1",
    "grade_spread",
    # 4) 風・波
    "wind_speed",
    "wave_height",
    "wind_calm",  # 0-3m
    "wind_tail",
    "wind_head",
    "wind_cross",
    "temperature_norm",
    "tide_sensitive",
    "near_high_tide",
    # 5) 企画・番組性質
    "is_special_program",  # 特選/選抜/優勝/準優など
    "is_finalish",  # 優勝戦・準優勝戦
    "is_yosen",
    "is_fixed_entry",
    "grade_number",
    "day_number",
    "race_no_norm",
    "same_day_in_win_rate",
    # 6) 展示
    "exhibition_complete",
    "fav_ex_advantage",
    "fav_ex_st_advantage",
    "ex_time_spread",
    "ex_st_spread",
    "fav_ex_clear",  # 展示タイム差が明確
]


def _program_flags(title: str) -> dict[str, float]:
    t = title or ""
    special = any(k in t for k in ("特選", "選抜", "企画", "優勝", "準優", "準優勝"))
    finalish = any(k in t for k in ("優勝戦", "優勝", "準優勝戦", "準優"))
    yosen = ("予選" in t) and not special
    return {
        "is_special_program": 1.0 if special else 0.0,
        "is_finalish": 1.0 if finalish else 0.0,
        "is_yosen": 1.0 if yosen else 0.0,
    }


def _entropy(probs: list[float]) -> float:
    vals = [max(float(p), 1e-12) for p in probs if p is not None]
    if not vals:
        return 0.0
    s = sum(vals) or 1.0
    vals = [v / s for v in vals]
    return float(-sum(v * math.log(v) for v in vals))


def _grade_strength(boat) -> float:
    g = (boat.values or {}).get("grade_strength")
    if g is not None:
        return float(g)
    code = str((boat.raw or {}).get("grade_code") or (boat.raw or {}).get("grade") or "").upper()
    return {"A1": 1.0, "A2": 0.72, "B1": 0.40, "B2": 0.18}.get(code, 0.4)


def _motor_q(boat) -> float:
    v = (boat.values or {}).get("motor_quinella_rate")
    if v is not None:
        return float(v)
    raw = (boat.raw or {}).get("motor_quinella_rate")
    if raw is None:
        raw = (boat.raw or {}).get("motor_recent_q")
    if raw is None:
        return 0.3
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return 0.3
    if x > 1.5:
        x = x / 100.0
    return max(0.0, min(1.0, x))


def extract_confidence_features(
    features: RaceFeatures,
    result: PredictionResult,
) -> dict[str, float]:
    """レース時点で使える特徴からメタ特徴を作る（6観点を明示含む）."""
    snap = result.feature_snapshot or {}
    env = features.env or snap.get("env") or {}
    tickets = result.tickets or snap.get("tickets") or {}
    win_probs = {int(k): float(v) for k, v in (result.win_probs or {}).items()}
    rankings = list(result.rankings or [])
    fav = rankings[0] if rankings else None
    second = rankings[1] if len(rankings) > 1 else None

    fav_p = float(win_probs.get(fav, 0.0)) if fav is not None else 0.0
    sec_p = float(win_probs.get(second, 0.0)) if second is not None else 0.0
    margin = max(0.0, fav_p - sec_p)

    sps = tickets.get("sanrenpuku") or []
    sts = tickets.get("sanrentan") or []
    top_sp_p = float((sps[0] or {}).get("prob") or 0.0) if sps else 0.0
    top_tf_p = float(
        snap.get("top_trifecta_prob")
        or ((sts[0] or {}).get("prob") if sts else 0.0)
        or 0.0
    )
    tf_mass = sum(float(t.get("prob") or 0.0) for t in sts[:3])

    trio_probs = {int(k): float(v) for k, v in (result.trio_probs or {}).items()}
    top3_boats = sorted(trio_probs, key=lambda w: trio_probs[w], reverse=True)[:3]
    top3_mass = sum(trio_probs.get(w, 0.0) for w in top3_boats)
    fav_t3 = float(trio_probs.get(fav, 0.0)) if fav is not None else 0.0

    boat_by_waku = {b.waku: b for b in features.boats}
    fav_boat = boat_by_waku.get(fav) if fav is not None else None
    fav_vals = (fav_boat.values if fav_boat else {}) or {}
    course1 = boat_by_waku.get(1)

    nat_rates: list[float] = []
    grade_scores: list[float] = []
    motor_qs: list[float] = []
    ex_times: list[float] = []
    ex_sts: list[float] = []
    for b in features.boats:
        nat = b.values.get("national_win_rate")
        if nat is None:
            nat = _safe_float(b.raw.get("national_win_rate"), default=None)
            if nat is not None and nat > 1.5:
                nat = nat / 10.0
        if nat is not None:
            nat_rates.append(float(nat))
        grade_scores.append(_grade_strength(b))
        motor_qs.append(_motor_q(b))
        et = b.raw.get("exhibition_time")
        if et is not None:
            try:
                ex_times.append(float(et))
            except (TypeError, ValueError):
                pass
        est = b.raw.get("exhibition_st")
        if est is not None:
            try:
                ex_sts.append(float(est))
            except (TypeError, ValueError):
                pass

    motor_sorted = sorted(motor_qs, reverse=True)
    motor_top2 = float(np.mean(motor_sorted[:2])) if len(motor_sorted) >= 2 else (
        motor_sorted[0] if motor_sorted else 0.3
    )
    motor_bottom2 = float(np.mean(motor_sorted[-2:])) if len(motor_sorted) >= 2 else motor_top2
    motor_spread = max(0.0, motor_top2 - motor_bottom2)
    motor_clear = 0.0
    if len(motor_sorted) >= 3 and (motor_sorted[0] - motor_sorted[2]) >= 0.08:
        motor_clear = min(1.0, (motor_sorted[0] - motor_sorted[2]) / 0.20)

    grade_std = float(np.std(grade_scores)) if grade_scores else 0.0
    grade_spread = (
        float(max(grade_scores) - min(grade_scores)) if len(grade_scores) >= 2 else 0.0
    )
    a1_c1 = 1.0 if course1 is not None and _grade_strength(course1) >= 0.95 else 0.0

    exhibition = snap.get("exhibition") or {}
    ex_complete = 1.0 if exhibition.get("complete") else (
        1.0 if exhibition.get("phase") in {"complete", "ready", "ok"} else 0.0
    )
    if not exhibition and len(ex_times) >= 6:
        ex_complete = 1.0
    ex_time_spread = (max(ex_times) - min(ex_times)) if len(ex_times) >= 2 else 0.0
    ex_st_spread = (max(ex_sts) - min(ex_sts)) if len(ex_sts) >= 2 else 0.0
    fav_ex_clear = 0.0
    if fav_boat is not None and ex_times:
        try:
            fav_et = float(fav_boat.raw.get("exhibition_time"))
            best = min(ex_times)
            # 展示タイムが最良寄り、かつ場内差がある
            if fav_et <= best + 0.06 and ex_time_spread >= 0.08:
                fav_ex_clear = min(1.0, ex_time_spread / 0.20)
        except (TypeError, ValueError):
            fav_ex_clear = 0.0

    fly = float(snap.get("course1_fly_risk") or env.get("course1_fly_risk") or 0.0)
    wb = str(env.get("wind_bucket") or "other")
    wind = float(env.get("wind_speed") or 0.0)
    prog = _program_flags(str(env.get("race_title") or ""))

    def envf(key: str, default: float = 0.0) -> float:
        v = env.get(key)
        return float(v) if v is not None else default

    temp = envf("temperature", 20.0)
    temp_norm = max(0.0, min(1.0, (temp - 5.0) / 30.0))
    venue_in = envf("venue_in_win_rate", 0.5)
    typical_in = envf("typical_in_advantage", 0.52)
    in_align = 0.0
    if fav == 1:
        in_align = max(0.0, (venue_in - 0.48) / 0.15) * (1.0 - fly)

    return {
        "win_margin": margin,
        "fav_win_prob": fav_p,
        "win_entropy": _entropy(list(win_probs.values())),
        "top_trifecta_prob": float(top_tf_p),
        "top_trio_ticket_prob": top_sp_p,
        "trifecta_mass_top3": float(tf_mass),
        "course1_fly_risk": fly,
        "fav_is_course1": 1.0 if fav == 1 else 0.0,
        "fav_top3_prob": fav_t3,
        "second_win_prob": sec_p,
        "top3_mass": float(top3_mass),
        "low_confidence_flag": 1.0 if snap.get("low_confidence") else 0.0,
        "has_upset_flag": 1.0 if result.has_upset else 0.0,
        "venue_in_win_rate": venue_in,
        "venue_nige_rate": envf("venue_nige_rate", 0.4),
        "venue_makuri_rate": envf("venue_makuri_rate", 0.2),
        "venue_sashi_rate": envf("venue_sashi_rate", 0.2),
        "venue_kado_strength": envf("venue_kado_strength", 0.2),
        "typical_in_advantage": typical_in,
        "in_fav_alignment": float(max(0.0, min(1.0, in_align))),
        "fav_motor_q": float(
            fav_vals.get("motor_quinella_rate")
            or (_motor_q(fav_boat) if fav_boat else 0.3)
        ),
        "motor_spread": float(motor_spread),
        "motor_top2_mean": float(motor_top2),
        "motor_clear_gap": float(motor_clear),
        "fav_local_win": float(fav_vals.get("local_win_rate") or 0.0),
        "fav_national_win": float(fav_vals.get("national_win_rate") or 0.0),
        "fav_racer_course_win": float(fav_vals.get("racer_course_win") or 0.0),
        "fav_grade_strength": float(_grade_strength(fav_boat) if fav_boat else 0.4),
        "field_nat_std": float(np.std(nat_rates)) if nat_rates else 0.0,
        "field_grade_mean": float(np.mean(grade_scores)) if grade_scores else 0.4,
        "field_grade_std": grade_std,
        "a1_on_course1": a1_c1,
        "grade_spread": grade_spread,
        "wind_speed": wind,
        "wave_height": envf("wave_height", 0.0),
        "wind_calm": 1.0 if wind <= 3.0 else 0.0,
        "wind_tail": 1.0 if wb == "tail" else 0.0,
        "wind_head": 1.0 if wb in {"head", "head_light"} else 0.0,
        "wind_cross": 1.0 if wb == "cross" else 0.0,
        "temperature_norm": temp_norm,
        "tide_sensitive": 1.0 if env.get("tide_sensitive") else 0.0,
        "near_high_tide": 1.0 if env.get("near_high_tide") else 0.0,
        "is_special_program": prog["is_special_program"],
        "is_finalish": prog["is_finalish"],
        "is_yosen": prog["is_yosen"],
        "is_fixed_entry": 1.0 if env.get("is_fixed_entry") else 0.0,
        "grade_number": envf("grade_number", 0.0),
        "day_number": envf("day_number", 0.0),
        "race_no_norm": float(features.race_no or 0) / 12.0,
        "same_day_in_win_rate": envf("same_day_in_win_rate", 0.5),
        "exhibition_complete": float(ex_complete),
        "fav_ex_advantage": float(fav_vals.get("exhibition_advantage") or 0.0),
        "fav_ex_st_advantage": float(fav_vals.get("exhibition_st_advantage") or 0.0),
        "ex_time_spread": float(ex_time_spread),
        "ex_st_spread": float(ex_st_spread),
        "fav_ex_clear": float(fav_ex_clear),
    }


def _safe_float(v: Any, default: float | None = 0.0) -> float | None:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def features_to_vector(feat: dict[str, float], columns: list[str] | None = None) -> np.ndarray:
    cols = columns or CONFIDENCE_FEATURES
    return np.asarray([float(feat.get(k, 0.0) or 0.0) for k in cols], dtype=float)


def rule_based_score(feat: dict[str, float]) -> float:
    """モデル未学習時のヒューリスティック信頼度."""
    score = 0.28
    score += 0.18 * min(1.0, feat.get("win_margin", 0.0) / 0.22)
    score += 0.08 * min(1.0, feat.get("fav_win_prob", 0.0) / 0.55)
    score += 0.08 * min(1.0, feat.get("top_trio_ticket_prob", 0.0) / 0.18)
    score += 0.07 * min(1.0, feat.get("top_trifecta_prob", 0.0) / 0.08)
    score += 0.06 * feat.get("exhibition_complete", 0.0)
    score += 0.05 * feat.get("fav_ex_clear", 0.0)
    score += 0.05 * min(1.0, feat.get("motor_clear_gap", 0.0))
    score += 0.05 * min(1.0, feat.get("grade_spread", 0.0) / 0.6)
    score += 0.04 * feat.get("is_special_program", 0.0)
    score += 0.03 * feat.get("is_finalish", 0.0)
    score += 0.05 * feat.get("in_fav_alignment", 0.0)
    score += 0.04 * feat.get("a1_on_course1", 0.0)
    score += 0.04 * feat.get("wind_calm", 0.0)
    score += 0.03 * feat.get("wind_tail", 0.0)
    score -= 0.16 * min(1.0, feat.get("course1_fly_risk", 0.0))
    score -= 0.06 * feat.get("wind_head", 0.0)
    score -= 0.04 * feat.get("wind_cross", 0.0)
    score -= 0.05 * feat.get("low_confidence_flag", 0.0)
    score -= 0.04 * feat.get("has_upset_flag", 0.0)
    score -= 0.04 * min(1.0, feat.get("wind_speed", 0.0) / 8.0)
    score -= 0.03 * min(1.0, feat.get("wave_height", 0.0) / 8.0)
    score -= 0.04 * min(1.0, feat.get("win_entropy", 0.0) / 1.8)
    return float(max(0.02, min(0.95, score)))


def explain_confidence(feat: dict[str, float], score: float) -> list[str]:
    """UI用の短い根拠（6観点を意識）."""
    reasons: list[str] = []
    if feat.get("in_fav_alignment", 0) >= 0.35 or (
        feat.get("fav_is_course1", 0) >= 0.5 and feat.get("venue_in_win_rate", 0) >= 0.52
    ):
        reasons.append(
            f"イン優勢水面（1コース勝率{feat.get('venue_in_win_rate', 0):.0%}）で本命1号"
        )
    if feat.get("motor_clear_gap", 0) >= 0.4 or feat.get("motor_spread", 0) >= 0.10:
        reasons.append(
            f"モーター差が明確（上位と下位の2連対差{feat.get('motor_spread', 0):.0%}）"
        )
    if feat.get("a1_on_course1", 0) >= 0.5:
        reasons.append("A1が1コースで実力差がはっきり")
    elif feat.get("grade_spread", 0) >= 0.4:
        reasons.append("級別の実力差が大きい番組")
    if feat.get("wind_calm", 0) >= 0.5:
        reasons.append(f"風が弱く荒れにくい（{feat.get('wind_speed', 0):.0f}m）")
    elif feat.get("wind_tail", 0) >= 0.5:
        reasons.append("追い風でインの逃げが残りやすい")
    elif feat.get("wind_head", 0) >= 0.5 or feat.get("wind_cross", 0) >= 0.5:
        reasons.append("向かい風・横風で展開が荒れやすい")
    if feat.get("is_finalish", 0) >= 0.5:
        reasons.append("優勝戦・準優級で実力上位が集まりやすい")
    elif feat.get("is_special_program", 0) >= 0.5:
        reasons.append("特選・選抜など公式が組みやすい番組")
    if feat.get("fav_ex_clear", 0) >= 0.4 or (
        feat.get("exhibition_complete", 0) >= 0.5 and feat.get("fav_ex_advantage", 0) > 0.05
    ):
        reasons.append("展示タイム・気配で本命が明確")
    if feat.get("course1_fly_risk", 0) >= 0.45:
        reasons.append(f"1号艇飛びリスク高め（{feat['course1_fly_risk']:.0%}）")
    if feat.get("win_margin", 0) >= 0.12:
        reasons.append(f"本命の勝率差がはっきり（差{feat['win_margin']:.0%}）")
    if score >= 0.62:
        reasons.insert(0, "過去同型では3連系が当たりやすい部類")
    elif score < 0.42:
        reasons.insert(0, "過去同型では外れが多く、見送り候補")
    # 重複除去しつつ最大5
    out: list[str] = []
    for r in reasons:
        if r not in out:
            out.append(r)
        if len(out) >= 5:
            break
    return out


def focus_trifecta_on_top_trios(
    tickets: dict[str, Any],
    *,
    limit: int = 5,
    primary_perms: int = 3,
) -> dict[str, Any]:
    """自信ありレース向け: 3連複上位セットの順列に3連単を集中させる.

    本命セット3点 + 対抗セット残り、で着順カバーを厚くする。
    """
    from itertools import permutations

    sps = list(tickets.get("sanrenpuku") or [])
    sts = list(tickets.get("sanrentan") or [])
    if not sps:
        return tickets
    limit = max(2, min(int(limit), 6))

    def build_perms(combo: list[int]) -> list[dict[str, Any]]:
        key = set(int(x) for x in combo)
        ranked: list[dict[str, Any]] = []
        have: set[tuple[int, ...]] = set()
        for t in sts:
            c = [int(x) for x in (t.get("combo") or [])]
            if len(c) == 3 and set(c) == key and tuple(c) not in have:
                row = dict(t)
                row["combo"] = c
                row["label"] = "-".join(map(str, c))
                ranked.append(row)
                have.add(tuple(c))
        base_p = float((ranked[0].get("prob") if ranked else 0.03) or 0.03)
        for p in permutations([int(x) for x in combo], 3):
            if p in have:
                continue
            ranked.append(
                {
                    "rank": 0,
                    "combo": list(p),
                    "label": "-".join(map(str, p)),
                    "prob": base_p * (0.92 ** len(ranked)),
                    "stake_share": 0.0,
                    "focused": True,
                    "why_short": "自信あり→上位3連複の着順カバー",
                }
            )
            have.add(p)
        return ranked

    out: list[dict[str, Any]] = []
    used: set[tuple[int, ...]] = set()
    primary = [int(x) for x in (sps[0].get("combo") or [])]
    if len(primary) == 3:
        for row in build_perms(primary)[: max(1, primary_perms)]:
            key = tuple(row["combo"])
            if key in used:
                continue
            out.append(row)
            used.add(key)

    if len(sps) > 1 and len(out) < limit:
        secondary = [int(x) for x in (sps[1].get("combo") or [])]
        if len(secondary) == 3:
            for row in build_perms(secondary):
                if len(out) >= limit:
                    break
                key = tuple(row["combo"])
                if key in used:
                    continue
                out.append(row)
                used.add(key)

    # 足りなければ本命セットの残り順列
    if len(primary) == 3 and len(out) < limit:
        for row in build_perms(primary):
            if len(out) >= limit:
                break
            key = tuple(row["combo"])
            if key in used:
                continue
            out.append(row)
            used.add(key)

    probs = [max(float(r.get("prob") or 1e-9), 1e-9) for r in out]
    s = sum(probs) or 1.0
    for i, r in enumerate(out):
        r["rank"] = i + 1
        r["stake_share"] = probs[i] / s
        if i == 0 and not r.get("role"):
            r["role"] = "honmei"
        elif i == 1 and not r.get("role"):
            r["role"] = "taikou"
        elif i >= 2 and not r.get("role"):
            r["role"] = "ana"

    new_tickets = dict(tickets)
    new_tickets["sanrentan"] = out[:limit]
    return new_tickets


@dataclass
class ConfidenceAssessment:
    score: float
    is_confident: bool
    label: str
    tier: str  # high / mid / low
    reasons: list[str]
    features: dict[str, float]
    source: str  # model / rules


class RaceConfidenceModel:
    def __init__(self, path: Path | None = None):
        self.path = path or DEFAULT_CONFIDENCE_PATH
        self.model: lgb.LGBMClassifier | None = None
        self.threshold: float = 0.55
        self.feature_columns: list[str] = list(CONFIDENCE_FEATURES)
        self.metrics: dict[str, Any] = {}
        self.version: str = "confidence_v1"
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = joblib.load(self.path)
            self.model = payload.get("model")
            self.threshold = float(payload.get("threshold", 0.55))
            self.feature_columns = list(payload.get("feature_columns") or CONFIDENCE_FEATURES)
            self.metrics = dict(payload.get("metrics") or {})
            self.version = str(payload.get("version") or "confidence_v1")
        except Exception as e:  # noqa: BLE001
            logger.warning("confidence_model_load_failed", error=str(e))
            self.model = None

    def save(self, path: Path | None = None) -> Path:
        out = path or self.path
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "threshold": self.threshold,
                "feature_columns": self.feature_columns,
                "metrics": self.metrics,
                "version": self.version,
            },
            out,
        )
        return out

    def predict_score(self, feat: dict[str, float]) -> tuple[float, str]:
        if self.model is None:
            return rule_based_score(feat), "rules"
        vec = np.asarray(
            [[float(feat.get(k, 0.0) or 0.0) for k in self.feature_columns]],
            dtype=float,
        )
        try:
            proba = self.model.predict_proba(vec)[0]
            # positive class last for binary
            score = float(proba[1] if len(proba) > 1 else proba[0])
            return score, "model"
        except Exception as e:  # noqa: BLE001
            logger.warning("confidence_predict_failed", error=str(e))
            return rule_based_score(feat), "rules"

    def assess(self, features: RaceFeatures, result: PredictionResult) -> ConfidenceAssessment:
        cfg = get_settings().prediction
        feat = extract_confidence_features(features, result)
        score, source = self.predict_score(feat)
        # 学習済みモデルの閾値を優先し、設定は上書き用
        thr = float(self.threshold or 0.55)
        cfg_thr = getattr(cfg, "confidence_threshold", None)
        if cfg_thr is not None and self.model is None:
            thr = float(cfg_thr)
        elif cfg_thr is not None and abs(float(cfg_thr) - thr) < 0.02:
            thr = float(cfg_thr)
        is_conf = score >= thr
        if score >= max(thr, 0.62):
            tier, label = "high", "自信あり"
        elif score >= thr * 0.85:
            tier, label = "mid", "普通"
        else:
            tier, label = "low", "厳しい"
        if is_conf:
            label = "自信あり"
            tier = "high"
        return ConfidenceAssessment(
            score=round(score, 4),
            is_confident=is_conf,
            label=label,
            tier=tier,
            reasons=explain_confidence(feat, score),
            features=feat,
            source=source,
        )


_MODEL: RaceConfidenceModel | None = None


def get_confidence_model(reload: bool = False) -> RaceConfidenceModel:
    global _MODEL
    if _MODEL is None or reload:
        _MODEL = RaceConfidenceModel()
    return _MODEL


def attach_confidence(
    features: RaceFeatures,
    result: PredictionResult,
) -> ConfidenceAssessment:
    """PredictionResult に confidence を書き込み、自信ありなら3連単を絞る."""
    assessment = get_confidence_model().assess(features, result)
    snap = dict(result.feature_snapshot or {})
    tickets = result.tickets or snap.get("tickets") or {}
    focused = False
    if assessment.is_confident:
        limit = int(get_settings().prediction.sanrentan_candidates or 5)
        tickets = focus_trifecta_on_top_trios(tickets, limit=limit, primary_perms=3)
        result.tickets = tickets
        snap["tickets"] = tickets
        snap["sanrentan"] = [t.get("combo") for t in tickets.get("sanrentan", []) if t.get("combo")]
        focused = True
    block = {
        "score": assessment.score,
        "is_confident": assessment.is_confident,
        "label": assessment.label,
        "tier": assessment.tier,
        "reasons": assessment.reasons,
        "source": assessment.source,
        "threshold": float(get_confidence_model().threshold or 0.65),
        "version": get_confidence_model().version,
        "trifecta_focused": focused,
        "factors": {
            "in_course": round(float(assessment.features.get("in_fav_alignment") or 0), 3),
            "motor_gap": round(float(assessment.features.get("motor_clear_gap") or 0), 3),
            "grade_gap": round(float(assessment.features.get("grade_spread") or 0), 3),
            "wind_calm_or_tail": round(
                float(
                    max(
                        assessment.features.get("wind_calm") or 0,
                        assessment.features.get("wind_tail") or 0,
                    )
                ),
                3,
            ),
            "special_program": round(float(assessment.features.get("is_special_program") or 0), 3),
            "exhibition_clear": round(float(assessment.features.get("fav_ex_clear") or 0), 3),
        },
    }
    snap["confidence"] = block
    snap["confidence_features"] = {
        k: round(float(v), 4)
        for k, v in assessment.features.items()
        if k
        in {
            "win_margin",
            "fav_win_prob",
            "course1_fly_risk",
            "venue_in_win_rate",
            "in_fav_alignment",
            "motor_spread",
            "motor_clear_gap",
            "grade_spread",
            "a1_on_course1",
            "wind_calm",
            "wind_tail",
            "wind_head",
            "is_special_program",
            "is_finalish",
            "exhibition_complete",
            "fav_ex_clear",
            "ex_time_spread",
            "top_trio_ticket_prob",
            "top_trifecta_prob",
        }
    }
    result.feature_snapshot = snap
    return assessment


def train_confidence_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    X_valid: np.ndarray | None = None,
    y_valid: np.ndarray | None = None,
    y_tf_valid: np.ndarray | None = None,
    target_coverage: float = 0.30,
    target_tf: float = 0.35,
) -> RaceConfidenceModel:
    """3連複的中ラベルでメタモデルを学習し、閾値を検証セットで合わせる."""
    model = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=40,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    fit_kw: dict[str, Any] = {}
    if X_valid is not None and y_valid is not None and len(y_valid):
        fit_kw["eval_set"] = [(X_valid, y_valid)]
        fit_kw["callbacks"] = [lgb.early_stopping(40, verbose=False)]
    model.fit(X, y, **fit_kw)

    # 閾値: 検証の3連単的中が target_tf 以上になる最大カバーを優先
    def _pick_threshold(
        scores: np.ndarray,
        labels: np.ndarray,
        coverage: float,
        tf_labels: np.ndarray | None = None,
        target_tf_rate: float = 0.35,
    ) -> tuple[float, dict]:
        order = np.argsort(-scores)
        n = len(scores)
        if n == 0:
            return 0.55, {}
        best_thr = float(np.quantile(scores, 1.0 - coverage))
        best = {"trio_rate": -1.0, "tf_rate": -1.0, "n": 0, "coverage": 0.0}
        for q in np.linspace(0.50, 0.92, 22):
            k = max(15, int(round(n * (1.0 - q))))
            idx = order[:k]
            rate = float(labels[idx].mean()) if len(idx) else 0.0
            tf_rate = float(tf_labels[idx].mean()) if tf_labels is not None and len(idx) else 0.0
            thr = float(scores[idx[-1]]) if len(idx) else best_thr
            cov = len(idx) / n
            if not (0.10 <= cov <= 0.40):
                continue
            better = False
            if tf_labels is not None:
                if tf_rate >= target_tf_rate and (
                    best["tf_rate"] < target_tf_rate
                    or cov > best["coverage"]
                    or (abs(cov - best["coverage"]) < 1e-9 and rate > best["trio_rate"])
                ):
                    better = True
                elif best["tf_rate"] < target_tf_rate and tf_rate > best["tf_rate"]:
                    better = True
            else:
                if rate > best["trio_rate"]:
                    better = True
            if better:
                best = {
                    "trio_rate": rate,
                    "tf_rate": tf_rate,
                    "n": int(len(idx)),
                    "coverage": cov,
                    "threshold": thr,
                }
                best_thr = thr
        if best["trio_rate"] < 0:
            k = max(15, int(round(n * coverage)))
            idx = order[:k]
            best_thr = float(scores[idx[-1]])
            best = {
                "trio_rate": float(labels[idx].mean()),
                "tf_rate": float(tf_labels[idx].mean()) if tf_labels is not None else 0.0,
                "n": int(len(idx)),
                "coverage": len(idx) / n,
                "threshold": best_thr,
            }
        return best_thr, best

    rc = RaceConfidenceModel()
    rc.model = model
    rc.feature_columns = list(CONFIDENCE_FEATURES)
    rc.version = "confidence_v2"

    metrics: dict[str, Any] = {"n_train": int(len(y)), "base_rate": float(y.mean()) if len(y) else 0.0}
    if X_valid is not None and y_valid is not None and len(y_valid):
        scores = model.predict_proba(X_valid)[:, 1]
        thr, sel = _pick_threshold(
            scores,
            y_valid.astype(float),
            target_coverage,
            tf_labels=y_tf_valid.astype(float) if y_tf_valid is not None else None,
            target_tf_rate=target_tf,
        )
        rc.threshold = thr
        metrics["valid"] = {
            "n": int(len(y_valid)),
            "base_trio_rate": float(y_valid.mean()),
            "selected": sel,
            "auc_proxy_gap": float(sel.get("trio_rate", 0) - float(y_valid.mean())),
        }
    else:
        scores = model.predict_proba(X)[:, 1]
        thr, sel = _pick_threshold(scores, y.astype(float), target_coverage)
        rc.threshold = thr
        metrics["train_select"] = sel

    # feature importance
    try:
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            pairs = sorted(
                zip(CONFIDENCE_FEATURES, [float(x) for x in imp]),
                key=lambda x: -x[1],
            )
            metrics["top_features"] = pairs[:12]
    except Exception:
        pass

    rc.metrics = metrics
    return rc
