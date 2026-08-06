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
    "win_margin",
    "fav_win_prob",
    "win_entropy",
    "top_trifecta_prob",
    "top_trio_ticket_prob",
    "trifecta_mass_top3",
    "course1_fly_risk",
    "fav_is_course1",
    "venue_in_win_rate",
    "venue_nige_rate",
    "venue_makuri_rate",
    "venue_sashi_rate",
    "venue_kado_strength",
    "wind_speed",
    "wave_height",
    "temperature_norm",
    "tide_sensitive",
    "near_high_tide",
    "exhibition_complete",
    "fav_ex_advantage",
    "fav_ex_st_advantage",
    "fav_local_win",
    "fav_national_win",
    "fav_motor_q",
    "fav_racer_course_win",
    "field_nat_std",
    "field_grade_mean",
    "top3_mass",
    "same_day_in_win_rate",
    "grade_number",
    "day_number",
    "race_no_norm",
    "low_confidence_flag",
    "has_upset_flag",
    "fav_top3_prob",
    "second_win_prob",
]


def _entropy(probs: list[float]) -> float:
    vals = [max(float(p), 1e-12) for p in probs if p is not None]
    if not vals:
        return 0.0
    s = sum(vals) or 1.0
    vals = [v / s for v in vals]
    return float(-sum(v * math.log(v) for v in vals))


def extract_confidence_features(
    features: RaceFeatures,
    result: PredictionResult,
) -> dict[str, float]:
    """レース時点で使える特徴からメタ特徴を作る."""
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
    top_tf_p = float(snap.get("top_trifecta_prob") or ((sts[0] or {}).get("prob") if sts else 0.0) or 0.0)
    tf_mass = sum(float(t.get("prob") or 0.0) for t in sts[:3])

    trio_probs = {int(k): float(v) for k, v in (result.trio_probs or {}).items()}
    top3_boats = sorted(trio_probs, key=lambda w: trio_probs[w], reverse=True)[:3]
    top3_mass = sum(trio_probs.get(w, 0.0) for w in top3_boats)
    fav_t3 = float(trio_probs.get(fav, 0.0)) if fav is not None else 0.0

    boat_by_waku = {b.waku: b for b in features.boats}
    fav_boat = boat_by_waku.get(fav) if fav is not None else None
    fav_vals = (fav_boat.values if fav_boat else {}) or {}
    fav_raw = (fav_boat.raw if fav_boat else {}) or {}

    nat_rates = []
    grade_scores = []
    for b in features.boats:
        nat = b.values.get("national_win_rate")
        if nat is None:
            nat = _safe_float(b.raw.get("national_win_raw"), default=None)
            if nat is not None and nat > 1.5:
                nat = nat / 10.0
        if nat is not None:
            nat_rates.append(float(nat))
        g = b.values.get("grade_strength")
        if g is None:
            g = {"A1": 1.0, "A2": 0.72, "B1": 0.40, "B2": 0.18}.get(
                str(b.raw.get("grade") or ""), 0.4
            )
        grade_scores.append(float(g))

    exhibition = snap.get("exhibition") or {}
    ex_complete = 1.0 if exhibition.get("complete") else (
        1.0 if exhibition.get("phase") in {"complete", "ready", "ok"} else 0.0
    )
    if not exhibition and all(
        (b.raw.get("exhibition_time") is not None) for b in features.boats
    ):
        ex_complete = 1.0

    fly = float(
        snap.get("course1_fly_risk")
        or env.get("course1_fly_risk")
        or 0.0
    )

    def envf(key: str, default: float = 0.0) -> float:
        v = env.get(key)
        return float(v) if v is not None else default

    temp = envf("temperature", 20.0)
    temp_norm = max(0.0, min(1.0, (temp - 5.0) / 30.0))

    return {
        "win_margin": margin,
        "fav_win_prob": fav_p,
        "win_entropy": _entropy(list(win_probs.values())),
        "top_trifecta_prob": float(top_tf_p),
        "top_trio_ticket_prob": top_sp_p,
        "trifecta_mass_top3": float(tf_mass),
        "course1_fly_risk": fly,
        "fav_is_course1": 1.0 if fav == 1 else 0.0,
        "venue_in_win_rate": envf("venue_in_win_rate", 0.5),
        "venue_nige_rate": envf("venue_nige_rate", 0.4),
        "venue_makuri_rate": envf("venue_makuri_rate", 0.2),
        "venue_sashi_rate": envf("venue_sashi_rate", 0.2),
        "venue_kado_strength": envf("venue_kado_strength", 0.2),
        "wind_speed": envf("wind_speed", 0.0),
        "wave_height": envf("wave_height", 0.0),
        "temperature_norm": temp_norm,
        "tide_sensitive": 1.0 if env.get("tide_sensitive") else 0.0,
        "near_high_tide": 1.0 if env.get("near_high_tide") else 0.0,
        "exhibition_complete": float(ex_complete),
        "fav_ex_advantage": float(fav_vals.get("exhibition_advantage") or 0.0),
        "fav_ex_st_advantage": float(fav_vals.get("exhibition_st_advantage") or 0.0),
        "fav_local_win": float(fav_vals.get("local_win_rate") or 0.0),
        "fav_national_win": float(fav_vals.get("national_win_rate") or 0.0),
        "fav_motor_q": float(fav_vals.get("motor_quinella_rate") or 0.0),
        "fav_racer_course_win": float(fav_vals.get("racer_course_win") or 0.0),
        "field_nat_std": float(np.std(nat_rates)) if nat_rates else 0.0,
        "field_grade_mean": float(np.mean(grade_scores)) if grade_scores else 0.4,
        "top3_mass": float(top3_mass),
        "same_day_in_win_rate": envf("same_day_in_win_rate", 0.5),
        "grade_number": envf("grade_number", 0.0),
        "day_number": envf("day_number", 0.0),
        "race_no_norm": float(features.race_no or 0) / 12.0,
        "low_confidence_flag": 1.0 if snap.get("low_confidence") else 0.0,
        "has_upset_flag": 1.0 if result.has_upset else 0.0,
        "fav_top3_prob": fav_t3,
        "second_win_prob": sec_p,
    }


def _safe_float(v: Any, default: float | None = 0.0) -> float | None:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def features_to_vector(feat: dict[str, float]) -> np.ndarray:
    return np.asarray([float(feat.get(k, 0.0) or 0.0) for k in CONFIDENCE_FEATURES], dtype=float)


def rule_based_score(feat: dict[str, float]) -> float:
    """モデル未学習時のヒューリスティック信頼度."""
    score = 0.35
    score += 0.25 * min(1.0, feat.get("win_margin", 0.0) / 0.25)
    score += 0.12 * min(1.0, feat.get("fav_win_prob", 0.0) / 0.55)
    score += 0.10 * min(1.0, feat.get("top_trio_ticket_prob", 0.0) / 0.20)
    score += 0.08 * min(1.0, feat.get("top_trifecta_prob", 0.0) / 0.08)
    score += 0.06 * feat.get("exhibition_complete", 0.0)
    score += 0.05 * min(1.0, feat.get("fav_racer_course_win", 0.0) / 0.4)
    if feat.get("fav_is_course1", 0.0) >= 0.5:
        score += 0.08 * max(0.0, feat.get("venue_in_win_rate", 0.5) - 0.45) / 0.25
    score -= 0.18 * min(1.0, feat.get("course1_fly_risk", 0.0))
    score -= 0.08 * feat.get("low_confidence_flag", 0.0)
    score -= 0.05 * feat.get("has_upset_flag", 0.0)
    score -= 0.04 * min(1.0, feat.get("wind_speed", 0.0) / 8.0)
    score -= 0.03 * min(1.0, feat.get("wave_height", 0.0) / 8.0)
    score -= 0.04 * min(1.0, feat.get("win_entropy", 0.0) / 1.8)
    return float(max(0.02, min(0.95, score)))


def explain_confidence(feat: dict[str, float], score: float) -> list[str]:
    """UI用の短い根拠."""
    reasons: list[str] = []
    if feat.get("win_margin", 0) >= 0.12:
        reasons.append(f"本命の勝率差がはっきりしている（差{feat['win_margin']:.0%}）")
    elif feat.get("win_margin", 0) < 0.05:
        reasons.append("本命と対抗の差が小さく、荒れやすい")
    if feat.get("course1_fly_risk", 0) >= 0.45:
        reasons.append(f"1号艇飛びリスクが高め（{feat['course1_fly_risk']:.0%}）")
    elif feat.get("fav_is_course1", 0) >= 0.5 and feat.get("venue_in_win_rate", 0) >= 0.52:
        reasons.append(f"イン優勢水面（場の1コース勝率{feat['venue_in_win_rate']:.0%}）")
    if feat.get("exhibition_complete", 0) >= 0.5 and feat.get("fav_ex_advantage", 0) > 0.05:
        reasons.append("展示でも本命が優位")
    if feat.get("fav_racer_course_win", 0) >= 0.30:
        reasons.append("本命選手のコース成績が安定")
    if feat.get("wind_speed", 0) >= 6 or feat.get("wave_height", 0) >= 5:
        reasons.append("風・波が強く展開が読みにくい")
    if feat.get("top_trio_ticket_prob", 0) >= 0.12:
        reasons.append("3連複本命の確率質量が集中")
    if score >= 0.62:
        reasons.insert(0, "過去同型レースでは3連複が当たりやすい部類")
    elif score < 0.42:
        reasons.insert(0, "過去同型では外れが多く、見送り候補")
    return reasons[:4]


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
    """PredictionResult に confidence ブロックを書き込む."""
    assessment = get_confidence_model().assess(features, result)
    snap = dict(result.feature_snapshot or {})
    block = {
        "score": assessment.score,
        "is_confident": assessment.is_confident,
        "label": assessment.label,
        "tier": assessment.tier,
        "reasons": assessment.reasons,
        "source": assessment.source,
        "threshold": float(
            getattr(get_settings().prediction, "confidence_threshold", None)
            or get_confidence_model().threshold
        ),
        "version": get_confidence_model().version,
    }
    snap["confidence"] = block
    # 主要特徴もデバッグ用に残す（肥大化を避けるため主要のみ）
    snap["confidence_features"] = {
        k: round(float(v), 4)
        for k, v in assessment.features.items()
        if k
        in {
            "win_margin",
            "fav_win_prob",
            "course1_fly_risk",
            "venue_in_win_rate",
            "top_trio_ticket_prob",
            "top_trifecta_prob",
            "exhibition_complete",
            "wind_speed",
            "wave_height",
            "fav_racer_course_win",
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
    target_coverage: float = 0.30,
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

    def _pick_threshold(scores: np.ndarray, labels: np.ndarray, coverage: float) -> tuple[float, dict]:
        order = np.argsort(-scores)
        n = len(scores)
        if n == 0:
            return 0.55, {}
        # 目標カバー率付近で的中率が最大になる閾値を探索
        best_thr = float(np.quantile(scores, 1.0 - coverage))
        best = {"trio_rate": -1.0, "n": 0, "coverage": 0.0}
        for q in np.linspace(0.55, 0.90, 15):
            k = max(20, int(round(n * (1.0 - q))))
            idx = order[:k]
            rate = float(labels[idx].mean()) if len(idx) else 0.0
            thr = float(scores[idx[-1]]) if len(idx) else best_thr
            # カバー 15〜40% を優先
            cov = len(idx) / n
            if 0.15 <= cov <= 0.40 and rate > best["trio_rate"]:
                best = {"trio_rate": rate, "n": int(len(idx)), "coverage": cov, "threshold": thr}
                best_thr = thr
        if best["trio_rate"] < 0:
            k = max(20, int(round(n * coverage)))
            idx = order[:k]
            best_thr = float(scores[idx[-1]])
            best = {
                "trio_rate": float(labels[idx].mean()),
                "n": int(len(idx)),
                "coverage": len(idx) / n,
                "threshold": best_thr,
            }
        return best_thr, best

    rc = RaceConfidenceModel()
    rc.model = model
    rc.feature_columns = list(CONFIDENCE_FEATURES)
    rc.version = "confidence_v1"

    metrics: dict[str, Any] = {"n_train": int(len(y)), "base_rate": float(y.mean()) if len(y) else 0.0}
    if X_valid is not None and y_valid is not None and len(y_valid):
        scores = model.predict_proba(X_valid)[:, 1]
        thr, sel = _pick_threshold(scores, y_valid.astype(float), target_coverage)
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
