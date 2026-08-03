"""機械学習モデル（LightGBM: 1着/2連対/3連対 + 組み合わせ）."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from boatrace.config import ROOT_DIR, get_settings
from boatrace.features.builder import RaceFeatures
from boatrace.models.base import BasePredictor, PredictionResult
from boatrace.models.combinations import build_combination_bundle
from boatrace.models.dataset import FEATURE_COLUMNS, boat_feature_vector
from boatrace.models.scoring import ScoringPredictor, softmax

DEFAULT_MODEL_PATH = ROOT_DIR / "data" / "models" / "lgbm_win_v1.joblib"


def _intra_ranks(features: RaceFeatures) -> dict[str, list[float]]:
    boats = features.boats

    def rank_asc(vals: list[float | None]) -> list[float]:
        indexed = [(i, v if v is not None else 1e9) for i, v in enumerate(vals)]
        indexed.sort(key=lambda x: x[1])
        out = [0.5] * len(vals)
        n = max(len(vals) - 1, 1)
        for rank, (i, _) in enumerate(indexed):
            out[i] = 1.0 - rank / n
        return out

    def rank_desc(vals: list[float | None]) -> list[float]:
        indexed = [(i, v if v is not None else -1e9) for i, v in enumerate(vals)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        out = [0.5] * len(vals)
        n = max(len(vals) - 1, 1)
        for rank, (i, _) in enumerate(indexed):
            out[i] = 1.0 - rank / n
        return out

    return {
        "ex": rank_asc([b.raw.get("exhibition_time") for b in boats]),
        "ex_st": rank_asc([b.raw.get("exhibition_st") for b in boats]),
        "local": rank_desc([b.raw.get("local_win_rate") for b in boats]),
        "motor": rank_desc([b.raw.get("motor_quinella_rate") for b in boats]),
    }


def _pos_proba(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba, dtype=float).reshape(-1)
    return np.asarray(model.predict(X), dtype=float)


class MLPredictor(BasePredictor):
    name = "lgbm_v1"

    def __init__(
        self,
        model_path: str | Path | None = None,
        session=None,
        blend_with_scoring: float = 0.20,
    ):
        self.win_model: Any | None = None
        self.top2_model: Any | None = None
        self.top3_model: Any | None = None
        self.feature_columns = FEATURE_COLUMNS
        self.importance: dict[str, float] = {}
        self.fallback = ScoringPredictor(session=session)
        self.blend = blend_with_scoring
        self.version = "win_only"
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if path.exists():
            self._load(path)

    @classmethod
    def with_defaults(cls, session=None) -> "MLPredictor":
        return cls(session=session, blend_with_scoring=0.20)

    def _load(self, path: Path) -> None:
        try:
            import joblib

            payload = joblib.load(path)
            if isinstance(payload, dict) and "model" in payload:
                models = payload.get("models") or {}
                self.win_model = models.get("win") or payload["model"]
                self.top2_model = models.get("top2")
                self.top3_model = models.get("top3")
                self.feature_columns = payload.get("feature_columns", FEATURE_COLUMNS)
                self.importance = payload.get("feature_importance") or {}
                self.version = payload.get("version") or (
                    "place_v2" if self.top3_model is not None else "win_only"
                )
            else:
                self.win_model = payload
        except Exception:
            self.win_model = None
            self.top2_model = None
            self.top3_model = None

    @property
    def model(self) -> Any | None:
        return self.win_model

    def predict(self, features: RaceFeatures) -> PredictionResult:
        scored = self.fallback.predict(features)
        if self.win_model is None or len(features.boats) != 6:
            scored.model_name = self.name + "_fallback"
            return scored

        ranks = _intra_ranks(features)
        X = np.asarray(
            [boat_feature_vector(features, i, ranks) for i in range(6)],
            dtype=np.float64,
        )
        n_in = getattr(self.win_model, "n_features_in_", None)
        if n_in is not None and X.shape[1] != n_in:
            scored.model_name = self.name + "_fallback"
            return scored

        wakus = [b.waku for b in features.boats]
        win_raw = _pos_proba(self.win_model, X)
        ml_probs = softmax(win_raw.tolist(), temperature=0.75)
        ml_map = {w: p for w, p in zip(wakus, ml_probs)}

        alpha = self.blend
        win_probs = {
            w: (1 - alpha) * ml_map[w] + alpha * scored.win_probs[w] for w in wakus
        }
        s = sum(win_probs.values()) or 1.0
        win_probs = {w: v / s for w, v in win_probs.items()}

        top2_probs = None
        top3_probs = None
        if self.top2_model is not None:
            r2 = _pos_proba(self.top2_model, X)
            # ルール側 quinella とブレンド
            top2_probs = {
                w: 0.8 * float(r2[i]) + 0.2 * float(scored.quinella_probs.get(w, 0.3))
                for i, w in enumerate(wakus)
            }
        if self.top3_model is not None:
            r3 = _pos_proba(self.top3_model, X)
            top3_probs = {
                w: 0.8 * float(r3[i]) + 0.2 * float(scored.trio_probs.get(w, 0.4))
                for i, w in enumerate(wakus)
            }

        bundle = build_combination_bundle(win_probs, top2_probs, top3_probs)
        rankings = bundle["rankings"]
        quinella = bundle["top2_probs"]
        trio = bundle["top3_probs"]

        top_imp = list(self.importance.items())[:3]
        reasons: dict[int, list[str]] = {}
        best_tf = bundle["sanrentan"][0] if bundle["sanrentan"] else rankings[:3]
        best_tr = bundle["sanrenpuku"][0] if bundle["sanrenpuku"] else rankings[:3]
        for boat in features.boats:
            msgs = [
                f"LightGBM勝率 {ml_map[boat.waku]*100:.1f}%",
                f"ブレンド後1着 {win_probs[boat.waku]*100:.1f}%",
            ]
            if top2_probs:
                msgs.append(f"2連対見込み {top2_probs[boat.waku]*100:.1f}%")
            if top3_probs:
                msgs.append(f"3連対見込み {top3_probs[boat.waku]*100:.1f}%")
            if boat.waku in best_tr:
                msgs.append(f"本命3連複 {'-'.join(map(str, best_tr))} に含む")
            if boat.waku in best_tf:
                msgs.append(f"本命3連単 {'-'.join(map(str, best_tf))}")
            for name, _imp in top_imp:
                if name in boat.values:
                    msgs.append(f"重要特徴 {name}={boat.values[name]:.3f}")
            msgs.extend(scored.reasons.get(boat.waku, [])[:1])
            reasons[boat.waku] = msgs

        margin = win_probs[rankings[0]] - win_probs[rankings[1]] if len(rankings) > 1 else 1.0
        has_upset = margin < get_settings().prediction.upset_margin_threshold
        upset = [w for w in rankings[1:4] if w >= 4] if has_upset else []

        snap = dict(scored.feature_snapshot)
        snap["ml_raw"] = {str(w): float(v) for w, v in zip(wakus, win_raw)}
        snap["model"] = self.name
        snap["version"] = self.version
        snap["sanrentan"] = bundle["sanrentan"]
        snap["sanrenpuku"] = bundle["sanrenpuku"]
        snap["sanrentan_probs"] = bundle["sanrentan_probs"]
        snap["sanrenpuku_probs"] = bundle["sanrenpuku_probs"]

        return PredictionResult(
            model_name=self.name,
            rankings=rankings,
            win_probs=win_probs,
            quinella_probs=quinella,
            trio_probs=trio,
            candidates_win=bundle["candidates_win"],
            candidates_quinella=bundle["candidates_quinella"],
            candidates_trio=bundle["candidates_trio"],
            upset_candidates=upset,
            has_upset=has_upset,
            reasons=reasons,
            scores={w: float(v) for w, v in zip(wakus, win_raw)},
            feature_snapshot=snap,
        )
