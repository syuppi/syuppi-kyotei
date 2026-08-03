"""機械学習モデル（LightGBM）."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from boatrace.config import ROOT_DIR, get_settings
from boatrace.features.builder import RaceFeatures
from boatrace.models.base import BasePredictor, PredictionResult
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


class MLPredictor(BasePredictor):
    name = "lgbm_v1"

    def __init__(
        self,
        model_path: str | Path | None = None,
        session=None,
        blend_with_scoring: float = 0.25,
    ):
        self.model: Any | None = None
        self.feature_columns = FEATURE_COLUMNS
        self.importance: dict[str, float] = {}
        self.fallback = ScoringPredictor(session=session)
        self.blend = blend_with_scoring
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if path.exists():
            self._load(path)

    @classmethod
    def with_defaults(cls, session=None) -> "MLPredictor":
        return cls(session=session, blend_with_scoring=0.25)

    def _load(self, path: Path) -> None:
        try:
            import joblib

            payload = joblib.load(path)
            if isinstance(payload, dict) and "model" in payload:
                self.model = payload["model"]
                self.feature_columns = payload.get("feature_columns", FEATURE_COLUMNS)
                self.importance = payload.get("feature_importance") or {}
            else:
                self.model = payload
        except Exception:
            self.model = None

    def predict(self, features: RaceFeatures) -> PredictionResult:
        scored = self.fallback.predict(features)
        if self.model is None or len(features.boats) != 6:
            scored.model_name = self.name + "_fallback"
            return scored

        ranks = _intra_ranks(features)
        X = np.asarray(
            [boat_feature_vector(features, i, ranks) for i in range(6)],
            dtype=np.float64,
        )
        # 列順が保存時と違う場合に備える
        if hasattr(self.model, "n_features_in_") and X.shape[1] != self.model.n_features_in_:
            scored.model_name = self.name + "_fallback"
            return scored

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            raw = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1)
        else:
            raw = np.asarray(self.model.predict(X), dtype=float)

        # レース内 softmax
        ml_probs = softmax(raw.tolist(), temperature=0.75)
        wakus = [b.waku for b in features.boats]
        ml_map = {w: p for w, p in zip(wakus, ml_probs)}

        # ルールベースとブレンド（説明可能性と安定性）
        alpha = self.blend
        win_probs = {
            w: (1 - alpha) * ml_map[w] + alpha * scored.win_probs[w] for w in wakus
        }
        # 再正規化
        s = sum(win_probs.values()) or 1.0
        win_probs = {w: v / s for w, v in win_probs.items()}
        rankings = sorted(wakus, key=lambda w: win_probs[w], reverse=True)

        # 理由: 重要特徴 + スコアリング理由
        top_imp = list(self.importance.items())[:3]
        reasons: dict[int, list[str]] = {}
        for i, boat in enumerate(features.boats):
            msgs = [
                f"LightGBM勝率スコア {ml_map[boat.waku]*100:.1f}%",
                f"ブレンド後1着確率 {win_probs[boat.waku]*100:.1f}%",
            ]
            for name, _imp in top_imp:
                if name in boat.values:
                    msgs.append(f"重要特徴 {name}={boat.values[name]:.3f}")
            msgs.extend(scored.reasons.get(boat.waku, [])[:2])
            reasons[boat.waku] = msgs

        margin = win_probs[rankings[0]] - win_probs[rankings[1]] if len(rankings) > 1 else 1.0
        has_upset = margin < get_settings().prediction.upset_margin_threshold
        upset = [w for w in rankings[1:4] if w >= 4] if has_upset else []

        # 連対近似はスコアリング側の式を流用しつつ順位はML
        quinella = self.fallback._place_probs(win_probs, top_n=2)
        trio = self.fallback._place_probs(win_probs, top_n=3)

        snap = dict(scored.feature_snapshot)
        snap["ml_raw"] = {str(w): float(v) for w, v in zip(wakus, raw)}
        snap["model"] = self.name

        return PredictionResult(
            model_name=self.name,
            rankings=rankings,
            win_probs=win_probs,
            quinella_probs=quinella,
            trio_probs=trio,
            candidates_win=rankings[:2],
            candidates_quinella=rankings[:3],
            candidates_trio=rankings[:4],
            upset_candidates=upset,
            has_upset=has_upset,
            reasons=reasons,
            scores={w: float(v) for w, v in zip(wakus, raw)},
            feature_snapshot=snap,
        )
