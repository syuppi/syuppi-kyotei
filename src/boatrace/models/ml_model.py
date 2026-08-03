"""機械学習モデルの差し替え口（将来 LightGBM 等）."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from boatrace.features.builder import DEFAULT_WEIGHTS, RaceFeatures
from boatrace.models.base import BasePredictor, PredictionResult
from boatrace.models.scoring import ScoringPredictor, softmax


FEATURE_ORDER = list(DEFAULT_WEIGHTS.keys())


class MLPredictor(BasePredictor):
    """
    sklearn / LightGBM を差し込めるプレースホルダ。
    モデル未学習時は ScoringPredictor にフォールバックする。
    """

    name = "ml_v1"

    def __init__(self, model_path: str | Path | None = None):
        self.model: Any | None = None
        self.fallback = ScoringPredictor()
        if model_path and Path(model_path).exists():
            self._load(Path(model_path))

    def _load(self, path: Path) -> None:
        try:
            import joblib

            self.model = joblib.load(path)
        except Exception:
            self.model = None

    def predict(self, features: RaceFeatures) -> PredictionResult:
        if self.model is None:
            result = self.fallback.predict(features)
            result.model_name = self.name + "_fallback"
            return result

        X = []
        wakus = []
        for boat in features.boats:
            X.append([boat.values.get(k, 0.5) for k in FEATURE_ORDER])
            wakus.append(boat.waku)
        X_arr = np.asarray(X, dtype=float)

        # 分類器がクラス確率を返す想定。回帰スコアでも可。
        if hasattr(self.model, "predict_proba"):
            # 多クラスではなく「勝つ確率」を各艇独立に出す二値器の場合
            proba = self.model.predict_proba(X_arr)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                raw = proba[:, 1].tolist()
            else:
                raw = proba.reshape(-1).tolist()
        else:
            raw = self.model.predict(X_arr).tolist()

        win_list = softmax(raw, temperature=0.9)
        win_probs = {w: p for w, p in zip(wakus, win_list)}
        rankings = sorted(wakus, key=lambda w: win_probs[w], reverse=True)

        # 理由は特徴量上位で簡易生成
        reasons = {
            w: [f"MLスコア寄与に基づく予測（モデル: {self.name}）"]
            for w in wakus
        }
        scorer = ScoringPredictor()
        # 特徴寄与の説明を併用
        scored = scorer.predict(features)
        for w, msgs in scored.reasons.items():
            reasons[w] = msgs[:2] + reasons[w]

        return PredictionResult(
            model_name=self.name,
            rankings=rankings,
            win_probs=win_probs,
            quinella_probs=scored.quinella_probs,
            trio_probs=scored.trio_probs,
            candidates_win=rankings[:2],
            candidates_quinella=rankings[:3],
            candidates_trio=rankings[:4],
            upset_candidates=scored.upset_candidates,
            has_upset=scored.has_upset,
            reasons=reasons,
            scores={w: float(s) for w, s in zip(wakus, raw)},
            feature_snapshot=scored.feature_snapshot,
        )


def matrix_from_features(features: RaceFeatures) -> tuple[np.ndarray, list[int]]:
    X = [[b.values.get(k, 0.5) for k in FEATURE_ORDER] for b in features.boats]
    wakus = [b.waku for b in features.boats]
    return np.asarray(X, dtype=float), wakus
