"""予測モデルの共通インターフェース."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from boatrace.features.builder import RaceFeatures


@dataclass
class PredictionResult:
    model_name: str
    rankings: list[int]
    win_probs: dict[int, float]
    quinella_probs: dict[int, float]
    trio_probs: dict[int, float]
    candidates_win: list[int]
    candidates_quinella: list[int]
    candidates_trio: list[int]
    upset_candidates: list[int]
    has_upset: bool
    reasons: dict[int, list[str]]
    scores: dict[int, float]
    feature_snapshot: dict[str, Any] = field(default_factory=dict)
    # 券種別の期待度順候補（単勝/3連複/3連単 各2〜3）
    tickets: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


class BasePredictor(ABC):
    name: str = "base"

    @abstractmethod
    def predict(self, features: RaceFeatures) -> PredictionResult:
        raise NotImplementedError
