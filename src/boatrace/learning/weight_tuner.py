"""重み自動調整の公開入口（LearningService.tune_weights の薄いラッパ）."""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from boatrace.learning.service import LearningService


def tune_model_weights(session: Session, race_date: date | None = None) -> dict[str, float]:
    """日次精度に基づきグローバル特徴量重みを更新する."""
    return LearningService(session).tune_weights(race_date or date.today())


def compare_slices(session: Session, race_date: date) -> dict[str, Any]:
    """条件帯別精度を返す（場別/風速帯/潮位帯/展示帯）."""
    rows = LearningService(session).evaluate_accuracy(race_date)
    return rows
