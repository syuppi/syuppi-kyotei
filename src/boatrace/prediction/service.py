"""予測サービス."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy.orm import Session, joinedload

from boatrace.config import get_settings
from boatrace.db.models import PredictHistory, RaceCard
from boatrace.features.builder import FeatureBuilder
from boatrace.logging_setup import get_logger
from boatrace.models.base import BasePredictor, PredictionResult
from boatrace.models.ml_model import MLPredictor
from boatrace.models.scoring import ScoringPredictor

logger = get_logger(__name__)


def get_predictor(session: Session, model: str = "scoring") -> BasePredictor:
    if model == "ml":
        return MLPredictor()
    return ScoringPredictor(session=session)


class PredictionService:
    def __init__(self, session: Session, model: str = "scoring"):
        self.session = session
        self.builder = FeatureBuilder(session)
        self.predictor = get_predictor(session, model=model)
        self.settings = get_settings()

    def predict_race(self, race_card_id: int, persist: bool = True) -> PredictionResult:
        features = self.builder.build(race_card_id)
        result = self.predictor.predict(features)
        if persist:
            self._save(race_card_id, result)
        return result

    def predict_day(
        self,
        race_date: date | None = None,
        venue_id: str | None = None,
        persist: bool = True,
    ) -> list[dict[str, Any]]:
        target = race_date or date.today()
        q = (
            self.session.query(RaceCard)
            .options(joinedload(RaceCard.entries), joinedload(RaceCard.venue))
            .filter(RaceCard.race_date == target)
        )
        if venue_id:
            q = q.filter(RaceCard.venue_id == venue_id)
        cards = q.order_by(RaceCard.venue_id, RaceCard.race_no).all()

        outputs: list[dict[str, Any]] = []
        for card in cards:
            if len(card.entries) < 6:
                logger.warning("skip_incomplete_card", race_card_id=card.id)
                continue
            try:
                result = self.predict_race(card.id, persist=persist)
                outputs.append(self.to_dict(card, result))
            except Exception as e:  # noqa: BLE001
                logger.exception("predict_failed", race_card_id=card.id, error=str(e))
        return outputs

    def _save(self, race_card_id: int, result: PredictionResult) -> None:
        row = (
            self.session.query(PredictHistory)
            .filter_by(race_card_id=race_card_id, model_name=result.model_name)
            .one_or_none()
        )
        payload = {
            "rankings": result.rankings,
            "win_probs": {str(k): v for k, v in result.win_probs.items()},
            "quinella_probs": {str(k): v for k, v in result.quinella_probs.items()},
            "trio_probs": {str(k): v for k, v in result.trio_probs.items()},
            "candidates_win": result.candidates_win,
            "candidates_quinella": result.candidates_quinella,
            "candidates_trio": result.candidates_trio,
            "upset_candidates": result.upset_candidates,
            "reasons": {str(k): v for k, v in result.reasons.items()},
            "feature_snapshot": result.feature_snapshot,
            "scores": {str(k): v for k, v in result.scores.items()},
            "has_upset": result.has_upset,
            "predicted_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        if row is None:
            row = PredictHistory(race_card_id=race_card_id, model_name=result.model_name, **payload)
            self.session.add(row)
        else:
            for k, v in payload.items():
                setattr(row, k, v)
        self.session.flush()

    @staticmethod
    def to_dict(card: RaceCard, result: PredictionResult) -> dict[str, Any]:
        return {
            "race_card_id": card.id,
            "venue_id": card.venue_id,
            "venue_name": card.venue.name if card.venue else card.venue_id,
            "race_date": card.race_date.isoformat(),
            "race_no": card.race_no,
            "race_title": card.race_title,
            "model_name": result.model_name,
            "rankings": result.rankings,
            "win_probs": result.win_probs,
            "quinella_probs": result.quinella_probs,
            "trio_probs": result.trio_probs,
            "candidates_win": result.candidates_win,
            "candidates_quinella": result.candidates_quinella,
            "candidates_trio": result.candidates_trio,
            "upset_candidates": result.upset_candidates,
            "has_upset": result.has_upset,
            "reasons": result.reasons,
            "scores": result.scores,
        }
