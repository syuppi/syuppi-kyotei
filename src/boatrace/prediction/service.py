"""予測サービス."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy.orm import Session, joinedload

from boatrace.config import get_settings
from boatrace.db.models import PredictHistory, RaceCard
from boatrace.features.builder import FeatureBuilder
from boatrace.features.exhibition import exhibition_status
from boatrace.logging_setup import get_logger
from boatrace.models.base import BasePredictor, PredictionResult
from boatrace.models.combinations import annotate_ticket_roles
from boatrace.models.ml_model import MLPredictor
from boatrace.models.scoring import ScoringPredictor
from boatrace.prediction.narrative import build_race_narrative
from boatrace.prediction.scenarios import build_exhibition_scenarios

logger = get_logger(__name__)


def get_predictor(session: Session, model: str = "auto") -> BasePredictor:
    from boatrace.models.ml_model import DEFAULT_MODEL_PATH

    if model == "auto":
        model = "lgbm" if DEFAULT_MODEL_PATH.exists() else "scoring"
    if model in {"ml", "lgbm", "lgbm_v1"}:
        return MLPredictor(session=session)
    return ScoringPredictor(session=session)


class PredictionService:
    def __init__(self, session: Session, model: str = "auto"):
        self.session = session
        self.builder = FeatureBuilder(session)
        self.predictor = get_predictor(session, model=model)
        self.settings = get_settings()

    def predict_race(self, race_card_id: int, persist: bool = True) -> PredictionResult:
        features = self.builder.build(race_card_id)
        result = self.predictor.predict(features)
        # 試走シナリオ（予測本体とは分離して再計算）
        try:
            weights = None
            if isinstance(self.predictor, MLPredictor) and getattr(
                self.predictor, "fallback", None
            ):
                weights = self.predictor.fallback.weights
            elif isinstance(self.predictor, ScoringPredictor):
                weights = self.predictor.weights
            scenario_predictor = ScoringPredictor(session=self.session, weights=weights)
            exhibition = build_exhibition_scenarios(
                scenario_predictor,
                features,
                baseline_win_probs=result.win_probs,
            )
            result.feature_snapshot = dict(result.feature_snapshot or {})
            result.feature_snapshot["exhibition"] = exhibition["status"]
            result.feature_snapshot["scenarios"] = exhibition
            pre_mode = bool((result.feature_snapshot or {}).get("pre_exhibition_mode"))
            # シナリオ上の本命遅れ受益艇を穴候補に合流（展示前は展示依存のためスキップ）
            scen_bens = exhibition.get("delay_beneficiaries") or []
            if scen_bens and not pre_mode:
                delay = dict(result.feature_snapshot.get("delay_upset") or {})
                merged = list(dict.fromkeys(list(delay.get("beneficiaries") or []) + list(scen_bens)))
                delay["beneficiaries"] = merged
                delay["scenario_beneficiaries"] = scen_bens
                result.feature_snapshot["delay_upset"] = delay
                from boatrace.prediction.delay_upset import inject_delay_ana_tickets

                fav = (result.rankings or [None])[0]
                enriched_t = inject_delay_ana_tickets(
                    result.tickets or result.feature_snapshot.get("tickets") or {},
                    beneficiaries=merged,
                    favorite=fav,
                    top3_probs=result.trio_probs,
                    strengths=None,
                    fly_risk=float(
                        (result.feature_snapshot or {}).get("course1_fly_risk")
                        or features.env.get("course1_fly_risk")
                        or 0.0
                    ),
                )
                result.tickets = enriched_t
                result.feature_snapshot["tickets"] = enriched_t
            # 理由の先頭に試走コメントを足す
            status_line = exhibition["status"].get("phase", "")
            extra = exhibition.get("status_comments") or []
            top_sc = [s["comment"] for s in exhibition.get("scenarios", [])[:3]]
            for waku, msgs in result.reasons.items():
                prefixed = []
                if status_line:
                    prefixed.append(f"試走状況: {status_line}")
                if not pre_mode:
                    prefixed.extend(extra[:1])
                    prefixed.extend(top_sc[:2])
                result.reasons[waku] = prefixed + list(msgs)
        except Exception as e:  # noqa: BLE001
            logger.warning("scenario_build_failed", race_card_id=race_card_id, error=str(e))
            result.feature_snapshot = dict(result.feature_snapshot or {})
            result.feature_snapshot["exhibition"] = exhibition_status(features)

        # オッズ×モデル確率で期待値を付与
        try:
            from boatrace.models.odds_ev import enrich_tickets_with_ev, top_ev_reasons

            card = self.session.get(RaceCard, race_card_id)
            odds = ((card.raw_payload or {}).get("odds") if card else None) or {}
            entry_win = {
                int(e.waku): float(e.win_odds)
                for e in (card.entries if card else [])
                if e.win_odds
            }
            enriched = enrich_tickets_with_ev(
                result.tickets or (result.feature_snapshot or {}).get("tickets"),
                odds,
                entry_win_odds=entry_win,
            )
            result.tickets = enriched
            result.feature_snapshot = dict(result.feature_snapshot or {})
            result.feature_snapshot["tickets"] = enriched
            result.feature_snapshot["odds"] = odds or (
                {"win": {str(k): v for k, v in entry_win.items()}} if entry_win else {}
            )
            result.feature_snapshot["has_odds"] = bool(odds or entry_win)
            ev_reasons = top_ev_reasons(enriched, limit=3)
            result.feature_snapshot["ev_reasons"] = ev_reasons
            if ev_reasons:
                for waku, msgs in list(result.reasons.items()):
                    # 全艇に同じ長文を付けず、本命艇にだけEV理由を載せる
                    if waku == result.rankings[0]:
                        result.reasons[waku] = list(ev_reasons) + list(msgs)
        except Exception as e:  # noqa: BLE001
            logger.warning("odds_ev_enrich_failed", race_card_id=race_card_id, error=str(e))

        # 本命/対抗/穴ラベル + 日本語根拠
        try:
            fly_risk = float(
                (result.feature_snapshot or {}).get("course1_fly_risk")
                or features.env.get("course1_fly_risk")
                or 0.0
            )
            labeled = annotate_ticket_roles(
                result.tickets or (result.feature_snapshot or {}).get("tickets"),
                fly_risk=fly_risk,
                upset_candidates=list(result.upset_candidates or []),
            )
            result.tickets = labeled
            result.feature_snapshot = dict(result.feature_snapshot or {})
            result.feature_snapshot["tickets"] = labeled
            narrative = build_race_narrative(features, result)
            result.feature_snapshot["race_thesis"] = narrative["race_thesis"]
            result.feature_snapshot["ticket_reasons"] = narrative["ticket_reasons"]
            result.feature_snapshot["styles"] = narrative["styles"]
            # narrative が why_short を更新した tickets を再保存
            result.tickets = result.feature_snapshot.get("tickets") or labeled
        except Exception as e:  # noqa: BLE001
            logger.warning("narrative_annotate_failed", race_card_id=race_card_id, error=str(e))

        # 的中しやすいレースの自信度（場・選手・天候・モデル出力から学習）
        if self.settings.prediction.confidence_enabled:
            try:
                from boatrace.prediction.confidence import attach_confidence

                attach_confidence(features, result)
            except Exception as e:  # noqa: BLE001
                logger.warning("confidence_attach_failed", race_card_id=race_card_id, error=str(e))

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
        # tickets は feature_snapshot 内にも保存済み
        if result.tickets:
            snap = dict(payload.get("feature_snapshot") or {})
            snap["tickets"] = result.tickets
            payload["feature_snapshot"] = snap
        if row is None:
            row = PredictHistory(race_card_id=race_card_id, model_name=result.model_name, **payload)
            self.session.add(row)
        else:
            for k, v in payload.items():
                setattr(row, k, v)
        self.session.flush()

    @staticmethod
    def to_dict(card: RaceCard, result: PredictionResult) -> dict[str, Any]:
        snap = result.feature_snapshot or {}
        tickets = result.tickets or snap.get("tickets") or {}
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
            "tickets": tickets,
            "sanrentan": [t["combo"] for t in tickets.get("sanrentan", [])]
            or snap.get("sanrentan")
            or [result.rankings[:3]],
            "sanrenpuku": [t["combo"] for t in tickets.get("sanrenpuku", [])]
            or snap.get("sanrenpuku")
            or [sorted(result.candidates_trio[:3])],
            "exhibition": snap.get("exhibition"),
            "scenarios": (snap.get("scenarios") or {}).get("comments")
            or [],
            "scenario_detail": snap.get("scenarios"),
            "ev_reasons": snap.get("ev_reasons") or [],
            "has_odds": bool(snap.get("has_odds")),
            "race_thesis": snap.get("race_thesis") or "",
            "ticket_reasons": snap.get("ticket_reasons") or {},
            "styles": snap.get("styles") or {},
            "delay_thesis": snap.get("delay_thesis") or "",
            "delay_upset": snap.get("delay_upset") or {},
            "confidence": snap.get("confidence") or {},
            "is_confident": bool((snap.get("confidence") or {}).get("is_confident")),
            "confidence_score": (snap.get("confidence") or {}).get("score"),
            "pre_exhibition_mode": bool(snap.get("pre_exhibition_mode")),
        }
