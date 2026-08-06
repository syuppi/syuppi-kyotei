"""機械学習モデル（LightGBM: 1着/連対/ランカー + 組み合わせ）."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from boatrace.config import ROOT_DIR, get_settings
from boatrace.features.builder import RaceFeatures
from boatrace.models.base import BasePredictor, PredictionResult
from boatrace.models.combinations import build_combination_bundle
from boatrace.models.course_prior import COURSE_PRIOR_BETA, apply_course_log_prior, select_favorite_probs
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
        blend_with_scoring: float = 0.08,
    ):
        self.win_model: Any | None = None
        self.top2_model: Any | None = None
        self.top3_model: Any | None = None
        self.ranker: Any | None = None
        self.fly_model: Any | None = None
        self.session = session
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
        return cls(session=session, blend_with_scoring=0.08)

    def _load(self, path: Path) -> None:
        try:
            import joblib

            payload = joblib.load(path)
            if isinstance(payload, dict) and "model" in payload:
                models = payload.get("models") or {}
                self.win_model = models.get("win") or payload["model"]
                self.top2_model = models.get("top2")
                self.top3_model = models.get("top3")
                self.ranker = models.get("ranker")
                self.fly_model = models.get("course1_fly") or payload.get("fly_model")
                self.feature_columns = payload.get("feature_columns", FEATURE_COLUMNS)
                self.importance = payload.get("feature_importance") or {}
                self.version = payload.get("version") or (
                    "rank_v3"
                    if self.ranker is not None
                    else ("place_v2" if self.top3_model is not None else "win_only")
                )
            else:
                self.win_model = payload
                self.fly_model = None
        except Exception:
            self.win_model = None
            self.top2_model = None
            self.top3_model = None
            self.ranker = None
            self.fly_model = None

    @property
    def model(self) -> Any | None:
        return self.win_model

    def _venue_prior(self, features: RaceFeatures):
        if self.session is None:
            return None
        try:
            from boatrace.db.models import RaceCard
            from boatrace.models.venue_cooccur import build_venue_trio_prior

            card = self.session.get(RaceCard, features.race_card_id)
            as_of = card.race_date if card else None
            return build_venue_trio_prior(self.session, features.venue_id, as_of)
        except Exception:
            return None

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
        # 線形正規化を使う（softmaxは差を潰してコース事前に負ける）
        raw_sum = float(np.sum(win_raw)) or 1.0
        ml_map = {w: float(win_raw[i] / raw_sum) for i, w in enumerate(wakus)}

        # ルールベースはコース偏りが強いのでブレンドしない（実力信号を維持）
        strength_probs = dict(ml_map)

        # 飛びリスク: 専用モデルがあれば優先（ヒューリスティックと軽くブレンド）
        heuristic_fly = float(features.env.get("course1_fly_risk") or 0.0)
        fly_risk = heuristic_fly
        if self.fly_model is not None:
            try:
                idx = wakus.index(1)
                ml_fly = float(_pos_proba(self.fly_model, X[idx : idx + 1])[0])
                fly_risk = 0.65 * ml_fly + 0.35 * heuristic_fly
            except Exception:
                pass
        venue_in = features.env.get("venue_in_win_rate")
        venue_in_f = float(venue_in) if venue_in is not None else None

        # 単勝: コース事後 + 非1本命ゲート（的中優先）
        win_probs = select_favorite_probs(
            strength_probs,
            fly_risk=fly_risk,
            venue_in_win=venue_in_f,
        )
        # 3連系: ゲートなしの広い実力分布
        combo_win = apply_course_log_prior(
            strength_probs,
            beta=0.15,
            fly_risk=fly_risk,
            venue_in_win=venue_in_f,
        )

        top2_probs = None
        top3_probs = None
        rank_scores = None
        if self.top2_model is not None:
            r2 = _pos_proba(self.top2_model, X)
            top2_probs = {w: float(r2[i]) for i, w in enumerate(wakus)}
        if self.top3_model is not None:
            r3 = _pos_proba(self.top3_model, X)
            top3_probs = {w: float(r3[i]) for i, w in enumerate(wakus)}
        # ランカーは枠リークが残りやすいので3連単には弱くだけ使う
        if self.ranker is not None:
            rs = np.asarray(self.ranker.predict(X), dtype=float)
            rs = rs - rs.max()
            rs = np.exp(rs)
            rank_scores = {w: float(rs[i]) for i, w in enumerate(wakus)}

        cfg = get_settings().prediction
        venue_prior = self._venue_prior(features)
        from boatrace.prediction.delay_upset import (
            compute_delay_beneficiaries,
            inject_delay_ana_tickets,
        )

        delay_info = compute_delay_beneficiaries(
            features,
            favorite=max(combo_win, key=combo_win.get) if combo_win else None,
            win_probs=combo_win,
            top3_probs=top3_probs,
            limit=3,
        )
        bundle = build_combination_bundle(
            combo_win,
            top2_probs,
            top3_probs,
            venue_prior=venue_prior,
            rank_scores=rank_scores,
            n_win=cfg.win_candidates,
            n_sanrenpuku=cfg.sanrenpuku_candidates,
            n_sanrentan=cfg.sanrentan_candidates,
            prior_weight=0.03,
            fly_risk=fly_risk,
            delay_beneficiaries=delay_info.get("beneficiaries") or [],
        )
        # 1着順位は勝率モデルを正とする（3連単本命の1着固定を避ける）
        rankings = sorted(wakus, key=lambda w: win_probs[w], reverse=True)
        quinella = bundle["top2_probs"]
        trio = bundle["top3_probs"]
        tickets = bundle.get("tickets") or {}
        tickets = inject_delay_ana_tickets(
            tickets,
            beneficiaries=delay_info.get("beneficiaries") or [],
            favorite=rankings[0] if rankings else delay_info.get("favorite"),
            top3_probs=top3_probs or trio,
            strengths=bundle.get("strengths"),
            fly_risk=fly_risk,
        )
        # 単勝チケットも勝率順で揃える
        if tickets.get("win"):
            win_sorted = rankings[: cfg.win_candidates]
            probs = [float(win_probs[w]) for w in win_sorted]
            psum = sum(probs) or 1.0
            tickets["win"] = [
                {
                    "rank": i + 1,
                    "combo": [w],
                    "label": str(w),
                    "prob": probs[i],
                    "stake_share": probs[i] / psum,
                    "odds": (tickets["win"][i].get("odds") if i < len(tickets["win"]) else None),
                    "ev": None,
                    "value_tag": None,
                }
                for i, w in enumerate(win_sorted)
            ]

        top_imp = list(self.importance.items())[:3]
        reasons: dict[int, list[str]] = {}
        win_labels = [t["label"] for t in tickets.get("win", [])]
        best_tf = tickets.get("sanrentan", [{}])[0].get("label") or ""
        best_tr = tickets.get("sanrenpuku", [{}])[0].get("label") or ""
        for boat in features.boats:
            msgs = [
                f"LightGBM勝率 {ml_map[boat.waku]*100:.1f}%",
                f"ブレンド後1着 {win_probs[boat.waku]*100:.1f}%",
            ]
            if top2_probs:
                msgs.append(f"2連対見込み {top2_probs[boat.waku]*100:.1f}%")
            if top3_probs:
                msgs.append(f"3連対見込み {top3_probs[boat.waku]*100:.1f}%")
            if win_labels:
                msgs.append(f"単勝候補 {', '.join(win_labels)}")
            if best_tr:
                msgs.append(f"本命3連複 {best_tr}")
            if best_tf:
                msgs.append(f"本命3連単 {best_tf}")
            if bundle.get("low_confidence"):
                msgs.append("3連単は低信頼（広め候補推奨）")
            for name, _imp in top_imp:
                if name in boat.values:
                    msgs.append(f"重要特徴 {name}={boat.values[name]:.3f}")
            msgs.extend(scored.reasons.get(boat.waku, [])[:1])
            reasons[boat.waku] = msgs

        margin = (
            win_probs[rankings[0]] - win_probs[rankings[1]] if len(rankings) > 1 else 1.0
        )
        has_upset = margin < get_settings().prediction.upset_margin_threshold
        upset = [w for w in rankings[1:4] if w >= 4] if has_upset else []
        fav0 = rankings[0] if rankings else None
        for w in delay_info.get("beneficiaries") or []:
            if int(w) not in upset and fav0 != int(w):
                upset.append(int(w))
        has_upset = has_upset or fly_risk >= 0.40 or bool(delay_info.get("beneficiaries"))

        snap = dict(scored.feature_snapshot)
        snap["ml_raw"] = {str(w): float(v) for w, v in zip(wakus, win_raw)}
        snap["strength_probs"] = {str(w): float(v) for w, v in strength_probs.items()}
        snap["course1_fly_risk"] = fly_risk
        snap["course1_fly_heuristic"] = heuristic_fly
        snap["has_fly_model"] = self.fly_model is not None
        snap["model"] = self.name
        snap["version"] = self.version
        snap["sanrentan"] = bundle["sanrentan"]
        snap["sanrenpuku"] = bundle["sanrenpuku"]
        snap["sanrentan_probs"] = bundle["sanrentan_probs"]
        snap["sanrenpuku_probs"] = bundle["sanrenpuku_probs"]
        snap["tickets"] = tickets
        snap["low_confidence"] = bundle.get("low_confidence")
        snap["top_trifecta_prob"] = bundle.get("top_trifecta_prob")
        snap["has_venue_prior"] = bool(venue_prior)
        snap["delay_upset"] = delay_info
        snap["delay_thesis"] = delay_info.get("thesis") or ""

        return PredictionResult(
            model_name=self.name,
            rankings=rankings,
            win_probs=win_probs,
            quinella_probs=quinella,
            trio_probs=trio,
            candidates_win=rankings[: cfg.win_candidates],
            candidates_quinella=bundle["candidates_quinella"],
            candidates_trio=bundle["candidates_trio"],
            upset_candidates=upset,
            has_upset=has_upset or bool(bundle.get("low_confidence")),
            reasons=reasons,
            scores={w: float(v) for w, v in zip(wakus, win_raw)},
            feature_snapshot=snap,
            tickets=tickets,
        )
