"""説明可能なルールベース・スコアリングモデル."""

from __future__ import annotations

import math
from typing import Any

from sqlalchemy.orm import Session

from boatrace.config import get_settings
from boatrace.db.models import ModelWeights
from boatrace.features.builder import DEFAULT_WEIGHTS, RaceFeatures
from boatrace.models.base import BasePredictor, PredictionResult
from boatrace.models.combinations import build_combination_bundle


def softmax(xs: list[float], temperature: float = 1.0) -> list[float]:
    if not xs:
        return []
    t = max(temperature, 1e-6)
    m = max(xs)
    exps = [math.exp((x - m) / t) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


class ScoringPredictor(BasePredictor):
    name = "scoring_v1"

    def __init__(self, session: Session | None = None, weights: dict[str, float] | None = None):
        self.settings = get_settings()
        # 単独利用時は scoring_v1。MLフォールバック時も内部名は維持
        self.name = "scoring_v1"
        if weights is not None:
            self.weights = weights
        elif session is not None:
            self.weights = self._load_weights(session)
        else:
            self.weights = dict(DEFAULT_WEIGHTS)

    def _load_weights(self, session: Session) -> dict[str, float]:
        # 重みは scoring_v1 と設定モデル名の両方を参照
        names = ["scoring_v1", self.settings.prediction.model_name]
        weights = dict(DEFAULT_WEIGHTS)
        for name in names:
            rows = session.query(ModelWeights).filter_by(model_name=name).all()
            for r in rows:
                weights[r.feature_key] = r.weight
        for k, v in DEFAULT_WEIGHTS.items():
            weights.setdefault(k, v)
        return weights

    def predict(self, features: RaceFeatures) -> PredictionResult:
        scores: dict[int, float] = {}
        contribs: dict[int, dict[str, float]] = {}

        for boat in features.boats:
            total = 0.0
            parts: dict[str, float] = {}
            for key, weight in self.weights.items():
                val = boat.values.get(key, 0.5)
                c = weight * val
                parts[key] = c
                total += c
            scores[boat.waku] = total
            contribs[boat.waku] = parts

        wakus = [b.waku for b in features.boats]
        raw = [scores[w] for w in wakus]
        win_list = softmax(raw, self.settings.prediction.temperature)
        win_probs = {w: p for w, p in zip(wakus, win_list)}

        quinella_probs = self._place_probs(win_probs, top_n=2)
        trio_probs = self._place_probs(win_probs, top_n=3)
        cfg = self.settings.prediction
        bundle = build_combination_bundle(
            win_probs,
            quinella_probs,
            trio_probs,
            n_win=cfg.win_candidates,
            n_sanrenpuku=cfg.sanrenpuku_candidates,
            n_sanrentan=cfg.sanrentan_candidates,
        )

        # 1着はスコア由来の勝率順（3連単の1着固定を避ける）
        rankings = sorted(wakus, key=lambda w: win_probs[w], reverse=True)
        candidates_win = rankings[: cfg.win_candidates]
        candidates_quinella = bundle["candidates_quinella"]
        candidates_trio = bundle["candidates_trio"]
        tickets = bundle.get("tickets") or {}
        if tickets.get("win"):
            probs = [float(win_probs[w]) for w in candidates_win]
            psum = sum(probs) or 1.0
            tickets["win"] = [
                {
                    "rank": i + 1,
                    "combo": [w],
                    "label": str(w),
                    "prob": probs[i],
                    "stake_share": probs[i] / psum,
                }
                for i, w in enumerate(candidates_win)
            ]

        margin = win_probs[rankings[0]] - win_probs[rankings[1]] if len(rankings) > 1 else 1.0
        has_upset = margin < self.settings.prediction.upset_margin_threshold
        upset_candidates: list[int] = []
        if has_upset:
            upset_candidates = [w for w in rankings[1:4] if w >= 4]
            for boat in features.boats:
                if boat.values.get("exhibition_advantage", 0) >= 0.95 and boat.waku >= 4:
                    if boat.waku not in upset_candidates:
                        upset_candidates.append(boat.waku)

        reasons = self._build_reasons(features, contribs, win_probs)
        win_labels = [t["label"] for t in tickets.get("win", [])]
        best_tf = tickets.get("sanrentan", [{}])[0].get("label", "")
        best_tr = tickets.get("sanrenpuku", [{}])[0].get("label", "")
        for w, msgs in reasons.items():
            if win_labels:
                msgs.insert(0, f"単勝候補 {', '.join(win_labels)}")
            if best_tr:
                msgs.insert(1 if win_labels else 0, f"本命3連複 {best_tr}")
            if best_tf:
                msgs.insert(2 if win_labels else 1, f"本命3連単 {best_tf}")

        feature_snapshot: dict[str, Any] = {
            "env": features.env,
            "condition_keys": features.condition_keys,
            "boats": {
                str(b.waku): {"values": b.values, "raw": b.raw, "missing": b.missing}
                for b in features.boats
            },
            "weights": self.weights,
            "sanrentan": bundle["sanrentan"],
            "sanrenpuku": bundle["sanrenpuku"],
            "sanrentan_probs": bundle["sanrentan_probs"],
            "sanrenpuku_probs": bundle["sanrenpuku_probs"],
            "tickets": tickets,
        }

        return PredictionResult(
            model_name=self.name,
            rankings=rankings,
            win_probs=win_probs,
            quinella_probs=quinella_probs,
            trio_probs=trio_probs,
            candidates_win=candidates_win,
            candidates_quinella=candidates_quinella,
            candidates_trio=candidates_trio,
            upset_candidates=upset_candidates,
            has_upset=has_upset,
            reasons=reasons,
            scores=scores,
            feature_snapshot=feature_snapshot,
            tickets=tickets,
        )

    def _place_probs(self, win_probs: dict[int, float], top_n: int) -> dict[int, float]:
        """着以内確率の簡易近似: P(in top_n) ∝ P(win)^(0.7) を正規化しつつ底上げ."""
        items = list(win_probs.items())
        raw = {w: (p**0.7) + 0.02 for w, p in items}
        # 上位に入りやすいようにスケール
        s = sum(raw.values()) or 1.0
        base = {w: v / s for w, v in raw.items()}
        # top_n に入る期待を反映して再スケール（合計は top_n 前後になるようクリップ）
        factor = top_n / max(sum(base.values()), 1e-9)
        out = {w: min(0.95, v * factor * 0.85) for w, v in base.items()}
        return out

    def _build_reasons(
        self,
        features: RaceFeatures,
        contribs: dict[int, dict[str, float]],
        win_probs: dict[int, float],
    ) -> dict[int, list[str]]:
        label = {
            "venue_course_win_rate": "この場のコース別1着率が高い",
            "local_win_rate": "当地1着率が高い",
            "exhibition_advantage": "展示タイムが上位",
            "exhibition_st_advantage": "スタート展示が速い",
            "motor_quinella_rate": "モーター2連対率が高い",
            "national_win_rate": "全国勝率が高い",
            "grade_strength": "級別が上位",
            "recent_form": "直近成績が良い",
            "boat_quinella_rate": "ボート2連対率が高い",
            "st_advantage": "平均STが優位",
            "tide_adjustment": "潮位条件がこのコースに有利",
            "wind_course_bias": "風向・風速がこのコースに有利",
            "same_day_course_form": "当日同場の流れがこのコースに合う",
        }
        neg_label = {
            "tide_adjustment": "満潮付近でインの信頼度が低下",
            "wind_course_bias": "向かい風で差し・まくり傾向",
        }

        reasons: dict[int, list[str]] = {}
        env = features.env
        for boat in features.boats:
            parts = contribs[boat.waku]
            top = sorted(parts.items(), key=lambda x: x[1], reverse=True)[:3]
            msgs: list[str] = []
            for key, val in top:
                msgs.append(f"{label.get(key, key)}（寄与 {val:.3f}）")

            # 環境特記
            if env.get("tide_sensitive") and env.get("near_high_tide") and boat.waku == 1:
                msgs.append(neg_label["tide_adjustment"])
            if env.get("wind_bucket") in {"head", "head_light"} and (env.get("wind_speed") or 0) >= 3:
                if boat.waku >= 4 and parts.get("wind_course_bias", 0) > 0.03:
                    msgs.append("向かい風3m以上で差し・まくり余地")
                if boat.waku == 1:
                    msgs.append(neg_label["wind_course_bias"])

            if boat.raw.get("exhibition_time") is not None:
                msgs.append(f"展示タイム {boat.raw['exhibition_time']:.2f}")
            if boat.raw.get("exhibition_st") is not None:
                msgs.append(f"スタート展示 {boat.raw['exhibition_st']:.2f}")
            if boat.raw.get("grade_code"):
                msgs.append(f"級別 {boat.raw['grade_code']}")
            if env.get("same_day_prev_count"):
                msgs.append(
                    f"当日前R {env['same_day_prev_count']}走・イン勝率"
                    f"{float(env.get('same_day_in_win_rate') or 0)*100:.0f}%"
                )

            msgs.append(f"1着確率 {win_probs[boat.waku]*100:.1f}%")
            reasons[boat.waku] = msgs
        return reasons
