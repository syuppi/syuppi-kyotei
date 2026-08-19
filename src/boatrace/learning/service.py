"""学習パイプライン: 場別統計・補正・精度集計・重み調整."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Optional

from sqlalchemy.orm import Session, joinedload

from boatrace.config import get_settings, get_venue_map
from boatrace.db.models import (
    AccuracyDaily,
    ModelWeights,
    PredictHistory,
    RaceCard,
    RaceResult,
    VenueBias,
    VenueCourseStats,
)
from boatrace.prediction.timing import is_hit_verifiable
from boatrace.logging_setup import get_logger

logger = get_logger(__name__)


def condition_keys_for_card(card: RaceCard) -> list[str]:
    keys = ["all"]
    weather = card.weather
    tide = card.tide
    if weather:
        venue_map = get_venue_map()
        heading = None
        vc = venue_map.get(card.venue_id)
        if vc is not None:
            heading = vc.course_heading_deg
        wb = wind_bucket(
            weather.wind_direction, weather.wind_speed, course_heading_deg=heading
        )
        if wb in {"head", "head_light"} and (weather.wind_speed or 0) >= 3:
            keys.append("wind_head_ge3")
        if wb == "tail":
            keys.append("wind_tail")
        if weather.wave_height is not None and weather.wave_height >= 5:
            keys.append("wave_ge5")
    if tide and tide.near_high_tide:
        keys.append("tide_high")
    return keys


class LearningService:
    def __init__(self, session: Session):
        self.session = session
        self.settings = get_settings()

    def learn_day(self, race_date: date | None = None) -> dict[str, Any]:
        target = race_date or date.today()
        updated_stats = self.update_venue_course_stats(target)
        updated_bias = self.update_venue_bias(target)
        accuracy = self.evaluate_accuracy(target)
        weights = self.tune_weights(target)
        return {
            "date": target.isoformat(),
            "course_stats_updated": updated_stats,
            "bias_updated": updated_bias,
            "accuracy": accuracy,
            "weights": weights,
        }

    def update_venue_course_stats(self, race_date: date) -> int:
        cards = (
            self.session.query(RaceCard)
            .options(
                joinedload(RaceCard.result),
                joinedload(RaceCard.weather),
                joinedload(RaceCard.tide),
            )
            .filter(RaceCard.race_date == race_date, RaceCard.status == "finished")
            .all()
        )
        # 再集計のため、対象場の all を含む条件を当日結果からインクリメンタル更新
        touched = 0
        for card in cards:
            result = card.result
            if not result or not result.entry_results:
                continue
            keys = condition_keys_for_card(card)
            for er in result.entry_results:
                course = int(er.get("course") or er.get("waku") or 0)
                rank = er.get("rank")
                waku = er.get("waku")
                if not course or not rank:
                    # course不明なら枠をコースとみなす
                    course = int(waku or 0)
                if not course or not rank:
                    continue
                for key in keys:
                    touched += self._bump_course_stat(
                        card.venue_id, course, key, int(rank)
                    )
        self.session.flush()
        return touched

    def _bump_course_stat(self, venue_id: str, course: int, key: str, rank: int) -> int:
        row = (
            self.session.query(VenueCourseStats)
            .filter_by(venue_id=venue_id, course=course, condition_key=key)
            .one_or_none()
        )
        if row is None:
            row = VenueCourseStats(
                venue_id=venue_id,
                course=course,
                condition_key=key,
                starts=0,
                wins=0,
                quinellas=0,
                trios=0,
                win_rate=0.0,
                quinella_rate=0.0,
                trio_rate=0.0,
            )
            self.session.add(row)
            self.session.flush()
        row.starts = int(row.starts or 0) + 1
        if rank == 1:
            row.wins = int(row.wins or 0) + 1
        if rank <= 2:
            row.quinellas = int(row.quinellas or 0) + 1
        if rank <= 3:
            row.trios = int(row.trios or 0) + 1
        row.win_rate = row.wins / row.starts
        row.quinella_rate = row.quinellas / row.starts
        row.trio_rate = row.trios / row.starts
        row.updated_at = datetime.utcnow()
        return 1

    def update_venue_bias(self, race_date: date) -> int:
        """
        予測の寄与特徴と実着順の相関から、場別補正係数を緩やかに更新。
        1着艇の上位寄与特徴を強化、外れ本命の特徴を減衰。
        """
        cards = (
            self.session.query(RaceCard)
            .options(joinedload(RaceCard.result))
            .filter(RaceCard.race_date == race_date, RaceCard.status == "finished")
            .all()
        )
        model_name = self.settings.prediction.model_name
        lr = self.settings.learning.weight_lr
        updates = 0

        for card in cards:
            if not card.result or not card.result.rank1_waku:
                continue
            pred = (
                self.session.query(PredictHistory)
                .filter_by(race_card_id=card.id, model_name=model_name)
                .one_or_none()
            )
            if not pred or not pred.feature_snapshot:
                continue
            boats = pred.feature_snapshot.get("boats") or {}
            winner = str(card.result.rank1_waku)
            favorite = str(pred.rankings[0]) if pred.rankings else None
            winner_vals = (boats.get(winner) or {}).get("values") or {}

            # コース系は場バイアス更新から除外（1号艇偏重の増幅を防ぐ）
            blocked = {
                "venue_course_win_rate",
                "tide_adjustment",
                "wind_course_bias",
                "same_day_course_form",
            }
            # 勝者の高い特徴を強化
            for feature_key, val in winner_vals.items():
                if feature_key in blocked:
                    continue
                if val >= 0.55:
                    updates += self._adjust_bias(card.venue_id, feature_key, +lr * 0.5)

            # 本命外れ時、本命の強い特徴を減衰
            if favorite and favorite != winner:
                fav_vals = (boats.get(favorite) or {}).get("values") or {}
                for feature_key, val in fav_vals.items():
                    if feature_key in blocked:
                        continue
                    if val >= 0.7:
                        updates += self._adjust_bias(card.venue_id, feature_key, -lr * 0.3)

        self.session.flush()
        return updates

    def _adjust_bias(self, venue_id: str, feature_key: str, delta: float) -> int:
        row = (
            self.session.query(VenueBias)
            .filter_by(venue_id=venue_id, feature_key=feature_key)
            .one_or_none()
        )
        if row is None:
            row = VenueBias(
                venue_id=venue_id,
                feature_key=feature_key,
                coefficient=1.0,
                sample_count=0,
            )
            self.session.add(row)
            self.session.flush()
        # 補正幅を抑制（以前 1.5 上限でコース特徴が飽和していた）
        row.coefficient = float(min(1.20, max(0.80, (row.coefficient or 1.0) + delta)))
        row.sample_count = int(row.sample_count or 0) + 1
        row.updated_at = datetime.utcnow()
        return 1

    def evaluate_accuracy(self, race_date: date) -> dict[str, Any]:
        model_name = self.settings.prediction.model_name
        cards = (
            self.session.query(RaceCard)
            .options(joinedload(RaceCard.result), joinedload(RaceCard.weather), joinedload(RaceCard.tide))
            .filter(RaceCard.race_date == race_date, RaceCard.status == "finished")
            .all()
        )

        buckets: dict[tuple[Optional[str], str], dict[str, int]] = defaultdict(
            lambda: {
                "n": 0,
                "win": 0,
                "quinella": 0,
                "trio": 0,
                "trifecta": 0,
                "fav_top3": 0,
                "fav1": 0,
                "always1": 0,
                "fly_pred": 0,
                "fly_hit": 0,
            }
        )

        for card in cards:
            pred = (
                self.session.query(PredictHistory)
                .filter_by(race_card_id=card.id, model_name=model_name)
                .one_or_none()
            )
            result = card.result
            if not pred or not result or not result.rank1_waku:
                continue
            if not is_hit_verifiable(card, pred):
                continue

            # 複数候補のカバー的中（単勝/3連複/3連単 各2〜3）
            snap = pred.feature_snapshot or {}
            tickets = snap.get("tickets") or {}

            win_cands = [
                int(t["combo"][0])
                for t in tickets.get("win", [])
                if t.get("combo")
            ] or list(pred.candidates_win or [])[:3] or (pred.rankings[:1] if pred.rankings else [])
            hit_win = int(result.rank1_waku in win_cands)

            top2_pred = set(pred.candidates_quinella[:2])
            top2_real = {result.rank1_waku, result.rank2_waku}
            hit_quinella = int(top2_real <= top2_pred) if result.rank2_waku else 0

            top3_real = {result.rank1_waku, result.rank2_waku, result.rank3_waku}
            hit_trio = 0
            if None not in top3_real:
                sp_list = tickets.get("sanrenpuku") or []
                if sp_list:
                    for t in sp_list:
                        if set(t.get("combo") or []) == top3_real:
                            hit_trio = 1
                            break
                else:
                    top3_pred = set(pred.candidates_trio[:3])
                    hit_trio = int(top3_real <= top3_pred)

            hit_trifecta = 0
            if (
                result.rank2_waku is not None
                and result.rank3_waku is not None
            ):
                true_order = [
                    result.rank1_waku,
                    result.rank2_waku,
                    result.rank3_waku,
                ]
                st_list = tickets.get("sanrentan") or []
                if st_list:
                    for t in st_list:
                        if list(t.get("combo") or []) == true_order:
                            hit_trifecta = 1
                            break
                elif pred.rankings and len(pred.rankings) >= 3:
                    hit_trifecta = int(pred.rankings[:3] == true_order)

            fav = pred.rankings[0] if pred.rankings else None
            top3_set = {result.rank1_waku, result.rank2_waku, result.rank3_waku}
            hit_fav_top3 = int(fav in top3_set) if fav and None not in top3_set else 0
            fly_risk = float((snap.get("course1_fly_risk") or 0.0))
            pred_fly = fly_risk >= 0.45
            actual_fly = result.rank1_waku != 1

            slice_keys = ["all"]
            if card.weather:
                venue_map = get_venue_map()
                vc = venue_map.get(card.venue_id)
                heading = vc.course_heading_deg if vc else None
                wb = wind_bucket(
                    card.weather.wind_direction,
                    card.weather.wind_speed,
                    course_heading_deg=heading,
                )
                if (card.weather.wind_speed or 0) >= 3:
                    slice_keys.append(f"wind_{wb}_ge3")
                else:
                    slice_keys.append(f"wind_{wb}")
            if card.tide and card.tide.near_high_tide:
                slice_keys.append("tide_high")
            # 展示タイム帯（1号艇）
            slice_keys.append(self._exhibition_band(card))
            if pred_fly:
                slice_keys.append("pred_course1_fly")
            if actual_fly:
                slice_keys.append("actual_course1_fly")

            for venue_key in (None, card.venue_id):
                for sk in slice_keys:
                    b = buckets[(venue_key, sk)]
                    b["n"] += 1
                    b["win"] += hit_win
                    b["quinella"] += hit_quinella
                    b["trio"] += hit_trio
                    b["trifecta"] += hit_trifecta
                    b["fav_top3"] += hit_fav_top3
                    b["fav1"] += int(fav == 1) if fav else 0
                    b["always1"] += int(result.rank1_waku == 1)
                    if pred_fly:
                        b["fly_pred"] += 1
                        b["fly_hit"] += int(actual_fly)

        summary = []
        for (venue_id, slice_key), b in buckets.items():
            n = b["n"] or 1
            row = (
                self.session.query(AccuracyDaily)
                .filter_by(
                    stat_date=race_date,
                    venue_id=venue_id,
                    slice_key=slice_key,
                    model_name=model_name,
                )
                .one_or_none()
            )
            if row is None:
                row = AccuracyDaily(
                    stat_date=race_date,
                    venue_id=venue_id,
                    slice_key=slice_key,
                    model_name=model_name,
                )
                self.session.add(row)
            row.n_races = b["n"]
            row.hit_win = b["win"]
            row.hit_quinella = b["quinella"]
            row.hit_trio = b["trio"]
            row.hit_trifecta = b["trifecta"]
            row.win_rate = b["win"] / n
            row.quinella_rate = b["quinella"] / n
            row.trio_rate = b["trio"] / n
            row.trifecta_rate = b["trifecta"] / n
            row.updated_at = datetime.utcnow()
            summary.append(
                {
                    "venue_id": venue_id,
                    "slice_key": slice_key,
                    "n": b["n"],
                    "win_rate": row.win_rate,
                    "quinella_rate": row.quinella_rate,
                    "trio_rate": row.trio_rate,
                    "trifecta_rate": row.trifecta_rate,
                    "favorite_in_top3": b["fav_top3"] / n,
                    "fav1_rate": b["fav1"] / n,
                    "always1_baseline": b["always1"] / n,
                    "course1_fly_precision": (
                        b["fly_hit"] / b["fly_pred"] if b["fly_pred"] else None
                    ),
                }
            )
        self.session.flush()
        return {"slices": summary}

    def _exhibition_band(self, card: RaceCard) -> str:
        times = [e.exhibition_time for e in card.entries if e.exhibition_time]
        if not times:
            return "ex_unknown"
        best = min(times)
        if best <= 6.70:
            return "ex_fast"
        if best <= 6.85:
            return "ex_mid"
        return "ex_slow"

    def tune_weights(self, race_date: date) -> dict[str, float]:
        """
        直近精度に基づきグローバル重みを微小調整。
        爆発的な偏りを防ぐため、各特徴の変動幅と下限・上限を制限する。
        """
        model_name = self.settings.prediction.model_name
        lr = min(self.settings.learning.weight_lr, 0.02)
        weights = self._ensure_weights(model_name)

        # 初期値からの乖離を抑える
        base = dict(DEFAULT_WEIGHTS)

        tide_acc = (
            self.session.query(AccuracyDaily)
            .filter_by(
                stat_date=race_date,
                slice_key="tide_high",
                model_name=model_name,
                venue_id=None,
            )
            .one_or_none()
        )
        all_acc = (
            self.session.query(AccuracyDaily)
            .filter_by(
                stat_date=race_date,
                slice_key="all",
                model_name=model_name,
                venue_id=None,
            )
            .one_or_none()
        )
        if tide_acc and all_acc and tide_acc.n_races >= 5:
            # 3連複を主指標に（単勝は参考）
            tide_trio = float(getattr(tide_acc, "trio_rate", 0) or 0)
            all_trio = float(getattr(all_acc, "trio_rate", 0) or 0)
            if tide_trio < all_trio - 0.05 or (
                tide_acc.win_rate < all_acc.win_rate - 0.05 and tide_trio <= all_trio
            ):
                weights["tide_adjustment"] = weights.get("tide_adjustment", 0.06) + lr

        head_acc = (
            self.session.query(AccuracyDaily)
            .filter_by(
                stat_date=race_date,
                slice_key="wind_head_ge3",
                model_name=model_name,
                venue_id=None,
            )
            .one_or_none()
        )
        if head_acc and all_acc and head_acc.n_races >= 5:
            head_trio = float(getattr(head_acc, "trio_rate", 0) or 0)
            all_trio = float(getattr(all_acc, "trio_rate", 0) or 0)
            if head_trio < all_trio - 0.05:
                weights["wind_course_bias"] = weights.get("wind_course_bias", 0.06) + lr
                weights["venue_course_win_rate"] = (
                    weights.get("venue_course_win_rate", 0.22) + lr * 0.5
                )
                # 向かい風で3連が崩れる → 展示STを厚く
                weights["exhibition_st_advantage"] = (
                    weights.get("exhibition_st_advantage", 0.11) + lr * 0.5
                )

        fly_acc = (
            self.session.query(AccuracyDaily)
            .filter_by(
                stat_date=race_date,
                slice_key="pred_course1_fly",
                model_name=model_name,
                venue_id=None,
            )
            .one_or_none()
        )
        if fly_acc and all_acc and fly_acc.n_races >= 5:
            fly_trio = float(getattr(fly_acc, "trio_rate", 0) or 0)
            all_trio = float(getattr(all_acc, "trio_rate", 0) or 0)
            # 飛び想定レースで3連が弱い → 展示・当地を強化
            if fly_trio < all_trio - 0.04:
                weights["exhibition_st_advantage"] = (
                    weights.get("exhibition_st_advantage", 0.11) + lr
                )
                weights["exhibition_advantage"] = (
                    weights.get("exhibition_advantage", 0.13) + lr * 0.5
                )

        # クリップ: 初期値の 0.4〜2.0 倍に制限
        for k, b in base.items():
            w = weights.get(k, b)
            weights[k] = float(min(b * 2.0, max(b * 0.4, w)))

        # 正規化
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}
        self._save_weights(model_name, weights)
        return weights

    def _ensure_weights(self, model_name: str) -> dict[str, float]:
        rows = self.session.query(ModelWeights).filter_by(model_name=model_name).all()
        weights = dict(DEFAULT_WEIGHTS)
        if not rows:
            for k, v in DEFAULT_WEIGHTS.items():
                self.session.add(ModelWeights(model_name=model_name, feature_key=k, weight=v))
            self.session.flush()
            return weights
        for r in rows:
            weights[r.feature_key] = r.weight
        # 新規キーをDBへ補完
        existing = {r.feature_key for r in rows}
        for k, v in DEFAULT_WEIGHTS.items():
            if k not in existing:
                self.session.add(ModelWeights(model_name=model_name, feature_key=k, weight=v))
                weights[k] = v
        self.session.flush()
        return weights

    def _save_weights(self, model_name: str, weights: dict[str, float]) -> None:
        for k, v in weights.items():
            row = (
                self.session.query(ModelWeights)
                .filter_by(model_name=model_name, feature_key=k)
                .one_or_none()
            )
            if row is None:
                row = ModelWeights(model_name=model_name, feature_key=k, weight=v)
                self.session.add(row)
            else:
                row.weight = v
                row.updated_at = datetime.utcnow()
        self.session.flush()

    def accuracy_summary(self, days: int | None = None) -> dict[str, Any]:
        window = days or self.settings.learning.accuracy_window_days
        since = date.today() - timedelta(days=window)
        model_name = self.settings.prediction.model_name
        rows = (
            self.session.query(AccuracyDaily)
            .filter(
                AccuracyDaily.stat_date >= since,
                AccuracyDaily.venue_id.is_(None),
                AccuracyDaily.slice_key == "all",
                AccuracyDaily.model_name == model_name,
            )
            .all()
        )
        n = sum(r.n_races for r in rows)
        if n == 0:
            return {
                "window_days": window,
                "n_races": 0,
                "win_rate": 0.0,
                "quinella_rate": 0.0,
                "trio_rate": 0.0,
                "trifecta_rate": 0.0,
                "model_name": model_name,
            }
        return {
            "window_days": window,
            "n_races": n,
            "win_rate": sum(r.hit_win for r in rows) / n,
            "quinella_rate": sum(r.hit_quinella for r in rows) / n,
            "trio_rate": sum(r.hit_trio for r in rows) / n,
            "trifecta_rate": sum(getattr(r, "hit_trifecta", 0) or 0 for r in rows) / n,
            "model_name": model_name,
        }
