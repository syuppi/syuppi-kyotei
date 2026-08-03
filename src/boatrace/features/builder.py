"""特徴量定義とビルダー."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from sqlalchemy.orm import Session, joinedload

from boatrace.config import get_settings, get_venue_map
from boatrace.db.models import RaceCard, VenueBias, VenueCourseStats


DEFAULT_WEIGHTS: dict[str, float] = {
    "venue_course_win_rate": 0.22,
    "local_win_rate": 0.14,
    "exhibition_advantage": 0.14,
    "motor_quinella_rate": 0.10,
    "national_win_rate": 0.08,
    "recent_form": 0.08,
    "boat_quinella_rate": 0.06,
    "st_advantage": 0.06,
    "tide_adjustment": 0.06,
    "wind_course_bias": 0.06,
}


@dataclass
class BoatFeatures:
    waku: int
    racer_id: str
    values: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


@dataclass
class RaceFeatures:
    race_card_id: int
    venue_id: str
    race_no: int
    boats: list[BoatFeatures]
    env: dict[str, Any] = field(default_factory=dict)
    condition_keys: list[str] = field(default_factory=list)


def _safe_rate(value: float | None, default: float = 0.0, scale: float = 100.0) -> float:
    """パーセントや勝率を 0-1 に正規化."""
    if value is None:
        return default
    if value > 1.5:  # パーセント表記
        return max(0.0, min(value / scale, 1.0))
    return max(0.0, min(float(value) / 10.0 if value > 1.0 else float(value), 1.0))


def _win_rate_norm(value: float | None, default: float = 0.05) -> float:
    """勝率(目安2〜8)を0-1に."""
    if value is None:
        return default
    return max(0.0, min(float(value) / 10.0, 1.0))


def wind_bucket(direction: str | None, speed: float | None) -> str:
    if speed is None or speed < 1.0:
        return "calm"
    d = (direction or "").strip()
    # 公式の相対風向
    if "向" in d:
        return "head" if speed >= 3.0 else "head_light"
    if "追" in d:
        return "tail"
    if "横" in d:
        return "cross"
    # OpenAPI等の方位表記は相対変換できないため other（場別学習で吸収）
    return "other"


class FeatureBuilder:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.venue_map = get_venue_map()

    def build(self, race_card_id: int) -> RaceFeatures:
        card = (
            self.session.query(RaceCard)
            .options(
                joinedload(RaceCard.entries),
                joinedload(RaceCard.weather),
                joinedload(RaceCard.tide),
                joinedload(RaceCard.venue),
            )
            .filter(RaceCard.id == race_card_id)
            .one()
        )
        weather = card.weather
        tide = card.tide
        venue_cfg = self.venue_map.get(card.venue_id)

        wind_spd = weather.wind_speed if weather else None
        wind_dir = weather.wind_direction if weather else None
        wb = wind_bucket(wind_dir, wind_spd)
        wave = weather.wave_height if weather else None

        condition_keys = ["all"]
        if wb in {"head", "head_light"} and (wind_spd or 0) >= 3:
            condition_keys.append("wind_head_ge3")
        if wb == "tail":
            condition_keys.append("wind_tail")
        if tide and tide.near_high_tide:
            condition_keys.append("tide_high")
        if wave is not None and wave >= 5:
            condition_keys.append("wave_ge5")

        env = {
            "wind_speed": wind_spd,
            "wind_direction": wind_dir,
            "wind_bucket": wb,
            "wave_height": wave,
            "temperature": weather.temperature if weather else None,
            "water_temperature": weather.water_temperature if weather else None,
            "weather": weather.weather if weather else None,
            "tide_level_cm": tide.tide_level_cm if tide else None,
            "tide_delta_cm": tide.tide_delta_cm if tide else None,
            "near_high_tide": bool(tide.near_high_tide) if tide else False,
            "near_low_tide": bool(tide.near_low_tide) if tide else False,
            "tide_sensitive": bool(venue_cfg.tide_sensitive) if venue_cfg else False,
            "is_fixed_entry": card.is_fixed_entry,
            "typical_in_advantage": (
                venue_cfg.typical_in_advantage if venue_cfg else 0.52
            ),
        }

        # リーク防止: 当該レース日より前の結果だけから場コース傾向を作る
        course_stats = self._load_course_stats(
            card.venue_id, condition_keys, as_of_date=card.race_date
        )
        biases = self._load_biases(card.venue_id)

        # 展示・ST の相対評価用
        times = [e.exhibition_time for e in card.entries if e.exhibition_time]
        best_time = min(times) if times else None
        sts = [e.avg_st for e in card.entries if e.avg_st is not None]
        best_st = min(sts) if sts else None

        boats: list[BoatFeatures] = []
        for entry in sorted(card.entries, key=lambda x: x.waku):
            course = entry.estimated_course or entry.waku
            missing: list[str] = []
            values: dict[str, float] = {}

            vcw = course_stats.get(course, {}).get("win_rate")
            if vcw is None:
                # デフォルトのコース別勝率（全国平均に近い初期値）
                defaults = {1: 0.55, 2: 0.14, 3: 0.11, 4: 0.10, 5: 0.06, 6: 0.04}
                vcw = defaults.get(course, 0.08)
                missing.append("venue_course_win_rate")
            values["venue_course_win_rate"] = float(vcw)

            if entry.local_win_rate is None:
                missing.append("local_win_rate")
            values["local_win_rate"] = _win_rate_norm(entry.local_win_rate, 0.05)

            if entry.national_win_rate is None:
                missing.append("national_win_rate")
            values["national_win_rate"] = _win_rate_norm(entry.national_win_rate, 0.05)

            if entry.motor_quinella_rate is None:
                missing.append("motor_quinella_rate")
            values["motor_quinella_rate"] = _safe_rate(entry.motor_quinella_rate, 0.3)

            if entry.boat_quinella_rate is None:
                missing.append("boat_quinella_rate")
            values["boat_quinella_rate"] = _safe_rate(entry.boat_quinella_rate, 0.3)

            # 展示優位（速いほど高い）
            if entry.exhibition_time is None or best_time is None:
                values["exhibition_advantage"] = 0.5
                missing.append("exhibition_advantage")
            else:
                gap = entry.exhibition_time - best_time
                values["exhibition_advantage"] = max(0.0, 1.0 - gap / 0.15)

            # ST優位（小さいほど高い）
            if entry.avg_st is None or best_st is None:
                values["st_advantage"] = 0.5
                missing.append("st_advantage")
            else:
                gap = entry.avg_st - best_st
                values["st_advantage"] = max(0.0, 1.0 - gap / 0.10)

            # 直近フォーム（前走着順を簡易利用）
            if entry.previous_rank is None:
                values["recent_form"] = 0.4
                missing.append("recent_form")
            else:
                values["recent_form"] = max(0.0, (7 - entry.previous_rank) / 6.0)

            # 潮位補正特徴（インほど満潮で減点、アウトは微加点）
            tide_adj = 0.5
            if env["tide_sensitive"] and env.get("near_high_tide"):
                if course == 1:
                    tide_adj = 0.30
                elif course <= 3:
                    tide_adj = 0.40
                else:
                    tide_adj = 0.60
            elif env["tide_sensitive"] and env.get("near_low_tide"):
                if course == 1:
                    tide_adj = 0.55
            values["tide_adjustment"] = tide_adj

            # 風向コースバイアス
            wind_bias = 0.5
            if wb in {"head", "head_light"} and (wind_spd or 0) >= 3:
                # 差し・まくり増 → アウト寄り
                wind_bias = 0.35 if course == 1 else (0.65 if course >= 4 else 0.55)
            elif wb == "tail":
                wind_bias = 0.62 if course == 1 else (0.40 if course >= 5 else 0.50)
            if env["is_fixed_entry"] and course == 1:
                wind_bias = min(1.0, wind_bias + 0.08)
            # 場のイン有利度
            in_adv = float(env["typical_in_advantage"])
            if course == 1:
                wind_bias = 0.5 * wind_bias + 0.5 * in_adv
            values["wind_course_bias"] = wind_bias

            # 場別補正係数を特徴に乗算（学習済み）
            for key in list(values.keys()):
                coef = biases.get(key, 1.0)
                values[key] = values[key] * coef

            boats.append(
                BoatFeatures(
                    waku=entry.waku,
                    racer_id=entry.racer_id,
                    values=values,
                    raw={
                        "exhibition_time": entry.exhibition_time,
                        "avg_st": entry.avg_st,
                        "local_win_rate": entry.local_win_rate,
                        "national_win_rate": entry.national_win_rate,
                        "motor_quinella_rate": entry.motor_quinella_rate,
                        "boat_quinella_rate": entry.boat_quinella_rate,
                        "course": course,
                        "previous_rank": entry.previous_rank,
                    },
                    missing=missing,
                )
            )

        return RaceFeatures(
            race_card_id=card.id,
            venue_id=card.venue_id,
            race_no=card.race_no,
            boats=boats,
            env=env,
            condition_keys=condition_keys,
        )

    def _load_course_stats(
        self,
        venue_id: str,
        condition_keys: list[str],
        as_of_date: date | None = None,
    ) -> dict[int, dict[str, float]]:
        """場×コース成績。as_of_date 指定時はその日より前の確定結果のみ使う（当日リーク防止）。"""
        historical = self._course_stats_from_history(venue_id, as_of_date)
        if historical:
            return historical

        # 履歴が足りない初期段階のみ、集計テーブル（学習済み）を参照
        rows = (
            self.session.query(VenueCourseStats)
            .filter(
                VenueCourseStats.venue_id == venue_id,
                VenueCourseStats.condition_key.in_(condition_keys),
            )
            .all()
        )
        by_course: dict[int, dict[str, float]] = {}
        all_stats: dict[int, dict[str, float]] = {}
        for r in rows:
            payload = {
                "win_rate": r.win_rate,
                "quinella_rate": r.quinella_rate,
                "trio_rate": r.trio_rate,
                "starts": float(r.starts),
            }
            if r.condition_key == "all":
                all_stats[r.course] = payload
            elif r.starts >= 5:
                by_course[r.course] = payload
        for course, payload in all_stats.items():
            by_course.setdefault(course, payload)
        return by_course

    def _course_stats_from_history(
        self, venue_id: str, as_of_date: date | None
    ) -> dict[int, dict[str, float]]:
        from boatrace.db.models import RaceResult

        q = (
            self.session.query(RaceCard, RaceResult)
            .join(RaceResult, RaceResult.race_card_id == RaceCard.id)
            .filter(RaceCard.venue_id == venue_id, RaceCard.status == "finished")
        )
        if as_of_date is not None:
            q = q.filter(RaceCard.race_date < as_of_date)

        starts = {c: 0 for c in range(1, 7)}
        wins = {c: 0 for c in range(1, 7)}
        quinellas = {c: 0 for c in range(1, 7)}
        trios = {c: 0 for c in range(1, 7)}
        n_races = 0
        for _card, result in q.all():
            n_races += 1
            entries = result.entry_results or []
            for er in entries:
                course = int(er.get("course") or er.get("waku") or 0)
                rank = er.get("rank")
                if not course or not rank or course not in starts:
                    continue
                starts[course] += 1
                if rank == 1:
                    wins[course] += 1
                if rank <= 2:
                    quinellas[course] += 1
                if rank <= 3:
                    trios[course] += 1

        if n_races < 3:
            return {}

        out: dict[int, dict[str, float]] = {}
        for c in range(1, 7):
            if starts[c] <= 0:
                continue
            out[c] = {
                "win_rate": wins[c] / starts[c],
                "quinella_rate": quinellas[c] / starts[c],
                "trio_rate": trios[c] / starts[c],
                "starts": float(starts[c]),
            }
        return out

    def _load_biases(self, venue_id: str) -> dict[str, float]:
        rows = self.session.query(VenueBias).filter_by(venue_id=venue_id).all()
        return {r.feature_key: r.coefficient for r in rows}
