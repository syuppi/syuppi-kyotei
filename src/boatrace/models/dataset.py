"""学習用データセット生成."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
from sqlalchemy.orm import Session, joinedload

from boatrace.db.models import RaceCard
from boatrace.features.builder import DEFAULT_WEIGHTS, FeatureBuilder
from boatrace.logging_setup import get_logger

logger = get_logger(__name__)

BASE_FEATURES = list(DEFAULT_WEIGHTS.keys())

# ML用に追加する特徴（プロ予想士が確認する項目を含む）
EXTRA_FEATURES = [
    "waku",
    "exhibition_time_raw",
    "exhibition_st_raw",
    "avg_st_raw",
    "local_win_raw",
    "national_win_raw",
    "motor_q_raw",
    "motor_trio_raw",
    "boat_q_raw",
    "boat_trio_raw",
    "grade_score",
    "f_count",
    "l_count",
    "tilt_raw",
    "weight_adjustment",
    "parts_changed_flag",
    "win_odds_inv",
    "wind_speed",
    "wave_height",
    "tide_level_norm",
    "near_high_tide",
    "tide_sensitive",
    "is_fixed_entry",
    "day_number",
    "grade_number",
    "same_day_prev_count",
    "same_day_in_win_rate",
    "same_day_nige_rate",
    "same_day_last_winner_course",
    "same_day_wind_delta",
    "ex_rank",
    "ex_st_rank",
    "local_rank",
    "motor_rank",
]

FEATURE_COLUMNS = BASE_FEATURES + EXTRA_FEATURES


@dataclass
class RaceSample:
    race_card_id: int
    venue_id: str
    race_date: date
    race_no: int
    X: np.ndarray  # (6, n_features)
    y: np.ndarray  # (6,) 1=win else 0
    y_rank: np.ndarray  # (6,) 着順 1-6
    wakus: list[int]


def _rank_desc(values: list[float | None]) -> list[float]:
    """大きいほど良い値のレース内順位を 0-1 に（1位=1.0）."""
    indexed = [(i, v if v is not None else -1e9) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: x[1], reverse=True)
    out = [0.5] * len(values)
    n = max(len(values) - 1, 1)
    for rank, (i, _) in enumerate(indexed):
        out[i] = 1.0 - rank / n
    return out


def _rank_asc(values: list[float | None]) -> list[float]:
    """小さいほど良い値のレース内順位."""
    indexed = [(i, v if v is not None else 1e9) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: x[1])
    out = [0.5] * len(values)
    n = max(len(values) - 1, 1)
    for rank, (i, _) in enumerate(indexed):
        out[i] = 1.0 - rank / n
    return out


def _grade_to_score(grade: str | None) -> float:
    from boatrace.features.builder import GRADE_SCORE

    if not grade:
        return 0.45
    return GRADE_SCORE.get(str(grade).upper(), 0.45)


def boat_feature_vector(features, boat_idx: int, ranks: dict[str, list[float]]) -> list[float]:
    boat = features.boats[boat_idx]
    env = features.env
    vec = [float(boat.values.get(k, 0.5)) for k in BASE_FEATURES]
    raw = boat.raw
    win_odds = raw.get("win_odds")
    # 単勝オッズは逆数（人気度）。未取得時は中立
    win_odds_inv = (1.0 / float(win_odds)) if win_odds and float(win_odds) > 0 else 0.15
    extra = {
        "waku": float(boat.waku),
        "exhibition_time_raw": float(raw.get("exhibition_time") or 6.9),
        "exhibition_st_raw": float(raw.get("exhibition_st") if raw.get("exhibition_st") is not None else 0.18),
        "avg_st_raw": float(raw.get("avg_st") or 0.18),
        "local_win_raw": float(raw.get("local_win_rate") or 5.0),
        "national_win_raw": float(raw.get("national_win_rate") or 5.0),
        "motor_q_raw": float(raw.get("motor_quinella_rate") or 30.0),
        "motor_trio_raw": float(raw.get("motor_trio_rate") or 45.0),
        "boat_q_raw": float(raw.get("boat_quinella_rate") or 30.0),
        "boat_trio_raw": float(raw.get("boat_trio_rate") or 45.0),
        "grade_score": _grade_to_score(raw.get("grade_code")),
        "f_count": float(raw.get("f_count") or 0),
        "l_count": float(raw.get("l_count") or 0),
        "tilt_raw": float(raw.get("tilt") if raw.get("tilt") is not None else 0.0),
        "weight_adjustment": float(raw.get("weight_adjustment") or 0.0),
        "parts_changed_flag": 1.0 if raw.get("parts_changed_flag") else 0.0,
        "win_odds_inv": float(win_odds_inv),
        "wind_speed": float(env.get("wind_speed") or 0.0),
        "wave_height": float(env.get("wave_height") or 0.0),
        "tide_level_norm": float(env.get("tide_level_cm") or 100.0) / 200.0,
        "near_high_tide": 1.0 if env.get("near_high_tide") else 0.0,
        "tide_sensitive": 1.0 if env.get("tide_sensitive") else 0.0,
        "is_fixed_entry": 1.0 if env.get("is_fixed_entry") else 0.0,
        "day_number": float(env.get("day_number") or 0),
        "grade_number": float(env.get("grade_number") or 0),
        "same_day_prev_count": float(env.get("same_day_prev_count") or 0),
        "same_day_in_win_rate": float(env.get("same_day_in_win_rate") or 0.55),
        "same_day_nige_rate": float(env.get("same_day_nige_rate") or 0.5),
        "same_day_last_winner_course": float(env.get("same_day_last_winner_course") or 0),
        "same_day_wind_delta": float(env.get("same_day_wind_delta") or 0.0),
        "ex_rank": ranks["ex"][boat_idx],
        "ex_st_rank": ranks["ex_st"][boat_idx],
        "local_rank": ranks["local"][boat_idx],
        "motor_rank": ranks["motor"][boat_idx],
    }
    vec.extend(float(extra[k]) for k in EXTRA_FEATURES)
    return vec


def build_race_sample(session: Session, card: RaceCard) -> RaceSample | None:
    if not card.result or not card.result.rank1_waku:
        return None
    if len(card.entries) < 6:
        return None
    builder = FeatureBuilder(session)
    try:
        feats = builder.build(card.id)
    except Exception as e:  # noqa: BLE001
        logger.debug("feature_build_failed", race_card_id=card.id, error=str(e))
        return None
    if len(feats.boats) != 6:
        return None

    boats = feats.boats
    ranks = {
        "ex": _rank_asc([b.raw.get("exhibition_time") for b in boats]),
        "ex_st": _rank_asc([b.raw.get("exhibition_st") for b in boats]),
        "local": _rank_desc([b.raw.get("local_win_rate") for b in boats]),
        "motor": _rank_desc([b.raw.get("motor_quinella_rate") for b in boats]),
    }
    X = np.asarray(
        [boat_feature_vector(feats, i, ranks) for i in range(6)],
        dtype=np.float64,
    )
    place_by_waku: dict[int, int] = {}
    if card.result.entry_results:
        for er in card.result.entry_results:
            if er.get("waku") and er.get("rank"):
                place_by_waku[int(er["waku"])] = int(er["rank"])
    if not place_by_waku:
        place_by_waku[card.result.rank1_waku] = 1
        if card.result.rank2_waku:
            place_by_waku[card.result.rank2_waku] = 2
        if card.result.rank3_waku:
            place_by_waku[card.result.rank3_waku] = 3

    y = np.zeros(6, dtype=np.int32)
    y_rank = np.full(6, 6, dtype=np.int32)
    wakus = [b.waku for b in boats]
    for i, w in enumerate(wakus):
        if w == card.result.rank1_waku:
            y[i] = 1
        y_rank[i] = place_by_waku.get(w, 6)

    return RaceSample(
        race_card_id=card.id,
        venue_id=card.venue_id,
        race_date=card.race_date,
        race_no=card.race_no,
        X=X,
        y=y,
        y_rank=y_rank,
        wakus=wakus,
    )


def iter_finished_cards(
    session: Session,
    start: date,
    end: date,
) -> list[RaceCard]:
    return (
        session.query(RaceCard)
        .options(
            joinedload(RaceCard.entries),
            joinedload(RaceCard.weather),
            joinedload(RaceCard.tide),
            joinedload(RaceCard.result),
            joinedload(RaceCard.venue),
        )
        .filter(
            RaceCard.race_date >= start,
            RaceCard.race_date <= end,
            RaceCard.status == "finished",
        )
        .order_by(RaceCard.race_date, RaceCard.venue_id, RaceCard.race_no)
        .all()
    )


def build_dataset(
    session: Session,
    start: date,
    end: date,
) -> tuple[list[RaceSample], dict[str, Any]]:
    cards = iter_finished_cards(session, start, end)
    samples: list[RaceSample] = []
    for i, card in enumerate(cards):
        sample = build_race_sample(session, card)
        if sample is not None:
            samples.append(sample)
        if (i + 1) % 500 == 0:
            logger.info("dataset_progress", done=i + 1, total=len(cards), kept=len(samples))
    meta = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "cards": len(cards),
        "samples": len(samples),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
    }
    return samples, meta


def stack_samples(samples: list[RaceSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """艇単位に flatten. 戻り値: X, y_win, race_ids."""
    Xs = []
    ys = []
    race_ids = []
    for s in samples:
        Xs.append(s.X)
        ys.append(s.y)
        race_ids.append(np.full(6, s.race_card_id, dtype=np.int64))
    return (
        np.vstack(Xs),
        np.concatenate(ys),
        np.concatenate(race_ids),
    )
