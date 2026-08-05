"""LightGBM 学習・評価（1着/連対 + 着順ランカー + 3連単カバー評価）."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
from sqlalchemy.orm import Session

from boatrace.config import ROOT_DIR
from boatrace.logging_setup import get_logger
from boatrace.models.combinations import build_combination_bundle
from boatrace.models.dataset import (
    FEATURE_COLUMNS,
    RaceSample,
    stack_place_labels,
    stack_samples,
)
from boatrace.models.dataset_cache import build_dataset_cached

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = ROOT_DIR / "data" / "models" / "lgbm_win_v1.joblib"
SMOKE_MODEL_PATH = ROOT_DIR / "data" / "models" / "lgbm_win_smoke.joblib"


@dataclass
class TrainResult:
    model_path: Path
    metrics: dict[str, Any]
    feature_importance: dict[str, float]


def _fit_binary(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    n_estimators: int = 400,
) -> lgb.LGBMClassifier:
    pos = max(float(y_tr.sum()), 1.0)
    neg = max(float(len(y_tr) - y_tr.sum()), 1.0)
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=40,
        reg_alpha=0.1,
        reg_lambda=0.3,
        scale_pos_weight=float(neg / pos),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(40, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def _fit_ranker(
    train_samples: list[RaceSample],
    valid_samples: list[RaceSample],
) -> lgb.LGBMRanker:
    """着順ランカー: 関連度は 1着=3, 2着=2, 3着=1, それ以外=0."""

    def pack(samples: list[RaceSample]):
        Xs = []
        ys = []
        groups = []
        for s in samples:
            Xs.append(s.X)
            # y_rank: 1..6
            rel = np.clip(4 - s.y_rank, 0, 3).astype(np.int32)
            ys.append(rel)
            groups.append(6)
        return np.vstack(Xs), np.concatenate(ys), groups

    X_tr, y_tr, g_tr = pack(train_samples)
    X_va, y_va, g_va = pack(valid_samples)
    model = lgb.LGBMRanker(
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=40,
        reg_alpha=0.1,
        reg_lambda=0.3,
        random_state=42,
        n_jobs=-1,
        objective="lambdarank",
        metric="ndcg",
        importance_type="gain",
    )
    model.fit(
        X_tr,
        y_tr,
        group=g_tr,
        eval_set=[(X_va, y_va)],
        eval_group=[g_va],
        eval_at=[3],
        callbacks=[
            lgb.early_stopping(40, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def _predict_pos_proba(model: lgb.LGBMClassifier, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    return proba.reshape(-1)


def _bundle_from_models(
    s: RaceSample,
    win_model: lgb.LGBMClassifier,
    top2_model: lgb.LGBMClassifier | None,
    top3_model: lgb.LGBMClassifier | None,
    ranker: lgb.LGBMRanker | None,
    fly_model: lgb.LGBMClassifier | None = None,
) -> dict[str, Any]:
    from boatrace.models.course_prior import apply_course_log_prior, select_favorite_probs

    win_raw = _predict_pos_proba(win_model, s.X)
    win_sum = float(win_raw.sum()) or 1.0
    strength = {s.wakus[i]: float(win_raw[i] / win_sum) for i in range(6)}
    fly_risk = _resolve_fly_risk(s, fly_model)
    # 3連系はゲートなしの広い分布
    combo_win = apply_course_log_prior(
        strength,
        beta=0.15,
        fly_risk=fly_risk,
        venue_in_win=getattr(s, "venue_in_win", None),
    )
    top2_probs = None
    top3_probs = None
    rank_scores = None
    if top2_model is not None:
        r2 = _predict_pos_proba(top2_model, s.X)
        top2_probs = {s.wakus[i]: float(r2[i]) for i in range(6)}
    if top3_model is not None:
        r3 = _predict_pos_proba(top3_model, s.X)
        top3_probs = {s.wakus[i]: float(r3[i]) for i in range(6)}
    if ranker is not None:
        rs = np.asarray(ranker.predict(s.X), dtype=float)
        rs = rs - rs.max()
        rs = np.exp(rs)
        rank_scores = {s.wakus[i]: float(rs[i]) for i in range(6)}
    return build_combination_bundle(
        combo_win, top2_probs, top3_probs, rank_scores=rank_scores
    )


def _resolve_fly_risk(
    s: RaceSample, fly_model: lgb.LGBMClassifier | None
) -> float:
    heuristic = float(getattr(s, "fly_risk", 0.0) or 0.0)
    if fly_model is None:
        return heuristic
    try:
        idx = s.wakus.index(1)
        ml_fly = float(_predict_pos_proba(fly_model, s.X[idx : idx + 1])[0])
        return 0.65 * ml_fly + 0.35 * heuristic
    except Exception:
        return heuristic


def _race_hit_rate_multi(
    samples: list[RaceSample],
    win_model: lgb.LGBMClassifier,
    top2_model: lgb.LGBMClassifier | None = None,
    top3_model: lgb.LGBMClassifier | None = None,
    ranker: lgb.LGBMRanker | None = None,
    fly_model: lgb.LGBMClassifier | None = None,
) -> dict[str, float]:
    hit1 = hit2 = hit3 = hit_tf = hit_tf3 = 0
    fav_top3 = always1 = fav1 = fly_pred = fly_hit = 0
    n = 0
    from boatrace.models.course_prior import select_favorite_probs

    for s in samples:
        bundle = _bundle_from_models(
            s, win_model, top2_model, top3_model, ranker, fly_model=fly_model
        )
        win_raw = _predict_pos_proba(win_model, s.X)
        win_sum = float(win_raw.sum()) or 1.0
        strength = {s.wakus[i]: float(win_raw[i] / win_sum) for i in range(6)}
        fly_risk = _resolve_fly_risk(s, fly_model)
        win_probs = select_favorite_probs(
            strength,
            fly_risk=fly_risk,
            venue_in_win=getattr(s, "venue_in_win", None),
        )
        rankings = sorted(s.wakus, key=lambda w: win_probs[w], reverse=True)
        winner = s.wakus[int(np.argmax(s.y))]
        by_rank = sorted(range(6), key=lambda i: s.y_rank[i])
        true_top2 = {s.wakus[i] for i in by_rank[:2]}
        true_top3 = {s.wakus[i] for i in by_rank[:3]}
        true_order = [s.wakus[i] for i in by_rank[:3]]

        tickets = bundle.get("tickets") or {}
        tf_cands = [t["combo"] for t in tickets.get("sanrentan", [])]
        sp_cands = [set(t["combo"]) for t in tickets.get("sanrenpuku", [])]

        n += 1
        hit1 += int(rankings[0] == winner)
        fav_top3 += int(rankings[0] in true_top3)
        always1 += int(winner == 1)
        fav1 += int(rankings[0] == 1)
        pred_fly = fly_risk >= 0.50
        actual_fly = winner != 1
        if pred_fly:
            fly_pred += 1
            fly_hit += int(actual_fly)
        hit2 += int(true_top2 == set(bundle["candidates_quinella"][:2]))
        if sp_cands:
            hit3 += int(true_top3 in sp_cands)
        else:
            hit3 += int(true_top3 == set(bundle["candidates_trio"][:3]))
        hit_tf += int(bool(tf_cands) and tf_cands[0] == true_order)
        hit_tf3 += int(any(c == true_order for c in tf_cands))

    if n == 0:
        return {
            "n": 0,
            "win_rate": 0.0,
            "quinella_rate": 0.0,
            "trio_rate": 0.0,
            "trifecta_rate": 0.0,
            "trifecta_top3_rate": 0.0,
            "favorite_in_top3": 0.0,
            "fav1_rate": 0.0,
            "always1_baseline": 0.0,
            "course1_fly_precision": 0.0,
        }
    return {
        "n": n,
        "win_rate": hit1 / n,
        "quinella_rate": hit2 / n,
        "trio_rate": hit3 / n,
        "trifecta_rate": hit_tf / n,
        "trifecta_top3_rate": hit_tf3 / n,
        "favorite_in_top3": fav_top3 / n,
        "fav1_rate": fav1 / n,
        "always1_baseline": always1 / n,
        "course1_fly_precision": (fly_hit / fly_pred) if fly_pred else 0.0,
        "course1_fly_pred_rate": fly_pred / n,
    }


def _race_hit_rate(samples: list[RaceSample], model: lgb.LGBMClassifier) -> dict[str, float]:
    return _race_hit_rate_multi(samples, model)


def _stack_fly_dataset(
    samples: list[RaceSample],
) -> tuple[np.ndarray, np.ndarray]:
    """レース単位: 1号艇特徴 → ラベル=1号艇が飛んだ(1着≠1)."""
    Xs: list[np.ndarray] = []
    ys: list[int] = []
    for s in samples:
        try:
            idx = s.wakus.index(1)
        except ValueError:
            continue
        Xs.append(s.X[idx])
        winner = s.wakus[int(np.argmax(s.y))]
        ys.append(1 if winner != 1 else 0)
    if not Xs:
        return np.zeros((0, len(FEATURE_COLUMNS))), np.zeros((0,), dtype=np.int32)
    return np.vstack(Xs), np.asarray(ys, dtype=np.int32)


def _train_place_models(
    train_samples: list[RaceSample],
    valid_samples: list[RaceSample],
) -> tuple[
    lgb.LGBMClassifier,
    lgb.LGBMClassifier,
    lgb.LGBMClassifier,
    lgb.LGBMRanker,
    lgb.LGBMClassifier | None,
    dict[str, float],
]:
    X_train, y_win_tr, _ = stack_samples(train_samples)
    y_top2_tr, y_top3_tr = stack_place_labels(train_samples)
    X_valid, y_win_va, _ = stack_samples(valid_samples)
    y_top2_va, y_top3_va = stack_place_labels(valid_samples)

    win_model = _fit_binary(X_train, y_win_tr, X_valid, y_win_va)
    top2_model = _fit_binary(X_train, y_top2_tr, X_valid, y_top2_va)
    top3_model = _fit_binary(X_train, y_top3_tr, X_valid, y_top3_va)
    ranker = _fit_ranker(train_samples, valid_samples)

    X_fly_tr, y_fly_tr = _stack_fly_dataset(train_samples)
    X_fly_va, y_fly_va = _stack_fly_dataset(valid_samples)
    fly_model = None
    if len(y_fly_tr) >= 200 and y_fly_tr.sum() >= 20:
        fly_model = _fit_binary(X_fly_tr, y_fly_tr, X_fly_va, y_fly_va, n_estimators=300)

    importance = {
        FEATURE_COLUMNS[i]: float(v)
        for i, v in enumerate(win_model.feature_importances_)
    }
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    return win_model, top2_model, top3_model, ranker, fly_model, importance


def train_lgbm(
    session: Session,
    train_start: date,
    train_end: date,
    valid_start: date,
    valid_end: date,
    model_path: Path | None = None,
    *,
    force_rebuild_cache: bool = False,
    samples: list[RaceSample] | None = None,
) -> TrainResult:
    """期間指定で学習。samples を渡すとデータセット再構築をスキップ."""
    model_path = model_path or DEFAULT_MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if samples is None:
        # train+valid を一度に作って分割（二重ビルド回避）
        full_start = min(train_start, valid_start)
        full_end = max(train_end, valid_end)
        logger.info(
            "build_dataset_cached",
            start=full_start.isoformat(),
            end=full_end.isoformat(),
        )
        all_samples, all_meta = build_dataset_cached(
            session, full_start, full_end, force_rebuild=force_rebuild_cache
        )
    else:
        all_samples = samples
        all_meta = {"samples": len(samples), "n_features": len(FEATURE_COLUMNS)}

    train_samples = [s for s in all_samples if train_start <= s.race_date <= train_end]
    valid_samples = [s for s in all_samples if valid_start <= s.race_date <= valid_end]
    if not train_samples:
        raise ValueError(f"no train samples in {train_start}..{train_end}")
    if not valid_samples:
        # 末尾10%を検証に回す
        cut = max(1, int(len(train_samples) * 0.9))
        valid_samples = train_samples[cut:]
        train_samples = train_samples[:cut]

    train_meta = {
        "start": train_start.isoformat(),
        "end": train_end.isoformat(),
        "samples": len(train_samples),
        "source_meta": all_meta,
    }
    valid_meta = {
        "start": valid_start.isoformat(),
        "end": valid_end.isoformat(),
        "samples": len(valid_samples),
    }

    win_model, top2_model, top3_model, ranker, fly_model, importance = _train_place_models(
        train_samples, valid_samples
    )

    # 学習側メトリクスは最大3000件に間引き（評価コスト削減）
    train_eval = train_samples[-3000:] if len(train_samples) > 3000 else train_samples
    train_m = _race_hit_rate_multi(
        train_eval, win_model, top2_model, top3_model, ranker, fly_model=fly_model
    )
    valid_m = _race_hit_rate_multi(
        valid_samples, win_model, top2_model, top3_model, ranker, fly_model=fly_model
    )
    valid_no_ranker = _race_hit_rate_multi(
        valid_samples, win_model, top2_model, top3_model, None, fly_model=fly_model
    )

    payload = {
        "model": win_model,
        "models": {
            "win": win_model,
            "top2": top2_model,
            "top3": top3_model,
            "ranker": ranker,
            "course1_fly": fly_model,
        },
        "feature_columns": FEATURE_COLUMNS,
        "trained_at": date.today().isoformat(),
        "train_meta": train_meta,
        "valid_meta": valid_meta,
        "metrics": {
            "train": train_m,
            "valid": valid_m,
            "valid_no_ranker": valid_no_ranker,
        },
        "feature_importance": importance,
        "version": "hitrate_v3",
    }
    joblib.dump(payload, model_path)
    logger.info("model_saved", path=str(model_path), valid=valid_m)

    return TrainResult(
        model_path=model_path,
        metrics={
            "train": train_m,
            "valid": valid_m,
            "valid_no_ranker": valid_no_ranker,
            "train_meta": train_meta,
            "valid_meta": valid_meta,
        },
        feature_importance=importance,
    )


def train_default_split(session: Session, days: int = 90, model_path: Path | None = None) -> TrainResult:
    today = date.today()
    valid_end = today - timedelta(days=1)
    valid_start = today - timedelta(days=7)
    train_end = valid_start - timedelta(days=1)
    train_start = today - timedelta(days=days)
    return train_lgbm(
        session,
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        model_path=model_path,
    )


def smoke_retrain(
    session: Session,
    *,
    days: int = 45,
    model_path: Path | None = None,
    force_rebuild_cache: bool = False,
) -> TrainResult:
    """フル学習の前に回す軽量検証（直近N日・本番モデルは上書きしない）."""
    today = date.today()
    end = today - timedelta(days=1)
    start = end - timedelta(days=max(14, days) - 1)
    path = model_path or SMOKE_MODEL_PATH
    logger.info("smoke_retrain_start", start=start.isoformat(), end=end.isoformat())
    return train_lgbm(
        session,
        train_start=start,
        train_end=end - timedelta(days=7),
        valid_start=end - timedelta(days=6),
        valid_end=end,
        model_path=path,
        force_rebuild_cache=force_rebuild_cache,
    )


def retrain_all_before_today(
    session: Session,
    days: int | None = None,
    model_path: Path | None = None,
    start: date | None = None,
    *,
    force_rebuild_cache: bool = False,
    skip_holdout: bool = False,
) -> TrainResult:
    """検証後、本日以外の全期間で再学習して本番用モデルを保存。

    データセットは1回だけ構築し、holdout検証と最終学習で再利用する。
    """
    today = date.today()
    end = today - timedelta(days=1)
    if start is None:
        if days is None:
            start = date(2026, 1, 1)
        else:
            start = today - timedelta(days=days)

    logger.info("build_full_dataset_once", start=start.isoformat(), end=end.isoformat())
    samples, meta = build_dataset_cached(
        session, start, end, force_rebuild=force_rebuild_cache
    )
    if not samples:
        raise ValueError(f"no samples in {start}..{end}")

    holdout_start = end - timedelta(days=6)
    holdout_train = [s for s in samples if s.race_date < holdout_start]
    holdout_valid = [s for s in samples if s.race_date >= holdout_start]
    if not holdout_train or not holdout_valid:
        cut = int(len(samples) * 0.85)
        holdout_train, holdout_valid = samples[:cut], samples[cut:]

    first_metrics: dict[str, Any] = {}
    if not skip_holdout:
        first = train_lgbm(
            session,
            train_start=start,
            train_end=holdout_start - timedelta(days=1),
            valid_start=holdout_start,
            valid_end=end,
            model_path=model_path,
            samples=samples,  # 再ビルドしない
        )
        first_metrics = {
            "holdout_valid": first.metrics.get("valid"),
            "holdout_valid_no_ranker": first.metrics.get("valid_no_ranker"),
        }

    cut = int(len(samples) * 0.85)
    train_s, valid_s = samples[:cut], samples[cut:]
    if not valid_s:
        valid_s = train_s[-max(1, len(train_s) // 10) :]

    win_model, top2_model, top3_model, ranker, fly_model, importance = _train_place_models(
        train_s, valid_s
    )
    path = model_path or DEFAULT_MODEL_PATH
    metrics = {
        **first_metrics,
        "final_train_race_hit": _race_hit_rate_multi(
            train_s[-3000:] if len(train_s) > 3000 else train_s,
            win_model,
            top2_model,
            top3_model,
            ranker,
            fly_model=fly_model,
        ),
        "final_tail_hit": _race_hit_rate_multi(
            valid_s, win_model, top2_model, top3_model, ranker, fly_model=fly_model
        ),
        "meta": meta,
        "n_samples": len(samples),
    }
    joblib.dump(
        {
            "model": win_model,
            "models": {
                "win": win_model,
                "top2": top2_model,
                "top3": top3_model,
                "ranker": ranker,
                "course1_fly": fly_model,
            },
            "feature_columns": FEATURE_COLUMNS,
            "trained_at": date.today().isoformat(),
            "metrics": metrics,
            "feature_importance": importance,
            "version": "hitrate_v3",
        },
        path,
    )
    return TrainResult(model_path=path, metrics=metrics, feature_importance=importance)
