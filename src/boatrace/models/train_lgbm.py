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
    build_dataset,
    stack_place_labels,
    stack_samples,
)

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = ROOT_DIR / "data" / "models" / "lgbm_win_v1.joblib"


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
) -> dict[str, Any]:
    win_raw = _predict_pos_proba(win_model, s.X)
    win_sum = float(win_raw.sum()) or 1.0
    win_probs = {s.wakus[i]: float(win_raw[i] / win_sum) for i in range(6)}
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
        # softmax-ish positive
        rs = rs - rs.max()
        rs = np.exp(rs)
        rank_scores = {s.wakus[i]: float(rs[i]) for i in range(6)}
    return build_combination_bundle(
        win_probs, top2_probs, top3_probs, rank_scores=rank_scores
    )


def _race_hit_rate_multi(
    samples: list[RaceSample],
    win_model: lgb.LGBMClassifier,
    top2_model: lgb.LGBMClassifier | None = None,
    top3_model: lgb.LGBMClassifier | None = None,
    ranker: lgb.LGBMRanker | None = None,
) -> dict[str, float]:
    hit1 = hit2 = hit3 = hit_tf = hit_tf3 = 0
    n = 0
    for s in samples:
        bundle = _bundle_from_models(s, win_model, top2_model, top3_model, ranker)
        rankings = bundle["rankings"]
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
        }
    return {
        "n": n,
        "win_rate": hit1 / n,
        "quinella_rate": hit2 / n,
        "trio_rate": hit3 / n,
        "trifecta_rate": hit_tf / n,
        "trifecta_top3_rate": hit_tf3 / n,
    }


def _race_hit_rate(samples: list[RaceSample], model: lgb.LGBMClassifier) -> dict[str, float]:
    return _race_hit_rate_multi(samples, model)


def _train_place_models(
    train_samples: list[RaceSample],
    valid_samples: list[RaceSample],
) -> tuple[
    lgb.LGBMClassifier,
    lgb.LGBMClassifier,
    lgb.LGBMClassifier,
    lgb.LGBMRanker,
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

    importance = {
        FEATURE_COLUMNS[i]: float(v)
        for i, v in enumerate(win_model.feature_importances_)
    }
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    return win_model, top2_model, top3_model, ranker, importance


def train_lgbm(
    session: Session,
    train_start: date,
    train_end: date,
    valid_start: date,
    valid_end: date,
    model_path: Path | None = None,
) -> TrainResult:
    model_path = model_path or DEFAULT_MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("build_train_dataset", start=train_start.isoformat(), end=train_end.isoformat())
    train_samples, train_meta = build_dataset(session, train_start, train_end)
    logger.info("build_valid_dataset", start=valid_start.isoformat(), end=valid_end.isoformat())
    valid_samples, valid_meta = build_dataset(session, valid_start, valid_end)

    win_model, top2_model, top3_model, ranker, importance = _train_place_models(
        train_samples, valid_samples
    )

    train_m = _race_hit_rate_multi(
        train_samples, win_model, top2_model, top3_model, ranker
    )
    valid_m = _race_hit_rate_multi(
        valid_samples, win_model, top2_model, top3_model, ranker
    )

    payload = {
        "model": win_model,
        "models": {
            "win": win_model,
            "top2": top2_model,
            "top3": top3_model,
            "ranker": ranker,
        },
        "feature_columns": FEATURE_COLUMNS,
        "trained_at": date.today().isoformat(),
        "train_meta": train_meta,
        "valid_meta": valid_meta,
        "metrics": {"train": train_m, "valid": valid_m},
        "feature_importance": importance,
        "version": "rank_v3",
    }
    joblib.dump(payload, model_path)
    logger.info("model_saved", path=str(model_path), valid=valid_m)

    return TrainResult(
        model_path=model_path,
        metrics={
            "train": train_m,
            "valid": valid_m,
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


def retrain_all_before_today(
    session: Session,
    days: int | None = None,
    model_path: Path | None = None,
    start: date | None = None,
) -> TrainResult:
    """検証後、本日以外の全期間で再学習して本番用モデルを保存。"""
    today = date.today()
    end = today - timedelta(days=1)
    if start is None:
        if days is None:
            start = date(2026, 1, 1)
        else:
            start = today - timedelta(days=days)
    first = train_lgbm(
        session,
        train_start=start,
        train_end=end - timedelta(days=7),
        valid_start=end - timedelta(days=6),
        valid_end=end,
        model_path=model_path,
    )
    samples, meta = build_dataset(session, start, end)
    cut = int(len(samples) * 0.85)
    train_s, valid_s = samples[:cut], samples[cut:]
    if not valid_s:
        valid_s = train_s[-max(1, len(train_s) // 10) :]

    win_model, top2_model, top3_model, ranker, importance = _train_place_models(
        train_s, valid_s
    )
    path = model_path or DEFAULT_MODEL_PATH
    metrics = {
        "holdout_valid": first.metrics.get("valid"),
        "final_train_race_hit": _race_hit_rate_multi(
            train_s, win_model, top2_model, top3_model, ranker
        ),
        "final_tail_hit": _race_hit_rate_multi(
            valid_s, win_model, top2_model, top3_model, ranker
        ),
        "meta": meta,
    }
    joblib.dump(
        {
            "model": win_model,
            "models": {
                "win": win_model,
                "top2": top2_model,
                "top3": top3_model,
                "ranker": ranker,
            },
            "feature_columns": FEATURE_COLUMNS,
            "trained_at": date.today().isoformat(),
            "metrics": metrics,
            "feature_importance": importance,
            "version": "rank_v3",
        },
        path,
    )
    return TrainResult(model_path=path, metrics=metrics, feature_importance=importance)
