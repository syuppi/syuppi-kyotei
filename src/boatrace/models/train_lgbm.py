"""LightGBM 学習・評価."""

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
from boatrace.models.dataset import (
    FEATURE_COLUMNS,
    RaceSample,
    build_dataset,
    stack_samples,
)

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = ROOT_DIR / "data" / "models" / "lgbm_win_v1.joblib"


@dataclass
class TrainResult:
    model_path: Path
    metrics: dict[str, Any]
    feature_importance: dict[str, float]


def _race_hit_rate(samples: list[RaceSample], model: lgb.LGBMClassifier) -> dict[str, float]:
    hit1 = hit2 = hit3 = 0
    n = 0
    for s in samples:
        proba = model.predict_proba(s.X)[:, 1]
        order = [s.wakus[i] for i in np.argsort(-proba)]
        winner = s.wakus[int(np.argmax(s.y))]
        # true top2/top3 from y_rank
        by_rank = sorted(range(6), key=lambda i: s.y_rank[i])
        true_top2 = {s.wakus[i] for i in by_rank[:2]}
        true_top3 = {s.wakus[i] for i in by_rank[:3]}
        n += 1
        hit1 += int(order[0] == winner)
        hit2 += int(true_top2 <= set(order[:2]))
        hit3 += int(true_top3 <= set(order[:3]))
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "quinella_rate": 0.0, "trio_rate": 0.0}
    return {
        "n": n,
        "win_rate": hit1 / n,
        "quinella_rate": hit2 / n,
        "trio_rate": hit3 / n,
    }


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

    X_train, y_train, _ = stack_samples(train_samples)
    X_valid, y_valid, _ = stack_samples(valid_samples)

    # クラス不均衡: 1着は1/6
    pos = max(y_train.sum(), 1)
    neg = max(len(y_train) - pos, 1)
    spw = float(neg / pos)

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=40,
        reg_alpha=0.1,
        reg_lambda=0.3,
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(40, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    train_m = _race_hit_rate(train_samples, model)
    valid_m = _race_hit_rate(valid_samples, model)
    importance = {
        FEATURE_COLUMNS[i]: float(v)
        for i, v in enumerate(model.feature_importances_)
    }
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    payload = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "trained_at": date.today().isoformat(),
        "train_meta": train_meta,
        "valid_meta": valid_meta,
        "metrics": {"train": train_m, "valid": valid_m},
        "feature_importance": importance,
    }
    joblib.dump(payload, model_path)
    logger.info("model_saved", path=str(model_path), valid=valid_m)

    return TrainResult(
        model_path=model_path,
        metrics={"train": train_m, "valid": valid_m, "train_meta": train_meta, "valid_meta": valid_meta},
        feature_importance=importance,
    )


def train_default_split(session: Session, days: int = 90, model_path: Path | None = None) -> TrainResult:
    """直近days日のうち、最後の7日を検証、それ以前を学習。本日は含めない。"""
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


def retrain_all_before_today(session: Session, days: int = 90, model_path: Path | None = None) -> TrainResult:
    """検証後、本日以外の全期間で再学習して本番用モデルを保存。"""
    today = date.today()
    end = today - timedelta(days=1)
    start = today - timedelta(days=days)
    # 最終7日を軽く内部検証しつつ全期間fit
    # ここでは train=start..end-7, valid=end-6..end で一度評価後、start..end で再fit
    first = train_lgbm(
        session,
        train_start=start,
        train_end=end - timedelta(days=7),
        valid_start=end - timedelta(days=6),
        valid_end=end,
        model_path=model_path,
    )
    # 本番用: 本日以外全部
    samples, meta = build_dataset(session, start, end)
    X, y, _ = stack_samples(samples)
    pos = max(y.sum(), 1)
    neg = max(len(y) - pos, 1)
    model = lgb.LGBMClassifier(
        n_estimators=max(100, int(getattr(first, "metrics", {}).get("n_estimators", 300) or 300)),
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
    # early stopping用に末尾20%をvalidに
    cut = int(len(samples) * 0.85)
    train_s, valid_s = samples[:cut], samples[cut:]
    X_tr, y_tr, _ = stack_samples(train_s)
    X_va, y_va, _ = stack_samples(valid_s) if valid_s else (X_tr[:6], y_tr[:6])
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    path = model_path or DEFAULT_MODEL_PATH
    importance = {
        FEATURE_COLUMNS[i]: float(v) for i, v in enumerate(model.feature_importances_)
    }
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    metrics = {
        "holdout_valid": first.metrics.get("valid"),
        "final_train_race_hit": _race_hit_rate(train_s, model),
        "final_tail_hit": _race_hit_rate(valid_s, model) if valid_s else {},
        "meta": meta,
    }
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "trained_at": date.today().isoformat(),
            "metrics": metrics,
            "feature_importance": importance,
        },
        path,
    )
    return TrainResult(model_path=path, metrics=metrics, feature_importance=importance)
