#!/usr/bin/env python3
"""的中しやすいレースの信頼度メタモデルを学習する."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.config import get_settings
from boatrace.db.models import RaceCard
from boatrace.db.session import session_scope
from boatrace.features.builder import FeatureBuilder
from boatrace.logging_setup import setup_logging, get_logger
from boatrace.models.ml_model import MLPredictor
from boatrace.prediction.confidence import (
    CONFIDENCE_FEATURES,
    DEFAULT_CONFIDENCE_PATH,
    features_to_vector,
    extract_confidence_features,
    get_confidence_model,
    train_confidence_model,
)

logger = get_logger(__name__)


def _trio_hit(pred, card: RaceCard) -> bool | None:
    r = card.result
    if not r or not r.rank1_waku or not r.rank2_waku or not r.rank3_waku:
        return None
    true3 = {r.rank1_waku, r.rank2_waku, r.rank3_waku}
    tickets = pred.tickets or (pred.feature_snapshot or {}).get("tickets") or {}
    sps = [set(t["combo"]) for t in tickets.get("sanrenpuku", []) if t.get("combo")]
    if not sps:
        return None
    return any(s == true3 for s in sps)


def _tf_hit(pred, card: RaceCard) -> bool | None:
    r = card.result
    if not r or not r.rank1_waku or not r.rank2_waku or not r.rank3_waku:
        return None
    true_ord = [r.rank1_waku, r.rank2_waku, r.rank3_waku]
    tickets = pred.tickets or (pred.feature_snapshot or {}).get("tickets") or {}
    sts = [list(t["combo"]) for t in tickets.get("sanrentan", []) if t.get("combo")]
    if not sts:
        return None
    return any(s == true_ord for s in sts)


def collect_rows(
    session,
    start: date,
    end: date,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """start..end の確定レースから (X, y_trio, y_tf, meta) を収集.

    サービス層（叙事・オッズ）は省略し、ML本体の買い目でラベル付けする。
    """
    builder = FeatureBuilder(session)
    predictor = MLPredictor(session=session)
    Xs: list[np.ndarray] = []
    ys: list[int] = []
    ytf: list[int] = []
    meta: list[dict] = []
    d = start
    while d <= end:
        cards = (
            session.query(RaceCard)
            .filter(RaceCard.race_date == d, RaceCard.status == "finished")
            .all()
        )
        day_n = 0
        for card in cards:
            if not card.result or len(card.entries or []) < 6:
                continue
            try:
                features = builder.build(card.id)
                pred = predictor.predict(features)
            except Exception as e:  # noqa: BLE001
                logger.warning("predict_failed", race_card_id=card.id, error=str(e))
                continue
            hit = _trio_hit(pred, card)
            if hit is None:
                continue
            feat = extract_confidence_features(features, pred)
            Xs.append(features_to_vector(feat))
            ys.append(int(hit))
            # 自信あり時と同じ3連単集中を当てた場合の的中で閾値を合わせる
            from boatrace.prediction.confidence import focus_trifecta_on_top_trios

            focused_tickets = focus_trifecta_on_top_trios(
                pred.tickets or {}, limit=5, primary_perms=3
            )
            pred.tickets = focused_tickets
            tf = _tf_hit(pred, card)
            ytf.append(int(tf) if tf is not None else 0)
            meta.append(
                {
                    "race_card_id": card.id,
                    "date": d.isoformat(),
                    "venue_id": card.venue_id,
                    "race_no": card.race_no,
                }
            )
            day_n += 1
        logger.info("confidence_collect_day", date=d.isoformat(), n=day_n, total=len(ys))
        print(f"collect {d.isoformat()} +{day_n} total={len(ys)}", flush=True)
        d += timedelta(days=1)
    if not Xs:
        return (
            np.zeros((0, len(CONFIDENCE_FEATURES))),
            np.zeros(0),
            np.zeros(0),
            [],
        )
    return np.vstack(Xs), np.asarray(ys, dtype=int), np.asarray(ytf, dtype=int), meta


def evaluate_selection(
    scores: np.ndarray,
    y_trio: np.ndarray,
    y_tf: np.ndarray,
    threshold: float,
) -> dict:
    mask = scores >= threshold
    n = int(len(scores))
    n_sel = int(mask.sum())
    return {
        "n_all": n,
        "trio_all": float(y_trio.mean()) if n else 0.0,
        "tf_all": float(y_tf.mean()) if n else 0.0,
        "n_confident": n_sel,
        "coverage": n_sel / n if n else 0.0,
        "trio_confident": float(y_trio[mask].mean()) if n_sel else 0.0,
        "tf_confident": float(y_tf[mask].mean()) if n_sel else 0.0,
        "threshold": float(threshold),
    }


def main() -> None:
    setup_logging()
    get_settings.cache_clear()
    today = date.today()
    # holdout: 直近7日 / train: その前 60 日（収集コストと精度のバランス）
    valid_end = today - timedelta(days=1)
    valid_start = today - timedelta(days=7)
    train_end = valid_start - timedelta(days=1)
    train_start = train_end - timedelta(days=59)

    print(
        json.dumps(
            {
                "train": [train_start.isoformat(), train_end.isoformat()],
                "valid": [valid_start.isoformat(), valid_end.isoformat()],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with session_scope() as session:
        X_tr, y_tr, ytf_tr, _ = collect_rows(session, train_start, train_end)
        X_va, y_va, ytf_va, _ = collect_rows(session, valid_start, valid_end)

    print(
        {
            "train_n": int(len(y_tr)),
            "train_trio_rate": float(y_tr.mean()) if len(y_tr) else None,
            "valid_n": int(len(y_va)),
            "valid_trio_rate": float(y_va.mean()) if len(y_va) else None,
        },
        flush=True,
    )
    if len(y_tr) < 200:
        raise SystemExit(f"not enough train samples: {len(y_tr)}")

    target_cov = float(
        getattr(get_settings().prediction, "confidence_target_coverage", 0.30) or 0.30
    )
    rc = train_confidence_model(
        X_tr,
        y_tr,
        X_valid=X_va if len(y_va) else None,
        y_valid=y_va if len(y_va) else None,
        y_tf_valid=ytf_va if len(ytf_va) else None,
        target_coverage=target_cov,
        target_tf=0.35,
    )
    path = rc.save(DEFAULT_CONFIDENCE_PATH)

    if len(y_va):
        scores = rc.model.predict_proba(X_va)[:, 1]
        report = evaluate_selection(scores, y_va, ytf_va, rc.threshold)
        order = np.argsort(-scores)
        bands = {}
        for cov in (0.20, 0.30, 0.40):
            k = max(10, int(round(len(scores) * cov)))
            idx = order[:k]
            bands[f"top_{int(cov * 100)}pct"] = {
                "n": int(k),
                "trio": float(y_va[idx].mean()),
                "tf": float(ytf_va[idx].mean()),
            }
        rc.metrics["holdout_report"] = report
        rc.metrics["holdout_bands"] = bands
        rc.save(path)
        get_confidence_model(reload=True)
        print(
            json.dumps(
                {
                    "path": str(path),
                    "metrics": rc.metrics,
                    "holdout": report,
                    "bands": bands,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    else:
        print(
            json.dumps(
                {"path": str(path), "metrics": rc.metrics},
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
