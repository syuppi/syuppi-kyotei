"""学習用データセットのディスクキャッシュ.

特徴量生成が再学習の大半を占めるため、FEATURE_COLUMNS と期間・件数で
キーを切り、再利用／末尾差分だけ追記する。
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy import func
from sqlalchemy.orm import Session

from boatrace.config import ROOT_DIR
from boatrace.db.models import RaceCard, RaceResult
from boatrace.logging_setup import get_logger
from boatrace.models.dataset import FEATURE_COLUMNS, RaceSample, build_dataset

logger = get_logger(__name__)

CACHE_DIR = ROOT_DIR / "data" / "cache" / "datasets"
CACHE_VERSION = "v1"


def _feature_fingerprint() -> str:
    raw = "|".join(FEATURE_COLUMNS)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _range_stats(session: Session, start: date, end: date) -> dict[str, Any]:
    cards = (
        session.query(func.count(RaceCard.id))
        .filter(RaceCard.race_date >= start, RaceCard.race_date <= end)
        .scalar()
    ) or 0
    with_res = (
        session.query(func.count(RaceCard.id))
        .join(RaceResult, RaceResult.race_card_id == RaceCard.id)
        .filter(
            RaceCard.race_date >= start,
            RaceCard.race_date <= end,
            RaceResult.rank1_waku.isnot(None),
        )
        .scalar()
    ) or 0
    max_rid = (
        session.query(func.max(RaceResult.id))
        .join(RaceCard, RaceCard.id == RaceResult.race_card_id)
        .filter(RaceCard.race_date >= start, RaceCard.race_date <= end)
        .scalar()
    ) or 0
    return {"cards": int(cards), "with_result": int(with_res), "max_result_id": int(max_rid)}


def cache_key(start: date, end: date, stats: dict[str, Any]) -> str:
    payload = {
        "v": CACHE_VERSION,
        "feat": _feature_fingerprint(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "cards": stats["cards"],
        "with_result": stats["with_result"],
        "max_result_id": stats["max_result_id"],
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return f"{start.isoformat()}_{end.isoformat()}_{digest}"


def _cache_paths(key: str) -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.joblib", CACHE_DIR / f"{key}.meta.json"


def save_dataset_cache(
    key: str,
    samples: list[RaceSample],
    meta: dict[str, Any],
) -> Path:
    data_path, meta_path = _cache_paths(key)
    joblib.dump({"samples": samples, "meta": meta}, data_path, compress=3)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("dataset_cache_saved", path=str(data_path), samples=len(samples))
    return data_path


def load_dataset_cache(key: str) -> tuple[list[RaceSample], dict[str, Any]] | None:
    data_path, _meta_path = _cache_paths(key)
    if not data_path.exists():
        return None
    try:
        payload = joblib.load(data_path)
        samples = list(payload.get("samples") or [])
        meta = dict(payload.get("meta") or {})
        if not samples:
            return None
        # 特徴量が変わっていたら無効
        if meta.get("feature_fingerprint") != _feature_fingerprint():
            logger.info("dataset_cache_stale_features", path=str(data_path))
            return None
        logger.info("dataset_cache_hit", path=str(data_path), samples=len(samples))
        return samples, meta
    except Exception as e:  # noqa: BLE001
        logger.warning("dataset_cache_load_failed", path=str(data_path), error=str(e))
        return None


def _find_prefix_cache(
    session: Session, start: date, end: date
) -> tuple[list[RaceSample], dict[str, Any], date] | None:
    """同一 start・同一特徴量で end が手前のキャッシュがあれば差分追記用に返す."""
    if not CACHE_DIR.exists():
        return None
    feat = _feature_fingerprint()
    best: tuple[list[RaceSample], dict[str, Any], date] | None = None
    for meta_path in CACHE_DIR.glob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if meta.get("feature_fingerprint") != feat:
            continue
        if meta.get("start") != start.isoformat():
            continue
        try:
            cached_end = date.fromisoformat(str(meta["end"]))
        except Exception:  # noqa: BLE001
            continue
        if cached_end < start or cached_end >= end:
            continue
        # キャッシュ時点の stats がまだ有効か（その期間の結果件数が増えていないか）
        stats = _range_stats(session, start, cached_end)
        if (
            stats["with_result"] != int(meta.get("with_result") or -1)
            or stats["max_result_id"] != int(meta.get("max_result_id") or -1)
        ):
            continue
        data_path = CACHE_DIR / meta_path.name.replace(".meta.json", ".joblib")
        if not data_path.exists():
            continue
        try:
            payload = joblib.load(data_path)
            samples = list(payload.get("samples") or [])
        except Exception:  # noqa: BLE001
            continue
        if not samples:
            continue
        if best is None or cached_end > best[2]:
            best = (samples, meta, cached_end)
    return best


def build_dataset_cached(
    session: Session,
    start: date,
    end: date,
    *,
    force_rebuild: bool = False,
) -> tuple[list[RaceSample], dict[str, Any]]:
    """キャッシュ優先でデータセットを構築。末尾差分のみ再計算する."""
    stats = _range_stats(session, start, end)
    key = cache_key(start, end, stats)
    if not force_rebuild:
        hit = load_dataset_cache(key)
        if hit is not None:
            return hit

        prefix = _find_prefix_cache(session, start, end)
        if prefix is not None:
            base_samples, base_meta, cached_end = prefix
            incr_start = cached_end + timedelta(days=1)
            logger.info(
                "dataset_cache_incremental",
                cached_end=cached_end.isoformat(),
                incr_start=incr_start.isoformat(),
                end=end.isoformat(),
                base=len(base_samples),
            )
            new_samples, new_meta = build_dataset(session, incr_start, end)
            # 重複防止（同日境界）
            seen = {s.race_card_id for s in base_samples}
            merged = list(base_samples)
            for s in new_samples:
                if s.race_card_id not in seen:
                    merged.append(s)
                    seen.add(s.race_card_id)
            meta = {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "cards": stats["cards"],
                "samples": len(merged),
                "n_features": len(FEATURE_COLUMNS),
                "feature_columns": FEATURE_COLUMNS,
                "feature_fingerprint": _feature_fingerprint(),
                "with_result": stats["with_result"],
                "max_result_id": stats["max_result_id"],
                "cache_key": key,
                "incremental_from": cached_end.isoformat(),
                "base_samples": len(base_samples),
                "new_samples": len(new_samples),
            }
            save_dataset_cache(key, merged, meta)
            return merged, meta

    samples, meta = build_dataset(session, start, end)
    meta = {
        **meta,
        "feature_fingerprint": _feature_fingerprint(),
        "with_result": stats["with_result"],
        "max_result_id": stats["max_result_id"],
        "cache_key": key,
    }
    save_dataset_cache(key, samples, meta)
    return samples, meta
