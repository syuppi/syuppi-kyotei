"""日次収集パイプライン（出走表 + オッズ）."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from boatrace.collectors.official import OfficialCollector
from boatrace.collectors.official_odds import OfficialOddsCollector
from boatrace.collectors.openapi import OpenApiCollector
from boatrace.collectors.tide import TideCollector
from boatrace.collectors.turnmark import TurnmarkOddsCollector
from boatrace.config import get_settings
from boatrace.logging_setup import get_logger
from boatrace.timeutil import japan_today

logger = get_logger(__name__)


def collect_daily(
    race_date: date | None = None,
    venue_ids: list[str] | None = None,
    include_lookback: bool = True,
    prefer_openapi: bool = True,
    include_odds: bool = True,
) -> dict[str, Any]:
    """
    当日（＋直近N日）データを収集。
    既定では Boatrace Open API(JSON) を優先し、失敗時は公式HTMLへフォールバック。
    オッズは turnmark（過去）＋公式単勝（当日）で補完。
    """
    settings = get_settings()
    target = race_date or japan_today()
    openapi = OpenApiCollector()
    official = OfficialCollector()
    tide = TideCollector()
    turnmark = TurnmarkOddsCollector()
    official_odds = OfficialOddsCollector()

    results: dict[str, Any] = {"date": target.isoformat(), "days": []}

    dates = [target]
    if include_lookback:
        for i in range(1, settings.collect.lookback_days + 1):
            dates.append(target - timedelta(days=i))

    for d in dates:
        logger.info("collect_start", date=d.isoformat())
        day: dict[str, Any] = {"date": d.isoformat()}

        used = None
        if prefer_openapi:
            try:
                day["openapi"] = openapi.collect(d, venue_ids=venue_ids)
                used = "openapi"
            except Exception as e:  # noqa: BLE001
                logger.warning("openapi_collect_failed", date=d.isoformat(), error=str(e))
                day["openapi_error"] = str(e)

        if used is None:
            try:
                day["official"] = official.collect(d, venue_ids=venue_ids)
                used = "official"
            except Exception as e:  # noqa: BLE001
                logger.exception("official_collect_error", date=d.isoformat(), error=str(e))
                day["official_error"] = str(e)

        if include_odds:
            # 過去日: turnmark（3連単/3連複含む）
            try:
                day["turnmark_odds"] = turnmark.collect(d, venue_ids=venue_ids)
            except Exception as e:  # noqa: BLE001
                logger.info("turnmark_odds_skip", date=d.isoformat(), error=str(e))
                day["turnmark_odds_error"] = str(e)

            # 当日（または turnmark が薄いとき）: 公式単勝
            if d == japan_today() or not (day.get("turnmark_odds") or {}).get("cards_updated"):
                try:
                    day["official_odds"] = official_odds.collect(d, venue_ids=venue_ids)
                except Exception as e:  # noqa: BLE001
                    logger.warning("official_odds_failed", date=d.isoformat(), error=str(e))
                    day["official_odds_error"] = str(e)

        try:
            day["tide"] = tide.collect(d, venue_ids=venue_ids)
        except Exception as e:  # noqa: BLE001
            logger.exception("tide_collect_error", date=d.isoformat(), error=str(e))
            day["tide_error"] = str(e)

        # 着順未取得を公式HTMLで補完（OpenAPIが空結果を返すケース対策）
        try:
            day["missing_results"] = official.collect_missing_results(d, venue_ids=venue_ids)
        except Exception as e:  # noqa: BLE001
            logger.warning("missing_results_failed", date=d.isoformat(), error=str(e))
            day["missing_results_error"] = str(e)

        day["primary_source"] = used
        results["days"].append(day)
        logger.info("collect_done", date=d.isoformat(), primary=used)

    return results


def prepare_day(
    race_date: date | None = None,
    venue_ids: list[str] | None = None,
    *,
    fast: bool = True,
) -> dict[str, Any]:
    """
    UIの日付変更用: その日だけ収集（lookbackなし）。
    fast=True では潮汐・公式オッズを省略し、OpenAPI + turnmark のみ（524回避）。
    """
    target = race_date or japan_today()
    openapi = OpenApiCollector()
    turnmark = TurnmarkOddsCollector()
    summary: dict[str, Any] = {"date": target.isoformat(), "fast": fast, "steps": {}}

    try:
        summary["steps"]["openapi"] = openapi.collect(target, venue_ids=venue_ids)
    except Exception as e:  # noqa: BLE001
        logger.warning("prepare_openapi_failed", date=target.isoformat(), error=str(e))
        summary["steps"]["openapi_error"] = str(e)
        # OpenAPI失敗時のみ公式HTMLへ
        try:
            summary["steps"]["official"] = OfficialCollector().collect(target, venue_ids=venue_ids)
        except Exception as e2:  # noqa: BLE001
            summary["steps"]["official_error"] = str(e2)

    # OpenAPIで着順が欠けることがあるため、未取得分は公式結果で補完
    try:
        summary["steps"]["missing_results"] = OfficialCollector().collect_missing_results(
            target, venue_ids=venue_ids
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("prepare_missing_results_failed", date=target.isoformat(), error=str(e))
        summary["steps"]["missing_results_error"] = str(e)

    try:
        summary["steps"]["turnmark_odds"] = turnmark.collect(target, venue_ids=venue_ids)
    except Exception as e:  # noqa: BLE001
        summary["steps"]["turnmark_odds_error"] = str(e)

    if not fast:
        try:
            summary["steps"]["official_odds"] = OfficialOddsCollector(
                time_budget_sec=45, workers=8
            ).collect(target, venue_ids=venue_ids)
        except Exception as e:  # noqa: BLE001
            summary["steps"]["official_odds_error"] = str(e)
        try:
            summary["steps"]["tide"] = TideCollector().collect(target, venue_ids=venue_ids)
        except Exception as e:  # noqa: BLE001
            summary["steps"]["tide_error"] = str(e)

    # 前走情報。学習は predict_only ではスキップ（Render等は別環境で学習）
    try:
        from boatrace.config import is_predict_only
        from boatrace.db.session import session_scope
        from boatrace.features.previous_starts import backfill_previous_for_day
        from boatrace.learning.service import LearningService

        with session_scope() as session:
            summary["steps"]["previous_starts"] = backfill_previous_for_day(
                session, target, venue_ids=venue_ids
            )
            if is_predict_only():
                summary["steps"]["learn"] = {"skipped": True, "reason": "predict_only"}
            else:
                learn = LearningService(session).learn_day(target)
                summary["steps"]["learn"] = {
                    "date": learn.get("date"),
                    "course_stats_updated": learn.get("course_stats_updated"),
                    "bias_updated": learn.get("bias_updated"),
                }
            session.commit()
    except Exception as e:  # noqa: BLE001
        logger.warning("prepare_learn_failed", date=target.isoformat(), error=str(e))
        summary["steps"]["learn_error"] = str(e)

    return summary

