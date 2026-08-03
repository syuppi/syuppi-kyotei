"""潮位データ収集 (JMA / 海上保安庁系)."""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta
from typing import Any

from boatrace.collectors.base import BaseCollector, CollectorError
from boatrace.config import VenueConfig, get_settings, get_venue_map
from boatrace.db.models import RaceCard, TideSnapshot
from boatrace.db.session import session_scope
from boatrace.logging_setup import get_logger

logger = get_logger(__name__)


# 簡易天文潮モデル用の場ごとの基準潮位振幅(cm)と位相オフセット(時間)
_VENUE_TIDE_PARAMS: dict[str, tuple[float, float]] = {
    "03": (90, 0.0),
    "04": (95, 0.2),
    "06": (70, 1.0),
    "07": (100, 0.5),
    "08": (110, 0.4),
    "09": (105, 0.6),
    "10": (40, 2.0),
    "14": (80, 1.2),
    "15": (120, 0.8),
    "16": (130, 0.7),
    "17": (160, 0.9),
    "18": (140, 1.0),
    "19": (100, 1.1),
    "20": (90, 1.3),
    "21": (85, 1.4),
    "22": (95, 1.5),
    "23": (80, 1.6),
}


class TideCollector(BaseCollector):
    """
    JMA潮位表 / 海上保安庁系の取得を試み、失敗時は天文近似で補完する。
    本番運用では観測点コード対応表を拡充すること。
    """

    source_name = "tide"

    def __init__(self) -> None:
        super().__init__()
        settings = get_settings()
        self.jma_base = settings.tide.jma_base_url
        self.kaiho_base = settings.tide.kaiho_base_url
        self.timeout = settings.tide.timeout_sec
        self.max_retries = settings.tide.max_retries

    def collect(self, race_date: date, venue_ids: list[str] | None = None) -> dict[str, Any]:
        venue_map = get_venue_map()
        targets = venue_ids or list(venue_map.keys())
        updated = 0
        for vid in targets:
            venue = venue_map.get(vid)
            if venue is None:
                continue
            if not venue.tide_sensitive and venue.tide_station is None:
                # 非感潮でも0埋めスナップショットを残し欠損を明示
                tide_info = self._neutral_tide()
            else:
                tide_info = self._fetch_or_estimate(race_date, venue)
            updated += self._apply_to_races(race_date, vid, tide_info, venue)
        return {"updated": updated}

    def _fetch_or_estimate(self, race_date: date, venue: VenueConfig) -> dict[str, Any]:
        # 1) JMA 試行
        try:
            info = self._try_jma(race_date, venue)
            if info:
                info["source"] = "jma"
                return info
        except Exception as e:  # noqa: BLE001
            logger.info("jma_tide_fallback", venue=venue.id, error=str(e))

        # 2) 海上保安庁系 試行
        try:
            info = self._try_kaiho(race_date, venue)
            if info:
                info["source"] = "kaiho"
                return info
        except Exception as e:  # noqa: BLE001
            logger.info("kaiho_tide_fallback", venue=venue.id, error=str(e))

        # 3) 天文近似
        info = self._estimate_astronomical(race_date, venue)
        info["source"] = "estimate"
        return info

    def _try_jma(self, race_date: date, venue: VenueConfig) -> dict[str, Any] | None:
        """
        JMA潮位関連ページはHTML構造が複雑なため、取得できた場合のみ部分パース。
        失敗時は None を返し推定にフォールバック。
        """
        if not venue.tide_station:
            return None
        # 公開インデックスを叩いて到達性確認（詳細パースは場×観測点マッピング拡充後に強化）
        url = f"{self.jma_base}/gmd/kaiyou/db/tide/suisan/index.php"
        html = self.fetch_text(url)
        if "潮位" not in html and "潮汐" not in html:
            return None
        # 現状は到達確認のみ → 推定に委譲（観測点コード対応後に本実装）
        return None

    def _try_kaiho(self, race_date: date, venue: VenueConfig) -> dict[str, Any] | None:
        url = f"{self.kaiho_base}/KANKYO/TIDE/real_time_tide/"
        try:
            html = self.fetch_text(url)
        except CollectorError:
            return None
        if not html:
            return None
        # 実況ページの構造依存が強いため、ここでは到達時も推定併用
        return None

    def _estimate_astronomical(self, race_date: date, venue: VenueConfig) -> dict[str, Any]:
        amp, phase_h = _VENUE_TIDE_PARAMS.get(venue.id, (80.0, 0.0))
        # 半日周潮の簡易モデル（M2近似）
        # 基準: 日付からの時間を12.42h周期で計算し、レース時刻は12時想定で代表値
        day_index = (race_date - date(2000, 1, 1)).days
        hour = 12.0 + phase_h
        period = 12.42
        angle = 2 * math.pi * ((day_index * 24 + hour) / period)
        mean = 100.0
        level = mean + amp * math.sin(angle)
        # 変化量: 1時間後との差
        angle2 = 2 * math.pi * ((day_index * 24 + hour + 1) / period)
        level2 = mean + amp * math.sin(angle2)
        delta = level2 - level

        # 満潮・干潮の近似時刻
        # sin最大 ≈ 満潮
        high_offset_h = (math.pi / 2 - (angle % (2 * math.pi))) / (2 * math.pi) * period
        low_offset_h = (math.pi * 1.5 - (angle % (2 * math.pi))) / (2 * math.pi) * period
        base_dt = datetime.combine(race_date, datetime.min.time()) + timedelta(hours=hour)
        high_at = base_dt + timedelta(hours=high_offset_h)
        low_at = base_dt + timedelta(hours=low_offset_h)

        near_high = abs((high_at - base_dt).total_seconds()) < 90 * 60
        near_low = abs((low_at - base_dt).total_seconds()) < 90 * 60

        return {
            "station_name": venue.tide_station or venue.name,
            "tide_level_cm": round(level, 1),
            "tide_delta_cm": round(delta, 1),
            "near_high_tide": near_high,
            "near_low_tide": near_low,
            "high_tide_at": high_at,
            "low_tide_at": low_at,
        }

    def _neutral_tide(self) -> dict[str, Any]:
        return {
            "station_name": None,
            "tide_level_cm": None,
            "tide_delta_cm": None,
            "near_high_tide": False,
            "near_low_tide": False,
            "high_tide_at": None,
            "low_tide_at": None,
            "source": "n/a",
        }

    def _apply_to_races(
        self,
        race_date: date,
        venue_id: str,
        tide_info: dict[str, Any],
        venue: VenueConfig,
    ) -> int:
        with session_scope() as session:
            cards = (
                session.query(RaceCard)
                .filter_by(venue_id=venue_id, race_date=race_date)
                .all()
            )
            now = datetime.utcnow()
            count = 0
            for card in cards:
                # レース番号で潮位をわずかに時間シフト
                info = dict(tide_info)
                if info.get("tide_level_cm") is not None:
                    amp, phase_h = _VENUE_TIDE_PARAMS.get(venue_id, (80.0, 0.0))
                    hour = 9.0 + card.race_no * 0.75 + phase_h
                    day_index = (race_date - date(2000, 1, 1)).days
                    period = 12.42
                    angle = 2 * math.pi * ((day_index * 24 + hour) / period)
                    mean = 100.0
                    level = mean + amp * math.sin(angle)
                    angle2 = 2 * math.pi * ((day_index * 24 + hour + 1) / period)
                    delta = (mean + amp * math.sin(angle2)) - level
                    info["tide_level_cm"] = round(level, 1)
                    info["tide_delta_cm"] = round(delta, 1)
                    info["near_high_tide"] = math.sin(angle) > 0.85
                    info["near_low_tide"] = math.sin(angle) < -0.85

                ts = session.query(TideSnapshot).filter_by(race_card_id=card.id).one_or_none()
                if ts is None:
                    ts = TideSnapshot(race_card_id=card.id)
                    session.add(ts)
                ts.station_name = info.get("station_name")
                ts.tide_level_cm = info.get("tide_level_cm")
                ts.tide_delta_cm = info.get("tide_delta_cm")
                ts.near_high_tide = bool(info.get("near_high_tide"))
                ts.near_low_tide = bool(info.get("near_low_tide"))
                ts.high_tide_at = info.get("high_tide_at")
                ts.low_tide_at = info.get("low_tide_at")
                ts.source = info.get("source", "estimate")
                ts.fetched_at = now
                ts.updated_at = now
                count += 1
            return count
