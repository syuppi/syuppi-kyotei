"""Boatrace Open API (非公式JSON) からの実データ収集."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import httpx

from boatrace.collectors.base import BaseCollector, CollectorError
from boatrace.config import get_venue_map
from boatrace.db.models import RaceCard, RaceEntry, RaceResult, Racer, WeatherSnapshot
from boatrace.db.session import session_scope
from boatrace.logging_setup import get_logger

logger = get_logger(__name__)

OPENAPI_BASE = "https://boatraceopenapi.github.io/api/v1"

# 決まり手番号 → 文言（schema / 実データより）
TECHNIQUE_MAP = {
    1: "逃げ",
    2: "差し",
    3: "まくり",
    4: "まくり差し",
    5: "抜き",
    6: "恵まれ",
}


def _vid(stadium_number: int | str) -> str:
    return f"{int(stadium_number):02d}"


class OpenApiCollector(BaseCollector):
    """
    https://boatraceopenapi.github.io/api/v1/today.json
    出走表・直前情報・結果を一括取得して正規化保存する。
    ※非公式・数分遅延あり。公式が必要な場合は OfficialCollector を併用。
    """

    source_name = "boatrace_openapi"

    def collect(
        self,
        race_date: date | None = None,
        venue_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        target = race_date or date.today()
        payload = self.fetch_day_json(target)
        return self.ingest(payload, venue_ids=venue_ids)

    def fetch_day_json(self, race_date: date) -> dict[str, Any]:
        if race_date == date.today():
            url = f"{OPENAPI_BASE}/today.json"
        else:
            url = f"{OPENAPI_BASE}/{race_date.year}/{race_date.strftime('%Y%m%d')}.json"
        try:
            text = self.fetch_text(url)
        except CollectorError:
            # today 失敗時は日付URLへフォールバック
            if race_date == date.today():
                url = f"{OPENAPI_BASE}/{race_date.year}/{race_date.strftime('%Y%m%d')}.json"
                text = self.fetch_text(url)
            else:
                raise
        import json

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise CollectorError(f"invalid json from {url}: {e}") from e
        if not isinstance(data, dict) or "programs" not in data:
            raise CollectorError(f"unexpected payload from {url}")
        return data

    def ingest(
        self, payload: dict[str, Any], venue_ids: list[str] | None = None
    ) -> dict[str, Any]:
        venue_map = get_venue_map()
        allowed = set(venue_ids) if venue_ids else set(venue_map.keys())
        stadiums = ((payload.get("programs") or {}).get("stadiums") or {})
        summary = {
            "stadiums": 0,
            "cards": 0,
            "entries": 0,
            "previews": 0,
            "results": 0,
            "source": self.source_name,
        }

        for sid, stadium in stadiums.items():
            vid = _vid(sid)
            if vid not in allowed:
                continue
            if vid not in venue_map:
                logger.warning("unknown_venue_in_openapi", venue_id=vid)
                continue
            summary["stadiums"] += 1
            races = (stadium or {}).get("races") or {}
            for _rno, race in races.items():
                try:
                    c, e, p, r = self._upsert_race(vid, race)
                    summary["cards"] += c
                    summary["entries"] += e
                    summary["previews"] += p
                    summary["results"] += r
                except Exception as ex:  # noqa: BLE001
                    logger.exception(
                        "openapi_race_ingest_failed",
                        venue_id=vid,
                        race=race.get("race_number"),
                        error=str(ex),
                    )
        return summary

    def _upsert_race(self, venue_id: str, race: dict[str, Any]) -> tuple[int, int, int, int]:
        race_date = date.fromisoformat(str(race["date"]))
        rno = int(race["race_number"])
        now = datetime.utcnow()
        card_n = entry_n = preview_n = result_n = 0

        with session_scope() as session:
            card = (
                session.query(RaceCard)
                .filter_by(venue_id=venue_id, race_date=race_date, race_no=rno)
                .one_or_none()
            )
            if card is None:
                card = RaceCard(venue_id=venue_id, race_date=race_date, race_no=rno)
                session.add(card)
                session.flush()
                card_n = 1

            title = " ".join(
                x for x in [race.get("title") or "", race.get("subtitle") or ""] if x
            ).strip()
            card.race_title = title or card.race_title
            card.race_grade = str(race.get("grade_number_source") or card.race_grade or "")
            if race.get("grade_number") is not None:
                card.grade_number = int(race["grade_number"])
            if race.get("day_number") is not None:
                card.day_number = int(race["day_number"])
            if race.get("distance") is not None:
                card.distance_m = int(race["distance"])
            closed = race.get("closed_at")
            if closed:
                try:
                    card.deadline_at = datetime.strptime(closed, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass
            card.fetched_at = now
            card.updated_at = now
            odds_blob = race.get("odds")
            from boatrace.models.odds_ev import normalize_odds_blob

            odds_norm = normalize_odds_blob(odds_blob or {})
            card.raw_payload = {
                "source": self.source_name,
                "has_preview": bool(race.get("preview")),
                "has_result": bool(race.get("result") and (race["result"] or {}).get("racers")),
                "has_odds": bool(odds_norm),
                "odds": odds_norm or None,
            }

            for _waku, rd in (race.get("racers") or {}).items():
                if rd.get("number") is None:
                    continue
                waku = int(rd.get("entry_number") or _waku)
                racer_id = f"{int(rd['number']):04d}"
                name = (rd.get("name") or f"選手{racer_id}").replace("　", " ").strip()
                grade = rd.get("rank_number_source") or ""
                racer = session.get(Racer, racer_id)
                if racer is None:
                    session.add(
                        Racer(
                            id=racer_id,
                            name=name,
                            grade=grade,
                            branch=str(rd.get("branch_number_source") or ""),
                        )
                    )
                    session.flush()
                else:
                    racer.name = name
                    if grade:
                        racer.grade = grade

                entry = (
                    session.query(RaceEntry)
                    .filter_by(race_card_id=card.id, waku=waku)
                    .one_or_none()
                )
                if entry is None:
                    entry = RaceEntry(race_card_id=card.id, waku=waku, racer_id=racer_id)
                    session.add(entry)
                    entry_n += 1
                entry.racer_id = racer_id
                entry.age = rd.get("age")
                entry.weight = rd.get("weight")
                entry.f_count = rd.get("flying_count")
                entry.l_count = rd.get("late_count")
                entry.avg_st = rd.get("average_start_timing")
                entry.national_win_rate = rd.get("national_win_rate")
                entry.national_quinella_rate = rd.get("national_top_2_percent")
                entry.national_trio_rate = rd.get("national_top_3_percent")
                entry.local_win_rate = rd.get("local_win_rate")
                entry.local_quinella_rate = rd.get("local_top_2_percent")
                entry.local_trio_rate = rd.get("local_top_3_percent")
                entry.motor_no = rd.get("motor_number")
                entry.motor_quinella_rate = rd.get("motor_top_2_percent")
                entry.motor_trio_rate = rd.get("motor_top_3_percent")
                entry.boat_no = rd.get("boat_number")
                entry.boat_quinella_rate = rd.get("boat_top_2_percent")
                entry.boat_trio_rate = rd.get("boat_top_3_percent")
                entry.grade_code = grade or entry.grade_code
                entry.updated_at = now

            # 単勝オッズ（dict {"1":2.0} / list 両対応）
            for waku, odd_v in self._parse_win_odds(race.get("odds")).items():
                entry = (
                    session.query(RaceEntry)
                    .filter_by(race_card_id=card.id, waku=waku)
                    .one_or_none()
                )
                if entry is not None:
                    entry.win_odds = odd_v

            preview = race.get("preview") or {}
            result = race.get("result") or {}
            weather_src = result if result.get("air_temperature") is not None else preview

            if weather_src:
                preview_n = 1 if preview else preview_n
                ws = (
                    session.query(WeatherSnapshot)
                    .filter_by(race_card_id=card.id)
                    .one_or_none()
                )
                if ws is None:
                    ws = WeatherSnapshot(race_card_id=card.id, source=self.source_name)
                    session.add(ws)
                    session.flush()
                if weather_src.get("air_temperature") is not None:
                    ws.temperature = weather_src.get("air_temperature")
                ws.weather = weather_src.get("weather_number_source") or ws.weather
                if weather_src.get("wind_speed") is not None:
                    ws.wind_speed = float(weather_src["wind_speed"])
                ws.wind_direction = (
                    weather_src.get("wind_direction_number_source") or ws.wind_direction
                )
                if weather_src.get("water_temperature") is not None:
                    ws.water_temperature = weather_src.get("water_temperature")
                if weather_src.get("wave_height") is not None:
                    ws.wave_height = float(weather_src["wave_height"])
                ws.source = self.source_name
                ws.fetched_at = now
                ws.updated_at = now

            if preview:
                for _waku, pr in (preview.get("racers") or {}).items():
                    waku = int(pr.get("entry_number") or _waku)
                    entry = (
                        session.query(RaceEntry)
                        .filter_by(race_card_id=card.id, waku=waku)
                        .one_or_none()
                    )
                    if entry is None:
                        continue
                    if pr.get("exhibition_time") is not None:
                        entry.exhibition_time = float(pr["exhibition_time"])
                    if pr.get("tilt_adjustment") is not None:
                        entry.tilt = float(pr["tilt_adjustment"])
                    if pr.get("course_number") is not None:
                        entry.estimated_course = int(pr["course_number"])
                    if pr.get("weight") is not None:
                        entry.weight = float(pr["weight"])
                    if pr.get("weight_adjustment") is not None:
                        entry.weight_adjustment = float(pr["weight_adjustment"])
                    if pr.get("start_timing") is not None:
                        entry.exhibition_st = float(pr["start_timing"])
                    entry.updated_at = now

            racers_res = (result or {}).get("racers") or {}
            if racers_res:
                result_n = 1
                by_place: dict[int, int] = {}
                entry_results: list[dict[str, Any]] = []
                for _waku, rr in racers_res.items():
                    waku = int(rr.get("entry_number") or _waku)
                    place = rr.get("place_number")
                    if place is None:
                        continue
                    place = int(place)
                    by_place[place] = waku
                    entry_results.append(
                        {
                            "waku": waku,
                            "rank": place,
                            "st": rr.get("start_timing"),
                            "course": rr.get("course_number") or waku,
                        }
                    )

                rr_row = (
                    session.query(RaceResult)
                    .filter_by(race_card_id=card.id)
                    .one_or_none()
                )
                if rr_row is None:
                    rr_row = RaceResult(race_card_id=card.id)
                    session.add(rr_row)
                rr_row.rank1_waku = by_place.get(1)
                rr_row.rank2_waku = by_place.get(2)
                rr_row.rank3_waku = by_place.get(3)
                tech = result.get("technique_number_source") or TECHNIQUE_MAP.get(
                    result.get("technique_number")
                )
                rr_row.kimarite = tech
                rr_row.payouts = result.get("payouts")
                rr_row.entry_results = entry_results
                rr_row.fetched_at = now
                rr_row.updated_at = now
                card.status = "finished"
                card.updated_at = now

        return card_n, entry_n, preview_n, result_n

    @staticmethod
    def _parse_win_odds(odds_payload: object) -> dict[int, float]:
        """単勝オッズを艇番→オッズに正規化（dict / list 両対応）。"""
        win_by_boat: dict[int, float] = {}
        if not isinstance(odds_payload, dict):
            return win_by_boat
        win = odds_payload.get("win")
        if isinstance(win, dict):
            for k, v in win.items():
                try:
                    waku = int(str(k).split("-")[0])
                    odd_v = float(v)
                    if waku and odd_v > 0:
                        win_by_boat[waku] = odd_v
                except (TypeError, ValueError):
                    continue
            return win_by_boat
        if isinstance(win, list):
            for item in win:
                if not isinstance(item, dict):
                    continue
                comb = str(item.get("combination") or item.get("key") or item.get("boat_number") or "")
                try:
                    waku = int(comb.split("-")[0].split("=")[0])
                except ValueError:
                    continue
                odd_v = item.get("odds") or item.get("value")
                if odd_v is None:
                    continue
                try:
                    odd_f = float(odd_v)
                except (TypeError, ValueError):
                    continue
                if waku and odd_f > 0:
                    win_by_boat[waku] = odd_f
        return win_by_boat
