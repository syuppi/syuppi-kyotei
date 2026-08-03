"""Turnmark API からオッズを取得して既存レースにマージ."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from boatrace.collectors.base import BaseCollector, CollectorError
from boatrace.config import get_venue_map
from boatrace.db.models import RaceCard, RaceEntry
from boatrace.db.session import session_scope
from boatrace.logging_setup import get_logger
from boatrace.models.odds_ev import normalize_odds_blob

logger = get_logger(__name__)

TURNMARK_BASE = "https://turnmark.github.io/api/v1"


def _vid(stadium_number: int | str) -> str:
    return f"{int(stadium_number):02d}"


class TurnmarkOddsCollector(BaseCollector):
    """
    turnmark.github.io の日次JSONからオッズを取り込む。
    基本は前日まで。本日は未公開のことが多い。
    """

    source_name = "turnmark_odds"

    def collect(
        self,
        race_date: date | None = None,
        venue_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        target = race_date or date.today()
        payload = self.fetch_day_json(target)
        return self.ingest(payload, venue_ids=venue_ids, race_date=target)

    def fetch_day_json(self, race_date: date) -> dict[str, Any]:
        url = f"{TURNMARK_BASE}/{race_date.year}/{race_date.strftime('%Y%m%d')}.json"
        # today.json は無い想定。日付URLのみ。
        text = self.fetch_text(url)
        import json

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise CollectorError(f"invalid json from {url}: {e}") from e
        if not isinstance(data, dict) or "programs" not in data:
            raise CollectorError(f"unexpected payload from {url}")
        return data

    def ingest(
        self,
        payload: dict[str, Any],
        venue_ids: list[str] | None = None,
        race_date: date | None = None,
    ) -> dict[str, Any]:
        venue_map = get_venue_map()
        allowed = set(venue_ids) if venue_ids else set(venue_map.keys())
        stadiums = ((payload.get("programs") or {}).get("stadiums") or {})
        summary = {
            "stadiums": 0,
            "cards_updated": 0,
            "win_odds": 0,
            "with_combo_odds": 0,
            "source": self.source_name,
            "date": (race_date or date.today()).isoformat(),
        }
        now = datetime.utcnow()

        with session_scope() as session:
            for sid, stadium in stadiums.items():
                vid = _vid(sid)
                if vid not in allowed or vid not in venue_map:
                    continue
                summary["stadiums"] += 1
                races = (stadium or {}).get("races") or {}
                for rno, race in races.items():
                    try:
                        race_no = int(rno)
                    except (TypeError, ValueError):
                        continue
                    odds_raw = race.get("odds")
                    if not odds_raw and any(k in race for k in ("win", "trifecta", "trio")):
                        odds_raw = {
                            k: race.get(k)
                            for k in (
                                "win",
                                "place",
                                "quinella",
                                "quinella_place",
                                "exacta",
                                "trio",
                                "trifecta",
                            )
                            if race.get(k) is not None
                        }
                    odds = normalize_odds_blob(odds_raw or {})
                    if not odds:
                        continue

                    d = race_date
                    if race.get("date"):
                        try:
                            d = date.fromisoformat(str(race["date"])[:10])
                        except ValueError:
                            d = race_date
                    if d is None:
                        continue

                    card = (
                        session.query(RaceCard)
                        .filter_by(venue_id=vid, race_date=d, race_no=race_no)
                        .one_or_none()
                    )
                    if card is None:
                        continue

                    raw = dict(card.raw_payload or {})
                    raw["odds"] = odds
                    raw["has_odds"] = True
                    raw["odds_source"] = self.source_name
                    raw["odds_fetched_at"] = now.isoformat()
                    card.raw_payload = raw
                    card.updated_at = now
                    summary["cards_updated"] += 1
                    if odds.get("trifecta") or odds.get("trio"):
                        summary["with_combo_odds"] += 1

                    win = odds.get("win") or {}
                    if isinstance(win, dict):
                        for k, v in win.items():
                            try:
                                waku = int(str(k))
                                odd_v = float(v)
                            except (TypeError, ValueError):
                                continue
                            if waku < 1 or odd_v <= 0:
                                continue
                            entry = (
                                session.query(RaceEntry)
                                .filter_by(race_card_id=card.id, waku=waku)
                                .one_or_none()
                            )
                            if entry is None:
                                continue
                            entry.win_odds = odd_v
                            entry.updated_at = now
                            summary["win_odds"] += 1

        logger.info("turnmark_odds_ingested", **summary)
        return summary
