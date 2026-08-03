"""公式サイトから単勝オッズを取得して既存カードに反映."""

from __future__ import annotations

import re
import warnings
from datetime import date, datetime
from typing import Any

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from boatrace.collectors.base import BaseCollector, CollectorError
from boatrace.config import get_settings, get_venues
from boatrace.db.models import RaceCard, RaceEntry
from boatrace.db.session import session_scope
from boatrace.logging_setup import get_logger
from boatrace.models.odds_ev import normalize_odds_blob

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
logger = get_logger(__name__)

_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")


def _norm(text: str | None) -> str:
    if text is None:
        return ""
    return text.translate(_ZEN).replace("\u3000", " ").strip()


def _to_float(text: str | None) -> float | None:
    t = _norm(text).replace(",", "")
    if t in {"", "-", "－", "---", "欠場"}:
        return None
    try:
        return float(t)
    except ValueError:
        return None


class OfficialOddsCollector(BaseCollector):
    """単勝オッズ（oddstf）を公式から取得。本日のライブ向け."""

    source_name = "boatrace_official_odds"

    def __init__(self) -> None:
        super().__init__()
        self.base_url = get_settings().collect.base_url
        # 単勝オッズは件数が多いので短間隔
        self.interval = 0.12
        self.max_retries = 2

    def _log_fetch(
        self,
        target_key: str,
        success: bool,
        attempts: int,
        error_message: str | None,
        summary: dict[str, Any] | None,
    ) -> None:
        # レース数が多いため個別ログは省略
        return

    def collect(
        self,
        race_date: date | None = None,
        venue_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        target = race_date or date.today()
        venues = venue_ids or [v.id for v in get_venues()]
        summary: dict[str, Any] = {
            "date": target.isoformat(),
            "cards_updated": 0,
            "win_odds": 0,
            "errors": [],
            "source": self.source_name,
        }
        hd = target.strftime("%Y%m%d")
        with session_scope() as session:
            for vid in venues:
                cards = (
                    session.query(RaceCard)
                    .filter_by(venue_id=vid, race_date=target)
                    .order_by(RaceCard.race_no)
                    .all()
                )
                for card in cards:
                    try:
                        win = self.fetch_win_odds(hd, vid, card.race_no)
                        if not win:
                            continue
                        raw = dict(card.raw_payload or {})
                        odds = normalize_odds_blob(raw.get("odds") or {})
                        odds["win"] = {str(k): v for k, v in win.items()}
                        raw["odds"] = odds
                        raw["has_odds"] = True
                        raw["odds_source"] = self.source_name
                        raw["odds_fetched_at"] = datetime.utcnow().isoformat()
                        card.raw_payload = raw
                        card.updated_at = datetime.utcnow()
                        summary["cards_updated"] += 1
                        for waku, odd_v in win.items():
                            entry = (
                                session.query(RaceEntry)
                                .filter_by(race_card_id=card.id, waku=waku)
                                .one_or_none()
                            )
                            if entry is None:
                                continue
                            entry.win_odds = odd_v
                            entry.updated_at = datetime.utcnow()
                            summary["win_odds"] += 1
                    except Exception as e:  # noqa: BLE001
                        logger.debug(
                            "official_odds_failed",
                            venue_id=vid,
                            race_no=card.race_no,
                            error=str(e),
                        )
                        summary["errors"].append(
                            {"venue_id": vid, "race_no": card.race_no, "error": str(e)}
                        )
        logger.info("official_odds_done", **{k: v for k, v in summary.items() if k != "errors"})
        return summary

    def fetch_win_odds(self, hd: str, venue_id: str, race_no: int) -> dict[int, float]:
        url = f"{self.base_url}/owpc/pc/race/oddstf"
        html = self.fetch_text(url, params={"rno": race_no, "jcd": venue_id, "hd": hd})
        soup = BeautifulSoup(html, "lxml")
        out: dict[int, float] = {}
        # 単勝テーブル: 枠 / 選手 / 単勝オッズ
        for table in soup.select("table"):
            header = " ".join(th.get_text(strip=True) for th in table.select("tr th"))
            if "単勝" not in header:
                continue
            for tr in table.select("tbody tr, tr"):
                tds = tr.select("td")
                if len(tds) < 2:
                    continue
                waku = None
                # 先頭セルが枠番
                w_txt = _norm(tds[0].get_text())
                m = re.match(r"^(\d+)$", w_txt)
                if m:
                    waku = int(m.group(1))
                odd = None
                # oddsPoint 優先、なければ末尾セル
                point = tr.select_one(".oddsPoint, td.oddsPoint")
                if point is not None:
                    odd = _to_float(point.get_text())
                if odd is None:
                    odd = _to_float(tds[-1].get_text())
                if waku and odd:
                    out[waku] = odd
        return out
