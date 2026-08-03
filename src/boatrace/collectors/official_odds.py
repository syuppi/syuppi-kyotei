"""公式サイトから単勝オッズを取得して既存カードに反映."""

from __future__ import annotations

import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any

import httpx
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


def _parse_win_odds_html(html: str) -> dict[int, float]:
    soup = BeautifulSoup(html, "lxml")
    out: dict[int, float] = {}
    for table in soup.select("table"):
        header = " ".join(th.get_text(strip=True) for th in table.select("tr th"))
        if "単勝" not in header:
            continue
        for tr in table.select("tbody tr, tr"):
            tds = tr.select("td")
            if len(tds) < 2:
                continue
            w_txt = _norm(tds[0].get_text())
            m = re.match(r"^(\d+)$", w_txt)
            if not m:
                continue
            waku = int(m.group(1))
            point = tr.select_one(".oddsPoint, td.oddsPoint")
            odd = _to_float(point.get_text()) if point is not None else _to_float(tds[-1].get_text())
            if odd:
                out[waku] = odd
    return out


class OfficialOddsCollector(BaseCollector):
    """単勝オッズ（oddstf）を公式から取得。本日のライブ向け."""

    source_name = "boatrace_official_odds"

    def __init__(self, time_budget_sec: float = 90.0, workers: int = 6) -> None:
        super().__init__()
        self.base_url = get_settings().collect.base_url
        self.interval = 0.0
        self.max_retries = 1
        self.timeout = 20.0
        self.time_budget_sec = time_budget_sec
        self.workers = workers

    def _log_fetch(
        self,
        target_key: str,
        success: bool,
        attempts: int,
        error_message: str | None,
        summary: dict[str, Any] | None,
    ) -> None:
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
            "errors": 0,
            "source": self.source_name,
        }
        hd = target.strftime("%Y%m%d")
        headers = {"User-Agent": get_settings().collect.user_agent}
        started = time.monotonic()

        jobs: list[tuple[int, str, int]] = []  # card_id, venue, race_no
        with session_scope() as session:
            for vid in venues:
                cards = (
                    session.query(RaceCard)
                    .filter_by(venue_id=vid, race_date=target)
                    .order_by(RaceCard.race_no)
                    .all()
                )
                for card in cards:
                    jobs.append((card.id, vid, card.race_no))

        results: dict[int, dict[int, float]] = {}

        def _one(job: tuple[int, str, int]) -> tuple[int, dict[int, float] | None]:
            card_id, vid, rno = job
            if time.monotonic() - started > self.time_budget_sec:
                return card_id, None
            url = f"{self.base_url}/owpc/pc/race/oddstf"
            try:
                with httpx.Client(timeout=self.timeout, headers=headers, follow_redirects=True) as client:
                    resp = client.get(url, params={"rno": rno, "jcd": vid, "hd": hd})
                    if resp.status_code >= 400:
                        return card_id, None
                    return card_id, _parse_win_odds_html(resp.text)
            except Exception:
                return card_id, None

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = [ex.submit(_one, j) for j in jobs]
            for fut in as_completed(futs):
                if time.monotonic() - started > self.time_budget_sec + 5:
                    break
                card_id, win = fut.result()
                if win:
                    results[card_id] = win
                else:
                    summary["errors"] += 1

        now = datetime.utcnow()
        with session_scope() as session:
            for card_id, win in results.items():
                card = session.get(RaceCard, card_id)
                if card is None:
                    continue
                raw = dict(card.raw_payload or {})
                odds = normalize_odds_blob(raw.get("odds") or {})
                odds["win"] = {str(k): v for k, v in win.items()}
                raw["odds"] = odds
                raw["has_odds"] = True
                raw["odds_source"] = self.source_name
                raw["odds_fetched_at"] = now.isoformat()
                card.raw_payload = raw
                card.updated_at = now
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
                    entry.updated_at = now
                    summary["win_odds"] += 1

        logger.info("official_odds_done", **summary)
        return summary

    def fetch_win_odds(self, hd: str, venue_id: str, race_no: int) -> dict[int, float]:
        url = f"{self.base_url}/owpc/pc/race/oddstf"
        html = self.fetch_text(url, params={"rno": race_no, "jcd": venue_id, "hd": hd})
        return _parse_win_odds_html(html)
