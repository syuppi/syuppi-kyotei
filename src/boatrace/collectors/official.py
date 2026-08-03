"""BOAT RACE公式サイトからの収集."""

from __future__ import annotations

import re
import warnings
from datetime import date, datetime
from typing import Any

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from boatrace.collectors.base import BaseCollector, CollectorError
from boatrace.config import get_settings
from boatrace.db.models import RaceCard, RaceEntry, RaceResult, Racer, WeatherSnapshot
from boatrace.db.session import session_scope
from boatrace.logging_setup import get_logger

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
logger = get_logger(__name__)

_ZEN_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def _norm(text: str | None) -> str:
    if text is None:
        return ""
    return text.translate(_ZEN_DIGITS).replace("\u3000", " ").strip()


def _to_float(text: str | None) -> float | None:
    if text is None:
        return None
    t = _norm(text).replace("%", "")
    if t in {"", "-", "－", "---"}:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _to_int(text: str | None) -> int | None:
    if text is None:
        return None
    t = re.sub(r"[^\d-]", "", _norm(text))
    if not t or t == "-":
        return None
    try:
        return int(t)
    except ValueError:
        return None


def _parse_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


class OfficialCollector(BaseCollector):
    source_name = "boatrace_official"

    def __init__(self) -> None:
        super().__init__()
        self.base_url = get_settings().collect.base_url

    def collect(self, race_date: date, venue_ids: list[str] | None = None) -> dict[str, Any]:
        from boatrace.config import get_venues

        venues = venue_ids or [v.id for v in get_venues()]
        summary: dict[str, Any] = {"cards": 0, "entries": 0, "results": 0, "errors": []}
        for vid in venues:
            try:
                s = self.collect_venue_day(race_date, vid)
                summary["cards"] += s["cards"]
                summary["entries"] += s["entries"]
                summary["results"] += s["results"]
            except Exception as e:  # noqa: BLE001
                logger.exception("venue_collect_failed", venue_id=vid, error=str(e))
                summary["errors"].append({"venue_id": vid, "error": str(e)})
        return summary

    def collect_venue_day(self, race_date: date, venue_id: str) -> dict[str, int]:
        hd = race_date.strftime("%Y%m%d")
        race_nos = self._discover_race_nos(hd, venue_id)
        cards = entries = results = 0
        for rno in race_nos:
            try:
                card_data = self.fetch_racelist(hd, venue_id, rno)
                if not card_data.get("entries"):
                    logger.info("no_entries", venue=venue_id, rno=rno)
                    continue
                before = self.fetch_beforeinfo(hd, venue_id, rno)
                self._merge_before(card_data, before)
                c, e = self._upsert_card(race_date, venue_id, rno, card_data)
                cards += c
                entries += e
            except Exception as e:  # noqa: BLE001
                logger.warning("racelist_failed", venue=venue_id, rno=rno, error=str(e))

            try:
                result = self.fetch_result(hd, venue_id, rno)
                if result:
                    results += self._upsert_result(race_date, venue_id, rno, result)
            except Exception as e:  # noqa: BLE001
                logger.debug("result_not_ready", venue=venue_id, rno=rno, error=str(e))
        return {"cards": cards, "entries": entries, "results": results}

    def _discover_race_nos(self, hd: str, venue_id: str) -> list[int]:
        url = f"{self.base_url}/owpc/pc/race/raceindex"
        try:
            html = self.fetch_text(url, params={"jcd": venue_id, "hd": hd})
            soup = _parse_soup(html)
            nos: list[int] = []
            for a in soup.select("a[href*='racelist']"):
                href = a.get("href", "")
                m = re.search(r"rno=(\d+)", href)
                if m:
                    nos.append(int(m.group(1)))
            if nos:
                return sorted(set(nos))
        except CollectorError:
            pass
        return list(range(1, 13))

    def fetch_racelist(self, hd: str, venue_id: str, rno: int) -> dict[str, Any]:
        url = f"{self.base_url}/owpc/pc/race/racelist"
        html = self.fetch_text(url, params={"rno": rno, "jcd": venue_id, "hd": hd})
        return self._parse_racelist(html, venue_id, rno)

    def fetch_beforeinfo(self, hd: str, venue_id: str, rno: int) -> dict[str, Any]:
        url = f"{self.base_url}/owpc/pc/race/beforeinfo"
        try:
            html = self.fetch_text(url, params={"rno": rno, "jcd": venue_id, "hd": hd})
            return self._parse_beforeinfo(html)
        except CollectorError as e:
            logger.warning("beforeinfo_unavailable", error=str(e))
            return {}

    def fetch_result(self, hd: str, venue_id: str, rno: int) -> dict[str, Any] | None:
        url = f"{self.base_url}/owpc/pc/race/raceresult"
        html = self.fetch_text(url, params={"rno": rno, "jcd": venue_id, "hd": hd})
        return self._parse_result(html)

    def _parse_racelist(self, html: str, venue_id: str, rno: int) -> dict[str, Any]:
        soup = _parse_soup(html)
        entries: list[dict[str, Any]] = []

        bodies = soup.select("tbody.is-fs12")
        for tb in bodies:
            tr = tb.find("tr")
            if not tr:
                continue
            tds = tr.find_all("td")
            if len(tds) < 8:
                continue
            waku = _to_int(tds[0].get_text(strip=True))
            if not waku or not (1 <= waku <= 6):
                continue

            # 選手リンク
            racer_id = ""
            name = ""
            for a in tr.select("a[href*='toban=']"):
                href = a.get("href", "")
                m = re.search(r"toban=(\d{4})", href)
                if not m:
                    continue
                racer_id = m.group(1)
                txt = _norm(a.get_text())
                if txt and not txt.isdigit():
                    name = txt
            if not racer_id:
                continue

            info = _norm(tds[2].get_text(" ", strip=True))
            age = None
            weight = None
            grade = ""
            am = re.search(r"(\d{1,2})歳", info)
            wm = re.search(r"([\d.]+)\s*kg", info, re.I)
            gm = re.search(r"\b([AB][12])\b", info)
            if am:
                age = int(am.group(1))
            if wm:
                weight = float(wm.group(1))
            if gm:
                grade = gm.group(1)
            if not name:
                nm = re.search(r"[AB][12]\s+(.+?)\s+\S+/", info)
                name = _norm(nm.group(1)) if nm else f"選手{racer_id}"

            st_cell = _norm(tds[3].get_text(" ", strip=True))
            f_count = _to_int(re.search(r"F(\d+)", st_cell).group(1)) if re.search(r"F(\d+)", st_cell) else None
            l_count = _to_int(re.search(r"L(\d+)", st_cell).group(1)) if re.search(r"L(\d+)", st_cell) else None
            avg_st = None
            stm = re.search(r"(\d\.\d{2})", st_cell)
            if stm:
                avg_st = float(stm.group(1))

            nat = _norm(tds[4].get_text(" ", strip=True)).split()
            loc = _norm(tds[5].get_text(" ", strip=True)).split()
            mot = _norm(tds[6].get_text(" ", strip=True)).split()
            boat = _norm(tds[7].get_text(" ", strip=True)).split()

            entry = {
                "waku": waku,
                "racer_id": racer_id,
                "racer_name": name,
                "grade": grade,
                "age": age,
                "weight": weight,
                "f_count": f_count,
                "l_count": l_count,
                "avg_st": avg_st,
                "national_win_rate": _to_float(nat[0]) if len(nat) > 0 else None,
                "national_quinella_rate": _to_float(nat[1]) if len(nat) > 1 else None,
                "national_trio_rate": _to_float(nat[2]) if len(nat) > 2 else None,
                "local_win_rate": _to_float(loc[0]) if len(loc) > 0 else None,
                "local_quinella_rate": _to_float(loc[1]) if len(loc) > 1 else None,
                "local_trio_rate": _to_float(loc[2]) if len(loc) > 2 else None,
                "motor_no": _to_int(mot[0]) if len(mot) > 0 else None,
                "motor_quinella_rate": _to_float(mot[1]) if len(mot) > 1 else None,
                "boat_no": _to_int(boat[0]) if len(boat) > 0 else None,
                "boat_quinella_rate": _to_float(boat[1]) if len(boat) > 1 else None,
            }
            entries.append(entry)

        # 重複枠を除去し1-6のみ
        by_waku = {e["waku"]: e for e in entries if 1 <= e["waku"] <= 6}
        entries = [by_waku[w] for w in sorted(by_waku)]

        title_el = soup.select_one("h3") or soup.select_one(".heading2_titleName")
        title = title_el.get_text(strip=True) if title_el else f"{rno}R"
        return {
            "venue_id": venue_id,
            "race_no": rno,
            "race_title": title,
            "entries": entries,
            "is_fixed_entry": "進入固定" in soup.get_text(),
        }

    def _parse_beforeinfo(self, html: str) -> dict[str, Any]:
        soup = _parse_soup(html)
        exhibition: dict[int, dict[str, Any]] = {}
        weather: dict[str, Any] = {}

        # 天候ブロック
        weather_text = ""
        root = soup.select_one(".weather1_body") or soup.select_one(".weather1")
        if root:
            weather_text = root.get_text("\n", strip=True)
        else:
            weather_text = soup.get_text("\n", strip=True)

        wm = re.search(r"気温\s*([\d.]+)", weather_text)
        if wm:
            weather["temperature"] = float(wm.group(1))
        wm = re.search(r"水温\s*([\d.]+)", weather_text)
        if wm:
            weather["water_temperature"] = float(wm.group(1))
        wm = re.search(r"風速\s*([\d.]+)", weather_text)
        if wm:
            weather["wind_speed"] = float(wm.group(1))
        wm = re.search(r"波高\s*([\d.]+)", weather_text)
        if wm:
            weather["wave_height"] = float(wm.group(1))
        for label in ["晴", "曇り", "曇", "雨", "雪", "霧"]:
            if label in weather_text:
                weather["weather"] = label
                break

        # 風向: 画像ファイル名や alt から推定（取れなければ空）
        for img in soup.select(".weather1_body img, .weather1 img"):
            src = img.get("src") or ""
            alt = img.get("alt") or ""
            if "向" in alt or "追" in alt or "横" in alt:
                weather["wind_direction"] = alt
            elif "corner" in src:
                # 公式は風向を画像で示すことが多い。番号から粗い分類はしない。
                weather["wind_direction"] = weather.get("wind_direction") or "不明"

        # 展示タイム行
        for tr in soup.select("table.is-w748 tbody tr"):
            tds = [_norm(td.get_text(strip=True)) for td in tr.find_all("td")]
            if len(tds) < 5:
                continue
            waku = _to_int(tds[0])
            if not waku or not (1 <= waku <= 6):
                continue
            ex: dict[str, Any] = {"waku": waku}
            # 典型: 枠, 写真, 名前, 体重, 展示, チルト, ...
            for cell in tds:
                if re.fullmatch(r"\d\.\d{2}", cell):
                    ex["exhibition_time"] = float(cell)
                elif re.fullmatch(r"-?\d\.\d", cell):
                    v = float(cell)
                    if -1.5 <= v <= 3.5:
                        ex["tilt"] = v
            # 部品交換列
            for cell in tds:
                if any(k in cell for k in ("交換", "電気", "キャブ", "ピストン", "リング")):
                    ex["parts_changed"] = cell
            exhibition[waku] = ex

        return {"exhibition": exhibition, "weather": weather}

    def _parse_result(self, html: str) -> dict[str, Any] | None:
        soup = _parse_soup(html)
        text = soup.get_text(" ", strip=True)
        if "レース中止" in text:
            return None

        entry_results: list[dict[str, Any]] = []
        ranks: dict[int, int] = {}

        # 着順テーブル
        for tr in soup.select("table tbody tr"):
            tds = [_norm(td.get_text(strip=True)) for td in tr.find_all("td")]
            if len(tds) < 3:
                continue
            rank = _to_int(tds[0])
            waku = _to_int(tds[1])
            if not rank or not waku:
                continue
            if not (1 <= rank <= 6 and 1 <= waku <= 6):
                continue
            st = None
            for cell in tds:
                if re.fullmatch(r"0\.\d{2}", cell):
                    st = float(cell)
                elif cell == "F":
                    st = 0.0
            course = None
            # コース列がある場合
            for cell in tds[2:5]:
                c = _to_int(cell)
                if c and 1 <= c <= 6:
                    course = c
                    break
            entry_results.append(
                {"waku": waku, "rank": rank, "st": st, "course": course or waku}
            )
            ranks[rank] = waku

        if len(ranks) < 3:
            return None

        kimarite = None
        m = re.search(r"(逃げ|差し|まくり|まくり差し|抜き|恵まれ)", text)
        if m:
            kimarite = m.group(1)

        return {
            "rank1_waku": ranks.get(1),
            "rank2_waku": ranks.get(2),
            "rank3_waku": ranks.get(3),
            "kimarite": kimarite,
            "entry_results": entry_results,
        }

    def _merge_before(self, card: dict[str, Any], before: dict[str, Any]) -> None:
        exhibition = before.get("exhibition") or {}
        for e in card.get("entries", []):
            ex = exhibition.get(e["waku"])
            if not ex:
                continue
            e.update({k: v for k, v in ex.items() if k != "waku" and v is not None})
        if before.get("weather"):
            card["weather"] = before["weather"]

    def _ensure_racer(self, session: Any, racer_id: str, name: str, grade: str = "") -> None:
        racer = session.get(Racer, racer_id)
        if racer is None:
            session.add(Racer(id=racer_id, name=name or f"選手{racer_id}", grade=grade or ""))
            session.flush()
        else:
            if name:
                racer.name = name
            if grade:
                racer.grade = grade

    def _upsert_card(
        self, race_date: date, venue_id: str, rno: int, data: dict[str, Any]
    ) -> tuple[int, int]:
        with session_scope() as session:
            card = (
                session.query(RaceCard)
                .filter_by(venue_id=venue_id, race_date=race_date, race_no=rno)
                .one_or_none()
            )
            now = datetime.utcnow()
            if card is None:
                card = RaceCard(venue_id=venue_id, race_date=race_date, race_no=rno)
                session.add(card)
                session.flush()
                card_count = 1
            else:
                card_count = 0

            card.race_title = data.get("race_title") or card.race_title
            card.is_fixed_entry = bool(data.get("is_fixed_entry"))
            card.fetched_at = now
            card.updated_at = now
            card.raw_payload = {"entries_count": len(data.get("entries", []))}

            entry_count = 0
            for ed in data.get("entries", []):
                racer_id = str(ed["racer_id"])[:4]
                self._ensure_racer(
                    session,
                    racer_id,
                    ed.get("racer_name") or f"選手{racer_id}",
                    ed.get("grade") or "",
                )

                entry = (
                    session.query(RaceEntry)
                    .filter_by(race_card_id=card.id, waku=ed["waku"])
                    .one_or_none()
                )
                if entry is None:
                    entry = RaceEntry(race_card_id=card.id, waku=ed["waku"], racer_id=racer_id)
                    session.add(entry)
                    entry_count += 1
                else:
                    entry.racer_id = racer_id

                for field in [
                    "avg_st",
                    "national_win_rate",
                    "national_quinella_rate",
                    "national_trio_rate",
                    "local_win_rate",
                    "local_quinella_rate",
                    "local_trio_rate",
                    "motor_no",
                    "motor_quinella_rate",
                    "motor_trio_rate",
                    "boat_no",
                    "boat_quinella_rate",
                    "boat_trio_rate",
                    "exhibition_time",
                    "exhibition_st",
                    "tilt",
                    "weight_adjustment",
                    "parts_changed",
                    "previous_rank",
                    "previous_st",
                    "previous_course",
                    "estimated_course",
                    "age",
                    "weight",
                    "f_count",
                    "l_count",
                    "grade_code",
                    "win_odds",
                ]:
                    if ed.get(field) is not None:
                        setattr(entry, field, ed[field])
                if ed.get("grade") and not ed.get("grade_code"):
                    entry.grade_code = ed["grade"]
                if ed.get("parts_changed"):
                    entry.parts_changed_flag = True
                    entry.parts_changed = ed["parts_changed"]
                elif ed.get("parts_changed_flag") is not None:
                    entry.parts_changed_flag = bool(ed["parts_changed_flag"])
                entry.updated_at = now

            weather = data.get("weather")
            if weather:
                ws = (
                    session.query(WeatherSnapshot)
                    .filter_by(race_card_id=card.id)
                    .one_or_none()
                )
                if ws is None:
                    ws = WeatherSnapshot(race_card_id=card.id)
                    session.add(ws)
                for k in [
                    "temperature",
                    "weather",
                    "wind_speed",
                    "wind_direction",
                    "water_temperature",
                    "wave_height",
                ]:
                    if weather.get(k) is not None:
                        setattr(ws, k, weather[k])
                ws.fetched_at = now
                ws.updated_at = now

            return card_count, entry_count

    def _upsert_result(
        self, race_date: date, venue_id: str, rno: int, result: dict[str, Any]
    ) -> int:
        with session_scope() as session:
            card = (
                session.query(RaceCard)
                .filter_by(venue_id=venue_id, race_date=race_date, race_no=rno)
                .one_or_none()
            )
            if card is None:
                return 0
            rr = session.query(RaceResult).filter_by(race_card_id=card.id).one_or_none()
            now = datetime.utcnow()
            created = 0
            if rr is None:
                rr = RaceResult(race_card_id=card.id)
                session.add(rr)
                created = 1
            rr.rank1_waku = result.get("rank1_waku")
            rr.rank2_waku = result.get("rank2_waku")
            rr.rank3_waku = result.get("rank3_waku")
            rr.kimarite = result.get("kimarite")
            rr.entry_results = result.get("entry_results")
            rr.fetched_at = now
            rr.updated_at = now
            card.status = "finished"
            card.updated_at = now
            return created
