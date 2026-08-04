"""過去成績からの高速集計（リーク防止: as_of_date より前のみ）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from sqlalchemy.orm import Session, joinedload

from boatrace.db.models import RaceCard, RaceEntry, RaceResult
from boatrace.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class RacerRaceRecord:
    race_date: date
    course: int
    rank: int
    st: float | None
    venue_id: str
    motor_no: int | None
    race_no: int = 0


@dataclass
class VenueRaceRecord:
    race_date: date
    kimarite: str | None
    rank1_course: int | None


@dataclass
class HistoryIndex:
    """セッション内で再利用する過去成績インデックス."""

    racer_races: dict[str, list[RacerRaceRecord]] = field(default_factory=dict)
    venue_races: dict[str, list[VenueRaceRecord]] = field(default_factory=dict)
    motor_races: dict[tuple[str, int], list[tuple[date, int]]] = field(default_factory=dict)
    built: bool = False

    def ensure_built(self, session: Session) -> None:
        if self.built:
            return
        logger.info("history_index_build_start")
        cards = (
            session.query(RaceCard)
            .options(
                joinedload(RaceCard.result),
                joinedload(RaceCard.entries),
            )
            .filter(RaceCard.status == "finished")
            .all()
        )
        for card in cards:
            result = card.result
            if not result:
                continue
            er_by_waku: dict[int, dict[str, Any]] = {}
            for er in result.entry_results or []:
                w = int(er.get("waku") or 0)
                if w:
                    er_by_waku[w] = er
            rank1_course = None
            for er in er_by_waku.values():
                if int(er.get("rank") or 0) == 1:
                    rank1_course = int(er.get("course") or er.get("waku") or 0) or None
                    break
            if rank1_course is None and result.rank1_waku:
                rank1_course = int(result.rank1_waku)

            self.venue_races.setdefault(card.venue_id, []).append(
                VenueRaceRecord(
                    race_date=card.race_date,
                    kimarite=result.kimarite,
                    rank1_course=rank1_course,
                )
            )

            for entry in card.entries:
                er = er_by_waku.get(entry.waku) or {}
                course = int(er.get("course") or entry.estimated_course or entry.waku or 0)
                rank = int(er.get("rank") or 0)
                st = er.get("st")
                st_f = float(st) if st is not None else None
                if not entry.racer_id or not course or not rank:
                    continue
                rec = RacerRaceRecord(
                    race_date=card.race_date,
                    course=course,
                    rank=rank,
                    st=st_f,
                    venue_id=card.venue_id,
                    motor_no=entry.motor_no,
                    race_no=int(card.race_no or 0),
                )
                self.racer_races.setdefault(entry.racer_id, []).append(rec)
                if entry.motor_no is not None:
                    key = (card.venue_id, int(entry.motor_no))
                    self.motor_races.setdefault(key, []).append((card.race_date, rank))

        for rid in self.racer_races:
            self.racer_races[rid].sort(key=lambda r: (r.race_date, r.race_no))
        for vid in self.venue_races:
            self.venue_races[vid].sort(key=lambda r: r.race_date)
        for key in self.motor_races:
            self.motor_races[key].sort(key=lambda x: x[0])

        self.built = True
        logger.info(
            "history_index_built",
            racers=len(self.racer_races),
            venues=len(self.venue_races),
            motors=len(self.motor_races),
        )

    def last_start(
        self,
        racer_id: str,
        as_of: date,
        before_race_no: int | None = None,
    ) -> RacerRaceRecord | None:
        """as_of より前（同日なら before_race_no より前）の直近1走."""
        rows = self.racer_races.get(racer_id) or []
        past: list[RacerRaceRecord] = []
        for r in rows:
            if r.race_date < as_of:
                past.append(r)
            elif (
                r.race_date == as_of
                and before_race_no is not None
                and r.race_no < before_race_no
            ):
                past.append(r)
        return past[-1] if past else None

    def racer_course_stats(
        self, racer_id: str, course: int, as_of: date, lookback_days: int = 120
    ) -> dict[str, float]:
        """選手×コースの勝率・平均ST（shrink付き）。"""
        rows = self.racer_races.get(racer_id) or []
        start = as_of - timedelta(days=lookback_days)
        wins = starts = 0
        st_sum = 0.0
        st_n = 0
        recent_ranks: list[int] = []
        for r in rows:
            if r.race_date >= as_of or r.race_date < start:
                continue
            recent_ranks.append(r.rank)
            if r.course != course:
                continue
            starts += 1
            if r.rank == 1:
                wins += 1
            if r.st is not None:
                st_sum += r.st
                st_n += 1
        # 全国コース事前で shrink
        from boatrace.features.builder import GLOBAL_COURSE_WIN_PRIOR

        prior = GLOBAL_COURSE_WIN_PRIOR.get(course, 0.08)
        pseudo = 8.0
        win_rate = (wins + prior * pseudo) / (starts + pseudo)
        avg_st = (st_sum / st_n) if st_n else 0.18
        # 直近調子（コース不問の直近6走）
        last6 = recent_ranks[-6:] if recent_ranks else []
        if last6:
            form = sum(max(0.0, (7 - rk) / 6.0) for rk in last6) / len(last6)
        else:
            form = 0.4
        return {
            "course_win_rate": float(win_rate),
            "course_starts": float(starts),
            "course_avg_st": float(avg_st),
            "recent_form": float(form),
        }

    def venue_kimarite_stats(
        self, venue_id: str, as_of: date, lookback_days: int = 90
    ) -> dict[str, float]:
        rows = self.venue_races.get(venue_id) or []
        start = as_of - timedelta(days=lookback_days)
        n = nige = makuri = sashi = makurisashi = in_wins = 0
        for r in rows:
            if r.race_date >= as_of or r.race_date < start:
                continue
            n += 1
            kim = r.kimarite or ""
            if "逃" in kim:
                nige += 1
            elif "まくり差し" in kim:
                makurisashi += 1
            elif "まくり" in kim:
                makuri += 1
            elif "差" in kim:
                sashi += 1
            if r.rank1_course == 1:
                in_wins += 1
        if n <= 0:
            return {
                "nige_rate": 0.52,
                "makuri_rate": 0.15,
                "sashi_rate": 0.12,
                "makurisashi_rate": 0.12,
                "in_win_rate": 0.55,
                "kado_strength": 0.5,
                "n": 0.0,
            }
        # カド(4コース)強さの近似: まくり+まくり差し率
        kado = (makuri + makurisashi) / n
        return {
            "nige_rate": nige / n,
            "makuri_rate": makuri / n,
            "sashi_rate": sashi / n,
            "makurisashi_rate": makurisashi / n,
            "in_win_rate": in_wins / n,
            "kado_strength": kado,
            "n": float(n),
        }

    def motor_recent_quinella(
        self, venue_id: str, motor_no: int | None, as_of: date, last_n: int = 12
    ) -> float | None:
        if motor_no is None:
            return None
        rows = self.motor_races.get((venue_id, int(motor_no))) or []
        past = [(d, rk) for d, rk in rows if d < as_of][-last_n:]
        if len(past) < 3:
            return None
        q = sum(1 for _, rk in past if rk <= 2)
        return q / len(past)


_INDEX: HistoryIndex | None = None


def get_history_index(session: Session) -> HistoryIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = HistoryIndex()
    _INDEX.ensure_built(session)
    return _INDEX


def reset_history_index() -> None:
    global _INDEX
    _INDEX = None
