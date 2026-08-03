"""場×コースの上位3着共起（リーク防止付き）."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from boatrace.db.models import RaceCard, RaceResult


def _course_of(entry_results: list[dict[str, Any]] | None, waku: int | None) -> int | None:
    if not waku:
        return None
    for er in entry_results or []:
        if int(er.get("waku") or 0) == int(waku):
            return int(er.get("course") or er.get("waku") or 0) or None
    return int(waku)


def build_venue_trio_prior(
    session: Session,
    venue_id: str,
    as_of_date: date | None,
    *,
    min_samples: int = 30,
) -> dict[frozenset[int], float]:
    """
    場における上位3着の「枠番セット」頻度を事前分布にする。
    as_of_date より前の確定結果のみ使用。
    """
    q = (
        session.query(RaceCard, RaceResult)
        .join(RaceResult, RaceResult.race_card_id == RaceCard.id)
        .filter(RaceCard.venue_id == venue_id, RaceCard.status == "finished")
    )
    if as_of_date is not None:
        q = q.filter(RaceCard.race_date < as_of_date)

    counts: dict[frozenset[int], int] = defaultdict(int)
    n = 0
    for _card, result in q.all():
        top = {result.rank1_waku, result.rank2_waku, result.rank3_waku}
        if None in top or len(top) < 3:
            continue
        counts[frozenset(int(x) for x in top)] += 1
        n += 1

    if n < min_samples:
        return {}

    # Laplace平滑化
    all_keys = list(counts.keys())
    total = n + len(all_keys)
    return {k: (counts[k] + 1) / total for k in all_keys}


def build_venue_course_trio_prior(
    session: Session,
    venue_id: str,
    as_of_date: date | None,
    *,
    min_samples: int = 40,
) -> dict[frozenset[int], float]:
    """上位3着の進入コースセット頻度。"""
    q = (
        session.query(RaceCard, RaceResult)
        .join(RaceResult, RaceResult.race_card_id == RaceCard.id)
        .filter(RaceCard.venue_id == venue_id, RaceCard.status == "finished")
    )
    if as_of_date is not None:
        q = q.filter(RaceCard.race_date < as_of_date)

    counts: dict[frozenset[int], int] = defaultdict(int)
    n = 0
    for _card, result in q.all():
        courses = []
        for waku in (result.rank1_waku, result.rank2_waku, result.rank3_waku):
            c = _course_of(result.entry_results, waku)
            if c:
                courses.append(c)
        if len(set(courses)) < 3:
            continue
        counts[frozenset(courses)] += 1
        n += 1

    if n < min_samples:
        return {}
    total = n + len(counts)
    return {k: (v + 1) / total for k, v in counts.items()}
