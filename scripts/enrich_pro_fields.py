#!/usr/bin/env python3
"""既存DBへ不足カラムを追加し、OpenAPIからプロ視点項目を再取込する."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.collectors.openapi import OpenApiCollector
from boatrace.db.migrate import migrate
from boatrace.db.models import RaceCard, RaceEntry
from boatrace.db.session import session_scope
from boatrace.logging_setup import setup_logging
from sqlalchemy import func


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def main() -> None:
    setup_logging()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    print("migrate:", migrate())

    today = date.today()
    start = today - timedelta(days=days)
    openapi = OpenApiCollector()
    ok = fail = 0
    for d in daterange(start, today):
        try:
            summary = openapi.collect(d)
            ok += 1
            if ok % 10 == 0:
                print(f"ingest {d.isoformat()} ok={ok} summary={summary}")
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"ingest fail {d.isoformat()}: {e}")

    with session_scope() as session:
        total = session.query(RaceEntry).count()
        stats = {
            "entries": total,
            "exhibition_st": session.query(RaceEntry)
            .filter(RaceEntry.exhibition_st.isnot(None))
            .count(),
            "motor_trio": session.query(RaceEntry)
            .filter(RaceEntry.motor_trio_rate.isnot(None))
            .count(),
            "grade_code": session.query(RaceEntry)
            .filter(RaceEntry.grade_code.isnot(None))
            .count(),
            "weight_adj": session.query(RaceEntry)
            .filter(RaceEntry.weight_adjustment.isnot(None))
            .count(),
            "win_odds": session.query(RaceEntry)
            .filter(RaceEntry.win_odds.isnot(None))
            .count(),
            "grade_number_cards": session.query(RaceCard)
            .filter(RaceCard.grade_number.isnot(None))
            .count(),
            "date_range": session.query(
                func.min(RaceCard.race_date), func.max(RaceCard.race_date)
            ).one(),
        }
    print("ingest_days", {"ok": ok, "fail": fail})
    print("fill_stats", stats)


if __name__ == "__main__":
    main()
