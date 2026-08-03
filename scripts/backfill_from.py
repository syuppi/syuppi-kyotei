#!/usr/bin/env python3
"""OpenAPIで取得可能な全期間（既定: 2026-01-01〜前日）をバックフィルする."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.cli.app import init_db
from boatrace.collectors.openapi import OpenApiCollector
from boatrace.collectors.tide import TideCollector
from boatrace.db.migrate import migrate
from boatrace.db.models import RaceCard
from boatrace.db.session import session_scope
from boatrace.logging_setup import get_logger, setup_logging
from sqlalchemy import func

logger = get_logger(__name__)


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def main() -> None:
    setup_logging()
    init_db()
    migrate()
    start = date(2026, 1, 1)
    if len(sys.argv) > 1:
        start = date.fromisoformat(sys.argv[1])
    end = date.today()
    openapi = OpenApiCollector()
    tide = TideCollector()
    ok = fail = 0
    for d in daterange(start, end):
        try:
            summary = openapi.collect(d)
            tide.collect(d, estimate_only=True)
            ok += 1
            if ok % 15 == 0:
                print(f"ok {d.isoformat()} days={ok} summary={summary}")
                logger.info("backfill_progress", date=d.isoformat(), ok=ok, summary=summary)
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"fail {d.isoformat()}: {e}")
            logger.warning("backfill_fail", date=d.isoformat(), error=str(e))

    with session_scope() as session:
        mn, mx = session.query(func.min(RaceCard.race_date), func.max(RaceCard.race_date)).one()
        n = session.query(RaceCard).count()
        n_fin = session.query(RaceCard).filter_by(status="finished").count()
    print({"ok": ok, "fail": fail, "min": str(mn), "max": str(mx), "cards": n, "finished": n_fin})


if __name__ == "__main__":
    main()
