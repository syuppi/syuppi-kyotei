#!/usr/bin/env python3
"""保存予想の公表整合性監査（締切前/後の的中率を分離）."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.db.session import session_scope
from boatrace.logging_setup import setup_logging
from boatrace.prediction.integrity_audit import audit_predictions


def main() -> None:
    setup_logging()
    day = None
    days = 14
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--day" and i + 2 <= len(sys.argv[1:]):
            day = datetime.strptime(sys.argv[i + 2], "%Y-%m-%d").date()
        if arg == "--days" and i + 2 <= len(sys.argv[1:]):
            days = int(sys.argv[i + 2])

    with session_scope() as session:
        report = audit_predictions(session, days=days, day=day)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
