#!/usr/bin/env python3
"""起動前に日本時間の当日出走表を投入する（Render空DB対策）."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.jobs.today_bootstrap import ensure_today_cards  # noqa: E402
from boatrace.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    setup_logging()
    result = ensure_today_cards(force=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
