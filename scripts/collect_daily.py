#!/usr/bin/env python3
"""当日データ収集."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.cli.app import collect

if __name__ == "__main__":
    collect()
