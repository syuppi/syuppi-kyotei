#!/usr/bin/env python3
"""結果学習."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.cli.app import learn

if __name__ == "__main__":
    learn()
