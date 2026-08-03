#!/usr/bin/env python3
"""本日予測."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.cli.app import predict

if __name__ == "__main__":
    predict()
