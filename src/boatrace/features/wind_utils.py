"""風向ユーティリティ: 絶対方位 → 相対（追/向/横）."""

from __future__ import annotations

import math
import re

# 日本語方位 → 度（北=0, 東=90）
_COMPASS_DEG: dict[str, float] = {
    "北": 0.0,
    "北北東": 22.5,
    "北東": 45.0,
    "東北東": 67.5,
    "東": 90.0,
    "東南東": 112.5,
    "南東": 135.0,
    "南南東": 157.5,
    "南": 180.0,
    "南南西": 202.5,
    "南西": 225.0,
    "西南西": 247.5,
    "西": 270.0,
    "西北西": 292.5,
    "北西": 315.0,
    "北北西": 337.5,
}


def parse_wind_degrees(direction: str | None) -> float | None:
    """絶対方位文字列を度に。相対表記や不明は None."""
    if not direction:
        return None
    d = str(direction).strip()
    if any(x in d for x in ("向", "追", "横")):
        return None
    # 長いキーから優先マッチ
    for key in sorted(_COMPASS_DEG.keys(), key=len, reverse=True):
        if key in d:
            return _COMPASS_DEG[key]
    # 数字のみ
    m = re.search(r"(\d+(?:\.\d)?)", d)
    if m:
        return float(m.group(1)) % 360.0
    return None


def relative_angle_deg(wind_from_deg: float, course_heading_deg: float) -> float:
    """風の吹来方位とホームストレッチ進行方位の差（0-180）."""
    # 追い風: 風が進行方向と同じ側から吹く ≈ wind_from が course+180 付近
    # 風向「北」=北から吹く → 南向きに進む艇が追い風
    tail_from = (course_heading_deg + 180.0) % 360.0
    delta = abs(wind_from_deg - tail_from) % 360.0
    if delta > 180.0:
        delta = 360.0 - delta
    return delta


def absolute_to_relative_label(
    direction: str | None,
    course_heading_deg: float | None,
) -> str | None:
    """絶対方位を追風/向風/横風ラベルに。変換不可なら None."""
    if course_heading_deg is None:
        return None
    deg = parse_wind_degrees(direction)
    if deg is None:
        return None
    # delta≈0 → 追い風、delta≈180 → 向かい風
    delta = relative_angle_deg(deg, float(course_heading_deg))
    if delta <= 45.0:
        return "追"
    if delta >= 135.0:
        return "向"
    return "横"


def wind_components(direction: str | None) -> tuple[float, float]:
    """絶対風向の cos/sin（北基準）。不明は (0,0)."""
    deg = parse_wind_degrees(direction)
    if deg is None:
        return 0.0, 0.0
    rad = math.radians(deg)
    return math.cos(rad), math.sin(rad)
