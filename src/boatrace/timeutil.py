"""日本時間（JST）基準の日付ヘルパー.

Render など UTC ホストでも、ボートレースの「本日」は JST で揃える。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))


def japan_now() -> datetime:
    return datetime.now(JST)


def japan_today() -> date:
    return japan_now().date()
