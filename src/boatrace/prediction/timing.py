"""予想時刻とレース締切の整合 — 結果後再予想の「偽的中」表示を防ぐ."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from boatrace.timeutil import JST

# 締切時刻と predicted_at の多少のズレ（秒）
_DEADLINE_GRACE = timedelta(minutes=3)


def _as_jst(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(JST)


def _deadline_jst(card: Any) -> datetime | None:
    raw = getattr(card, "deadline_at", None)
    if raw is None:
        return None
    if raw.tzinfo is None:
        return raw.replace(tzinfo=JST)
    return raw.astimezone(JST)


def classify_prediction_timing(card: Any, pred: Any) -> dict[str, Any]:
    """
    予想が締切前かどうかを判定する。

    Returns:
        status: pre_close | post_close | no_deadline | no_prediction_time | pending
        hit_verifiable: 的中表示・精度集計に使ってよいか
    """
    predicted_at = getattr(pred, "predicted_at", None) if pred is not None else None
    pred_jst = _as_jst(predicted_at)
    deadline = _deadline_jst(card)
    status = getattr(card, "status", None) or "scheduled"
    has_result = bool(
        getattr(card, "result", None) is not None
        and getattr(card.result, "rank1_waku", None) is not None
    )

    out: dict[str, Any] = {
        "predicted_at": predicted_at.isoformat() if predicted_at else None,
        "deadline_at": deadline.isoformat() if deadline else None,
        "race_status": status,
        "has_result": has_result,
    }

    if pred is None or pred_jst is None:
        out.update(
            {
                "status": "no_prediction_time",
                "hit_verifiable": False,
                "message": "予想時刻が記録されていないため、的中表示は行いません。",
            }
        )
        return out

    if not has_result and status != "finished":
        out.update(
            {
                "status": "pending",
                "hit_verifiable": True,
                "message": "レース未確定。確定後に締切前予想のみ的中として表示します。",
            }
        )
        return out

    if deadline is None:
        out.update(
            {
                "status": "no_deadline",
                "hit_verifiable": False,
                "message": "締切時刻が不明なため、結果があるレースの的中表示は行いません（再予想の見かけ的中を防ぐため）。",
            }
        )
        return out

    if pred_jst <= deadline + _DEADLINE_GRACE:
        out.update(
            {
                "status": "pre_close",
                "hit_verifiable": True,
                "message": "締切前の予想です。的中表示は有効です。",
            }
        )
        return out

    out.update(
        {
            "status": "post_close",
            "hit_verifiable": False,
            "message": "結果確定後（または締切後）の再予想です。的中表示は参考外です。",
        }
    )
    return out


def is_hit_verifiable(card: Any, pred: Any) -> bool:
    return bool(classify_prediction_timing(card, pred).get("hit_verifiable"))
