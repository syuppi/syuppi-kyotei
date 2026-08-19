"""候補順位（1〜5番手）ごとの過去的中率.

UI向けのガイド指標。ホールドアウト検証の基準値を持ち、
保存済み予想が十分あればライブ集計も返す。
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from sqlalchemy.orm import Session

from boatrace.prediction.timing import is_hit_verifiable

# 2026-07-30〜08-05 / 現行5点カバー再予想（展示あり）
BASELINE_TICKET_RANK_STATS: dict[str, Any] = {
    "source": "baseline",
    "label": "検証ベースライン（5点カバー）",
    "period": {"start": "2026-07-30", "end": "2026-08-05"},
    "n_races": 1020,
    "note": "候補のうち当該順位の点が的中した割合。全体の「いずれか的中」とは別指標。",
    "sanrenpuku": {
        "any_rate": 0.626,
        "ranks": [
            {"rank": 1, "hit_rate": 0.228, "label": "本命"},
            {"rank": 2, "hit_rate": 0.153, "label": "2番手"},
            {"rank": 3, "hit_rate": 0.124, "label": "3番手"},
            {"rank": 4, "hit_rate": 0.070, "label": "4番手"},
            {"rank": 5, "hit_rate": 0.052, "label": "5番手"},
        ],
    },
    "sanrentan": {
        "any_rate": 0.217,
        "ranks": [
            {"rank": 1, "hit_rate": 0.065, "label": "本命"},
            {"rank": 2, "hit_rate": 0.045, "label": "2番手"},
            {"rank": 3, "hit_rate": 0.048, "label": "3番手"},
            {"rank": 4, "hit_rate": 0.027, "label": "4番手"},
            {"rank": 5, "hit_rate": 0.031, "label": "5番手"},
        ],
    },
    "pre_exhibition": {
        "n_races": 867,
        "note": "同じ期間で展示を消した展示前モード",
        "sanrenpuku": {
            "any_rate": 0.627,
            "ranks": [
                {"rank": 1, "hit_rate": 0.232},
                {"rank": 2, "hit_rate": 0.158},
                {"rank": 3, "hit_rate": 0.127},
                {"rank": 4, "hit_rate": 0.053},
                {"rank": 5, "hit_rate": 0.058},
            ],
        },
        "sanrentan": {
            "any_rate": 0.218,
            "ranks": [
                {"rank": 1, "hit_rate": 0.077},
                {"rank": 2, "hit_rate": 0.053},
                {"rank": 3, "hit_rate": 0.045},
                {"rank": 4, "hit_rate": 0.023},
                {"rank": 5, "hit_rate": 0.020},
            ],
        },
    },
}


def _empty_counters() -> dict[str, Any]:
    return {
        "n": 0,
        "any": 0,
        "ranks": {i: {"offered": 0, "hit": 0} for i in range(1, 6)},
    }


def compute_live_ticket_rank_stats(
    session: Session,
    *,
    days: int = 14,
    model_name: str = "lgbm_v1",
) -> dict[str, Any] | None:
    """PredictHistory から候補順位別的中を集計。サンプル不足なら None."""
    since = date.today() - timedelta(days=max(1, days))
    cards = (
        session.query(RaceCard)
        .join(RaceResult)
        .filter(RaceCard.race_date >= since, RaceResult.rank1_waku.isnot(None))
        .all()
    )
    trio = _empty_counters()
    tf = _empty_counters()
    min_d: date | None = None
    max_d: date | None = None

    for card in cards:
        pred = (
            session.query(PredictHistory)
            .filter_by(race_card_id=card.id, model_name=model_name)
            .order_by(PredictHistory.predicted_at.desc())
            .first()
        )
        if not pred:
            continue
        if not is_hit_verifiable(card, pred):
            continue
        tickets = (pred.feature_snapshot or {}).get("tickets") or {}
        sps = [t for t in (tickets.get("sanrenpuku") or []) if t.get("combo")][:5]
        sts = [t for t in (tickets.get("sanrentan") or []) if t.get("combo")][:5]
        if not sps and not sts:
            continue

        true3 = {card.result.rank1_waku, card.result.rank2_waku, card.result.rank3_waku}
        true_ord = [card.result.rank1_waku, card.result.rank2_waku, card.result.rank3_waku]
        min_d = card.race_date if min_d is None else min(min_d, card.race_date)
        max_d = card.race_date if max_d is None else max(max_d, card.race_date)

        if sps and None not in true3:
            trio["n"] += 1
            hit_any = False
            for i, t in enumerate(sps, 1):
                trio["ranks"][i]["offered"] += 1
                if set(t["combo"]) == true3:
                    trio["ranks"][i]["hit"] += 1
                    hit_any = True
            if hit_any:
                trio["any"] += 1

        if sts and true_ord[1] and true_ord[2]:
            tf["n"] += 1
            hit_any = False
            for i, t in enumerate(sts, 1):
                tf["ranks"][i]["offered"] += 1
                if list(t["combo"]) == true_ord:
                    tf["ranks"][i]["hit"] += 1
                    hit_any = True
            if hit_any:
                tf["any"] += 1

    if trio["n"] < 80 and tf["n"] < 80:
        return None

    def pack(block: dict[str, Any], labels: dict[int, str]) -> dict[str, Any]:
        n = int(block["n"])
        ranks = []
        for i in range(1, 6):
            off = int(block["ranks"][i]["offered"])
            hit = int(block["ranks"][i]["hit"])
            ranks.append(
                {
                    "rank": i,
                    "label": labels.get(i, f"{i}番手"),
                    "offered": off,
                    "hit": hit,
                    "hit_rate": (hit / off) if off else 0.0,
                }
            )
        return {
            "n_races": n,
            "any_rate": (int(block["any"]) / n) if n else 0.0,
            "ranks": ranks,
        }

    labels = {1: "本命", 2: "2番手", 3: "3番手", 4: "4番手", 5: "5番手"}
    return {
        "source": "live",
        "label": f"直近{days}日の保存予想",
        "period": {
            "start": min_d.isoformat() if min_d else None,
            "end": max_d.isoformat() if max_d else None,
        },
        "n_races": max(int(trio["n"]), int(tf["n"])),
        "note": "保存済み予想の候補順位別的中。候補点数はレースにより異なる場合あり。",
        "sanrenpuku": pack(trio, labels),
        "sanrentan": pack(tf, labels),
    }


def get_ticket_rank_stats(session: Session | None = None, *, days: int = 14) -> dict[str, Any]:
    """UI用: ベースライン + 可能ならライブ集計."""
    out: dict[str, Any] = {
        "baseline": BASELINE_TICKET_RANK_STATS,
        "display": BASELINE_TICKET_RANK_STATS,
    }
    if session is None:
        return out
    live = compute_live_ticket_rank_stats(session, days=days)
    if live:
        out["live"] = live
        # 5点カバーが十分あるときだけライブを主表示に
        trio_ranks = live.get("sanrenpuku", {}).get("ranks") or []
        has_r5 = any(r.get("rank") == 5 and int(r.get("offered") or 0) >= 80 for r in trio_ranks)
        if has_r5:
            out["display"] = live
    return out


def rank_hit_rate_map(stats: dict[str, Any] | None = None) -> dict[str, dict[int, float]]:
    """kind -> rank -> hit_rate."""
    src = (stats or BASELINE_TICKET_RANK_STATS).get("display") or stats or BASELINE_TICKET_RANK_STATS
    if "sanrenpuku" not in src and "baseline" in (stats or {}):
        src = (stats or {})["display"]
    out: dict[str, dict[int, float]] = {"sanrenpuku": {}, "sanrentan": {}}
    for kind in ("sanrenpuku", "sanrentan"):
        for row in (src.get(kind) or {}).get("ranks") or []:
            out[kind][int(row["rank"])] = float(row.get("hit_rate") or 0)
    return out
