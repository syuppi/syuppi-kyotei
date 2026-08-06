#!/usr/bin/env python3
"""展示あり vs 展示前モードの3連系的中率を同一レースで比較."""

from __future__ import annotations

import sys
from copy import deepcopy
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boatrace.db.models import RaceCard, RaceResult
from boatrace.db.session import session_scope
from boatrace.features.exhibition import exhibition_status
from boatrace.logging_setup import setup_logging
from boatrace.models.ml_model import MLPredictor


def _wipe_exhibition_raw(features):
    """DB由来の展示を消し、展示前状態を再現."""
    feats = deepcopy(features)
    for boat in feats.boats:
        boat.raw["exhibition_time"] = None
        boat.raw["exhibition_st"] = None
        boat.raw["ex_time_gap"] = 0.0
        boat.raw["ex_st_gap_vs_best"] = 0.0
        boat.raw["st_gap_vs_inner"] = 0.0
        boat.values["exhibition_advantage"] = 0.5
        boat.values["exhibition_st_advantage"] = 0.5
    env = dict(feats.env or {})
    env["exhibition_complete"] = False
    feats.env = env
    return feats


def _hit(result, card: RaceCard) -> dict[str, bool]:
    r = card.result
    tickets = result.tickets or (result.feature_snapshot or {}).get("tickets") or {}
    sps = [set(t["combo"]) for t in tickets.get("sanrenpuku", []) if t.get("combo")]
    sts = [list(t["combo"]) for t in tickets.get("sanrentan", []) if t.get("combo")]
    wins = [t["combo"][0] for t in tickets.get("win", []) if t.get("combo")]
    true3 = {r.rank1_waku, r.rank2_waku, r.rank3_waku}
    true_ord = [r.rank1_waku, r.rank2_waku, r.rank3_waku]
    return {
        "win": bool(wins) and r.rank1_waku in wins,
        "trio": None not in true3 and any(s == true3 for s in sps),
        "tf": bool(r.rank2_waku and r.rank3_waku) and any(s == true_ord for s in sts),
        "pre": bool((result.feature_snapshot or {}).get("pre_exhibition_mode")),
    }


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "win": sum(r["win"] for r in rows) / n,
        "trio": sum(r["trio"] for r in rows) / n,
        "tf": sum(r["tf"] for r in rows) / n,
        "pre_mode_frac": sum(r["pre"] for r in rows) / n,
    }


def main() -> None:
    setup_logging()
    start = date(2026, 7, 30)
    end = date(2026, 8, 5)
    post_rows: list[dict] = []
    pre_rows: list[dict] = []

    with session_scope() as session:
        predictor = MLPredictor(session=session)
        from boatrace.features.builder import FeatureBuilder

        builder = FeatureBuilder(session)
        cards = (
            session.query(RaceCard)
            .join(RaceResult)
            .filter(RaceCard.race_date >= start, RaceCard.race_date <= end)
            .order_by(RaceCard.race_date, RaceCard.venue_id, RaceCard.race_no)
            .all()
        )
        used = 0
        for card in cards:
            if not card.result or not card.result.rank1_waku:
                continue
            if len(card.entries or []) < 6:
                continue
            feats = builder.build(card.id)
            st = exhibition_status(feats)
            if not st.get("complete"):
                continue
            used += 1

            post = predictor.predict(deepcopy(feats))
            post_rows.append(_hit(post, card))

            # 正しい展示前モード（中立化 + 重み再配分）
            wiped = _wipe_exhibition_raw(feats)
            pre = predictor.predict(wiped)
            pre_rows.append(_hit(pre, card))

            # 参考: 生 wipe のみ（旧リーク経路の有無確認用）— neutralize を明示適用しない
            # predictor.predict 内で prepare が動くので pre と同義になる。差分確認はスキップ。

            if used % 100 == 0:
                print(f"... {used} races", flush=True)

        print("range", start.isoformat(), "~", end.isoformat())
        print("with_complete_exhibition", used)
        print("POST exhibition:", _agg(post_rows))
        print("PRE  exhibition:", _agg(pre_rows))
        # 差分
        p, q = _agg(post_rows), _agg(pre_rows)
        if p["n"] and q["n"]:
            print(
                "delta (pre-post):",
                {
                    "win": q["win"] - p["win"],
                    "trio": q["trio"] - p["trio"],
                    "tf": q["tf"] - p["tf"],
                },
            )


if __name__ == "__main__":
    main()
