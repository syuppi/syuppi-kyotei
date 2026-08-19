"""公表精度・予想の整合性監査（締切前予想のみを有効とする）."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from sqlalchemy.orm import Session, joinedload

from boatrace.db.models import PredictHistory, RaceCard, RaceResult
from boatrace.prediction.review import compute_hit_flags
from boatrace.prediction.timing import classify_prediction_timing
from boatrace.prediction.offline_eval import get_holdout_reference
from boatrace.prediction.ticket_rank_stats import get_baseline_stats


def _empty_counters() -> dict[str, Any]:
    return {
        "n": 0,
        "any": 0,
        "ranks": {i: {"offered": 0, "hit": 0} for i in range(1, 6)},
    }


def _ticket_rank_block(
    session: Session,
    cards: list[RaceCard],
    *,
    model_name: str = "lgbm_v1",
) -> dict[str, Any]:
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
        if not pred or not card.result or not card.result.rank1_waku:
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

    labels = {1: "本命", 2: "2番手", 3: "3番手", 4: "4番手", 5: "5番手"}

    def pack(block: dict[str, Any]) -> dict[str, Any]:
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

    return {
        "period": {
            "start": min_d.isoformat() if min_d else None,
            "end": max_d.isoformat() if max_d else None,
        },
        "sanrenpuku": pack(trio),
        "sanrentan": pack(tf),
    }


def _race_hit_summary(card: RaceCard, pred: PredictHistory) -> dict[str, Any] | None:
    if not card.result or not card.result.rank1_waku:
        return None
    tickets = (pred.feature_snapshot or {}).get("tickets") or {}
    hits = compute_hit_flags(
        {
            "rank1": card.result.rank1_waku,
            "rank2": card.result.rank2_waku,
            "rank3": card.result.rank3_waku,
        },
        tickets,
    )
    conf = (pred.feature_snapshot or {}).get("confidence") or {}
    return {
        "race_card_id": card.id,
        "race_date": card.race_date.isoformat(),
        "venue_id": card.venue_id,
        "race_no": card.race_no,
        "timing": classify_prediction_timing(card, pred),
        "is_confident": bool(conf.get("is_confident")),
        "hit_win": hits["hit_win"],
        "hit_trio": hits["hit_trio"],
        "hit_tf": hits["hit_tf"],
        "any_hit": hits["any_hit"],
    }


def audit_predictions(
    session: Session,
    *,
    days: int = 14,
    day: date | None = None,
    model_name: str = "lgbm_v1",
) -> dict[str, Any]:
    """保存予想を締切前/後に分類し、的中率を再集計する."""
    if day is not None:
        since = day
        until = day
    else:
        until = date.today()
        since = until - timedelta(days=max(1, days) - 1)

    cards = (
        session.query(RaceCard)
        .options(joinedload(RaceCard.result))
        .filter(RaceCard.race_date >= since, RaceCard.race_date <= until)
        .order_by(RaceCard.race_date, RaceCard.venue_id, RaceCard.race_no)
        .all()
    )

    timing_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    verifiable_cards: list[RaceCard] = []
    all_with_pred: list[tuple[RaceCard, PredictHistory]] = []

    for card in cards:
        pred = (
            session.query(PredictHistory)
            .filter_by(race_card_id=card.id, model_name=model_name)
            .order_by(PredictHistory.predicted_at.desc())
            .first()
        )
        if not pred:
            continue
        timing = classify_prediction_timing(card, pred)
        status = str(timing.get("status") or "unknown")
        timing_counts[status] = timing_counts.get(status, 0) + 1
        summary = _race_hit_summary(card, pred)
        if summary:
            rows.append(summary)
        all_with_pred.append((card, pred))
        if timing.get("hit_verifiable"):
            verifiable_cards.append(card)

    def agg(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        if n == 0:
            return {"n": 0, "win_rate": 0.0, "trio_rate": 0.0, "trifecta_rate": 0.0, "any_rate": 0.0}
        return {
            "n": n,
            "win_rate": sum(int(x["hit_win"]) for x in items) / n,
            "trio_rate": sum(int(x["hit_trio"]) for x in items) / n,
            "trifecta_rate": sum(int(x["hit_tf"]) for x in items) / n,
            "combo_any_rate": sum(int(x["any_hit"]) for x in items) / n,
            "any_rate": sum(int(x["any_hit"]) for x in items) / n,
            "confident_n": sum(int(x["is_confident"]) for x in items),
            "confident_combo_any_rate": (
                sum(int(x["any_hit"]) for x in items if x["is_confident"])
                / max(1, sum(int(x["is_confident"]) for x in items))
            ),
        }

    verifiable_rows = [r for r in rows if (r.get("timing") or {}).get("hit_verifiable")]
    post_close_rows = [r for r in rows if (r.get("timing") or {}).get("status") == "post_close"]
    unverified_rows = [r for r in rows if not (r.get("timing") or {}).get("hit_verifiable")]

    verifiable_ids = {c.id for c in verifiable_cards}
    verifiable_only = [c for c in cards if c.id in verifiable_ids and c.result and c.result.rank1_waku]
    baseline = get_baseline_stats()

    return {
        "window": {"start": since.isoformat(), "end": until.isoformat(), "days": days if day is None else 1},
        "model_name": model_name,
        "prediction_counts": {
            "cards_in_window": len(cards),
            "with_prediction": len(all_with_pred),
            "with_result": sum(1 for c in cards if c.result and c.result.rank1_waku),
            "timing_status": timing_counts,
        },
        "published_risk": {
            "live_stats_unreliable": timing_counts.get("post_close", 0) > 0
            or timing_counts.get("no_deadline", 0) > 0,
            "note": (
                "締切後の再予想が含まれると、UI上の的中表示・ライブ的中率は"
                "「事前予想の実績」として公表できません。"
            ),
        },
        "hit_rates": {
            "all_saved_predictions": agg(rows),
            "pre_close_only": agg(verifiable_rows),
            "post_close_only": agg(post_close_rows),
            "unverified_timing": agg(unverified_rows),
        },
        "ticket_ranks": {
            "pre_close_only": _ticket_rank_block(session, verifiable_only, model_name=model_name),
            "all_saved_predictions": _ticket_rank_block(
                session,
                [c for c in cards if c.result and c.result.rank1_waku],
                model_name=model_name,
            ),
        },
        "baseline_reference": {
            "label": baseline.get("label"),
            "period": baseline.get("period"),
            "n_races": baseline.get("n_races"),
            "sanrenpuku_any_rate": (baseline.get("sanrenpuku") or {}).get("any_rate"),
            "sanrentan_any_rate": (baseline.get("sanrentan") or {}).get("any_rate"),
            "source": baseline.get("report_path") or baseline.get("source"),
            "note": (
                "GitHub 同梱の trifecta_eval_report.json から読み込み。"
                "確定レースを再予想したバックテストで、結果は買い目生成に使っていません。"
            ),
        },
        "holdout_reference": get_holdout_reference(),
    }
