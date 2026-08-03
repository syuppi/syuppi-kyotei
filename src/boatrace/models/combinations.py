"""3連単・3連複などの組み合わせ確率（Plackett–Luce近似）."""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Any


def _norm_strengths(strengths: dict[int, float], floor: float = 1e-6) -> dict[int, float]:
    out = {int(k): max(float(v), floor) for k, v in strengths.items()}
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}


def plackett_luce_ordered(
    strengths: dict[int, float],
) -> list[tuple[tuple[int, int, int], float]]:
    """全120通りの3連単確率を降順で返す。"""
    boats = list(strengths.keys())
    if len(boats) < 3:
        return []
    st = _norm_strengths(strengths)
    scored: list[tuple[tuple[int, int, int], float]] = []
    for a, b, c in permutations(boats, 3):
        rem = 1.0
        p1 = st[a] / rem
        rem -= st[a]
        p2 = st[b] / max(rem, 1e-12)
        rem -= st[b]
        p3 = st[c] / max(rem, 1e-12)
        scored.append(((a, b, c), p1 * p2 * p3))
    total = sum(p for _, p in scored) or 1.0
    scored = [(t, p / total) for t, p in scored]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def unordered_trio_probs(
    ordered: list[tuple[tuple[int, int, int], float]],
) -> list[tuple[frozenset[int], float]]:
    """3連単確率を集約して3連複確率にする。"""
    acc: dict[frozenset[int], float] = {}
    for ticket, p in ordered:
        key = frozenset(ticket)
        acc[key] = acc.get(key, 0.0) + p
    items = [(k, v) for k, v in acc.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def blend_place_strengths(
    win_probs: dict[int, float],
    top2_probs: dict[int, float] | None = None,
    top3_probs: dict[int, float] | None = None,
    w_win: float = 0.55,
    w_top2: float = 0.25,
    w_top3: float = 0.20,
) -> dict[int, float]:
    """1着・2連対・3連対確率を合成して順位強度にする。"""
    boats = list(win_probs.keys())
    top2 = top2_probs or win_probs
    top3 = top3_probs or win_probs
    raw = {
        w: (
            w_win * float(win_probs.get(w, 0.0))
            + w_top2 * float(top2.get(w, 0.0))
            + w_top3 * float(top3.get(w, 0.0))
        )
        for w in boats
    }
    return _norm_strengths(raw)


def best_trio_by_coverage(
    top3_probs: dict[int, float],
    strengths: dict[int, float] | None = None,
    venue_prior: dict[frozenset[int], float] | None = None,
    prior_weight: float = 0.15,
) -> list[tuple[frozenset[int], float]]:
    """
    3連複候補をスコアリング。
    基本は top3 確率の幾何平均、必要なら場のコース共起先验を混合。
    """
    boats = list(top3_probs.keys())
    scored: list[tuple[frozenset[int], float]] = []
    for combo in combinations(boats, 3):
        key = frozenset(combo)
        vals = [max(float(top3_probs[w]), 1e-6) for w in combo]
        geo = (vals[0] * vals[1] * vals[2]) ** (1.0 / 3.0)
        if strengths:
            geo *= 0.85 + 0.15 * sum(float(strengths.get(w, 0.0)) for w in combo)
        if venue_prior and key in venue_prior:
            geo = (1.0 - prior_weight) * geo + prior_weight * float(venue_prior[key])
        scored.append((key, geo))
    total = sum(s for _, s in scored) or 1.0
    scored = [(k, s / total) for k, s in scored]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _stake_shares(probs: list[float]) -> list[float]:
    """候補間の資金配分比率（合計1）。"""
    cleaned = [max(float(p), 1e-9) for p in probs]
    s = sum(cleaned) or 1.0
    return [p / s for p in cleaned]


def make_ticket_rows(
    combos: list[list[int]],
    probs: list[float],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """期待度順チケット行（確率・配分比率付き）。"""
    n = min(limit, len(combos), len(probs))
    if n <= 0:
        return []
    shares = _stake_shares(probs[:n])
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "rank": i + 1,
                "combo": list(combos[i]),
                "prob": float(probs[i]),
                "stake_share": float(shares[i]),
                "label": "-".join(map(str, combos[i])),
            }
        )
    return rows


def build_combination_bundle(
    win_probs: dict[int, float],
    top2_probs: dict[int, float] | None = None,
    top3_probs: dict[int, float] | None = None,
    venue_prior: dict[frozenset[int], float] | None = None,
    *,
    n_win: int = 3,
    n_sanrenpuku: int = 3,
    n_sanrentan: int = 3,
) -> dict[str, Any]:
    """
    3連単・3連複の本命と候補をまとめて返す。

    tickets:
      win / sanrenpuku / sanrentan を期待度順 2〜3 候補（確率・資金配分付き）
    """
    top2 = top2_probs or {w: min(0.95, p * 1.6 + 0.05) for w, p in win_probs.items()}
    top3 = top3_probs or {w: min(0.95, p * 2.2 + 0.08) for w, p in win_probs.items()}
    top2 = {w: float(max(0.01, min(0.98, v))) for w, v in top2.items()}
    top3 = {w: float(max(0.01, min(0.98, v))) for w, v in top3.items()}

    strengths = blend_place_strengths(win_probs, top2, top3)
    ordered = plackett_luce_ordered(strengths)
    unordered = best_trio_by_coverage(top3, strengths=strengths, venue_prior=venue_prior)
    unordered_pl = unordered_trio_probs(ordered)
    merged: dict[frozenset[int], float] = {}
    for key, p in unordered:
        merged[key] = 0.65 * p
    for key, p in unordered_pl:
        merged[key] = merged.get(key, 0.0) + 0.35 * p
    trio_ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

    best_trio = (
        sorted(trio_ranked[0][0])
        if trio_ranked
        else sorted(win_probs, key=win_probs.get, reverse=True)[:3]
    )
    best_order = None
    if ordered:
        for ticket, _p in ordered:
            if set(ticket) == set(best_trio):
                best_order = list(ticket)
                break
        if best_order is None:
            best_order = list(ordered[0][0])
    else:
        best_order = list(best_trio)

    used = set(best_order)
    rest = sorted(
        (w for w in strengths if w not in used),
        key=lambda w: strengths[w],
        reverse=True,
    )
    rankings = best_order + rest

    by_top2 = sorted(top2.keys(), key=lambda w: top2[w], reverse=True)
    quinella = sorted(
        set(best_order[:2]) | set(by_top2[:2]),
        key=lambda w: strengths[w],
        reverse=True,
    )[:2]
    if len(quinella) < 2:
        quinella = by_top2[:2]

    # 単勝候補: 1着確率の上位
    win_sorted = sorted(win_probs.keys(), key=lambda w: win_probs[w], reverse=True)
    n_win = max(2, min(int(n_win), 3, len(win_sorted)))
    win_combos = [[w] for w in win_sorted[:n_win]]
    win_ticket_probs = [float(win_probs[w]) for w in win_sorted[:n_win]]
    win_tickets = make_ticket_rows(win_combos, win_ticket_probs, limit=n_win)

    # 3連複候補
    n_sanrenpuku = max(2, min(int(n_sanrenpuku), 3, max(len(trio_ranked), 1)))
    sp_combos = [sorted(list(k)) for k, _ in trio_ranked[:n_sanrenpuku]]
    sp_probs = [float(p) for _, p in trio_ranked[:n_sanrenpuku]]
    sanrenpuku_tickets = make_ticket_rows(sp_combos, sp_probs, limit=n_sanrenpuku)

    # 3連単候補
    n_sanrentan = max(2, min(int(n_sanrentan), 3, max(len(ordered), 1)))
    st_combos = [list(t) for t, _ in ordered[:n_sanrentan]]
    st_probs = [float(p) for _, p in ordered[:n_sanrentan]]
    sanrentan_tickets = make_ticket_rows(st_combos, st_probs, limit=n_sanrentan)

    # 後方互換: 長いリストも残す
    sanrentan = [list(t) for t, _ in ordered[:8]]
    sanrenpuku = [sorted(list(k)) for k, _ in trio_ranked[:8]]

    # candidates_trio: 上位3連複をカバーする艇集合（最大4）
    covered: list[int] = []
    for combo in sp_combos:
        for w in sorted(combo, key=lambda x: strengths.get(x, 0.0), reverse=True):
            if w not in covered:
                covered.append(w)
            if len(covered) >= 4:
                break
        if len(covered) >= 4:
            break

    tickets = {
        "win": win_tickets,
        "sanrenpuku": sanrenpuku_tickets,
        "sanrentan": sanrentan_tickets,
    }

    return {
        "strengths": strengths,
        "rankings": rankings,
        "candidates_win": [t["combo"][0] for t in win_tickets],
        "candidates_quinella": quinella,
        "candidates_trio": covered or list(best_trio),
        "sanrentan": sanrentan,
        "sanrenpuku": sanrenpuku,
        "sanrentan_probs": {f"{a}-{b}-{c}": float(p) for (a, b, c), p in ordered[:20]},
        "sanrenpuku_probs": {
            "-".join(map(str, sorted(k))): float(p) for k, p in trio_ranked[:20]
        },
        "top2_probs": top2,
        "top3_probs": top3,
        "tickets": tickets,
    }
