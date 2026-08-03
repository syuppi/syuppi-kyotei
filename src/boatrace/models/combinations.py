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
        # 幾何平均（外れ1艇を強く罰する）
        vals = [max(float(top3_probs[w]), 1e-6) for w in combo]
        geo = (vals[0] * vals[1] * vals[2]) ** (1.0 / 3.0)
        if strengths:
            # 強度の合計でタイブレーク
            geo *= 0.85 + 0.15 * sum(float(strengths.get(w, 0.0)) for w in combo)
        if venue_prior and key in venue_prior:
            geo = (1.0 - prior_weight) * geo + prior_weight * float(venue_prior[key])
        scored.append((key, geo))
    total = sum(s for _, s in scored) or 1.0
    scored = [(k, s / total) for k, s in scored]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def build_combination_bundle(
    win_probs: dict[int, float],
    top2_probs: dict[int, float] | None = None,
    top3_probs: dict[int, float] | None = None,
    venue_prior: dict[frozenset[int], float] | None = None,
) -> dict[str, Any]:
    """
    3連単・3連複の本命と候補をまとめて返す。

    戻り値:
      - strengths
      - sanrentan: [(1,2,3), ...] 上位
      - sanrenpuku: [[1,2,3], ...] 上位
      - rankings: 6艇の推奨着順
      - candidates_quinella / candidates_trio
      - probs
    """
    top2 = top2_probs or {w: min(0.95, p * 1.6 + 0.05) for w, p in win_probs.items()}
    top3 = top3_probs or {w: min(0.95, p * 2.2 + 0.08) for w, p in win_probs.items()}
    # 再正規化は不要（周辺確率）だが極端値を抑える
    top2 = {w: float(max(0.01, min(0.98, v))) for w, v in top2.items()}
    top3 = {w: float(max(0.01, min(0.98, v))) for w, v in top3.items()}

    strengths = blend_place_strengths(win_probs, top2, top3)
    ordered = plackett_luce_ordered(strengths)
    unordered = best_trio_by_coverage(top3, strengths=strengths, venue_prior=venue_prior)
    # Plackett集約も混ぜて安定化
    unordered_pl = unordered_trio_probs(ordered)
    merged: dict[frozenset[int], float] = {}
    for key, p in unordered:
        merged[key] = 0.65 * p
    for key, p in unordered_pl:
        merged[key] = merged.get(key, 0.0) + 0.35 * p
    trio_ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

    best_trio = sorted(trio_ranked[0][0]) if trio_ranked else sorted(win_probs, key=win_probs.get, reverse=True)[:3]
    # 本命3連単: 本命3連複の並びを優先しつつ全体PL上位も参照
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

    # 6艇順位: 上位3は本命3連単、残りは強度順
    used = set(best_order)
    rest = sorted((w for w in strengths if w not in used), key=lambda w: strengths[w], reverse=True)
    rankings = best_order + rest

    # 2連複本命: 3連単の1-2着、または top2 上位2
    by_top2 = sorted(top2.keys(), key=lambda w: top2[w], reverse=True)
    quinella = sorted(set(best_order[:2]) | set(by_top2[:2]), key=lambda w: strengths[w], reverse=True)[:2]
    if len(quinella) < 2:
        quinella = by_top2[:2]

    # 3連複候補: 本命3 + 次点1（カバー用）
    trio_candidates = list(best_trio)
    if len(trio_ranked) > 1:
        for w in sorted(trio_ranked[1][0], key=lambda x: strengths.get(x, 0.0), reverse=True):
            if w not in trio_candidates:
                trio_candidates.append(w)
                break
    while len(trio_candidates) < 4 and len(rankings) >= len(trio_candidates) + 1:
        nxt = rankings[len(trio_candidates)]
        if nxt not in trio_candidates:
            trio_candidates.append(nxt)
        else:
            break

    sanrentan = [list(t) for t, _ in ordered[:8]]
    sanrenpuku = [sorted(list(k)) for k, _ in trio_ranked[:8]]

    return {
        "strengths": strengths,
        "rankings": rankings,
        "candidates_win": rankings[:2],
        "candidates_quinella": quinella,
        "candidates_trio": trio_candidates,
        "sanrentan": sanrentan,
        "sanrenpuku": sanrenpuku,
        "sanrentan_probs": {f"{a}-{b}-{c}": float(p) for (a, b, c), p in ordered[:20]},
        "sanrenpuku_probs": {"-".join(map(str, sorted(k))): float(p) for k, p in trio_ranked[:20]},
        "top2_probs": top2,
        "top3_probs": top3,
    }
