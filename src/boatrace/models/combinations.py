"""3連単・3連複などの組み合わせ確率（着順強度 + 条件付きチェーン + 共起）."""

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


def conditional_chain_ordered(
    win_probs: dict[int, float],
    second_strengths: dict[int, float],
    third_strengths: dict[int, float] | None = None,
) -> list[tuple[tuple[int, int, int], float]]:
    """
    条件付きチェーン:
      P(a,b,c) ∝ P1(a) * S2(b|≠a) * S3(c|≠a,b)
    """
    boats = list(win_probs.keys())
    if len(boats) < 3:
        return []
    p1 = _norm_strengths(win_probs)
    s2 = _norm_strengths(second_strengths)
    s3 = _norm_strengths(third_strengths or second_strengths)
    scored: list[tuple[tuple[int, int, int], float]] = []
    for a, b, c in permutations(boats, 3):
        # 残りの正規化
        rem2 = {w: s2[w] for w in boats if w != a}
        z2 = sum(rem2.values()) or 1.0
        rem3 = {w: s3[w] for w in boats if w not in {a, b}}
        z3 = sum(rem3.values()) or 1.0
        scored.append(((a, b, c), p1[a] * (rem2[b] / z2) * (rem3[c] / z3)))
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
    rank_scores: dict[int, float] | None = None,
    w_win: float = 0.40,
    w_top2: float = 0.20,
    w_top3: float = 0.15,
    w_rank: float = 0.25,
) -> dict[int, float]:
    """1着・連対・ランカー強度を合成。"""
    boats = list(win_probs.keys())
    top2 = top2_probs or win_probs
    top3 = top3_probs or win_probs
    rank = rank_scores or win_probs
    # rank scores を 0-1 に
    rank_n = _norm_strengths(rank)
    raw = {
        w: (
            w_win * float(win_probs.get(w, 0.0))
            + w_top2 * float(top2.get(w, 0.0))
            + w_top3 * float(top3.get(w, 0.0))
            + w_rank * float(rank_n.get(w, 0.0))
        )
        for w in boats
    }
    return _norm_strengths(raw)


def best_trio_by_coverage(
    top3_probs: dict[int, float],
    strengths: dict[int, float] | None = None,
    venue_prior: dict[frozenset[int], float] | None = None,
    prior_weight: float = 0.20,
) -> list[tuple[frozenset[int], float]]:
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
        elif venue_prior:
            geo *= 1.0 - prior_weight * 0.35
        scored.append((key, geo))
    total = sum(s for _, s in scored) or 1.0
    scored = [(k, s / total) for k, s in scored]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _stake_shares(probs: list[float]) -> list[float]:
    cleaned = [max(float(p), 1e-9) for p in probs]
    s = sum(cleaned) or 1.0
    return [p / s for p in cleaned]


def make_ticket_rows(
    combos: list[list[int]],
    probs: list[float],
    *,
    limit: int,
) -> list[dict[str, Any]]:
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


def select_diverse_trifectas(
    ordered: list[tuple[tuple[int, int, int], float]],
    *,
    limit: int = 3,
) -> list[tuple[list[int], float]]:
    """
    カバー効率の良い3連単候補を選ぶ。
    - 1点目: 最高確率
    - 2点目以降: 同じ3艇の別順列より、異なる艇セット / 異なる1着 を優先
    """
    if not ordered:
        return []
    selected: list[tuple[list[int], float]] = []
    used_sets: list[frozenset[int]] = []
    used_winners: set[int] = set()

    # まず本命
    top_t, top_p = ordered[0]
    selected.append((list(top_t), float(top_p)))
    used_sets.append(frozenset(top_t))
    used_winners.add(top_t[0])

    for ticket, p in ordered[1:]:
        if len(selected) >= limit:
            break
        key = frozenset(ticket)
        # すでに選んだセットの別順列は後回し
        if key in used_sets:
            continue
        selected.append((list(ticket), float(p)))
        used_sets.append(key)
        used_winners.add(ticket[0])

    # 足りなければ、異なる1着の順列を追加
    if len(selected) < limit:
        for ticket, p in ordered[1:]:
            if len(selected) >= limit:
                break
            if list(ticket) in [s[0] for s in selected]:
                continue
            if ticket[0] in used_winners and frozenset(ticket) in used_sets:
                continue
            # 同じセットでも1着が違えばカバーになる場合あり
            if frozenset(ticket) in used_sets and ticket[0] in used_winners:
                continue
            selected.append((list(ticket), float(p)))
            used_sets.append(frozenset(ticket))
            used_winners.add(ticket[0])

    # それでも足りなければ確率順で埋める
    if len(selected) < limit:
        have = {tuple(s[0]) for s in selected}
        for ticket, p in ordered:
            if len(selected) >= limit:
                break
            if tuple(ticket) in have:
                continue
            selected.append((list(ticket), float(p)))

    return selected[:limit]


def select_diverse_trios(
    trio_ranked: list[tuple[frozenset[int], float]],
    *,
    limit: int = 3,
) -> list[tuple[list[int], float]]:
    """3連複: 重複の少ないセットを上位から."""
    out: list[tuple[list[int], float]] = []
    for key, p in trio_ranked:
        if len(out) >= limit:
            break
        out.append((sorted(key), float(p)))
    return out


def merge_ordered_lists(
    primary: list[tuple[tuple[int, int, int], float]],
    secondary: list[tuple[tuple[int, int, int], float]],
    w_primary: float = 0.65,
) -> list[tuple[tuple[int, int, int], float]]:
    acc: dict[tuple[int, int, int], float] = {}
    for t, p in primary:
        acc[t] = acc.get(t, 0.0) + w_primary * p
    for t, p in secondary:
        acc[t] = acc.get(t, 0.0) + (1.0 - w_primary) * p
    items = list(acc.items())
    total = sum(p for _, p in items) or 1.0
    items = [(t, p / total) for t, p in items]
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def build_combination_bundle(
    win_probs: dict[int, float],
    top2_probs: dict[int, float] | None = None,
    top3_probs: dict[int, float] | None = None,
    venue_prior: dict[frozenset[int], float] | None = None,
    rank_scores: dict[int, float] | None = None,
    *,
    n_win: int = 3,
    n_sanrenpuku: int = 3,
    n_sanrentan: int = 3,
    prior_weight: float = 0.08,
) -> dict[str, Any]:
    """3連単・3連複の本命と候補をまとめて返す。"""
    top2 = top2_probs or {w: min(0.95, p * 1.6 + 0.05) for w, p in win_probs.items()}
    top3 = top3_probs or {w: min(0.95, p * 2.2 + 0.08) for w, p in win_probs.items()}
    top2 = {w: float(max(0.01, min(0.98, v))) for w, v in top2.items()}
    top3 = {w: float(max(0.01, min(0.98, v))) for w, v in top3.items()}

    # ランカーは枠リークが残りやすいので弱め
    strengths = blend_place_strengths(
        win_probs, top2, top3, rank_scores=rank_scores, w_win=0.50, w_top2=0.22, w_top3=0.18, w_rank=0.10
    )
    second_strengths = blend_place_strengths(
        win_probs, top2, top3, rank_scores=rank_scores, w_win=0.20, w_top2=0.50, w_top3=0.20, w_rank=0.10
    )
    third_strengths = blend_place_strengths(
        win_probs, top2, top3, rank_scores=rank_scores, w_win=0.12, w_top2=0.23, w_top3=0.55, w_rank=0.10
    )

    ordered_pl = plackett_luce_ordered(strengths)
    ordered_cond = conditional_chain_ordered(win_probs, second_strengths, third_strengths)
    ordered = merge_ordered_lists(ordered_cond, ordered_pl, w_primary=0.70)

    # 場共起は弱めに混合（強すぎると本命を崩す）
    pw = max(0.0, min(0.25, float(prior_weight)))
    if venue_prior and pw > 0:
        reweighted = []
        for ticket, p in ordered:
            key = frozenset(ticket)
            prior = float(venue_prior.get(key, 0.0))
            # prior は頻度分布。無いセットはわずかに減衰
            boost = 1.0 + pw * ((prior * 15.0) - 0.35) if prior else (1.0 - pw * 0.15)
            reweighted.append((ticket, p * max(0.5, boost)))
        total = sum(p for _, p in reweighted) or 1.0
        ordered = sorted(((t, p / total) for t, p in reweighted), key=lambda x: -x[1])

    unordered = best_trio_by_coverage(
        top3, strengths=strengths, venue_prior=venue_prior, prior_weight=pw
    )
    unordered_pl = unordered_trio_probs(ordered)
    merged: dict[frozenset[int], float] = {}
    for key, p in unordered:
        merged[key] = 0.55 * p
    for key, p in unordered_pl:
        merged[key] = merged.get(key, 0.0) + 0.45 * p
    trio_ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

    diverse_tf = select_diverse_trifectas(ordered, limit=max(2, min(int(n_sanrentan), 3)))
    diverse_sp = select_diverse_trios(trio_ranked, limit=max(2, min(int(n_sanrenpuku), 3)))

    best_order = diverse_tf[0][0] if diverse_tf else list(ordered[0][0])
    strengths_rank = strengths
    used = set(best_order)
    rest = sorted(
        (w for w in strengths_rank if w not in used),
        key=lambda w: strengths_rank[w],
        reverse=True,
    )
    rankings = best_order + rest

    by_top2 = sorted(top2.keys(), key=lambda w: top2[w], reverse=True)
    quinella = sorted(
        set(best_order[:2]) | set(by_top2[:2]),
        key=lambda w: strengths_rank[w],
        reverse=True,
    )[:2]
    if len(quinella) < 2:
        quinella = by_top2[:2]

    win_sorted = sorted(win_probs.keys(), key=lambda w: win_probs[w], reverse=True)
    n_win = max(2, min(int(n_win), 3, len(win_sorted)))
    win_combos = [[w] for w in win_sorted[:n_win]]
    win_ticket_probs = [float(win_probs[w]) for w in win_sorted[:n_win]]
    win_tickets = make_ticket_rows(win_combos, win_ticket_probs, limit=n_win)

    sp_combos = [c for c, _ in diverse_sp]
    sp_probs = [p for _, p in diverse_sp]
    sanrenpuku_tickets = make_ticket_rows(sp_combos, sp_probs, limit=len(sp_combos))

    st_combos = [c for c, _ in diverse_tf]
    st_probs = [p for _, p in diverse_tf]
    sanrentan_tickets = make_ticket_rows(st_combos, st_probs, limit=len(st_combos))

    sanrentan = [list(t) for t, _ in ordered[:8]]
    sanrenpuku = [sorted(list(k)) for k, _ in trio_ranked[:8]]

    covered: list[int] = []
    for combo in sp_combos:
        for w in sorted(combo, key=lambda x: strengths_rank.get(x, 0.0), reverse=True):
            if w not in covered:
                covered.append(w)
            if len(covered) >= 4:
                break
        if len(covered) >= 4:
            break

    # 信頼度: 本命3連単確率が薄い場合は広めフラグ
    top_tf_p = st_probs[0] if st_probs else 0.0
    low_confidence = top_tf_p < 0.025

    tickets = {
        "win": win_tickets,
        "sanrenpuku": sanrenpuku_tickets,
        "sanrentan": sanrentan_tickets,
    }

    return {
        "strengths": strengths_rank,
        "rankings": rankings,
        "candidates_win": [t["combo"][0] for t in win_tickets],
        "candidates_quinella": quinella,
        "candidates_trio": covered or (sp_combos[0] if sp_combos else rankings[:3]),
        "sanrentan": sanrentan,
        "sanrenpuku": sanrenpuku,
        "sanrentan_probs": {f"{a}-{b}-{c}": float(p) for (a, b, c), p in ordered[:20]},
        "sanrenpuku_probs": {
            "-".join(map(str, sorted(k))): float(p) for k, p in trio_ranked[:20]
        },
        "top2_probs": top2,
        "top3_probs": top3,
        "tickets": tickets,
        "low_confidence": low_confidence,
        "top_trifecta_prob": float(top_tf_p),
    }
