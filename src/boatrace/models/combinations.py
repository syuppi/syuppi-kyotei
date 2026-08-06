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


_ROLE_BY_RANK: dict[int, dict[str, str]] = {
    1: {
        "role": "honmei",
        "role_label": "本命",
        "style": "nobu",
        "style_label": "ノブ型（的中）",
    },
    2: {
        "role": "taikou",
        "role_label": "対抗",
        "style": "nobu",
        "style_label": "ノブ型（的中）",
    },
    3: {
        "role": "ana",
        "role_label": "穴",
        "style": "wonder",
        "style_label": "ワンダー型（展開）",
    },
}


def _short_why_for_row(
    row: dict[str, Any],
    *,
    kind: str,
    fly_risk: float,
    upset_candidates: list[int] | None,
    honmei_combo: list[int] | None,
) -> str:
    combo = [int(x) for x in (row.get("combo") or [])]
    role = row.get("role")
    if role == "honmei":
        if kind == "win":
            return f"{combo[0]}号艇を軸に的中重視で厚く見る"
        return f"{'-'.join(map(str, combo))} を本線（実績・コース重視）"
    if role == "taikou":
        if honmei_combo and combo and set(combo) == set(honmei_combo) and combo != honmei_combo:
            return "同軸のヒモ違いで保険"
        if honmei_combo and combo and combo[0] == honmei_combo[0]:
            return "本命と同軸の軽微な展開代替"
        return "本命に次ぐ実力帯の対抗候補"
    # ana / wonder
    head = combo[0] if combo else None
    ups = set(upset_candidates or [])
    if fly_risk >= 0.45 and head and head >= 2:
        if head >= 4:
            return f"1号艇飛びリスク高め（{fly_risk:.0%}）→外枠頭の展開崩れ"
        return f"1号艇飛びリスク高め（{fly_risk:.0%}）→{head}号頭の展開崩れ"
    if head is not None and head in ups:
        return "穴候補艇を頭にした展開狙い"
    if head is not None and head >= 4:
        return "外枠頭のワンダー型・展開崩れ狙い"
    return "本命崩れ時の保険の流し"


def annotate_ticket_roles(
    tickets: dict[str, Any] | None,
    *,
    fly_risk: float = 0.0,
    upset_candidates: list[int] | None = None,
) -> dict[str, Any]:
    """3候補に本命/対抗/穴とノブ/ワンダー意味付けを付与."""
    out: dict[str, Any] = {}
    if not tickets:
        return out
    for kind in ("win", "sanrenpuku", "sanrentan"):
        rows = list(tickets.get(kind) or [])
        honmei_combo: list[int] | None = None
        annotated: list[dict[str, Any]] = []
        for row in rows:
            r = dict(row)
            rank = int(r.get("rank") or (len(annotated) + 1))
            meta = _ROLE_BY_RANK.get(rank, _ROLE_BY_RANK[3])
            r.update(meta)
            if rank == 1:
                honmei_combo = [int(x) for x in (r.get("combo") or [])]
            # 3連の3本目はワンダー固定。単勝3位も穴扱い。
            if rank == 3:
                r["style"] = "wonder"
                r["style_label"] = "ワンダー型（展開）"
                r["role"] = "ana"
                r["role_label"] = "穴"
            r["why_short"] = _short_why_for_row(
                r,
                kind=kind,
                fly_risk=float(fly_risk or 0.0),
                upset_candidates=upset_candidates,
                honmei_combo=honmei_combo,
            )
            annotated.append(r)
        out[kind] = annotated
    # その他キーはそのまま
    for k, v in tickets.items():
        if k not in out:
            out[k] = v
    return out


def select_diverse_trifectas(
    ordered: list[tuple[tuple[int, int, int], float]],
    *,
    limit: int = 3,
    delay_heads: list[int] | None = None,
) -> list[tuple[list[int], float]]:
    """
    カバー効率の良い3連単候補を選ぶ。
    - 1点目: 最高確率
    - 2点目以降: 同じ3艇の別順列より、異なる艇セット / 異なる1着 を優先
    - 3点目候補に delay_heads があれば優先採用
    """
    if not ordered:
        return []
    selected: list[tuple[list[int], float]] = []
    used_sets: list[frozenset[int]] = []
    used_winners: set[int] = set()

    top_t, top_p = ordered[0]
    selected.append((list(top_t), float(top_p)))
    used_sets.append(frozenset(top_t))
    used_winners.add(top_t[0])

    for ticket, p in ordered[1:]:
        if len(selected) >= limit:
            break
        key = frozenset(ticket)
        if key in used_sets:
            continue
        selected.append((list(ticket), float(p)))
        used_sets.append(key)
        used_winners.add(ticket[0])

    if len(selected) < limit:
        prefer = set(int(x) for x in (delay_heads or []))
        for ticket, p in ordered[1:]:
            if len(selected) >= limit:
                break
            if prefer and ticket[0] not in prefer:
                continue
            if list(ticket) in [s[0] for s in selected]:
                continue
            if ticket[0] in used_winners and frozenset(ticket) in used_sets:
                continue
            if frozenset(ticket) in used_sets and ticket[0] in used_winners:
                continue
            selected.append((list(ticket), float(p)))
            used_sets.append(frozenset(ticket))
            used_winners.add(ticket[0])

    if len(selected) < limit:
        for ticket, p in ordered[1:]:
            if len(selected) >= limit:
                break
            if list(ticket) in [s[0] for s in selected]:
                continue
            if ticket[0] in used_winners and frozenset(ticket) in used_sets:
                continue
            if frozenset(ticket) in used_sets and ticket[0] in used_winners:
                continue
            selected.append((list(ticket), float(p)))
            used_sets.append(frozenset(ticket))
            used_winners.add(ticket[0])

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
    delay_boats: list[int] | None = None,
) -> list[tuple[list[int], float]]:
    """3連複: 重複の少ないセットを上位から。穴枠は受益艇含有を優先."""
    out: list[tuple[list[int], float]] = []
    used: set[frozenset[int]] = set()
    for key, p in trio_ranked:
        if len(out) >= max(1, limit - (1 if delay_boats else 0)):
            break
        if key in used:
            continue
        out.append((sorted(key), float(p)))
        used.add(key)
    delay_set = set(int(x) for x in (delay_boats or []))
    if len(out) < limit and delay_set:
        for key, p in trio_ranked:
            if key in used:
                continue
            if set(key) & delay_set:
                out.append((sorted(key), float(p)))
                used.add(key)
                break
    for key, p in trio_ranked:
        if len(out) >= limit:
            break
        if key in used:
            continue
        out.append((sorted(key), float(p)))
        used.add(key)
    return out[:limit]


def expand_trio_coverage(
    trio_ranked: list[tuple[frozenset[int], float]],
    selected: list[tuple[list[int], float]],
    *,
    limit: int,
    top3_probs: dict[int, float] | None = None,
) -> list[tuple[list[int], float]]:
    """4点目以降: 上位5艇の C(5,3) を優先してカバー拡張（本線は維持）."""
    out = list(selected)
    used = {frozenset(c) for c, _ in out}
    covered: set[int] = set().union(*[set(c) for c, _ in out]) if out else set()
    t3 = top3_probs or {}
    p_lookup = {k: float(p) for k, p in trio_ranked}
    ranked_boats = sorted(t3.keys(), key=lambda w: float(t3.get(w, 0.0)), reverse=True)
    pool = ranked_boats[:5]

    def geo(key: frozenset[int]) -> float:
        vals = [max(float(t3.get(w, 0.05)), 1e-6) for w in key]
        return (vals[0] * vals[1] * vals[2]) ** (1.0 / 3.0)

    # 1) プール組み合わせを geo × 新規性で追加
    pool_keys = [frozenset(c) for c in combinations(pool, 3)]
    pool_keys.sort(
        key=lambda k: -(
            0.55 * geo(k)
            + 0.25 * p_lookup.get(k, 0.0)
            + 0.35 * len(k - covered)
        )
    )
    for key in pool_keys:
        if len(out) >= limit:
            break
        if key in used:
            continue
        out.append((sorted(key), float(p_lookup.get(key, geo(key) * 0.05))))
        used.add(key)
        covered |= set(key)

    # 2) 足りなければ通常の高確率候補で埋める
    for key, p in trio_ranked:
        if len(out) >= limit:
            break
        if key in used:
            continue
        out.append((sorted(key), float(p)))
        used.add(key)
        covered |= set(key)
    return out[:limit]


def expand_trifecta_coverage(
    ordered: list[tuple[tuple[int, int, int], float]],
    selected: list[tuple[list[int], float]],
    *,
    limit: int,
) -> list[tuple[list[int], float]]:
    """本命セットの別順列→その他高確率並びで3連単カバーを拡張."""
    out = list(selected)
    have = {tuple(c) for c, _ in out}
    if not out and ordered:
        return [(list(t), float(p)) for t, p in ordered[:limit]]

    top_set = frozenset(out[0][0])
    for ticket, p in ordered:
        if len(out) >= limit:
            break
        if frozenset(ticket) != top_set or tuple(ticket) in have:
            continue
        out.append((list(ticket), float(p)))
        have.add(tuple(ticket))

    if len(selected) > 1 and len(out) < limit:
        second_set = frozenset(selected[1][0])
        for ticket, p in ordered:
            if len(out) >= limit:
                break
            if frozenset(ticket) != second_set or tuple(ticket) in have:
                continue
            out.append((list(ticket), float(p)))
            have.add(tuple(ticket))

    for ticket, p in ordered:
        if len(out) >= limit:
            break
        if tuple(ticket) in have:
            continue
        out.append((list(ticket), float(p)))
        have.add(tuple(ticket))
    return out[:limit]


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
    fly_risk: float = 0.0,
    delay_beneficiaries: list[int] | None = None,
) -> dict[str, Any]:
    """3連単・3連複の本命と候補をまとめて返す（3着以内カバー優先）."""
    top2 = top2_probs or {w: min(0.95, p * 1.6 + 0.05) for w, p in win_probs.items()}
    top3 = top3_probs or {w: min(0.95, p * 2.2 + 0.08) for w, p in win_probs.items()}
    top2 = {w: float(max(0.01, min(0.98, v))) for w, v in top2.items()}
    top3 = {w: float(max(0.01, min(0.98, v))) for w, v in top3.items()}

    strengths = blend_place_strengths(
        win_probs, top2, top3, rank_scores=rank_scores, w_win=0.42, w_top2=0.22, w_top3=0.26, w_rank=0.10
    )
    second_strengths = blend_place_strengths(
        win_probs, top2, top3, rank_scores=rank_scores, w_win=0.18, w_top2=0.48, w_top3=0.24, w_rank=0.10
    )
    third_strengths = blend_place_strengths(
        win_probs, top2, top3, rank_scores=rank_scores, w_win=0.10, w_top2=0.22, w_top3=0.58, w_rank=0.10
    )

    ordered_pl = plackett_luce_ordered(strengths)
    ordered_cond = conditional_chain_ordered(win_probs, second_strengths, third_strengths)
    ordered = merge_ordered_lists(ordered_cond, ordered_pl, w_primary=0.70)

    fr = float(fly_risk or 0.0)
    # 穴選抜への受益艇バイアスは飛びリスクが高いときだけ
    delay_heads = [int(x) for x in (delay_beneficiaries or [])] if fr >= 0.40 else []
    # 飛びが高いときだけ、受益頭の並びを軽く押し上げ（本線は壊さない）
    if delay_heads and fr >= 0.45:
        boost_set = set(delay_heads[:2])
        reweighted = []
        for ticket, p in ordered:
            mult = 1.0 + (0.18 * min(1.0, fr / 0.7) if ticket[0] in boost_set else 0.0)
            reweighted.append((ticket, p * mult))
        total = sum(p for _, p in reweighted) or 1.0
        ordered = sorted(((t, p / total) for t, p in reweighted), key=lambda x: -x[1])

    pw = max(0.0, min(0.25, float(prior_weight)))
    if venue_prior and pw > 0:
        reweighted = []
        for ticket, p in ordered:
            key = frozenset(ticket)
            prior = float(venue_prior.get(key, 0.0))
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
    if delay_heads:
        dset = set(delay_heads[:2])
        for key in list(merged.keys()):
            if key & dset and fr >= 0.40:
                merged[key] *= 1.0 + 0.12 * min(1.0, fr)
        total = sum(merged.values()) or 1.0
        merged = {k: v / total for k, v in merged.items()}
    trio_ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

    diverse_tf = select_diverse_trifectas(
        ordered,
        limit=max(2, min(int(n_sanrentan), 5)),
        delay_heads=delay_heads,
    )
    diverse_tf = expand_trifecta_coverage(
        ordered, diverse_tf, limit=max(2, min(int(n_sanrentan), 5))
    )
    # 本命〜対抗は確率上位を確保し、4点目以降で上位艇プールを拡張
    primary_n = min(int(n_sanrenpuku), 3)
    diverse_sp = select_diverse_trios(
        trio_ranked,
        limit=max(1, primary_n),
        delay_boats=delay_heads,
    )
    diverse_sp = expand_trio_coverage(
        trio_ranked,
        diverse_sp,
        limit=max(2, min(int(n_sanrenpuku), 5)),
        top3_probs=top3,
    )

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
            if len(covered) >= 5:
                break
        if len(covered) >= 5:
            break
    for w in delay_heads:
        if w not in covered:
            covered.append(w)

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
        "delay_beneficiaries": delay_heads,
        "fly_risk": fr,
    }
