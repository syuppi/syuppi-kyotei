"""オッズ参照と期待値（EV）計算."""

from __future__ import annotations

from typing import Any


def _as_float(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def normalize_odds_blob(odds: object) -> dict[str, Any]:
    """turnmark / openapi / 公式から得たオッズを共通形に寄せる."""
    if not isinstance(odds, dict):
        return {}
    # turnmark は odds 直下に win/trio/trifecta
    out: dict[str, Any] = {}
    for key in ("win", "place", "quinella", "quinella_place", "exacta", "trio", "trifecta"):
        if key in odds and odds[key] is not None:
            out[key] = odds[key]
    return out


def lookup_win_odds(odds: dict[str, Any] | None, waku: int) -> float | None:
    if not odds:
        return None
    win = odds.get("win")
    if isinstance(win, dict):
        if str(waku) in win:
            return _as_float(win.get(str(waku)))
        if waku in win:
            return _as_float(win.get(waku))
    return None


def lookup_trio_odds(odds: dict[str, Any] | None, combo: list[int]) -> float | None:
    if not odds or len(combo) != 3:
        return None
    trio = odds.get("trio")
    a, b, c = sorted(int(x) for x in combo)
    key = f"{a}-{b}-{c}"
    if isinstance(trio, dict):
        # flat "1-2-3" or nested
        if key in trio:
            return _as_float(trio[key])
        # nested a -> b -> c (unordered variants)
        node = trio.get(str(a)) or trio.get(a)
        if isinstance(node, dict):
            for x, y in ((b, c), (c, b)):
                mid = node.get(str(x)) or node.get(x)
                if isinstance(mid, dict):
                    val = mid.get(str(y)) or mid.get(y)
                    got = _as_float(val)
                    if got:
                        return got
                got = _as_float(mid) if not isinstance(mid, dict) and int(x) == b else None
                if got and int(x) == b and int(y) == c:
                    return got
    return None


def lookup_trifecta_odds(odds: dict[str, Any] | None, combo: list[int]) -> float | None:
    if not odds or len(combo) != 3:
        return None
    tf = odds.get("trifecta")
    a, b, c = (int(x) for x in combo)
    key = f"{a}-{b}-{c}"
    if isinstance(tf, dict):
        if key in tf:
            return _as_float(tf[key])
        node = tf.get(str(a)) or tf.get(a)
        if isinstance(node, dict):
            mid = node.get(str(b)) or node.get(b)
            if isinstance(mid, dict):
                return _as_float(mid.get(str(c)) or mid.get(c))
            return _as_float(mid)
    return None


def enrich_tickets_with_ev(
    tickets: dict[str, list[dict[str, Any]]] | None,
    odds: dict[str, Any] | None,
    *,
    entry_win_odds: dict[int, float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """ticket 行に odds / ev / value_tag / reason を付与."""
    tickets = tickets or {}
    odds = normalize_odds_blob(odds or {})
    entry_win_odds = entry_win_odds or {}
    out: dict[str, list[dict[str, Any]]] = {}

    def _tag(ev: float | None) -> str | None:
        if ev is None:
            return None
        if ev >= 1.20:
            return "割安"
        if ev >= 1.00:
            return "妥当〜やや割安"
        if ev >= 0.80:
            return "ほぼ妥当"
        return "割高"

    # win
    win_rows = []
    for row in tickets.get("win") or []:
        r = dict(row)
        waku = int((r.get("combo") or [0])[0] or 0)
        odd = lookup_win_odds(odds, waku) or entry_win_odds.get(waku)
        prob = float(r.get("prob") or 0.0)
        ev = (prob * odd) if odd else None
        r["odds"] = odd
        r["ev"] = round(ev, 3) if ev is not None else None
        r["value_tag"] = _tag(ev)
        if odd and ev is not None:
            r["ev_reason"] = (
                f"単勝EV {ev:.2f}（モデル確率{prob*100:.1f}% × オッズ{odd:.1f}倍）"
                f" → {r['value_tag']}"
            )
        win_rows.append(r)
    out["win"] = win_rows

    # sanrenpuku
    sp_rows = []
    for row in tickets.get("sanrenpuku") or []:
        r = dict(row)
        combo = [int(x) for x in (r.get("combo") or [])]
        odd = lookup_trio_odds(odds, combo)
        prob = float(r.get("prob") or 0.0)
        ev = (prob * odd) if odd else None
        r["odds"] = odd
        r["ev"] = round(ev, 3) if ev is not None else None
        r["value_tag"] = _tag(ev)
        if odd and ev is not None:
            r["ev_reason"] = (
                f"3連複EV {ev:.2f}（確率{prob*100:.1f}% × オッズ{odd:.1f}倍）"
                f" → {r['value_tag']}"
            )
        sp_rows.append(r)
    out["sanrenpuku"] = sp_rows

    # sanrentan
    st_rows = []
    for row in tickets.get("sanrentan") or []:
        r = dict(row)
        combo = [int(x) for x in (r.get("combo") or [])]
        odd = lookup_trifecta_odds(odds, combo)
        prob = float(r.get("prob") or 0.0)
        ev = (prob * odd) if odd else None
        r["odds"] = odd
        r["ev"] = round(ev, 3) if ev is not None else None
        r["value_tag"] = _tag(ev)
        if odd and ev is not None:
            r["ev_reason"] = (
                f"3連単EV {ev:.2f}（確率{prob*100:.1f}% × オッズ{odd:.1f}倍）"
                f" → {r['value_tag']}"
            )
        st_rows.append(r)
    out["sanrentan"] = st_rows

    return out


def top_ev_reasons(tickets: dict[str, list[dict[str, Any]]], limit: int = 3) -> list[str]:
    """表示用にEV理由を数件だけ抜き出す（割安優先）."""
    cands: list[tuple[float, str]] = []
    for kind, label in (("win", "単勝"), ("sanrenpuku", "3連複"), ("sanrentan", "3連単")):
        for row in tickets.get(kind) or []:
            reason = row.get("ev_reason")
            ev = row.get("ev")
            if not reason or ev is None:
                continue
            # 割安を優先表示
            score = float(ev) + (0.5 if row.get("value_tag") == "割安" else 0.0)
            cands.append((score, f"{label}{row.get('label')}: {reason}"))
    cands.sort(key=lambda x: -x[0])
    return [t for _, t in cands[:limit]]
