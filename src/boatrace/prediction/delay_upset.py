"""本命がスタート/コーナーで遅れたときの展開受益艇を推定する."""

from __future__ import annotations

from typing import Any

from boatrace.features.builder import RaceFeatures


def favorite_from_probs(win_probs: dict[int, float] | None) -> int | None:
    if not win_probs:
        return None
    return max(win_probs.keys(), key=lambda w: float(win_probs[w]))


def compute_delay_beneficiaries(
    features: RaceFeatures,
    *,
    favorite: int | None = None,
    win_probs: dict[int, float] | None = None,
    top3_probs: dict[int, float] | None = None,
    limit: int = 3,
) -> dict[str, Any]:
    """
    本命がST遅れ・1角で置かれた場合に伸びやすい艇を返す。

    根拠:
    - インより速い展示ST（st_gap_vs_inner）
    - course1_upset_pressure / fly_risk
    - 場のまくり・差し・カド強度
    - top3見込みの高さ（3着以内に来る実力）
    """
    fav = favorite or favorite_from_probs(win_probs)
    boats = {b.waku: b for b in features.boats}
    env = features.env or {}
    fly = float(env.get("course1_fly_risk") or 0.0)
    makuri = float(env.get("venue_makuri_rate") or 0.0)
    sashi = float(env.get("venue_sashi_rate") or 0.0)
    kado = float(env.get("venue_kado_strength") or 0.0)
    in_wr = float(env.get("venue_in_win_rate") or 0.55)

    scored: list[tuple[int, float, list[str]]] = []
    for waku, boat in boats.items():
        if fav is not None and waku == fav:
            continue
        raw = boat.raw or {}
        reasons: list[str] = []
        score = 0.0

        # 3着以内実力
        t3 = float((top3_probs or {}).get(waku) or boat.values.get("racer_course_win") or 0.4)
        score += 1.2 * t3
        reasons.append(f"3連対見込み帯{t3:.0%}")

        st_gap = raw.get("st_gap_vs_inner")
        if st_gap is not None and float(st_gap) > 0.01:
            score += min(0.45, float(st_gap) / 0.08)
            reasons.append(f"展示STがインより優位（{float(st_gap):+.2f}）")

        pressure = float(raw.get("course1_upset_pressure") or 0.0)
        if pressure > 0:
            score += 0.8 * pressure
            reasons.append(f"イン崩れ外圧{pressure:.2f}")

        # 枠位置: 本命遅れ時は2-4が取りやすい、カド場なら3-4
        if waku in {2, 3}:
            score += 0.18 + 0.12 * (makuri + sashi)
            reasons.append("差し・まくりの利きやすい枠")
        elif waku == 4:
            score += 0.15 + 0.25 * kado
            reasons.append("カドからのまくり圧")
        elif waku >= 5:
            score += 0.08 + 0.10 * makuri
            reasons.append("外からの展開次第")

        if fly >= 0.40 and waku != 1:
            score += 0.20 * fly
            reasons.append(f"飛びリスク連動（{fly:.0%}）")

        if in_wr < 0.50 and waku != 1:
            score += 0.12
            reasons.append("場のインが弱い")

        # 本命が1でなく外本命の場合、イン残りも保険
        if fav is not None and fav >= 3 and waku == 1:
            score += 0.25
            reasons.append("外本命遅れ時のイン残り")

        scored.append((waku, score, reasons[:4]))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: max(1, limit)]
    return {
        "favorite": fav,
        "fly_risk": fly,
        "beneficiaries": [w for w, _, _ in top],
        "details": [
            {"waku": w, "score": float(s), "reasons": rs} for w, s, rs in top
        ],
        "thesis": _delay_thesis(fav, fly, [w for w, _, _ in top], makuri, sashi, kado),
    }


def _delay_thesis(
    fav: int | None,
    fly: float,
    bens: list[int],
    makuri: float,
    sashi: float,
    kado: float,
) -> str:
    fav_txt = f"{fav}号" if fav else "本命"
    head = bens[0] if bens else None
    parts = [
        f"{fav_txt}がスタートや1角で遅れれば、"
        f"{'・'.join(str(w) + '号' for w in bens[:3]) or '外枠'}が3着以内に伸びやすい。"
    ]
    if fly >= 0.40:
        parts.append(f"飛びリスクは{fly:.0%}。")
    if makuri + sashi >= 0.30 or kado >= 0.25:
        parts.append("場もまくり・差しが残りやすい水面。")
    if head:
        parts.append(f"穴の頭はまず{head}号を見る。")
    return "".join(parts)


def inject_delay_ana_tickets(
    tickets: dict[str, Any],
    *,
    beneficiaries: list[int],
    favorite: int | None,
    top3_probs: dict[int, float] | None = None,
    strengths: dict[int, float] | None = None,
    fly_risk: float = 0.0,
    force: bool = False,
) -> dict[str, Any]:
    """
    3連の3本目（穴）を、本命遅れ時の受益艇頭に寄せる。
    飛びリスクが低いときは既存の確率上位を崩さない。
    """
    if not beneficiaries:
        return tickets
    fr = float(fly_risk or 0.0)
    if not force and fr < 0.45:
        return tickets
    out = {k: list(v) if isinstance(v, list) else v for k, v in (tickets or {}).items()}
    fav = favorite
    bens = [int(w) for w in beneficiaries if fav is None or int(w) != fav]
    if not bens:
        return tickets

    strength = strengths or {}
    t3 = top3_probs or {}

    def _mates(head: int, need: int = 2) -> list[int]:
        pool = sorted(
            (w for w in set(list(t3.keys()) or list(strength.keys()) or range(1, 7)) if w != head),
            key=lambda w: float(t3.get(w) or 0) * 0.6 + float(strength.get(w) or 0) * 0.4,
            reverse=True,
        )
        mates: list[int] = []
        if fav is not None and fav != head:
            mates.append(fav)
        for w in pool:
            if w in mates or w == head:
                continue
            mates.append(w)
            if len(mates) >= need:
                break
        return mates[:need]

    st = list(out.get("sanrentan") or [])
    if len(st) >= 1:
        head = bens[0]
        # 既に受益艇が頭/メンバーに入っていれば本線を崩さない
        covered_heads = {int((r.get("combo") or [0])[0]) for r in st}
        covered_boats = {int(x) for r in st for x in (r.get("combo") or [])}
        if head in covered_heads or head in covered_boats:
            pass
        else:
            mates = _mates(head, 2)
            if len(mates) >= 2:
                ana = {
                    "rank": len(st),
                    "combo": [head, mates[0], mates[1]],
                    "label": f"{head}-{mates[0]}-{mates[1]}",
                    "prob": float(st[-1].get("prob") or 0.01) * 0.9,
                    "stake_share": float(st[-1].get("stake_share") or 0.2),
                    "delay_ana": True,
                    "why_short": f"本命遅れ想定→{head}号頭の展開穴",
                }
                # 本命・対抗は残し、末尾（穴枠）だけ差し替え / 追記
                if len(st) >= 2:
                    st[-1] = ana
                else:
                    ana["rank"] = len(st) + 1
                    st.append(ana)
                probs = [max(float(r.get("prob") or 1e-9), 1e-9) for r in st]
                s = sum(probs) or 1.0
                for i, r in enumerate(st):
                    r["stake_share"] = probs[i] / s
                    r["rank"] = i + 1
                out["sanrentan"] = st

    sp = list(out.get("sanrenpuku") or [])
    if len(sp) >= 1:
        head = bens[0]
        covered_boats = {int(x) for r in sp for x in (r.get("combo") or [])}
        if head in covered_boats:
            pass
        else:
            mates = _mates(head, 2)
            combo = sorted({head, *mates[:2]})
            if len(combo) == 3:
                ana = {
                    "rank": len(sp),
                    "combo": combo,
                    "label": "-".join(map(str, combo)),
                    "prob": float(sp[-1].get("prob") or 0.01) * 0.9,
                    "stake_share": float(sp[-1].get("stake_share") or 0.2),
                    "delay_ana": True,
                    "why_short": f"本命遅れ想定→{head}号を絡めた3連複穴",
                }
                if len(sp) >= 2:
                    sp[-1] = ana
                else:
                    ana["rank"] = len(sp) + 1
                    sp.append(ana)
                probs = [max(float(r.get("prob") or 1e-9), 1e-9) for r in sp]
                s = sum(probs) or 1.0
                for i, r in enumerate(sp):
                    r["stake_share"] = probs[i] / s
                    r["rank"] = i + 1
                out["sanrenpuku"] = sp

    return out
