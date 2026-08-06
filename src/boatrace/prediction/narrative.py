"""予想の日本語根拠文（ノブ型・ワンダー型）を生成する."""

from __future__ import annotations

from typing import Any

from boatrace.features.builder import RaceFeatures
from boatrace.models.base import PredictionResult


def _pct(v: float | None, digits: int = 0) -> str:
    if v is None:
        return "-"
    return f"{float(v) * 100:.{digits}f}%"


def _boat_map(features: RaceFeatures) -> dict[int, Any]:
    return {b.waku: b for b in features.boats}


def _grade_label(code: Any) -> str:
    if not code:
        return ""
    s = str(code).strip().upper()
    mapping = {"A1": "A1", "A2": "A2", "B1": "B1", "B2": "B2"}
    return mapping.get(s, s)


def _honmei_bullets(features: RaceFeatures, waku: int, win_prob: float) -> list[str]:
    boats = _boat_map(features)
    b = boats.get(waku)
    if not b:
        return [f"{waku}号艇を本命とするが、詳細データが不足"]
    raw = b.raw or {}
    values = b.values or {}
    env = features.env or {}
    bullets: list[str] = []

    course = int(raw.get("course") or waku)
    rc_win = raw.get("racer_course_win")
    rc_starts = raw.get("racer_course_starts")
    if rc_win is not None:
        starts_txt = f"（試走{int(rc_starts)}走）" if rc_starts else ""
        bullets.append(
            f"{waku}号艇は{course}コース1着率{_pct(float(rc_win), 1)}{starts_txt}"
        )
    resid = raw.get("racer_course_win_resid")
    if resid is not None and abs(float(resid)) >= 0.03:
        sign = "上回る" if float(resid) > 0 else "下回る"
        bullets.append(f"全国コース平均より{_pct(abs(float(resid)), 1)}{sign}")

    local = raw.get("local_win_rate")
    if local is not None and float(local) > 0:
        bullets.append(f"当地勝率{_pct(float(local), 1)}")
    nat = raw.get("national_win_rate")
    if (
        nat is not None
        and local is not None
        and float(local) > 0
        and float(local) - float(nat) >= 0.02
    ):
        bullets.append(f"全国勝率{_pct(float(nat), 1)}より当地が上振れ")

    grade = _grade_label(raw.get("grade_code"))
    if grade:
        bullets.append(f"級別{grade}")

    prev = raw.get("previous_rank")
    if prev is not None:
        pst = raw.get("previous_st")
        st_txt = f"・ST{float(pst):.2f}" if pst is not None else ""
        bullets.append(f"前走{int(prev)}着{st_txt}")
    form = values.get("recent_form")
    if form is not None and float(form) >= 0.65:
        bullets.append("直近フォーム良好")

    et = raw.get("exhibition_time")
    et_gap = raw.get("ex_time_gap")
    if et is not None:
        rank_hint = "展示最速級" if et_gap is not None and float(et_gap) <= 0.01 else "展示タイム反映"
        bullets.append(f"展示{float(et):.2f}（{rank_hint}）")
    est = raw.get("exhibition_st")
    if est is not None:
        bullets.append(f"展示ST{float(est):.2f}")

    mq = raw.get("motor_recent_q")
    if mq is None:
        mq = raw.get("motor_quinella_rate")
    if mq is not None and float(mq) >= 0.35:
        bullets.append(f"モーター直近2連対{_pct(float(mq), 0)}")

    in_wr = env.get("venue_in_win_rate")
    nige = env.get("venue_nige_rate")
    if waku == 1 and in_wr is not None:
        bullets.append(f"場のイン1着率{_pct(float(in_wr), 0)}・逃げ率{_pct(float(nige or 0), 0)}")

    same_in = env.get("same_day_in_win_rate")
    prev_n = int(env.get("same_day_prev_count") or 0)
    if prev_n >= 2 and same_in is not None and waku == 1:
        bullets.append(f"当日同場イン率{_pct(float(same_in), 0)}（{prev_n}R消化）")

    wind = env.get("wind_speed")
    wb = env.get("wind_bucket")
    if wind is not None:
        wlabel = {"head": "向かい風", "head_light": "弱い向かい風", "tail": "追い風", "cross": "横風"}.get(
            str(wb or ""), "風"
        )
        bullets.append(f"{wlabel}{float(wind):.0f}m")

    bullets.append(f"単勝モデル確率{_pct(win_prob, 1)}で厚く見る（ノブ型）")
    return bullets[:7]


def _taikou_bullets(
    features: RaceFeatures,
    combo: list[int],
    honmei_combo: list[int],
    kind: str,
) -> list[str]:
    boats = _boat_map(features)
    bullets: list[str] = []
    if kind == "win" and combo:
        w = combo[0]
        b = boats.get(w)
        raw = (b.raw if b else {}) or {}
        if honmei_combo and w != honmei_combo[0]:
            bullets.append(f"本命{honmei_combo[0]}号に次ぐ実力帯として{w}号を対抗")
        local = raw.get("local_win_rate")
        if local is not None:
            bullets.append(f"当地勝率{_pct(float(local), 1)}")
        et_gap = raw.get("ex_time_gap")
        if et_gap is not None and float(et_gap) <= 0.05:
            bullets.append("展示タイム差が小さく崩しにくい")
        if not bullets:
            bullets.append(f"{w}号艇を対抗としてカバー")
        return bullets[:5]

    if honmei_combo and set(combo) == set(honmei_combo) and combo != honmei_combo:
        bullets.append("本命と同艇のヒモ違い（軸固定の保険）")
    elif honmei_combo and combo and combo[0] == honmei_combo[0]:
        bullets.append(f"{combo[0]}号軸は維持し、2・3着の並び替え")
    else:
        bullets.append("本命から軽く展開をずらした対抗候補")

    for w in combo[:2]:
        b = boats.get(w)
        if not b:
            continue
        rc = (b.raw or {}).get("racer_course_win")
        if rc is not None:
            bullets.append(f"{w}号のコース1着率{_pct(float(rc), 1)}")
    return bullets[:5]


def _ana_bullets(
    features: RaceFeatures,
    combo: list[int],
    *,
    fly_risk: float,
    upset_candidates: list[int],
    honmei_combo: list[int],
    kind: str,
) -> list[str]:
    boats = _boat_map(features)
    env = features.env or {}
    bullets: list[str] = []
    head = combo[0] if combo else None
    fav = honmei_combo[0] if honmei_combo else None

    if fav is not None:
        bullets.append(f"本命{fav}号がスタートや1角で遅れた場合の展開")
    if fly_risk >= 0.40:
        bullets.append(f"1号艇飛びリスク{_pct(fly_risk, 0)}でイン崩れを警戒（ワンダー型）")
    makuri = float(env.get("venue_makuri_rate") or 0)
    sashi = float(env.get("venue_sashi_rate") or 0)
    kado = float(env.get("venue_kado_strength") or 0)
    if makuri + sashi >= 0.30 or kado >= 0.25:
        bullets.append(f"場傾向はまくり{_pct(makuri, 0)}・差し{_pct(sashi, 0)}・カド{_pct(kado, 0)}")
    wb = str(env.get("wind_bucket") or "")
    if wb in {"head", "head_light", "cross"}:
        bullets.append("向かい風・横風帯で差し/まくりが伸びやすい")

    if head is not None and head >= 2:
        b = boats.get(head)
        raw = (b.raw if b else {}) or {}
        st_gap = raw.get("st_gap_vs_inner")
        if st_gap is not None and float(st_gap) > 0.02:
            bullets.append(f"{head}号の展示STがインより優位（差{float(st_gap):+.2f}）→遅れ時に差せる")
        pressure = raw.get("course1_upset_pressure")
        if pressure is not None and float(pressure) >= 0.20:
            bullets.append(f"{head}号にイン崩れの外圧あり")
        if head in set(upset_candidates or []):
            bullets.append(f"{head}号は本命遅れ時の受益候補")
        if head >= 4:
            bullets.append(f"外枠{head}号頭の展開崩れを拾う")

    if kind != "win" and honmei_combo and combo and set(combo) & set(honmei_combo):
        overlap = sorted(set(combo) & set(honmei_combo))
        bullets.append(f"本命との共通艇{'-'.join(map(str, overlap))}で完全矛盾を避ける")

    if not bullets:
        bullets.append("本命崩れ時の保険の流し（ワンダー型）")
    return bullets[:6]


def _why_short_from_bullets(role: str, bullets: list[str]) -> str:
    if not bullets:
        return {"honmei": "本命", "taikou": "対抗", "ana": "穴"}.get(role, "候補")
    first = bullets[0]
    if len(first) <= 40:
        return first
    return first[:38] + "…"


def build_race_narrative(
    features: RaceFeatures,
    result: PredictionResult,
) -> dict[str, Any]:
    """race_thesis / ticket_reasons / styles を返す."""
    env = features.env or {}
    fly_risk = float(
        (result.feature_snapshot or {}).get("course1_fly_risk")
        or env.get("course1_fly_risk")
        or 0.0
    )
    tickets = result.tickets or (result.feature_snapshot or {}).get("tickets") or {}
    rankings = result.rankings or []
    win_probs = result.win_probs or {}
    upset = list(result.upset_candidates or [])

    # レース全体の読み
    thesis_parts: list[str] = []
    top = rankings[0] if rankings else None
    if top is not None:
        p = float(win_probs.get(top) or 0)
        thesis_parts.append(
            f"本線は{top}号艇（単勝{_pct(p, 1)}）。実績・コース・展示を厚く見るノブ型。"
        )
    in_wr = env.get("venue_in_win_rate")
    if in_wr is not None:
        thesis_parts.append(f"この場のイン1着率は{_pct(float(in_wr), 0)}。")
    if fly_risk >= 0.45:
        thesis_parts.append(
            f"一方で1号艇飛びリスク{_pct(fly_risk, 0)}があり、本命がスタート/1角で遅れた場合の展開崩れ（ワンダー型）を3本目に用意する。"
        )
    elif fly_risk >= 0.30:
        thesis_parts.append(f"飛びリスクは中程度（{_pct(fly_risk, 0)}）。本命遅れ時の受益艇を穴でカバーする。")
    else:
        thesis_parts.append("インが残る流れを基調に、穴は本命遅れ時の保険に留める。")
    delay_thesis = (result.feature_snapshot or {}).get("delay_thesis") or ""
    if delay_thesis:
        thesis_parts.append(delay_thesis)
    wind = env.get("wind_speed")
    if wind is not None and float(wind) >= 4:
        thesis_parts.append(f"風速{float(wind):.0f}m帯で決まり手が流れやすい。")

    race_thesis = "".join(thesis_parts)

    ticket_reasons: dict[str, list[dict[str, Any]]] = {}
    for kind in ("win", "sanrenpuku", "sanrentan"):
        rows = list(tickets.get(kind) or [])
        honmei_combo = [int(x) for x in (rows[0].get("combo") or [])] if rows else []
        kind_reasons: list[dict[str, Any]] = []
        for row in rows:
            role = str(row.get("role") or "honmei")
            combo = [int(x) for x in (row.get("combo") or [])]
            if role == "honmei":
                w = combo[0] if combo else (top or 1)
                bullets = _honmei_bullets(features, w, float(win_probs.get(w) or 0))
            elif role == "taikou":
                bullets = _taikou_bullets(features, combo, honmei_combo, kind)
            else:
                bullets = _ana_bullets(
                    features,
                    combo,
                    fly_risk=fly_risk,
                    upset_candidates=upset,
                    honmei_combo=honmei_combo,
                    kind=kind,
                )
            why = row.get("why_short") or _why_short_from_bullets(role, bullets)
            # チケット側 why_short も同期
            row["why_short"] = why
            kind_reasons.append(
                {
                    "role": role,
                    "role_label": row.get("role_label"),
                    "style": row.get("style"),
                    "style_label": row.get("style_label"),
                    "label": row.get("label"),
                    "combo": combo,
                    "why_short": why,
                    "bullets": bullets,
                }
            )
        ticket_reasons[kind] = kind_reasons

    return {
        "race_thesis": race_thesis,
        "ticket_reasons": ticket_reasons,
        "styles": {"honmei": "nobu", "taikou": "nobu", "ana": "wonder"},
    }
