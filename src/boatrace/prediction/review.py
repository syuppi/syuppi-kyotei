"""確定結果との突き合わせ：的中/外れの復習文を生成する."""

from __future__ import annotations

from typing import Any


def _pct(v: float | None, digits: int = 0) -> str:
    if v is None:
        return "-"
    return f"{float(v) * 100:.{digits}f}%"


def _as_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _combo_list(row: dict[str, Any]) -> list[int]:
    raw = row.get("combo")
    if isinstance(raw, list):
        return [int(x) for x in raw]
    label = str(row.get("label") or "")
    if "-" in label:
        return [int(x) for x in label.split("-") if x.isdigit()]
    if label.isdigit():
        return [int(label)]
    return []


def compute_hit_flags(
    result: dict[str, Any] | None,
    tickets: dict[str, Any] | None,
) -> dict[str, Any]:
    """UI / LearningService と同じ候補カバー定義で的中判定."""
    r = result or {}
    rank1 = _as_int(r.get("rank1") if "rank1" in r else r.get("rank1_waku"))
    rank2 = _as_int(r.get("rank2") if "rank2" in r else r.get("rank2_waku"))
    rank3 = _as_int(r.get("rank3") if "rank3" in r else r.get("rank3_waku"))
    has_result = rank1 is not None
    true_top3 = [x for x in (rank1, rank2, rank3) if x is not None]
    t = tickets or {}

    win_rows = list(t.get("win") or [])
    sp_rows = list(t.get("sanrenpuku") or [])
    st_rows = list(t.get("sanrentan") or [])

    hit_win = False
    win_hit_label = None
    if has_result:
        for row in win_rows:
            combo = _combo_list(row)
            if combo and combo[0] == rank1:
                hit_win = True
                win_hit_label = row.get("label") or str(combo[0])
                break

    hit_trio = False
    trio_hit_label = None
    if len(true_top3) == 3:
        need = set(true_top3)
        for row in sp_rows:
            combo = _combo_list(row)
            if len(combo) == 3 and set(combo) == need:
                hit_trio = True
                trio_hit_label = row.get("label") or "-".join(map(str, sorted(combo)))
                break

    hit_tf = False
    tf_hit_label = None
    if len(true_top3) == 3:
        for row in st_rows:
            combo = _combo_list(row)
            if combo == true_top3:
                hit_tf = True
                tf_hit_label = row.get("label") or "-".join(map(str, combo))
                break

    return {
        "has_result": has_result,
        "rank1": rank1,
        "rank2": rank2,
        "rank3": rank3,
        "hit_win": hit_win,
        "hit_trio": hit_trio,
        "hit_tf": hit_tf,
        "any_hit": hit_win or hit_trio or hit_tf,
        "win_hit_label": win_hit_label,
        "trio_hit_label": trio_hit_label,
        "tf_hit_label": tf_hit_label,
    }


def _boat_raw(snap: dict[str, Any], waku: int) -> dict[str, Any]:
    boats = (snap or {}).get("boats") or {}
    node = boats.get(str(waku)) or boats.get(waku) or {}
    if isinstance(node, dict):
        return dict(node.get("raw") or {})
    return {}


def _boat_values(snap: dict[str, Any], waku: int) -> dict[str, Any]:
    boats = (snap or {}).get("boats") or {}
    node = boats.get(str(waku)) or boats.get(waku) or {}
    if isinstance(node, dict):
        return dict(node.get("values") or {})
    return {}


def _entry_map(entry_results: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in entry_results or []:
        w = _as_int(row.get("waku"))
        if w is not None:
            out[w] = row
    return out


def _ticket_summary(tickets: dict[str, Any] | None) -> dict[str, list[str]]:
    t = tickets or {}
    return {
        "win": [str(r.get("label") or "") for r in (t.get("win") or [])[:3]],
        "sanrenpuku": [str(r.get("label") or "") for r in (t.get("sanrenpuku") or [])[:3]],
        "sanrentan": [str(r.get("label") or "") for r in (t.get("sanrentan") or [])[:3]],
    }


def _role_of_winning_ticket(tickets: dict[str, Any] | None, kind: str, label: str | None) -> str | None:
    if not label:
        return None
    for row in (tickets or {}).get(kind) or []:
        if str(row.get("label")) == str(label):
            return row.get("role_label") or row.get("role")
    return None


def _kimarite_lesson(kimarite: str | None, rank1: int | None, env: dict[str, Any]) -> str | None:
    if not kimarite:
        return None
    in_wr = env.get("venue_in_win_rate")
    makuri = float(env.get("venue_makuri_rate") or 0)
    sashi = float(env.get("venue_sashi_rate") or 0)
    if kimarite == "逃げ" and rank1 == 1:
        base = "決まり手は逃げ。インが残る流れだった"
        if in_wr is not None:
            base += f"（場のイン1着率{_pct(float(in_wr), 0)}）"
        return base
    if kimarite in {"まくり", "まくり差し"}:
        return f"決まり手は{kimarite}。インが崩され外から差した展開"
    if kimarite == "差し":
        hint = f"場の差し率{_pct(sashi, 0)}" if sashi else ""
        return f"決まり手は差し。{hint}差しが通る水面だった".strip()
    return f"決まり手は{kimarite}"


def _fly_lesson(
    fly_risk: float,
    rank1: int | None,
    entry_by_waku: dict[int, dict[str, Any]],
) -> str | None:
    st1 = (entry_by_waku.get(1) or {}).get("st")
    if rank1 == 1:
        if fly_risk >= 0.45:
            return f"飛びリスクは高め（{_pct(fly_risk, 0)}）と見たが、実際は1号が残った"
        if st1 is not None:
            return f"1号艇のレースST{float(st1):.2f}でインが残った"
        return None
    if rank1 is not None and rank1 != 1:
        parts = [f"1号艇が頭を取れず{rank1}号が1着"]
        if fly_risk >= 0.40:
            parts.append(f"事前の飛びリスク{_pct(fly_risk, 0)}は方向として合っていた")
        elif fly_risk > 0:
            parts.append(f"事前飛びリスクは{_pct(fly_risk, 0)}と控えめで、イン崩れを薄く見過ぎた可能性")
        if st1 is not None:
            parts.append(f"1号の実ST{float(st1):.2f}")
        return "。".join(parts)
    return None


def _winner_strength_bullets(snap: dict[str, Any], winner: int) -> list[str]:
    raw = _boat_raw(snap, winner)
    values = _boat_values(snap, winner)
    bullets: list[str] = []
    rc = raw.get("racer_course_win")
    starts = raw.get("racer_course_starts")
    course = raw.get("course") or winner
    if rc is not None:
        st_txt = f"（{int(starts)}走）" if starts else ""
        bullets.append(f"勝者{winner}号の{course}コース1着率{_pct(float(rc), 1)}{st_txt}")
    et = raw.get("exhibition_time")
    et_gap = raw.get("ex_time_gap")
    if et is not None:
        tag = "展示上位" if et_gap is not None and float(et_gap) <= 0.03 else "展示反映"
        bullets.append(f"展示タイム{float(et):.2f}（{tag}）")
    est = raw.get("exhibition_st")
    if est is not None:
        bullets.append(f"展示ST{float(est):.2f}")
    form = values.get("recent_form")
    if form is not None and float(form) >= 0.6:
        bullets.append("直近フォームが相対的に良い帯")
    local = raw.get("local_win_rate")
    if local is not None and float(local) > 0:
        bullets.append(f"当地勝率{_pct(float(local), 1)}")
    return bullets[:4]


def _favorite_vs_winner(
    snap: dict[str, Any],
    fav: int | None,
    winner: int | None,
    win_probs: dict[Any, Any] | None,
) -> list[str]:
    if fav is None or winner is None:
        return []
    if fav == winner:
        return [f"本命{fav}号がそのまま1着。軸選定が結果と一致"]
    bullets = [f"本命は{fav}号だったが実際の1着は{winner}号"]
    probs = win_probs or {}
    pf = probs.get(fav, probs.get(str(fav)))
    pw = probs.get(winner, probs.get(str(winner)))
    if pf is not None and pw is not None:
        bullets.append(
            f"単勝モデル確率 本命{_pct(float(pf), 1)} / 勝者{_pct(float(pw), 1)}"
        )
    fav_raw = _boat_raw(snap, fav)
    win_raw = _boat_raw(snap, winner)
    fest = fav_raw.get("exhibition_st")
    west = win_raw.get("exhibition_st")
    if fest is not None and west is not None and float(west) + 0.02 < float(fest):
        bullets.append(
            f"展示STは勝者の方が速かった（本命{float(fest):.2f} / 勝者{float(west):.2f}）"
        )
    fet = fav_raw.get("exhibition_time")
    wet = win_raw.get("exhibition_time")
    if fet is not None and wet is not None and float(wet) + 0.05 < float(fet):
        bullets.append(
            f"展示タイムでも勝者優位（本命{float(fet):.2f} / 勝者{float(wet):.2f}）"
        )
    pressure = win_raw.get("course1_upset_pressure")
    if winner != 1 and pressure is not None and float(pressure) >= 0.2:
        bullets.append(f"勝者側にイン崩れ外圧（{float(pressure):.2f}）があった")
    return bullets[:5]


def _ticket_hit_bullets(
    hits: dict[str, Any],
    tickets: dict[str, Any] | None,
) -> list[str]:
    out: list[str] = []
    if hits.get("hit_win"):
        role = _role_of_winning_ticket(tickets, "win", hits.get("win_hit_label"))
        role_txt = f"（{role}）" if role else ""
        out.append(f"単勝は候補{hits.get('win_hit_label')}{role_txt}で的中")
    else:
        labels = _ticket_summary(tickets)["win"]
        out.append(
            f"単勝は外れ（候補 {', '.join(labels) or 'なし'} / 実際 {hits.get('rank1')}）"
        )

    if hits.get("hit_trio"):
        role = _role_of_winning_ticket(tickets, "sanrenpuku", hits.get("trio_hit_label"))
        role_txt = f"（{role}）" if role else ""
        out.append(f"3連複は{hits.get('trio_hit_label')}{role_txt}で的中")
    else:
        actual = "-".join(
            str(x) for x in (hits.get("rank1"), hits.get("rank2"), hits.get("rank3")) if x
        )
        labels = _ticket_summary(tickets)["sanrenpuku"]
        out.append(f"3連複は外れ（候補 {', '.join(labels) or 'なし'} / 実際 {actual}）")

    if hits.get("hit_tf"):
        role = _role_of_winning_ticket(tickets, "sanrentan", hits.get("tf_hit_label"))
        role_txt = f"（{role}）" if role else ""
        out.append(f"3連単は{hits.get('tf_hit_label')}{role_txt}で的中")
    else:
        actual = "-".join(
            str(x) for x in (hits.get("rank1"), hits.get("rank2"), hits.get("rank3")) if x
        )
        labels = _ticket_summary(tickets)["sanrentan"]
        # 部分一致ヒント
        true_set = {
            x
            for x in (hits.get("rank1"), hits.get("rank2"), hits.get("rank3"))
            if x is not None
        }
        partial = []
        for lab in labels:
            parts = {int(x) for x in lab.split("-") if x.isdigit()}
            if parts and true_set and len(parts & true_set) >= 2:
                partial.append(lab)
        hint = f"。2艇一致あり: {', '.join(partial)}" if partial else ""
        out.append(f"3連単は外れ（候補 {', '.join(labels) or 'なし'} / 実際 {actual}）{hint}")
    return out


def _lesson_summary(hits: dict[str, Any], bullets: list[str]) -> str:
    if hits.get("hit_tf"):
        head = "3連単まで的中。並びまで読めた好例。"
    elif hits.get("hit_trio") and hits.get("hit_win"):
        head = "単勝と3連複が的中。軸とメンバーは当たったが並びは外れた。"
    elif hits.get("hit_win"):
        head = "単勝は的中。軸は正しいが3連のカバーが足りなかった。"
    elif hits.get("hit_trio"):
        head = "3連複のみ的中。軸の単勝を外しつつメンバーは拾えた。"
    else:
        head = "候補はいずれも外れ。展開または軸選定の見直し材料。"
    detail = bullets[0] if bullets else ""
    return f"{head}{detail}"


def build_result_review(
    *,
    result: dict[str, Any] | None,
    tickets: dict[str, Any] | None,
    feature_snapshot: dict[str, Any] | None = None,
    rankings: list[int] | None = None,
    win_probs: dict[Any, Any] | None = None,
    upset_candidates: list[int] | None = None,
    kimarite: str | None = None,
    entry_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """
    確定結果があるレースの復習オブジェクトを返す。
    {
      hit_flags, headline, bullets_hit, bullets_miss, lessons, kimarite, ...
    }
    """
    snap = feature_snapshot or {}
    env = dict(snap.get("env") or {})
    hits = compute_hit_flags(result, tickets)
    if not hits["has_result"]:
        return None

    rank1 = hits["rank1"]
    fly_risk = float(snap.get("course1_fly_risk") or env.get("course1_fly_risk") or 0.0)
    entry_by = _entry_map(entry_results)
    fav = None
    if rankings:
        fav = _as_int(rankings[0])
    elif tickets and (tickets.get("win") or []):
        combo = _combo_list((tickets.get("win") or [])[0])
        fav = combo[0] if combo else None

    ticket_bullets = _ticket_hit_bullets(hits, tickets)
    kim_line = _kimarite_lesson(kimarite, rank1, env)
    fly_line = _fly_lesson(fly_risk, rank1, entry_by)
    fav_vs = _favorite_vs_winner(snap, fav, rank1, win_probs)
    winner_strength = _winner_strength_bullets(snap, rank1) if rank1 else []

    bullets_hit: list[str] = []
    bullets_miss: list[str] = []
    for b in ticket_bullets:
        if "的中" in b:
            bullets_hit.append(b)
        else:
            bullets_miss.append(b)

    context: list[str] = []
    if kim_line:
        context.append(kim_line)
    if fly_line:
        context.append(fly_line)
    context.extend(fav_vs)
    if hits.get("any_hit"):
        context.extend(winner_strength)
    else:
        # 外れ時も勝者の強みを復習
        context.extend(winner_strength)

    if upset_candidates and rank1 in set(int(x) for x in upset_candidates):
        context.append(f"穴候補に挙げていた{rank1}号が実際に1着になった")
    elif upset_candidates and rank1 and rank1 not in set(int(x) for x in upset_candidates) and rank1 != fav:
        context.append(
            f"勝者{rank1}号は穴候補（{', '.join(map(str, upset_candidates))}）に入っていなかった"
        )

    # 実STの並び
    if entry_by and rank1 is not None:
        st_winner = (entry_by.get(rank1) or {}).get("st")
        if st_winner is not None:
            context.append(f"1着{rank1}号のレースSTは{float(st_winner):.2f}")

    lessons: list[str] = []
    if hits.get("hit_tf"):
        lessons.append("本線の並びが結果と一致。同型のコース・展示根拠を次回も優先してよい")
    elif hits.get("hit_trio") and not hits.get("hit_tf"):
        lessons.append("メンバーは合っているので、2・3着の入れ替えパターンをもう1点足す余地")
    elif hits.get("hit_win") and not hits.get("hit_trio"):
        lessons.append("軸は正しい。ヒモの広げ方（対抗・穴の艇セット）を見直す")
    elif not hits.get("any_hit"):
        if rank1 and fav and rank1 != fav:
            lessons.append("本命軸を外した。展示ST差・飛びリスク・決まり手傾向を次レースで再点検")
        else:
            lessons.append("軸は近いが3連カバー外。確率上位以外の並び多様性を確認")

    if kimarite in {"まくり", "まくり差し", "差し"} and fav == 1 and rank1 != 1:
        lessons.append("イン本命が崩れる決まり手。ワンダー型（外枠頭）の配分を厚くする材料")
    if kimarite == "逃げ" and hits.get("hit_win") and fav == 1:
        lessons.append("逃げ残り。ノブ型のイン厚買いが機能した例")

    headline = _lesson_summary(hits, context)

    verdict = "hit" if hits.get("any_hit") else "miss"
    if hits.get("hit_tf"):
        verdict = "tf_hit"
    elif hits.get("hit_trio"):
        verdict = "trio_hit"
    elif hits.get("hit_win"):
        verdict = "win_hit"

    return {
        "verdict": verdict,
        "headline": headline,
        "kimarite": kimarite,
        "hit_flags": {
            "hit_win": hits["hit_win"],
            "hit_trio": hits["hit_trio"],
            "hit_tf": hits["hit_tf"],
            "any_hit": hits["any_hit"],
        },
        "bullets_hit": bullets_hit,
        "bullets_miss": bullets_miss,
        "context": context[:8],
        "lessons": lessons[:4],
        "actual": f"{hits['rank1']}-{hits['rank2']}-{hits['rank3']}",
        "favorite": fav,
    }


def review_from_db_row(
    *,
    pred_tickets: dict[str, Any] | None,
    feature_snapshot: dict[str, Any] | None,
    rankings: list[int] | None,
    win_probs: dict[Any, Any] | None,
    upset_candidates: list[int] | None,
    race_result: Any | None,
) -> dict[str, Any] | None:
    """RaceResult ORM / 簡易オブジェクトから復習を組み立て."""
    if race_result is None:
        return None
    rank1 = getattr(race_result, "rank1_waku", None)
    if rank1 is None and isinstance(race_result, dict):
        rank1 = race_result.get("rank1") or race_result.get("rank1_waku")
    if rank1 is None:
        return None

    if isinstance(race_result, dict):
        result = {
            "rank1": race_result.get("rank1") or race_result.get("rank1_waku"),
            "rank2": race_result.get("rank2") or race_result.get("rank2_waku"),
            "rank3": race_result.get("rank3") or race_result.get("rank3_waku"),
        }
        kimarite = race_result.get("kimarite")
        entry_results = race_result.get("entry_results")
    else:
        result = {
            "rank1": race_result.rank1_waku,
            "rank2": race_result.rank2_waku,
            "rank3": race_result.rank3_waku,
        }
        kimarite = race_result.kimarite
        entry_results = race_result.entry_results

    return build_result_review(
        result=result,
        tickets=pred_tickets,
        feature_snapshot=feature_snapshot,
        rankings=list(rankings or []) if rankings else None,
        win_probs=win_probs,
        upset_candidates=list(upset_candidates or []) if upset_candidates else None,
        kimarite=kimarite,
        entry_results=list(entry_results or []) if entry_results else None,
    )
