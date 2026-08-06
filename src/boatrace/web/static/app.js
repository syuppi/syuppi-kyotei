async function jget(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function pct(v) {
  if (v == null) return "-";
  return `${(Number(v) * 100).toFixed(1)}%`;
}

function todayISO() {
  const d = new Date();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${m}-${day}`;
}

async function fillVenues(selectEl, includeAll = true) {
  const data = await jget("/api/venues");
  const keep = selectEl.value;
  selectEl.innerHTML = "";
  if (includeAll) {
    const all = document.createElement("option");
    all.value = "";
    all.textContent = "すべて";
    selectEl.appendChild(all);
  }
  for (const v of data.items) {
    const opt = document.createElement("option");
    opt.value = v.id;
    opt.textContent = `${v.id} ${v.name}`;
    selectEl.appendChild(opt);
  }
  if (keep && [...selectEl.options].some((o) => o.value === keep)) {
    selectEl.value = keep;
  }
}

/** 指定日の開催場だけをセレクトに入れる（非開催場は出さない） */
async function fillActiveVenues(selectEl, day) {
  const prev = selectEl.value;
  const data = await jget(`/api/venues?day=${encodeURIComponent(day)}`);
  selectEl.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = (data.items && data.items.length)
    ? "場を選択…"
    : "この日の開催場がありません";
  selectEl.appendChild(placeholder);
  for (const v of data.items || []) {
    const opt = document.createElement("option");
    opt.value = v.id;
    const n = v.race_count != null ? `（${v.race_count}R）` : "";
    opt.textContent = `${v.id} ${v.name}${n}`;
    selectEl.appendChild(opt);
  }
  if (prev && [...selectEl.options].some((o) => o.value === prev)) {
    selectEl.value = prev;
  } else {
    selectEl.value = "";
  }
  return data;
}

function fillShares(rows) {
  const list = Array.isArray(rows) ? rows.filter(Boolean) : [];
  const sum = list.reduce((a, r) => a + (Number(r.prob) || 0), 0) || 1;
  return list.map((r) => ({
    ...r,
    stake_share: r.stake_share || (Number(r.prob) || 0) / sum,
  }));
}

function normalizeTickets(item) {
  const tickets = item.tickets && typeof item.tickets === "object" ? item.tickets : {};
  const probs = item.win_probs || {};

  let winTickets = Array.isArray(tickets.win) ? tickets.win : [];
  if (!winTickets.length) {
    winTickets = (item.candidates_win || []).map((w, i) => ({
      rank: i + 1,
      combo: [w],
      label: String(w),
      prob: probs[w] ?? probs[String(w)] ?? 0,
      stake_share: 0,
    }));
  }

  let spTickets = Array.isArray(tickets.sanrenpuku) ? tickets.sanrenpuku : [];
  if (!spTickets.length) {
    const src = item.sanrenpuku || [];
    spTickets = src.slice(0, 3).map((c, i) => {
      const combo = Array.isArray(c) ? c.map(Number) : [];
      return {
        rank: i + 1,
        combo,
        label: [...combo].sort((a, b) => a - b).join("-"),
        prob: 0,
        stake_share: 0,
      };
    });
  }

  let stTickets = Array.isArray(tickets.sanrentan) ? tickets.sanrentan : [];
  if (!stTickets.length) {
    const src = item.sanrentan || (item.rankings ? [item.rankings.slice(0, 3)] : []);
    stTickets = src.slice(0, 3).map((c, i) => {
      const combo = Array.isArray(c) ? c.map(Number) : [];
      return {
        rank: i + 1,
        combo,
        label: combo.join("-"),
        prob: 0,
        stake_share: 0,
      };
    });
  }

  // trio fallback from candidates_trio if still empty
  if (!spTickets.length && item.candidates_trio?.length >= 3) {
    const combo = item.candidates_trio.slice(0, 3).map(Number);
    spTickets = [{
      rank: 1,
      combo: [...combo].sort((a, b) => a - b),
      label: [...combo].sort((a, b) => a - b).join("-"),
      prob: 0,
      stake_share: 1,
    }];
  }

  return {
    wins: fillShares(winTickets).slice(0, 3),
    sps: fillShares(spTickets).slice(0, 3),
    sts: fillShares(stTickets).slice(0, 3),
  };
}

function hitFlags(item, wins, sps, sts) {
  const r = item.result || {};
  const hasResult = r.rank1 != null;
  const trueTop3 = [r.rank1, r.rank2, r.rank3].filter((x) => x != null);
  const hitWin = hasResult && wins.some((t) => Number(t.combo?.[0]) === Number(r.rank1));
  const hitTrio = trueTop3.length === 3 && sps.some((t) => {
    const set = new Set((t.combo || []).map(Number));
    return trueTop3.every((w) => set.has(Number(w))) && set.size === 3;
  });
  const hitTf = trueTop3.length === 3 && sts.some((t) => {
    const c = (t.combo || []).map(Number);
    return c.length === 3
      && c[0] === Number(r.rank1)
      && c[1] === Number(r.rank2)
      && c[2] === Number(r.rank3);
  });
  return {
    hasResult,
    hitWin,
    hitTrio,
    hitTf,
    anyHit: hitWin || hitTrio || hitTf,
  };
}

let _predCache = { data: null, prepared: [] };

function prepareItems(data) {
  return (data.items || []).map((item) => {
    const { wins, sps, sts } = normalizeTickets(item);
    const hits = hitFlags(item, wins, sps, sts);
    return { item, wins, sps, sts, hits };
  });
}

function filterAndSortPrepared(prepared) {
  const filterEl = document.getElementById("hit-filter");
  const sortEl = document.getElementById("hit-sort");
  const filter = filterEl ? filterEl.value : "all";
  const sort = sortEl ? sortEl.value : "venue";

  let rows = prepared.slice();
  rows = rows.filter(({ hits }) => {
    if (filter === "all") return true;
    if (filter === "any_hit") return hits.anyHit;
    if (filter === "win_hit") return hits.hitWin;
    if (filter === "trio_hit") return hits.hitTrio;
    if (filter === "trifecta_hit") return hits.hitTf;
    if (filter === "miss") return hits.hasResult && !hits.anyHit;
    if (filter === "pending") return !hits.hasResult;
    return true;
  });

  const score = (h) => (h.hitTf ? 4 : 0) + (h.hitTrio ? 2 : 0) + (h.hitWin ? 1 : 0);
  rows.sort((a, b) => {
    if (sort === "hit_desc") {
      const d = score(b.hits) - score(a.hits);
      if (d) return d;
    } else if (sort === "hit_asc") {
      const d = score(a.hits) - score(b.hits);
      if (d) return d;
    } else if (sort === "race_no") {
      const d = a.item.race_no - b.item.race_no;
      if (d) return d;
    }
    // default / tie-break: venue then race
    const v = String(a.item.venue_id).localeCompare(String(b.item.venue_id));
    if (v) return v;
    return a.item.race_no - b.item.race_no;
  });
  return rows;
}

async function bootPredictions() {
  const day = document.getElementById("day");
  const venue = document.getElementById("venue");
  const refreshBtn = document.getElementById("refresh");
  const filterEl = document.getElementById("hit-filter");
  const sortEl = document.getElementById("hit-sort");
  const compactEl = document.getElementById("compact-mode");
  const statusEl = document.getElementById("load-status");
  const list = document.getElementById("list");
  const summary = document.getElementById("summary");
  day.value = todayISO();

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = msg || "";
  }

  function setAnalyzeEnabled(on) {
    if (refreshBtn) refreshBtn.disabled = !on;
  }

  function clearRaceView(msg) {
    _predCache = { data: null, prepared: [] };
    if (summary) summary.innerHTML = "";
    if (list) {
      list.innerHTML = `<p class="lede">${msg}</p>`;
    }
  }

  async function refreshVenueOptions() {
    setStatus("開催場を確認中…");
    const data = await fillActiveVenues(venue, day.value);
    const has = (data.items || []).length > 0;
    setAnalyzeEnabled(Boolean(venue.value));
    if (!has) {
      clearRaceView("この日の開催場がありません。日付を変えるか、先に出走表を取得してください。");
      setStatus("開催場なし");
      return false;
    }
    if (!venue.value) {
      clearRaceView("場を選択すると、その場だけ解析・表示します。");
      setStatus(`${data.items.length}場が開催中 — 場を選んでください`);
      return false;
    }
    return true;
  }

  async function load(opts = {}) {
    const { refresh = false, prepare = false, force = false } = opts;
    if (!venue.value) {
      clearRaceView("場を選択すると、その場だけ解析・表示します。");
      setStatus("場未選択");
      setAnalyzeEnabled(false);
      return;
    }
    setAnalyzeEnabled(true);
    const params = new URLSearchParams({ day: day.value, venue_id: venue.value });
    try {
      let data;
      if (prepare) {
        setStatus("出走表を取得して予想中…（この場のみ・高速モード）");
        const prep = new URLSearchParams(params);
        prep.set("fast", "true");
        if (force) prep.set("force", "true");
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 90000);
        try {
          const res = await fetch(`/api/day/prepare?${prep}`, {
            method: "POST",
            signal: controller.signal,
          });
          clearTimeout(timer);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          data = await res.json();
        } catch (e) {
          clearTimeout(timer);
          if (e.name === "AbortError") {
            throw new Error("タイムアウト（処理が長すぎます。もう一度お試しください）");
          }
          const msg = String(e.message || e);
          setStatus(`${msg} → 保存済み予想を表示します`);
          data = await jget(`/api/predictions/today?${params}`);
        }
      } else if (refresh) {
        setStatus("再予想中…");
        params.set("refresh", "true");
        data = await jget(`/api/predictions/today?${params}`);
      } else {
        setStatus("読み込み中…");
        data = await jget(`/api/predictions/today?${params}`);
        if (!data.items || data.items.length === 0) {
          setStatus("データが無いためこの場を取得・予想しています…");
          const prep = new URLSearchParams(params);
          prep.set("fast", "true");
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), 90000);
          try {
            const res = await fetch(`/api/day/prepare?${prep}`, {
              method: "POST",
              signal: controller.signal,
            });
            clearTimeout(timer);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            data = await res.json();
          } catch (e) {
            clearTimeout(timer);
            throw new Error(
              e.name === "AbortError"
                ? "タイムアウトしました。少し待って再読み込みしてください"
                : (e.message || e)
            );
          }
          // prepare 後に場一覧も更新（カードが増えた場合）
          await fillActiveVenues(venue, day.value);
          if (params.get("venue_id") && [...venue.options].some((o) => o.value === params.get("venue_id"))) {
            venue.value = params.get("venue_id");
          }
        }
      }
      _predCache = { data, prepared: prepareItems(data) };
      renderPredictions();
      const oddsN = (data.items || []).filter((x) => x.has_odds).length;
      const cached = data.cached ? "（キャッシュ）" : "";
      setStatus(`更新済${cached}（${venue.value} / ${data.count || 0}R / オッズあり ${oddsN}R）`);
    } catch (e) {
      setStatus(`エラー: ${e.message || e}`);
      throw e;
    }
  }

  refreshBtn.addEventListener("click", () => load({ prepare: true, force: true }));
  day.addEventListener("change", async () => {
    const ok = await refreshVenueOptions();
    if (ok) await load({ prepare: false });
  });
  venue.addEventListener("change", () => {
    setAnalyzeEnabled(Boolean(venue.value));
    if (venue.value) load({ prepare: false });
    else clearRaceView("場を選択すると、その場だけ解析・表示します。");
  });
  if (filterEl) filterEl.addEventListener("change", () => renderPredictions());
  if (sortEl) sortEl.addEventListener("change", () => renderPredictions());
  if (compactEl) compactEl.addEventListener("change", () => renderPredictions());

  const ready = await refreshVenueOptions();
  if (ready) await load({ prepare: false });
}

function renderPredictions() {
  const summary = document.getElementById("summary");
  const list = document.getElementById("list");
  const data = _predCache.data;
  if (!data) return;

  const prepared = _predCache.prepared;
  const filtered = filterAndSortPrepared(prepared);
  const compact = document.getElementById("compact-mode")?.checked;

  const upsetCount = prepared.filter((x) => x.item.has_upset).length;
  const hitAny = prepared.filter((x) => x.hits.anyHit).length;
  const hitTf = prepared.filter((x) => x.hits.hitTf).length;
  const hitTrio = prepared.filter((x) => x.hits.hitTrio).length;
  const settled = prepared.filter((x) => x.hits.hasResult).length;

  summary.innerHTML = `
    <div class="stat"><div class="label">対象レース</div><div class="value">${data.count}</div></div>
    <div class="stat"><div class="label">表示中</div><div class="value">${filtered.length}</div></div>
    <div class="stat"><div class="label">的中（いずれか）</div><div class="value">${hitAny}<span class="sub"> / ${settled}</span></div></div>
    <div class="stat"><div class="label">3連複 / 3連単的中</div><div class="value">${hitTrio} / ${hitTf}</div></div>
    <div class="stat"><div class="label">穴候補あり</div><div class="value">${upsetCount}</div></div>
    <div class="stat"><div class="label">日付</div><div class="value" style="font-size:1.1rem">${data.date}</div></div>
  `;

  if (!prepared.length) {
    list.innerHTML = `<p class="lede">予測がありません。デモデータを投入するか「再予測」を実行してください。</p>`;
    return;
  }
  if (!filtered.length) {
    list.innerHTML = `<p class="lede">条件に合うレースがありません。フィルタを変更してください。</p>`;
    return;
  }

  list.innerHTML = filtered.map(({ item, wins, sps, sts, hits }, idx) => {
    const probs = item.win_probs || {};
    const wakus = [1, 2, 3, 4, 5, 6].map((w) => {
      const p = probs[w] ?? probs[String(w)] ?? 0;
      const top = wins.some((t) => Number(t.combo?.[0]) === w);
      return `<div class="waku ${top ? "top" : ""}"><div class="num">${w}</div><div class="prob">${pct(p)}</div></div>`;
    }).join("");
    const reasonTop = (item.reasons?.[item.rankings?.[0]] || item.reasons?.[String(item.rankings?.[0])] || []).slice(0, compact ? 2 : 3)
      .map((r) => `<div>・ ${r}</div>`).join("");
    const evReasons = (item.ev_reasons || []).slice(0, 3)
      .map((r) => `<div class="ev-reason">・ ${r}</div>`).join("");
    const oddsBadge = item.has_odds ? '<span class="badge hit">オッズ反映</span>' : '<span class="badge upset">オッズ未取得</span>';
    const thesis = item.race_thesis
      ? `<div class="race-thesis"><strong>展開の読み</strong><p>${item.race_thesis}</p></div>`
      : "";
    const ticketReasons = item.ticket_reasons || {};

    const reasonDetails = (kind, t) => {
      const rows = ticketReasons[kind] || [];
      const match = rows.find((r) => r.label === t.label || (r.role && r.role === t.role && String(r.label) === String(t.label)));
      const byRole = rows.find((r) => r.role === t.role);
      const block = match || byRole;
      const bullets = block?.bullets || [];
      if (compact) {
        const short = t.why_short || block?.why_short || "";
        return short ? `<div class="why-short">${short}</div>` : "";
      }
      if (!bullets.length && !t.why_short) return "";
      const lis = (bullets.length ? bullets : [t.why_short]).map((b) => `<li>${b}</li>`).join("");
      return `<details class="ticket-why"><summary>根拠</summary><ul>${lis}</ul></details>`;
    };

    const fmtTicket = (t, kind) => {
      const share = Math.round((Number(t.stake_share) || 0) * 100);
      const pr = Number(t.prob) || 0;
      const odd = t.odds != null ? Number(t.odds) : null;
      const ev = t.ev != null ? Number(t.ev) : null;
      let mark = "";
      if (hits.hasResult) {
        if (kind === "win" && Number(t.combo?.[0]) === Number(item.result.rank1)) mark = ' <span class="badge hit">的中</span>';
        if (kind === "trio") {
          const trueTop3 = [item.result.rank1, item.result.rank2, item.result.rank3];
          const set = new Set((t.combo || []).map(Number));
          if (trueTop3.every((w) => set.has(Number(w))) && set.size === 3) mark = ' <span class="badge hit">的中</span>';
        }
        if (kind === "tf") {
          const c = (t.combo || []).map(Number);
          if (c[0] === Number(item.result.rank1) && c[1] === Number(item.result.rank2) && c[2] === Number(item.result.rank3)) {
            mark = ' <span class="badge hit">的中</span>';
          }
        }
      }
      const valueBadge = t.value_tag === "割安"
        ? ' <span class="badge hit">割安</span>'
        : (t.value_tag === "割高" ? ' <span class="badge upset">割高</span>' : "");
      const oddsTxt = odd != null ? ` / オッズ${odd.toFixed(1)}` : "";
      const evTxt = ev != null ? ` / EV${ev.toFixed(2)}` : "";
      const kindKey = kind === "trio" ? "sanrenpuku" : (kind === "tf" ? "sanrentan" : "win");
      const prefix = t.role_label
        ? `【${t.role_label}・${(t.style_label || "").replace(/型.*/, "型")}】`
        : "";
      return `<li>
        <div class="ticket-line"><strong>${prefix}${t.label}</strong>${mark}${valueBadge}
          <span class="muted">確率${pct(pr)}${oddsTxt}${evTxt} / 配分${share}%</span></div>
        ${reasonDetails(kindKey, t)}
      </li>`;
    };
    const ticketBlock = (title, rows, kind) => `
      <div class="ticket-block">
        <div class="ticket-title">${title}</div>
        <ol class="ticket-list">${rows.map((t) => fmtTicket(t, kind)).join("") || "<li>候補なし</li>"}</ol>
      </div>`;

    const ex = item.exhibition || {};
    const exEntries = ex.entries || [];
    const exPhase = ex.phase || (ex.complete ? "試走反映済" : "試走前");
    const exTable = (!compact && exEntries.length) ? `
      <div class="ex-wrap">
        <div class="ticket-title">試走・展示 <span class="badge">${exPhase}</span></div>
        <table class="ex-table">
          <thead><tr><th>枠</th><th>進入</th><th>展示</th><th>ST展示</th><th>チルト</th><th>調整重量</th></tr></thead>
          <tbody>
            ${exEntries.map((e) => {
              const bestT = Number(e.waku) === Number(ex.best_time_waku);
              const bestS = Number(e.waku) === Number(ex.best_st_waku);
              return `<tr class="${bestT || bestS ? "best" : ""}">
                <td>${e.waku}</td>
                <td>${e.course ?? "-"}</td>
                <td>${e.exhibition_time != null ? Number(e.exhibition_time).toFixed(2) : "-"}${bestT ? " ★" : ""}</td>
                <td>${e.exhibition_st != null ? Number(e.exhibition_st).toFixed(2) : "-"}${bestS ? " ★" : ""}</td>
                <td>${e.tilt != null ? e.tilt : "-"}</td>
                <td>${e.weight_adjustment != null ? e.weight_adjustment : "-"}</td>
              </tr>`;
            }).join("")}
          </tbody>
        </table>
      </div>` : "";

    const scenarioComments = item.scenarios || (item.scenario_detail && item.scenario_detail.comments) || [];
    const scenarioHtml = (!compact && scenarioComments.length) ? `
      <div class="scenario-wrap">
        <div class="ticket-title">試走シナリオ（期待度の変化）</div>
        <ul class="scenario-list">
          ${scenarioComments.slice(0, 8).map((c) => `<li>${c}</li>`).join("")}
        </ul>
      </div>` : "";

    const resultHtml = (() => {
      if (!hits.hasResult) {
        return `<div class="meta">結果: 未確定（予測のみ）</div>`;
      }
      return `<div class="meta">結果: ${item.result.rank1}-${item.result.rank2}-${item.result.rank3}
        <span class="badge ${hits.hitWin ? "hit" : "upset"}">${hits.hitWin ? "単勝的中" : "単勝外れ"}</span>
        <span class="badge ${hits.hitTrio ? "hit" : "upset"}">${hits.hitTrio ? "3連複的中" : "3連複外れ"}</span>
        <span class="badge ${hits.hitTf ? "hit" : "upset"}">${hits.hitTf ? "3連単的中" : "3連単外れ"}</span>
      </div>`;
    })();

    const hitClass = hits.hitTf ? "is-tf-hit" : hits.hitTrio ? "is-trio-hit" : hits.hitWin ? "is-win-hit" : "";

    return `
      <article class="race ${hitClass} ${compact ? "compact" : ""}" style="animation-delay:${Math.min(idx, 12) * 0.03}s">
        <div class="race-head">
          <h2>${item.venue_name} ${item.race_no}R <span class="badge">${item.model_name || ""}</span>
            ${oddsBadge}
            ${item.has_upset ? '<span class="badge upset">穴あり</span>' : ""}
            ${item.status === "scheduled" ? '<span class="badge">予想中</span>' : ""}
          </h2>
          ${resultHtml}
        </div>
        ${thesis}
        <div class="wakus">${wakus}</div>
        <div class="ticket-grid">
          ${ticketBlock("単勝 候補", wins, "win")}
          ${ticketBlock("3連複 候補", sps, "trio")}
          ${ticketBlock("3連単 候補", sts, "tf")}
        </div>
        ${exTable}
        ${scenarioHtml}
        <div class="reasons">${item.race_thesis ? "" : `<strong>本命の理由</strong>${reasonTop || "<div>・ データ不足</div>"}`}
          ${evReasons ? `<div class="ev-box"><strong>期待値の理由</strong>${evReasons}</div>` : ""}
          ${!compact && item.race_title ? `<div class="meta">番組: ${item.race_title}</div>` : ""}
        </div>
      </article>
    `;
  }).join("");
}

async function bootVenues() {
  const select = document.getElementById("venue");
  await fillVenues(select, false);
  async function load() {
    const data = await jget(`/api/venues/${select.value}/trends`);
    const el = document.getElementById("venue-detail");
    const stats = (data.course_stats || []).filter((s) => s.condition_key === "all");
    const rows = stats.map((s) => `
      <tr>
        <td>${s.course}コース</td>
        <td>${s.starts}</td>
        <td>${pct(s.win_rate)}</td>
        <td>${pct(s.quinella_rate)}</td>
        <td>${pct(s.trio_rate)}</td>
      </tr>`).join("");
    const biases = (data.biases || []).map((b) => `
      <tr><td>${b.feature_key}</td><td>${b.coefficient.toFixed(3)}</td><td>${b.sample_count}</td></tr>
    `).join("");
    el.innerHTML = `
      <div class="summary-row">
        <div class="stat"><div class="label">場</div><div class="value" style="font-size:1.2rem">${data.venue.name}</div></div>
        <div class="stat"><div class="label">潮汐影響</div><div class="value" style="font-size:1.2rem">${data.venue.tide_sensitive ? "強" : "弱"}</div></div>
        <div class="stat"><div class="label">イン有利度</div><div class="value">${pct(data.venue.typical_in_advantage)}</div></div>
      </div>
      <div class="table-wrap" style="margin-bottom:1rem">
        <table>
          <thead><tr><th>コース</th><th>出走</th><th>1着率</th><th>2連対率</th><th>3連対率</th></tr></thead>
          <tbody>${rows || "<tr><td colspan=5>データなし（結果学習後に更新）</td></tr>"}</tbody>
        </table>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>特徴量</th><th>場別補正</th><th>サンプル</th></tr></thead>
          <tbody>${biases || "<tr><td colspan=3>補正未学習</td></tr>"}</tbody>
        </table>
      </div>
    `;
  }
  select.addEventListener("change", load);
  if (select.options.length) await load();
}

async function bootAccuracy() {
  const summary = await jget("/api/accuracy/summary?days=30");
  document.getElementById("acc-summary").innerHTML = `
    <div class="stat"><div class="label">対象レース</div><div class="value">${summary.n_races}</div></div>
    <div class="stat"><div class="label">単勝(〜3候補)</div><div class="value">${pct(summary.win_rate)}</div></div>
    <div class="stat"><div class="label">2連対的中率</div><div class="value">${pct(summary.quinella_rate)}</div></div>
    <div class="stat"><div class="label">3連複(〜3候補)</div><div class="value">${pct(summary.trio_rate)}</div></div>
    <div class="stat"><div class="label">3連単(〜3候補)</div><div class="value">${pct(summary.trifecta_rate || 0)}</div></div>
  `;
  const daily = await jget("/api/accuracy/daily?days=14");
  const rows = daily.items.map((r) => `
    <tr>
      <td>${r.stat_date}</td>
      <td>${r.slice_key}</td>
      <td>${r.n_races}</td>
      <td>${pct(r.win_rate)}</td>
      <td>${pct(r.quinella_rate)}</td>
      <td>${pct(r.trio_rate)}</td>
      <td>${pct(r.trifecta_rate || 0)}</td>
    </tr>`).join("");
  document.getElementById("acc-table").innerHTML = `
    <table>
      <thead><tr><th>日付</th><th>条件帯</th><th>N</th><th>1着</th><th>2連対</th><th>3連複</th><th>3連単</th></tr></thead>
      <tbody>${rows || "<tr><td colspan=7>集計なし</td></tr>"}</tbody>
    </table>
  `;
}

async function bootSearch() {
  const venue = document.getElementById("venue");
  await fillVenues(venue, true);
  document.getElementById("run").addEventListener("click", async () => {
    const params = new URLSearchParams();
    if (venue.value) params.set("venue_id", venue.value);
    const waku = document.getElementById("waku").value;
    const racer = document.getElementById("racer").value;
    const wind = document.getElementById("wind").value;
    const tide = document.getElementById("tide").value;
    if (waku) params.set("waku", waku);
    if (racer) params.set("racer_id", racer);
    if (wind) params.set("min_wind", wind);
    if (tide) params.set("near_high_tide", tide);
    const data = await jget(`/api/search?${params}`);
    const rows = data.items.map((r) => `
      <tr>
        <td>${r.race_date}</td>
        <td>${r.venue_name}</td>
        <td>${r.race_no}R</td>
        <td>${r.waku}</td>
        <td>${r.racer_name || r.racer_id}</td>
        <td>${r.exhibition_time ?? "-"}</td>
        <td>${r.wind_speed ?? "-"} ${r.wind_direction || ""}</td>
        <td>${r.near_high_tide == null ? "-" : (r.near_high_tide ? "満潮近" : "")}</td>
        <td>${r.result_rank ?? "-"}</td>
      </tr>`).join("");
    document.getElementById("results").innerHTML = `
      <table>
        <thead><tr><th>日付</th><th>場</th><th>R</th><th>枠</th><th>選手</th><th>展示</th><th>風</th><th>潮</th><th>着</th></tr></thead>
        <tbody>${rows || "<tr><td colspan=9>該当なし</td></tr>"}</tbody>
      </table>
    `;
  });
}
