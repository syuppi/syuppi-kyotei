async function jget(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/** 取得・解析中の表示（PWAでも状態が分かるように） */
let _fetchProgressDepth = 0;

function isStandalonePwa() {
  return (
    window.matchMedia("(display-mode: standalone)").matches
    || window.navigator.standalone === true
  );
}

function reloadAppPage() {
  window.location.reload();
}

function initReloadButtons() {
  const bind = (id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", () => reloadAppPage());
  };
  bind("page-reload");
  bind("header-reload");
  bind("fetch-done-reload");

  const headerReload = document.getElementById("header-reload");
  if (headerReload && isStandalonePwa()) {
    headerReload.hidden = false;
  }

  const doneBanner = document.getElementById("fetch-done-banner");
  const doneContinue = document.getElementById("fetch-done-continue");
  if (doneContinue && doneBanner) {
    doneContinue.addEventListener("click", () => {
      doneBanner.hidden = true;
    });
  }
}

function setFetchProgress(active, { title, detail } = {}) {
  const panel = document.getElementById("fetch-progress");
  const titleEl = document.getElementById("fetch-progress-title");
  const detailEl = document.getElementById("fetch-progress-detail");
  if (!panel) return;

  if (active) {
    _fetchProgressDepth += 1;
    panel.hidden = false;
    if (titleEl) titleEl.textContent = title || "取得中…";
    if (detailEl) detailEl.textContent = detail || "";
    document.body.classList.add("is-fetching");
    hideFetchDoneBanner();
    return;
  }

  _fetchProgressDepth = Math.max(0, _fetchProgressDepth - 1);
  if (_fetchProgressDepth === 0) {
    panel.hidden = true;
    document.body.classList.remove("is-fetching");
  }
}

function updateFetchProgress({ title, detail } = {}) {
  const titleEl = document.getElementById("fetch-progress-title");
  const detailEl = document.getElementById("fetch-progress-detail");
  if (titleEl && title) titleEl.textContent = title;
  if (detailEl && detail != null) detailEl.textContent = detail;
}

function showFetchDoneBanner(message) {
  const banner = document.getElementById("fetch-done-banner");
  const msgEl = document.getElementById("fetch-done-message");
  if (!banner) return;
  if (msgEl) {
    msgEl.textContent = message || "取得が完了しました。ページ更新で最新表示にできます。";
  }
  banner.hidden = false;
}

function hideFetchDoneBanner() {
  const banner = document.getElementById("fetch-done-banner");
  if (banner) banner.hidden = true;
}

async function waitForServerReady(maxMs = 120000) {
  setFetchProgress(true, {
    title: "サーバー起動中…",
    detail: "スリープ復帰後は1〜2分かかることがあります。この画面は閉じずにお待ちください。",
  });
  const started = Date.now();
  let attempt = 0;
  while (Date.now() - started < maxMs) {
    attempt += 1;
    try {
      const res = await fetch("/health", { cache: "no-store" });
      if (res.ok) {
        const health = await res.json();
        if (health.status === "ok") {
          setFetchProgress(false);
          return health;
        }
      }
    } catch (_) {
      /* retry */
    }
    updateFetchProgress({
      detail: `起動待ち…（${attempt}回目）スリープ復帰後は1〜2分かかることがあります。`,
    });
    await new Promise((r) => setTimeout(r, 3000));
  }
  setFetchProgress(false);
  throw new Error("サーバー起動がタイムアウトしました。しばらく待って「ページ更新」を押してください。");
}

function todayISO() {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Tokyo",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
}

document.addEventListener("DOMContentLoaded", initReloadButtons);

function pct(v) {
  if (v == null) return "-";
  return `${(Number(v) * 100).toFixed(1)}%`;
}

let _ticketRankStats = null;

async function loadTicketRankStats() {
  if (_ticketRankStats) return _ticketRankStats;
  try {
    _ticketRankStats = await jget("/api/accuracy/ticket-ranks?days=14");
  } catch (_) {
    _ticketRankStats = null;
  }
  return _ticketRankStats;
}

function displayTicketRankBlock(payload) {
  if (!payload) return null;
  return payload.display || payload.baseline || payload;
}

function rankHitRate(kind, rank) {
  const block = displayTicketRankBlock(_ticketRankStats);
  if (!block) return null;
  const key = kind === "trio" || kind === "sanrenpuku" ? "sanrenpuku" : "sanrentan";
  const rows = (block[key] || {}).ranks || [];
  const row = rows.find((r) => Number(r.rank) === Number(rank));
  return row ? Number(row.hit_rate) : null;
}

function renderTicketRankPanel(el, payload, { compact = false } = {}) {
  if (!el) return;
  const block = displayTicketRankBlock(payload);
  if (!block || !block.sanrenpuku) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  el.hidden = false;
  const period = block.period || {};
  const trio = block.sanrenpuku;
  const tf = block.sanrentan || {};
  const fmtRanks = (rows) => (rows || []).map((r) => {
    const name = r.label || `${r.rank}番手`;
    return `<span class="rank-pill"><em>${name}</em> ${pct(r.hit_rate)}</span>`;
  }).join("");
  const pre = block.pre_exhibition;
  const preNote = (!compact && pre)
    ? `<p class="meta">展示前モードでも同傾向（3連複いずれか ${pct(pre.sanrenpuku?.any_rate)} / 本命 ${pct(pre.sanrenpuku?.ranks?.[0]?.hit_rate)}）</p>`
    : "";
  el.innerHTML = `
    <div class="ticket-rank-head">
      <strong>候補順位別の過去的中率</strong>
      <span class="meta">${block.label || ""} · ${period.start || "?"}〜${period.end || "?"} · n=${block.n_races || "-"}</span>
    </div>
    <p class="meta">${block.note || "各順位の候補が単体で的中した割合（カバー全体の的中率とは別）"}</p>
    <div class="ticket-rank-grid">
      <div>
        <div class="ticket-title">3連複（いずれか ${pct(trio.any_rate)}）</div>
        <div class="rank-pills">${fmtRanks(trio.ranks)}</div>
      </div>
      <div>
        <div class="ticket-title">3連単（いずれか ${pct(tf.any_rate)}）</div>
        <div class="rank-pills">${fmtRanks(tf.ranks)}</div>
      </div>
    </div>
    ${preNote}
  `;
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
async function fillActiveVenues(selectEl, day, opts = {}) {
  const prev = selectEl.value;
  const confidentOnly = Boolean(opts.confidentOnly);
  const qs = new URLSearchParams({ day });
  if (confidentOnly) qs.set("confident_only", "true");
  const data = await jget(`/api/venues?${qs}`);
  selectEl.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = (data.items && data.items.length)
    ? "場を選択…"
    : (confidentOnly ? "自信ありのある場がありません" : "この日の開催場がありません");
  selectEl.appendChild(placeholder);

  // 全場の自信ありだけ見るショートカット
  const confTotal = Number(data.confident_total || 0);
  if (confTotal > 0) {
    const allConf = document.createElement("option");
    allConf.value = "__all_confident__";
    allConf.textContent = `自信あり（全場・${confTotal}R）`;
    selectEl.appendChild(allConf);
  }

  for (const v of data.items || []) {
    const opt = document.createElement("option");
    opt.value = v.id;
    const n = v.race_count != null ? `${v.race_count}R` : "";
    const c = v.confident_count != null ? `自信${v.confident_count}` : "";
    const meta = [n, c].filter(Boolean).join(" / ");
    opt.textContent = meta ? `${v.id} ${v.name}（${meta}）` : `${v.id} ${v.name}`;
    if (v.confident_count > 0) opt.dataset.confident = "1";
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
    spTickets = src.slice(0, 5).map((c, i) => {
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
    stTickets = src.slice(0, 5).map((c, i) => {
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
    sps: fillShares(spTickets).slice(0, 5),
    sts: fillShares(stTickets).slice(0, 5),
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
  rows = rows.filter(({ hits, item }) => {
    if (filter === "all") return true;
    if (filter === "confident") return Boolean(item.is_confident || (item.confidence || {}).is_confident);
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
    if (sort === "confidence") {
      const ca = Number(a.item.confidence_score ?? a.item.confidence?.score ?? 0);
      const cb = Number(b.item.confidence_score ?? b.item.confidence?.score ?? 0);
      const d = cb - ca;
      if (d) return d;
    } else if (sort === "hit_desc") {
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
  const venueConfOnly = document.getElementById("venue-confident-only");
  const statusEl = document.getElementById("load-status");
  const list = document.getElementById("list");
  const summary = document.getElementById("summary");
  day.value = todayISO();
  loadTicketRankStats().then((stats) => {
    renderTicketRankPanel(document.getElementById("ticket-rank-stats"), stats);
    if (_predCache.data) renderPredictions();
  });

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

  function isAllConfidentMode() {
    return venue.value === "__all_confident__";
  }

  function venueQueryId() {
    return isAllConfidentMode() ? "" : venue.value;
  }

  async function refreshVenueOptions(opts = {}) {
    const { autoFetch = true } = opts;
    setFetchProgress(true, {
      title: "開催場を確認中…",
      detail: "出走表の有無を確認しています。",
    });
    setStatus("開催場を確認中…");
    try {
      let data = await fillActiveVenues(venue, day.value, {
        confidentOnly: Boolean(venueConfOnly && venueConfOnly.checked),
      });
      let hasList = (data.items || []).length > 0 || Number(data.confident_total || 0) > 0;

      if (!hasList && autoFetch && !(venueConfOnly && venueConfOnly.checked)) {
        updateFetchProgress({
          title: "出走表を取得中…",
          detail: "OpenAPIから本日分をダウンロードしています。1〜2分かかることがあります。",
        });
        setStatus("出走表を取得中…");
        try {
          const prep = new URLSearchParams({ day: day.value });
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), 120000);
          const res = await fetch(`/api/day/fetch-cards?${prep}`, {
            method: "POST",
            signal: controller.signal,
          });
          clearTimeout(timer);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const fetched = await res.json();
          const vcount = fetched.venue_count || 0;
          const rcount = fetched.race_count || 0;
          setStatus(
            vcount > 0
              ? `出走表取得完了: ${vcount}場 / ${rcount}R`
              : `出走表0件（${fetched.date || day.value}）`
          );
          if (vcount > 0) {
            showFetchDoneBanner(
              `出走表の取得が完了しました（${vcount}場 / ${rcount}R）。ホーム画面追加時は「ページ更新」を押してください。`
            );
          }
          updateFetchProgress({
            title: "場一覧を更新中…",
            detail: "取得した出走表を反映しています。",
          });
          data = await fillActiveVenues(venue, day.value, {
            confidentOnly: Boolean(venueConfOnly && venueConfOnly.checked),
          });
          hasList = (data.items || []).length > 0 || Number(data.confident_total || 0) > 0;
        } catch (e) {
          setStatus(`出走表の取得に失敗: ${e.message || e}`);
        }
      }

      setAnalyzeEnabled(Boolean(venue.value));
      if (!hasList) {
        clearRaceView(
          venueConfOnly && venueConfOnly.checked
            ? "この日の自信ありレースがありません。先に解析するか、チェックを外してください。"
            : "開催場がありません。「ページ更新」を押すか、1〜2分待ってからもう一度お試しください。"
        );
        setStatus(venueConfOnly && venueConfOnly.checked ? "自信ありの場なし" : "開催場なし");
        return false;
      }
      if (!venue.value) {
        const tip = Number(data.confident_total || 0) > 0
          ? `場を選ぶか「自信あり（全場）」を選んでください（自信あり ${data.confident_total}R）`
          : "場を選択すると、その場だけ解析・表示します。";
        clearRaceView(tip);
        setStatus(
          venueConfOnly && venueConfOnly.checked
            ? `自信ありのある場 ${(data.items || []).length} — 場を選んでください`
            : `${(data.items || []).length}場が開催中 — 場を選んでください`
        );
        return false;
      }
      return true;
    } finally {
      setFetchProgress(false);
    }
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
    const allConf = isAllConfidentMode();
    const params = new URLSearchParams({ day: day.value });
    const vid = venueQueryId();
    if (vid) params.set("venue_id", vid);
    if (allConf) params.set("confident_only", "true");
    const progressTitle = prepare
      ? (allConf ? "自信ありレースを読み込み中…" : "出走表を取得して予想中…")
      : refresh
        ? "再予想中…"
        : "予想を読み込み中…";
    setFetchProgress(true, {
      title: progressTitle,
      detail: prepare && !allConf
        ? "この場だけ解析します。完了まで数十秒〜2分かかることがあります。"
        : "処理中です。この画面は閉じずにお待ちください。",
    });
    try {
      let data;
      if (prepare) {
        if (allConf) {
          setStatus("全場の自信ありを表示するため、保存済み予想を読み込み中…");
          data = await jget(`/api/predictions/today?${params}`);
        } else {
          setStatus("出走表を取得して予想中…（この場のみ・高速モード）");
          updateFetchProgress({
            title: "出走表を取得して予想中…",
            detail: "OpenAPI取得と機械学習予想を実行しています。",
          });
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
        }
      } else if (refresh) {
        if (allConf) {
          setStatus("全場の自信ありを読み込み中…");
          data = await jget(`/api/predictions/today?${params}`);
        } else {
          setStatus("再予想中…");
          params.set("refresh", "true");
          data = await jget(`/api/predictions/today?${params}`);
        }
      } else {
        setStatus("読み込み中…");
        data = await jget(`/api/predictions/today?${params}`);
        if ((!data.items || data.items.length === 0) && !allConf) {
          setStatus("データが無いためこの場を取得・予想しています…");
          updateFetchProgress({
            title: "出走表を取得して予想中…",
            detail: "初回表示のため、この場のデータを取得しています。",
          });
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
          await fillActiveVenues(venue, day.value, {
            confidentOnly: Boolean(venueConfOnly && venueConfOnly.checked),
          });
          if (params.get("venue_id") && [...venue.options].some((o) => o.value === params.get("venue_id"))) {
            venue.value = params.get("venue_id");
          }
        }
      }
      // 全場自信あり、またはチェック連動で表示フィルタを自信ありに合わせる
      if (allConf && filterEl && filterEl.value === "all") {
        filterEl.value = "confident";
      }
      _predCache = { data, prepared: prepareItems(data) };
      renderPredictions();
      // 自信度を付与した直後に場リストの件数を更新
      try {
        const keep = venue.value;
        await fillActiveVenues(venue, day.value, {
          confidentOnly: Boolean(venueConfOnly && venueConfOnly.checked),
        });
        if (keep && [...venue.options].some((o) => o.value === keep)) venue.value = keep;
      } catch (_) { /* ignore */ }
      const oddsN = (data.items || []).filter((x) => x.has_odds).length;
      const confN = (data.items || []).filter((x) => x.is_confident || x.confidence?.is_confident).length;
      const cached = data.cached ? "（キャッシュ）" : "";
      const scope = allConf ? "全場・自信あり" : venue.value;
      setStatus(`更新済${cached}（${scope} / ${data.count || 0}R / 自信あり ${confN}R / オッズあり ${oddsN}R）`);
      if ((prepare || (data.items && data.items.length > 0)) && isStandalonePwa()) {
        showFetchDoneBanner("解析が完了しました。表示を最新にするには「ページ更新」を押してください。");
      }
    } catch (e) {
      setStatus(`エラー: ${e.message || e}`);
      throw e;
    } finally {
      setFetchProgress(false);
    }
  }

  refreshBtn.addEventListener("click", () => load({ prepare: true, force: true }));
  day.addEventListener("change", async () => {
    const ok = await refreshVenueOptions();
    if (ok) await load({ prepare: false });
  });
  venue.addEventListener("change", () => {
    setAnalyzeEnabled(Boolean(venue.value));
    if (isAllConfidentMode() && filterEl) filterEl.value = "confident";
    if (venue.value) load({ prepare: false });
    else clearRaceView("場を選択すると、その場だけ解析・表示します。");
  });
  if (venueConfOnly) {
    venueConfOnly.addEventListener("change", async () => {
      if (venueConfOnly.checked && filterEl && filterEl.value === "all") {
        filterEl.value = "confident";
      }
      const ok = await refreshVenueOptions();
      if (ok) await load({ prepare: false });
    });
  }
  if (filterEl) filterEl.addEventListener("change", () => renderPredictions());
  if (sortEl) sortEl.addEventListener("change", () => renderPredictions());
  if (compactEl) compactEl.addEventListener("change", () => renderPredictions());

  try {
    await waitForServerReady();
  } catch (e) {
    setStatus(String(e.message || e));
    clearRaceView("サーバー起動を待てませんでした。「ページ更新」を押して再度お試しください。");
    return;
  }

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
  const confidentRows = prepared.filter((x) => x.item.is_confident || x.item.confidence?.is_confident);
  const confidentSettled = confidentRows.filter((x) => x.hits.hasResult);
  const confidentTrio = confidentSettled.filter((x) => x.hits.hitTrio).length;
  const hitAny = prepared.filter((x) => x.hits.anyHit).length;
  const hitTf = prepared.filter((x) => x.hits.hitTf).length;
  const hitTrio = prepared.filter((x) => x.hits.hitTrio).length;
  const settled = prepared.filter((x) => x.hits.hasResult).length;

  summary.innerHTML = `
    <div class="stat"><div class="label">対象レース</div><div class="value">${data.count}</div></div>
    <div class="stat"><div class="label">表示中</div><div class="value">${filtered.length}</div></div>
    <div class="stat"><div class="label">自信あり</div><div class="value">${confidentRows.length}<span class="sub"> / ${prepared.length}</span></div></div>
    <div class="stat"><div class="label">自信あり3連複</div><div class="value">${confidentTrio}<span class="sub"> / ${confidentSettled.length || 0}</span></div></div>
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
    const delayThesis = item.delay_thesis
      ? `<div class="race-thesis delay"><strong>本命遅れ時</strong><p>${item.delay_thesis}</p></div>`
      : "";
    const conf = item.confidence || {};
    const confScore = item.confidence_score ?? conf.score;
    const confBadge = conf.is_confident || item.is_confident
      ? `<span class="badge hit">自信あり ${confScore != null ? Math.round(Number(confScore) * 100) + "%" : ""}</span>`
      : (conf.label
        ? `<span class="badge">${conf.label}${confScore != null ? " " + Math.round(Number(confScore) * 100) + "%" : ""}</span>`
        : "");
    const confReasons = (conf.reasons || []).slice(0, compact ? 2 : 4)
      .map((r) => `<li>${r}</li>`).join("");
    const confBlock = confScore != null
      ? `<div class="race-thesis confidence"><strong>自信度 ${Math.round(Number(confScore) * 100)}%（${conf.label || "—"}）</strong>
          ${confReasons ? `<ul>${confReasons}</ul>` : "<p>場・選手・天候・モデル出力から推定</p>"}
        </div>`
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

    const fmtTicket = (t, kind, rankIdx) => {
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
      const histRate = (kind === "trio" || kind === "tf") ? rankHitRate(kind, rankIdx) : null;
      const histTxt = histRate != null
        ? ` <span class="rank-hist" title="この順位の候補が過去検証で的中した割合">過去${pct(histRate)}</span>`
        : "";
      return `<li>
        <div class="ticket-line"><strong>${prefix}${t.label}</strong>${mark}${valueBadge}${histTxt}
          <span class="muted">確率${pct(pr)}${oddsTxt}${evTxt} / 配分${share}%</span></div>
        ${reasonDetails(kindKey, t)}
      </li>`;
    };
    const ticketBlock = (title, rows, kind) => {
      const anyRate = kind === "trio"
        ? displayTicketRankBlock(_ticketRankStats)?.sanrenpuku?.any_rate
        : (kind === "tf" ? displayTicketRankBlock(_ticketRankStats)?.sanrentan?.any_rate : null);
      const sub = anyRate != null
        ? ` <span class="muted">過去いずれか ${pct(anyRate)}</span>`
        : "";
      return `
      <div class="ticket-block">
        <div class="ticket-title">${title}${sub}</div>
        <ol class="ticket-list">${rows.map((t, i) => fmtTicket(t, kind, i + 1)).join("") || "<li>候補なし</li>"}</ol>
      </div>`;
    };

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
      const kim = item.result?.kimarite ? ` / ${item.result.kimarite}` : "";
      return `<div class="meta">結果: ${item.result.rank1}-${item.result.rank2}-${item.result.rank3}${kim}
        <span class="badge ${hits.hitWin ? "hit" : "upset"}">${hits.hitWin ? "単勝的中" : "単勝外れ"}</span>
        <span class="badge ${hits.hitTrio ? "hit" : "upset"}">${hits.hitTrio ? "3連複的中" : "3連複外れ"}</span>
        <span class="badge ${hits.hitTf ? "hit" : "upset"}">${hits.hitTf ? "3連単的中" : "3連単外れ"}</span>
      </div>`;
    })();

    const review = item.review;
    const reviewHtml = (() => {
      if (!review) return "";
      const hitLis = (review.bullets_hit || []).map((b) => `<li class="rev-hit">${b}</li>`).join("");
      const missLis = (review.bullets_miss || []).map((b) => `<li class="rev-miss">${b}</li>`).join("");
      const ctxLis = (review.context || []).map((b) => `<li>${b}</li>`).join("");
      const lessonLis = (review.lessons || []).map((b) => `<li>${b}</li>`).join("");
      const title = hits.anyHit ? "的中の復習" : "外れの復習";
      const cls = hits.anyHit ? "review-hit" : "review-miss";
      if (compact) {
        return `<div class="review-box ${cls} compact">
          <div class="ticket-title">${title}</div>
          <p class="review-headline">${review.headline || ""}</p>
        </div>`;
      }
      return `<div class="review-box ${cls}">
        <div class="ticket-title">${title}</div>
        <p class="review-headline">${review.headline || ""}</p>
        ${hitLis ? `<div class="review-sec"><strong>なぜ当たったか</strong><ul>${hitLis}</ul></div>` : ""}
        ${missLis ? `<div class="review-sec"><strong>なぜ外れたか</strong><ul>${missLis}</ul></div>` : ""}
        ${ctxLis ? `<div class="review-sec"><strong>展開の振り返り</strong><ul>${ctxLis}</ul></div>` : ""}
        ${lessonLis ? `<div class="review-sec"><strong>次に活かす点</strong><ul>${lessonLis}</ul></div>` : ""}
      </div>`;
    })();

    const hitClass = hits.hitTf ? "is-tf-hit" : hits.hitTrio ? "is-trio-hit" : hits.hitWin ? "is-win-hit" : "";

    return `
      <article class="race ${hitClass} ${compact ? "compact" : ""}" style="animation-delay:${Math.min(idx, 12) * 0.03}s">
        <div class="race-head">
          <h2>${item.venue_name} ${item.race_no}R <span class="badge">${item.model_name || ""}</span>
            ${oddsBadge}
            ${confBadge}
            ${(item.pre_exhibition_mode || item.exhibition?.phase === "試走前" || (item.exhibition && item.exhibition.complete === false)) ? '<span class="badge" title="展示タイム未反映。級別・モーター・選手・場傾向で予想">展示前</span>' : ""}
            ${item.has_upset ? '<span class="badge upset">穴あり</span>' : ""}
            ${item.status === "scheduled" ? '<span class="badge">予想中</span>' : ""}
          </h2>
          ${resultHtml}
        </div>
        ${thesis}
        ${confBlock}
        ${delayThesis}
        <div class="wakus">${wakus}</div>
        <div class="ticket-grid">
          ${ticketBlock("単勝 候補", wins, "win")}
          ${ticketBlock("3連複 候補", sps, "trio")}
          ${ticketBlock("3連単 候補", sts, "tf")}
        </div>
        ${reviewHtml}
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
  const rankStats = await loadTicketRankStats();
  const panel = document.getElementById("acc-ticket-ranks");
  renderTicketRankPanel(panel, rankStats);
  if (panel && rankStats) {
    const live = rankStats.live;
    const base = rankStats.baseline || rankStats.display;
    const tableFor = (title, block) => {
      if (!block) return "";
      const rows = [];
      const n = Math.max(
        (block.sanrenpuku?.ranks || []).length,
        (block.sanrentan?.ranks || []).length,
      );
      for (let i = 0; i < n; i++) {
        const tr = (block.sanrenpuku?.ranks || [])[i] || {};
        const tf = (block.sanrentan?.ranks || [])[i] || {};
        rows.push(`<tr>
          <td>${tr.rank || tf.rank || i + 1} ${tr.label || ""}</td>
          <td>${pct(tr.hit_rate)}</td>
          <td>${pct(tf.hit_rate)}</td>
        </tr>`);
      }
      return `
        <div class="table-wrap" style="margin-top:0.75rem">
          <h3 style="font-size:0.95rem;margin:0 0 0.4rem">${title}</h3>
          <table>
            <thead><tr><th>候補順位</th><th>3連複的中率</th><th>3連単的中率</th></tr></thead>
            <tbody>
              ${rows.join("")}
              <tr>
                <td>いずれか（最大5点）</td>
                <td>${pct(block.sanrenpuku?.any_rate)}</td>
                <td>${pct(block.sanrentan?.any_rate)}</td>
              </tr>
            </tbody>
          </table>
          <p class="meta">期間 ${block.period?.start || "?"}〜${block.period?.end || "?"} / n=${block.n_races || "-"}</p>
        </div>`;
    };
    panel.insertAdjacentHTML("beforeend", tableFor(base?.label || "ベースライン", base));
    if (live) panel.insertAdjacentHTML("beforeend", tableFor(live.label || "ライブ", live));
  }
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
