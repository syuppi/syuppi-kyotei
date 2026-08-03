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
  if (includeAll) {
    // keep existing all option if present
  }
  for (const v of data.items) {
    const opt = document.createElement("option");
    opt.value = v.id;
    opt.textContent = `${v.id} ${v.name}`;
    selectEl.appendChild(opt);
  }
}

async function bootPredictions() {
  const day = document.getElementById("day");
  const venue = document.getElementById("venue");
  const refreshBtn = document.getElementById("refresh");
  day.value = todayISO();
  await fillVenues(venue, true);

  async function load(refresh = false) {
    const params = new URLSearchParams({ day: day.value });
    if (venue.value) params.set("venue_id", venue.value);
    if (refresh) params.set("refresh", "true");
    const data = await jget(`/api/predictions/today?${params}`);
    renderPredictions(data);
  }

  refreshBtn.addEventListener("click", () => load(true));
  day.addEventListener("change", () => load(false));
  venue.addEventListener("change", () => load(false));
  await load(false);
}

function renderPredictions(data) {
  const summary = document.getElementById("summary");
  const list = document.getElementById("list");
  const upsetCount = data.items.filter((x) => x.has_upset).length;
  summary.innerHTML = `
    <div class="stat"><div class="label">対象レース</div><div class="value">${data.count}</div></div>
    <div class="stat"><div class="label">穴候補あり</div><div class="value">${upsetCount}</div></div>
    <div class="stat"><div class="label">日付</div><div class="value" style="font-size:1.1rem">${data.date}</div></div>
  `;

  if (!data.items.length) {
    list.innerHTML = `<p class="lede">予測がありません。デモデータを投入するか「再予測」を実行してください。</p>`;
    return;
  }

  list.innerHTML = data.items.map((item, idx) => {
    const probs = item.win_probs || {};
    const tickets = item.tickets || {};
    const winTickets = tickets.win || (item.candidates_win || []).map((w, i) => ({
      rank: i + 1, combo: [w], label: String(w),
      prob: probs[w] ?? probs[String(w)] ?? 0, stake_share: 0,
    }));
    const spTickets = tickets.sanrenpuku || (item.sanrenpuku || []).slice(0, 3).map((c, i) => ({
      rank: i + 1, combo: c, label: [...c].sort((a,b)=>a-b).join("-"), prob: 0, stake_share: 0,
    }));
    const stTickets = tickets.sanrentan || (item.sanrentan || []).slice(0, 3).map((c, i) => ({
      rank: i + 1, combo: c, label: c.join("-"), prob: 0, stake_share: 0,
    }));
    // stake_share が無い古いデータ向けに再計算
    const fillShares = (rows) => {
      const sum = rows.reduce((a, r) => a + (Number(r.prob) || 0), 0) || 1;
      return rows.map((r) => ({
        ...r,
        stake_share: r.stake_share || ((Number(r.prob) || 0) / sum),
      }));
    };
    const wins = fillShares(winTickets).slice(0, 3);
    const sps = fillShares(spTickets).slice(0, 3);
    const sts = fillShares(stTickets).slice(0, 3);

    const wakus = [1,2,3,4,5,6].map((w) => {
      const p = probs[w] ?? probs[String(w)] ?? 0;
      const top = wins.some((t) => Number(t.combo?.[0]) === w);
      return `<div class="waku ${top ? "top" : ""}"><div class="num">${w}</div><div class="prob">${pct(p)}</div></div>`;
    }).join("");
    const reasonTop = (item.reasons?.[item.rankings?.[0]] || item.reasons?.[String(item.rankings?.[0])] || []).slice(0, 3)
      .map((r) => `<div>・ ${r}</div>`).join("");

    const fmtTicket = (t) => {
      const share = Math.round((Number(t.stake_share) || 0) * 100);
      const pr = Number(t.prob) || 0;
      return `<li><strong>${t.label}</strong> <span class="muted">期待${pct(pr)} / 配分${share}%</span></li>`;
    };
    const ticketBlock = (title, rows) => `
      <div class="ticket-block">
        <div class="ticket-title">${title}</div>
        <ol class="ticket-list">${rows.map(fmtTicket).join("") || "<li>候補なし</li>"}</ol>
      </div>`;

    const resultHtml = (() => {
      if (!item.result?.rank1) {
        return `<div class="meta">結果: 未確定（予測のみ）</div>`;
      }
      const trueTop3 = [item.result.rank1, item.result.rank2, item.result.rank3].filter((x) => x != null);
      const hit1 = wins.some((t) => Number(t.combo?.[0]) === Number(item.result.rank1));
      const hitTrio = trueTop3.length === 3 && sps.some((t) => {
        const set = new Set((t.combo || []).map(Number));
        return trueTop3.every((w) => set.has(Number(w)));
      });
      const hitTf = trueTop3.length === 3 && sts.some((t) => {
        const c = (t.combo || []).map(Number);
        return c[0] === Number(item.result.rank1)
          && c[1] === Number(item.result.rank2)
          && c[2] === Number(item.result.rank3);
      });
      return `<div class="meta">結果: ${item.result.rank1}-${item.result.rank2}-${item.result.rank3}
        <span class="badge ${hit1 ? "" : "upset"}">${hit1 ? "単勝的中" : "単勝外れ"}</span>
        <span class="badge ${hitTrio ? "" : "upset"}">${hitTrio ? "3連複的中" : "3連複外れ"}</span>
        <span class="badge ${hitTf ? "" : "upset"}">${hitTf ? "3連単的中" : "3連単外れ"}</span>
      </div>`;
    })();

    return `
      <article class="race" style="animation-delay:${idx * 0.04}s">
        <div class="race-head">
          <h2>${item.venue_name} ${item.race_no}R <span class="badge">${item.model_name || ""}</span>
            ${item.has_upset ? '<span class="badge upset">穴あり</span>' : ""}
            ${item.status === "scheduled" ? '<span class="badge">予想中</span>' : ""}
          </h2>
          ${resultHtml}
        </div>
        <div class="wakus">${wakus}</div>
        <div class="ticket-grid">
          ${ticketBlock("単勝 候補", wins)}
          ${ticketBlock("3連複 候補", sps)}
          ${ticketBlock("3連単 候補", sts)}
        </div>
        <div class="reasons"><strong>本命の理由</strong>${reasonTop || "<div>・ データ不足</div>"}
          ${item.race_title ? `<div class="meta">番組: ${item.race_title}</div>` : ""}
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
