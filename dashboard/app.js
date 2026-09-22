"use strict";

/* ============================================================
   State
   ============================================================ */
const STATE = {
  schema: null,           // GET /v1/schema response
  history: [],            // [{ probability, decision, riskBand, amt, category, latencyMs, at }]
  mode: "form",           // "form" | "json"
  // Request-generation counters: changing the base URL fires a fresh
  // health/schema/model fetch while an older (e.g. still cold-start-retrying)
  // one may still be in flight. Each async loader captures its generation at
  // call time and discards its result if a newer call has since started, so
  // a slow, stale response can never clobber a faster, current one.
  gen: { health: 0, schema: 0, model: 0 },
};

const REQUIRED_HELP = "Required — no formula can fill this in.";
const DERIVABLE_HELP = "Optional — derived server-side from other fields if left blank.";
const ADVANCED_HELP = "Optional — filled by the model's trained imputer if left blank.";

/* ============================================================
   Small utilities
   ============================================================ */
function $(id) { return document.getElementById(id); }
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

function getBaseUrl() {
  return $("apiBaseUrl").value.trim().replace(/\/+$/, "");
}

function getApiKey() {
  return $("apiKey").value;
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmtPct(x) { return (x * 100).toFixed(2) + "%"; }

/* ============================================================
   Theme toggle
   ============================================================ */
function initTheme() {
  const saved = localStorage.getItem("gotcha_theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  $("themeToggle").addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const isDark = current === "dark" ||
      (!current && window.matchMedia("(prefers-color-scheme: dark)").matches);
    const next = isDark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("gotcha_theme", next);
  });
}

/* ============================================================
   Connection bar: base URL + API key (sessionStorage only)
   ============================================================ */
function initConnectionBar() {
  const savedBaseUrl = sessionStorage.getItem("gotcha_api_base_url");
  $("apiBaseUrl").value = savedBaseUrl || CONFIG.API_BASE_URL;
  $("apiBaseUrl").addEventListener("change", () => {
    sessionStorage.setItem("gotcha_api_base_url", getBaseUrl());
    refreshHealth();
    loadSchema();
    loadModelCard();
  });

  const savedKey = sessionStorage.getItem("gotcha_api_key");
  if (savedKey) $("apiKey").value = savedKey;
  $("apiKey").addEventListener("input", () => {
    sessionStorage.setItem("gotcha_api_key", getApiKey());
  });
}

/* ============================================================
   Cold-start-aware fetch
   ============================================================ */
function showWakeBanner() { $("wakeBanner").classList.add("show"); }
function hideWakeBanner() { $("wakeBanner").classList.remove("show"); }

async function apiFetch(path, options = {}, cfg = {}) {
  const {
    timeoutMs = 10000,
    coldStartRetries = 8,
    coldStartDelayMs = 4000,
    wake = true,
  } = cfg;

  const url = getBaseUrl() + path;
  let attempt = 0;

  while (true) {
    attempt++;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timer);
      if (wake) hideWakeBanner();
      return res;
    } catch (err) {
      clearTimeout(timer);
      if (attempt > coldStartRetries) {
        if (wake) hideWakeBanner();
        throw err;
      }
      if (wake) showWakeBanner();
      await sleep(coldStartDelayMs);
    }
  }
}

/* ============================================================
   Health pill
   ============================================================ */
function setHealthPill(kind, text) {
  const pill = $("healthPill");
  pill.className = "pill " + kind;
  $("healthPillText").textContent = text;
}

async function refreshHealth() {
  const gen = ++STATE.gen.health;
  setHealthPill("waking", "checking…");
  try {
    const res = await apiFetch("/v1/health", {}, { timeoutMs: 6000, coldStartRetries: 10, coldStartDelayMs: 3000 });
    if (gen !== STATE.gen.health) return; // a newer call has since started
    if (!res.ok) throw new Error("status " + res.status);
    const data = await res.json();
    if (gen !== STATE.gen.health) return;
    setHealthPill("ok", "ok · " + data.model_variant);
  } catch (err) {
    if (gen !== STATE.gen.health) return;
    setHealthPill("down", "unreachable");
  }
}

/* ============================================================
   Schema-driven form
   ============================================================ */
async function loadSchema() {
  const gen = ++STATE.gen.schema;
  $("formFields").innerHTML = '<p class="empty-state">Loading field schema…</p>';
  try {
    const res = await apiFetch("/v1/schema", {}, { timeoutMs: 8000, coldStartRetries: 8, coldStartDelayMs: 3500 });
    if (gen !== STATE.gen.schema) return; // a newer call has since started
    if (!res.ok) throw new Error("status " + res.status);
    STATE.schema = await res.json();
    if (gen !== STATE.gen.schema) return;
    buildForm(STATE.schema);
    buildPresets();
  } catch (err) {
    if (gen !== STATE.gen.schema) return;
    $("formFields").innerHTML =
      '<p class="empty-state">Could not load schema from the API (' + escapeHtml(err.message) +
      "). Check the base URL and try again.</p>";
  }
}

function fieldInputHtml(field) {
  const id = "f_" + field.name;
  if (field.type === "string" && Array.isArray(field.allowed_values)) {
    const opts = field.allowed_values
      .map((v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`)
      .join("");
    return `<select id="${id}" data-field="${field.name}"><option value="">— auto —</option>${opts}</select>`;
  }
  return `<input type="number" step="any" id="${id}" data-field="${field.name}" placeholder="${field.required ? "required" : "auto"}" />`;
}

function badgeHtml(field) {
  if (field.required) return `<span class="field-badge required" title="${REQUIRED_HELP}">required</span>`;
  if (field.derivable) return `<span class="field-badge derivable" title="${DERIVABLE_HELP}">derived</span>`;
  return `<span class="field-badge auto" title="${ADVANCED_HELP}">imputed</span>`;
}

function buildForm(schema) {
  const required = schema.fields.filter((f) => f.required);
  const derivable = schema.fields.filter((f) => !f.required && f.derivable);
  const advanced = schema.fields.filter((f) => !f.required && !f.derivable);

  const group = (title, fields) =>
    fields.length
      ? `<h3>${title}</h3><div class="field-group">${fields
          .map(
            (f) => `<div class="field"><label for="f_${f.name}">${f.name}${badgeHtml(f)}</label>${fieldInputHtml(f)}</div>`
          )
          .join("")}</div>`
      : "";

  const html = `
    <h3>Derivation helpers</h3>
    <div class="field-group">
      <div class="field">
        <label for="f_timestamp">timestamp</label>
        <input type="datetime-local" id="f_timestamp" data-field="timestamp" />
        <p class="hint">${escapeHtml(schema.derivation_inputs.timestamp)}</p>
      </div>
      <div class="field">
        <label for="f_dob">dob</label>
        <input type="date" id="f_dob" data-field="dob" />
        <p class="hint">${escapeHtml(schema.derivation_inputs.dob)}</p>
      </div>
    </div>
    ${group("Required", required)}
    ${group("Optional — auto-derived", derivable)}
    <details class="advanced">
      <summary>Advanced: history / context fields (optional, imputed if blank)</summary>
      ${group("", advanced)}
    </details>
  `;
  $("formFields").innerHTML = html;
}

function collectPayloadFromForm() {
  const payload = {};
  document.querySelectorAll("#formFields [data-field]").forEach((el) => {
    const name = el.dataset.field;
    const raw = el.value;
    if (raw === "" || raw === null || raw === undefined) return;
    if (el.tagName === "SELECT") {
      payload[name] = raw;
    } else if (name === "timestamp") {
      payload[name] = raw; // "YYYY-MM-DDTHH:mm", fromisoformat-compatible
    } else if (name === "dob") {
      payload[name] = raw; // "YYYY-MM-DD"
    } else {
      const num = Number(raw);
      if (!Number.isNaN(num)) payload[name] = num;
    }
  });
  return payload;
}

function applyValuesToForm(values) {
  Object.entries(values).forEach(([name, value]) => {
    const el = document.querySelector(`#formFields [data-field="${CSS.escape(name)}"]`);
    if (el) el.value = value;
  });
  const advancedKeys = Object.keys(values).filter((k) => {
    const el = document.querySelector(`#formFields [data-field="${CSS.escape(k)}"]`);
    return el && el.closest("details.advanced");
  });
  if (advancedKeys.length) {
    const details = document.querySelector("details.advanced");
    if (details) details.open = true;
  }
}

function clearForm() {
  document.querySelectorAll("#formFields [data-field]").forEach((el) => (el.value = ""));
}

/* ============================================================
   Presets
   ============================================================ */
function isoNow(hour, minute) {
  const d = new Date();
  d.setHours(hour, minute, 0, 0);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(hour)}:${pad(minute)}`;
}

const PRESETS = [
  {
    label: "Normal grocery purchase",
    values: {
      amt: 42.5, zip: 53703, lat: 43.0731, long: -89.4012, city_pop: 259680,
      merch_lat: 43.08, merch_long: -89.39, category: "grocery_pos", state: "WI",
      gender: "F", timestamp: isoNow(14, 30),
    },
  },
  {
    label: "Late-night online, high amount",
    values: {
      amt: 899.99, zip: 90001, lat: 34.0, long: -118.2, city_pop: 500000,
      merch_lat: 34.4, merch_long: -118.6, category: "shopping_net", state: "CA",
      gender: "M", timestamp: isoNow(2, 15),
    },
  },
  {
    label: "Rapid card-testing burst",
    values: {
      amt: 1.0, zip: 10001, lat: 40.7128, long: -74.006, city_pop: 8400000,
      merch_lat: 40.72, merch_long: -74.0, category: "misc_net", state: "NY",
      gender: "M", timestamp: isoNow(3, 5),
      transactions_last_10min: 8, transactions_last_1h: 15,
      distinct_categories_last_24h: 1, amount_spent_last_1h: 9.0,
    },
  },
  {
    label: "Impossible travel",
    values: {
      amt: 250.0, zip: 60601, lat: 41.8781, long: -87.6298, city_pop: 2700000,
      merch_lat: 41.9, merch_long: -87.65, category: "shopping_net", state: "IL",
      gender: "F", timestamp: isoNow(11, 0),
      distance_from_last_txn_km: 9000, time_since_last_txn_sec: 300,
    },
  },
];

function buildPresets() {
  $("presetRow").innerHTML = PRESETS.map(
    (p, i) => `<button type="button" class="preset-btn" data-preset="${i}">${escapeHtml(p.label)}</button>`
  ).join("");
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = PRESETS[Number(btn.dataset.preset)];
      clearForm();
      applyValuesToForm(preset.values);
      if (STATE.mode === "json") syncFormToJson();
    });
  });
}

/* ============================================================
   Form <-> raw JSON toggle
   ============================================================ */
function syncFormToJson() {
  $("jsonEditor").value = JSON.stringify(collectPayloadFromForm(), null, 2);
}

function initJsonToggle() {
  $("toggleJsonMode").addEventListener("click", () => {
    if (STATE.mode === "form") {
      syncFormToJson();
      $("txnForm").style.display = "none";
      $("jsonEditorWrap").classList.add("show");
      $("toggleJsonMode").textContent = "Form editor";
      STATE.mode = "json";
    } else {
      let parsed;
      try {
        parsed = JSON.parse($("jsonEditor").value || "{}");
      } catch (err) {
        alert("Raw JSON is invalid: " + err.message);
        return;
      }
      applyValuesToForm(parsed);
      $("txnForm").style.display = "";
      $("jsonEditorWrap").classList.remove("show");
      $("toggleJsonMode").textContent = "Raw JSON editor";
      STATE.mode = "form";
    }
  });

  $("clearForm").addEventListener("click", () => {
    clearForm();
    $("jsonEditor").value = "{}";
  });
}

/* ============================================================
   Scoring
   ============================================================ */
function currentPayload() {
  if (STATE.mode === "json") {
    return JSON.parse($("jsonEditor").value || "{}");
  }
  return collectPayloadFromForm();
}

async function submitScore() {
  let payload;
  try {
    payload = currentPayload();
  } catch (err) {
    renderResultError("Invalid JSON payload: " + err.message);
    return;
  }

  if (!getApiKey()) {
    renderResultError("Enter an API key in the connection bar first.");
    return;
  }

  const headers = { "Content-Type": "application/json", "X-API-Key": getApiKey() };
  renderApiCall("/v1/score", "POST", headers, payload, null);
  $("resultPanel").innerHTML = '<p class="empty-state">Scoring…</p>';

  try {
    const res = await apiFetch("/v1/score", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    renderApiCall("/v1/score", "POST", headers, payload, data);
    if (!res.ok) {
      renderResultError((data && (data.error || data.message)) || ("HTTP " + res.status));
      return;
    }
    renderResult(data);
    addToHistory(payload, data);
  } catch (err) {
    renderResultError("Request failed: " + err.message);
  }
}

function renderResultError(message) {
  $("resultPanel").innerHTML = `<p class="empty-state" style="color:var(--status-critical)">${escapeHtml(message)}</p>`;
}

/* ---------- Gauge + badge + top factors ---------- */
function statusColorFor(riskBand) {
  if (riskBand === "low_risk") return "var(--status-good)";
  if (riskBand === "medium_risk") return "var(--status-warning)";
  return "var(--status-critical)";
}

function decisionMeta(decision) {
  switch (decision) {
    case "approve": return { icon: "✅", label: "Approve" };
    case "step_up_verification": return { icon: "⚠️", label: "Step-up verification" };
    case "block": return { icon: "⛔", label: "Block" };
    default: return { icon: "•", label: decision };
  }
}

function renderGauge(probability, riskBand) {
  const pct = Math.min(Math.max(probability, 0), 1) * 100;
  const color = statusColorFor(riskBand);
  return `
    <div class="hero-figure" style="color:${color}">${fmtPct(probability)}</div>
    <div class="meter-wrap">
      <div class="meter-track" style="background: linear-gradient(to right,
          var(--status-good-bg) 0%, var(--status-good-bg) 40%,
          var(--status-warning-bg) 40%, var(--status-warning-bg) 75%,
          var(--status-critical-bg) 75%, var(--status-critical-bg) 100%);"
          role="img" aria-label="Fraud probability ${fmtPct(probability)}, band boundaries at 0.40 and 0.75">
        <div class="meter-fill" style="width:${pct}%; background:${color};"></div>
        <div class="meter-marker" style="left:40%;"><span class="meter-marker-label">0.40</span></div>
        <div class="meter-marker" style="left:75%;"><span class="meter-marker-label">0.75</span></div>
      </div>
      <div class="meter-labels"><span>0.0</span><span>1.0</span></div>
    </div>
  `;
}

function renderTopFactors(topFactors) {
  if (!topFactors || !topFactors.length) return '<p class="empty-state">No factor data.</p>';
  const maxAbs = Math.max(...topFactors.map((f) => Math.abs(f.contribution)), 1e-9);
  const rows = topFactors
    .map((f) => {
      const halfPct = (Math.abs(f.contribution) / maxAbs) * 45;
      const positive = f.contribution >= 0;
      const left = positive ? 50 : 50 - halfPct;
      const color = positive ? "var(--series-red)" : "var(--series-blue)";
      const sign = positive ? "+" : "";
      return `
        <div class="factor-row">
          <div class="factor-name">${escapeHtml(f.feature)}</div>
          <div class="factor-track">
            <div class="factor-zero"></div>
            <div class="factor-bar" style="left:${left}%; width:${halfPct}%; background:${color};"></div>
          </div>
          <div class="factor-value">${sign}${f.contribution.toFixed(4)}</div>
        </div>`;
    })
    .join("");

  const tableRows = topFactors
    .map((f) => `<tr><td>${escapeHtml(f.feature)}</td><td>${f.contribution.toFixed(6)}</td></tr>`)
    .join("");

  return `
    <div class="chart-legend">
      <span><span class="legend-swatch" style="background:var(--series-red)"></span>increases risk</span>
      <span><span class="legend-swatch" style="background:var(--series-blue)"></span>decreases risk</span>
    </div>
    ${rows}
    <details style="margin-top:8px;">
      <summary class="hint" style="cursor:pointer;">View as table</summary>
      <table class="data-table"><thead><tr><th>Feature</th><th>Contribution</th></tr></thead><tbody>${tableRows}</tbody></table>
    </details>
  `;
}

function renderResult(data) {
  const meta = decisionMeta(data.decision);
  $("resultPanel").innerHTML = `
    ${renderGauge(data.fraud_probability, data.risk_band)}
    <span class="decision-badge ${data.decision}">${meta.icon} ${meta.label}</span>
    <p class="action-text">${escapeHtml(data.action)}</p>
    <h3>Top contributing factors</h3>
    ${renderTopFactors(data.top_factors)}
    <div class="latency-row">
      <span>Latency: <strong>${data.latency_ms} ms</strong></span>
      <span>Model variant: <strong>${escapeHtml(data.model_variant)}</strong></span>
      <span>Request ID: <code>${escapeHtml(data.request_id || "")}</code></span>
    </div>
    ${data.derived_fields && data.derived_fields.length ? `<p class="hint">Derived: ${data.derived_fields.map(escapeHtml).join(", ")}</p>` : ""}
    ${data.imputed_fields && data.imputed_fields.length ? `<p class="hint">Imputed by model: ${data.imputed_fields.map(escapeHtml).join(", ")}</p>` : ""}
  `;
}

/* ============================================================
   API call panel (curl + raw response)
   ============================================================ */
function renderApiCall(path, method, headers, body, response) {
  const maskedHeaders = { ...headers, "X-API-Key": "<YOUR_API_KEY>" };
  const headerFlags = Object.entries(maskedHeaders)
    .map(([k, v]) => `  -H '${k}: ${v}'`)
    .join(" \\\n");
  const curl = `curl -X ${method} '${getBaseUrl()}${path}' \\\n${headerFlags} \\\n  -d '${JSON.stringify(body)}'`;
  $("curlBlock").textContent = curl;
  $("rawResponseBlock").textContent = response ? JSON.stringify(response, null, 2) : "—";
}

function initCopyCurl() {
  $("copyCurl").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("curlBlock").textContent);
      const btn = $("copyCurl");
      const old = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(() => (btn.textContent = old), 1200);
    } catch (err) {
      alert("Could not copy to clipboard: " + err.message);
    }
  });
}

/* ============================================================
   Session history + decision mix
   ============================================================ */
function addToHistory(payload, data) {
  STATE.history.push({
    at: new Date(),
    amt: payload.amt,
    category: payload.category,
    probability: data.fraud_probability,
    decision: data.decision,
    latencyMs: data.latency_ms,
  });
  renderHistory();
}

function renderHistory() {
  const rows = STATE.history
    .map((h, i) => {
      const meta = decisionMeta(h.decision);
      return `<tr>
        <td>${i + 1}</td>
        <td>${h.at.toLocaleTimeString()}</td>
        <td>${h.amt !== undefined ? "$" + Number(h.amt).toFixed(2) : "—"}</td>
        <td>${escapeHtml(h.category || "—")}</td>
        <td>${fmtPct(h.probability)}</td>
        <td><span class="badge-mini ${h.decision}">${meta.icon} ${meta.label}</span></td>
        <td>${h.latencyMs !== undefined ? h.latencyMs + " ms" : "—"}</td>
      </tr>`;
    })
    .reverse()
    .join("");

  $("historyBody").innerHTML = rows || '<tr><td colspan="7" class="empty-state">Nothing scored yet.</td></tr>';
  renderDecisionMix();
}

function renderDecisionMix() {
  if (!STATE.history.length) {
    $("decisionMix").innerHTML = '<p class="empty-state">No scored transactions yet this session.</p>';
    return;
  }
  const counts = { approve: 0, step_up_verification: 0, block: 0 };
  STATE.history.forEach((h) => { if (counts[h.decision] !== undefined) counts[h.decision]++; });
  const total = STATE.history.length;

  const segs = Object.entries(counts)
    .filter(([, count]) => count > 0)
    .map(([decision, count]) => {
      const pct = (count / total) * 100;
      const meta = decisionMeta(decision);
      return `<div class="stacked-seg ${decision}" style="flex-grow:${count}" title="${meta.label}: ${count}">${pct >= 12 ? count : ""}</div>`;
    })
    .join("");

  const legend = Object.entries(counts)
    .map(([decision, count]) => {
      const meta = decisionMeta(decision);
      return `<span style="margin-right:14px;"><span class="badge-mini ${decision}">${meta.icon} ${meta.label}</span> ${count}</span>`;
    })
    .join("");

  $("decisionMix").innerHTML = `
    <div class="stacked-bar" role="img" aria-label="Decision mix across ${total} scored transactions">${segs}</div>
    <p class="hint" style="margin-top:10px;">${legend}</p>
  `;
}

/* ============================================================
   Batch scoring via CSV upload
   ============================================================ */
function parseCsv(text) {
  const rows = [];
  let row = [], field = "", inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else { inQuotes = false; }
      } else field += c;
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === ",") {
      row.push(field); field = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  return rows;
}

function csvRowsToTransactions(rows) {
  if (!rows.length) return [];
  const headers = rows[0].map((h) => h.trim());
  return rows.slice(1).map((row) => {
    const obj = {};
    headers.forEach((h, i) => {
      const raw = (row[i] || "").trim();
      if (raw === "") return;
      const num = Number(raw);
      obj[h] = !Number.isNaN(num) && h !== "category" && h !== "state" && h !== "gender" &&
        h !== "merchant_category_risk_tier" && h !== "timestamp" && h !== "dob"
        ? num
        : raw;
    });
    return obj;
  });
}

function initCsvUpload() {
  $("csvInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (!getApiKey()) {
      $("batchStatus").innerHTML = '<p class="empty-state" style="color:var(--status-critical)">Enter an API key first.</p>';
      return;
    }
    const text = await file.text();
    const transactions = csvRowsToTransactions(parseCsv(text));
    if (!transactions.length) {
      $("batchStatus").innerHTML = '<p class="empty-state">No rows found in that CSV.</p>';
      return;
    }
    $("batchStatus").innerHTML = `<p class="empty-state">Scoring ${transactions.length} transactions…</p>`;

    const headers = { "Content-Type": "application/json", "X-API-Key": getApiKey() };
    try {
      const res = await apiFetch("/v1/score/batch", {
        method: "POST",
        headers,
        body: JSON.stringify({ transactions }),
      });
      const data = await res.json();
      renderApiCall("/v1/score/batch", "POST", headers, { transactions }, data);
      if (!res.ok) {
        $("batchStatus").innerHTML = `<p class="empty-state" style="color:var(--status-critical)">${escapeHtml(data.error || "Batch request failed")}</p>`;
        return;
      }
      let ok = 0, failed = 0;
      data.results.forEach((r, i) => {
        if (r.error) { failed++; return; }
        ok++;
        addToHistory(transactions[i], r);
      });
      $("batchStatus").innerHTML = `<p class="empty-state">Batch complete: ${ok} scored, ${failed} failed validation.</p>`;
    } catch (err) {
      $("batchStatus").innerHTML = `<p class="empty-state" style="color:var(--status-critical)">Request failed: ${escapeHtml(err.message)}</p>`;
    } finally {
      $("csvInput").value = "";
    }
  });

  $("downloadSampleCsv").addEventListener("click", () => {
    const header = "amt,zip,lat,long,city_pop,merch_lat,merch_long,category,state,gender,timestamp";
    const row1 = "42.50,53703,43.0731,-89.4012,259680,43.08,-89.39,grocery_pos,WI,F," + isoNow(14, 30);
    const row2 = "899.99,90001,34.0,-118.2,500000,34.4,-118.6,shopping_net,CA,M," + isoNow(2, 15);
    const csv = header + "\n" + row1 + "\n" + row2 + "\n";
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "gotcha_sample_transactions.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  });
}

/* ============================================================
   Model card
   ============================================================ */
async function loadModelCard() {
  const gen = ++STATE.gen.model;
  $("modelCard").innerHTML = '<p class="empty-state">Loading model info…</p>';
  try {
    const res = await apiFetch("/v1/model", {}, { timeoutMs: 8000, coldStartRetries: 8, coldStartDelayMs: 3500 });
    if (gen !== STATE.gen.model) return; // a newer call has since started
    if (!res.ok) throw new Error("status " + res.status);
    const data = await res.json();
    if (gen !== STATE.gen.model) return;
    const m = data.tuned_test_metrics;
    $("modelCard").innerHTML = `
      <div class="stat-grid">
        <div class="stat-tile"><div class="stat-label">Model variant</div><div class="stat-value" style="font-size:16px;">${escapeHtml(data.model_variant)}</div></div>
        ${m ? `
        <div class="stat-tile"><div class="stat-label">Precision</div><div class="stat-value">${(m.precision * 100).toFixed(1)}%</div></div>
        <div class="stat-tile"><div class="stat-label">Recall</div><div class="stat-value">${(m.recall * 100).toFixed(1)}%</div></div>
        <div class="stat-tile"><div class="stat-label">PR-AUC</div><div class="stat-value">${m.pr_auc.toFixed(3)}</div></div>
        <div class="stat-tile"><div class="stat-label">Review rate</div><div class="stat-value">${(m.review_rate * 100).toFixed(2)}%</div></div>
        ` : '<div class="stat-tile"><div class="stat-label">Tuned metrics</div><div class="stat-value" style="font-size:13px;">n/a for this variant</div></div>'}
      </div>
      <p class="note">
        Decision bands: approve below ${data.approve_below}, block at/above ${data.block_at_or_above}.
        Trained on a <strong>synthetic</strong> fraud-detection dataset (Kaggle,
        <code>kartik2112/fraud-detection</code>) — treat these metrics as a property
        of that dataset, not a guarantee on real production traffic.
      </p>
    `;
  } catch (err) {
    if (gen !== STATE.gen.model) return;
    $("modelCard").innerHTML = `<p class="empty-state">Could not load model info (${escapeHtml(err.message)}).</p>`;
  }
}

/* ============================================================
   Init
   ============================================================ */
function init() {
  initTheme();
  initConnectionBar();
  initJsonToggle();
  initCopyCurl();
  initCsvUpload();

  $("scoreBtn").addEventListener("click", submitScore);

  refreshHealth();
  loadSchema();
  loadModelCard();
}

document.addEventListener("DOMContentLoaded", init);
