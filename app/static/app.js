/* ───────── Transcription Pipeline UI ───────── */

// ─── Helpers ──────────────────────────────────

const $ = (id) => document.getElementById(id);
const qs = (sel, ctx) => (ctx || document).querySelector(sel);
const qsa = (sel, ctx) => (ctx || document).querySelectorAll(sel);
const esc = (s) => String(s ?? "")
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

function statusClass(suspicion) {
  if (suspicion <= 0.05) return 0;
  if (suspicion <= 0.15) return 1;
  if (suspicion <= 0.25) return 2;
  if (suspicion <= 0.35) return 3;
  if (suspicion <= 0.45) return 4;
  if (suspicion <= 0.55) return 5;
  if (suspicion <= 0.65) return 6;
  if (suspicion <= 0.75) return 7;
  if (suspicion <= 0.85) return 8;
  if (suspicion <= 0.95) return 9;
  return 10;
}

function scoreClass(suspicion) {
  return `score-${statusClass(suspicion)}`;
}

function escScore(s) {
  return Number(s || 0).toFixed(3);
}

function escPct(s) {
  return (Number(s || 0) * 100).toFixed(0);
}

// ─── State ────────────────────────────────────

let lastData = null;
let lastTranscript = "";
let lastWordEls = [];

// ─── Stage UI helpers ────────────────────────

const STAGE_NAMES = ["1-scoring","2-flagging","3-retrieval","4-deciding","5-correction"];

function setStage(stageNum, state) {
  const step = $(`step-${stageNum}`);
  const dot = $(`dot-${stageNum}`);
  const conn = $(`conn-${stageNum}`);
  step.className = "step-number";
  dot.className = "status-dot";
  if (state === "active") {
    step.classList.add("is-active");
    dot.classList.add("running");
    if (conn) conn.querySelector(".line")?.classList.add("active");
  } else if (state === "done") {
    step.classList.add("is-done");
    dot.classList.add("done");
    if (conn) conn.querySelector(".line")?.classList.add("done");
  } else if (state === "error") {
    step.classList.add("is-error");
    dot.classList.add("error");
  }
}

function resetAllStages() {
  for (let i = 1; i <= 5; i++) {
    $(`step-${i}`).className = "step-number";
    $(`dot-${i}`).className = "status-dot pending";
    $(`badge-${i}`).textContent = $(`badge-${i}`).dataset.default || "–";
    $(`body-${i}`).innerHTML = '<div class="empty-state">Results will appear here after running the pipeline.</div>';
    const conn = $(`conn-${i}`);
    if (conn) conn.querySelector(".line")?.classList.remove("active", "done");
  }
}

// ─── Stage renderers ─────────────────────────

function renderStage1(words) {
  const body = $("body-1");
  if (!words || !words.length) {
    body.innerHTML = '<div class="empty-state">No words to score.</div>';
    return;
  }

  // Build the token stream
  const stream = document.createElement("div");
  stream.style.lineHeight = "2.2";
  lastWordEls = [];

  words.forEach((w, i) => {
    const cls = w.in_lexicon ? `word-token in-lexicon ${scoreClass(w.suspicion)}`
      : `word-token ${scoreClass(w.suspicion)}`;
    const span = document.createElement("span");
    span.className = cls;
    span.dataset.index = w.index;
    span.textContent = w.text + " ";

    // Build tooltip
    const tip = document.createElement("span");
    tip.className = "tooltip";
    tip.textContent = `idx ${w.index} · score ${escScore(w.suspicion)}${w.in_lexicon ? " · in lexicon" : ""}`;
    span.appendChild(tip);

    span.addEventListener("click", () => {
      highlightToken(w.index);
    });

    stream.appendChild(span);
    lastWordEls.push(span);
  });

  // Legend
  const legend = document.createElement("div");
  legend.className = "token-legend";
  legend.innerHTML = `
    <span class="legend-item"><span class="legend-swatch" style="background:#34d399"></span> 0.00 – 0.05</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#6ee7b7"></span> 0.06 – 0.15</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#a7f3d0"></span> 0.16 – 0.25</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#fef3c7"></span> 0.45 – 0.55</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#fcd34d"></span> 0.65 – 0.75</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#f59e0b"></span> 0.85 – 0.95</span>
    <span class="legend-item"><span class="legend-swatch" style="background:var(--red-bg);border:1px solid var(--red-border)"></span> ≥ 0.95</span>
    <span class="legend-item"><span class="legend-swatch" style="background:var(--accent-dim);border:1px solid var(--accent)"></span> in lexicon</span>
  `;

  // Table: suspicious words only
  const flagged = words.filter(w => w.suspicion >= 0.5 && !w.in_lexicon);
  const tableWrap = document.createElement("div");
  tableWrap.style.marginTop = "12px";

  let tableHtml = `<div class="text-sm text-muted mb-1">Suspicious words (≥ 0.50) — ${flagged.length} of ${words.length} total</div>`;
  if (flagged.length) {
    tableHtml += '<table style="width:100%;border-collapse:collapse;font-size:12px">';
    tableHtml += '<thead><tr style="color:var(--text-muted);border-bottom:1px solid var(--border)"><th style="padding:4px 6px;text-align:left">Word</th><th style="padding:4px 6px;text-align:left">Index</th><th style="padding:4px 6px;text-align:right">Score</th><th style="padding:4px 6px;text-align:right">Lexicon</th></tr></thead><tbody>';
    flagged.forEach(w => {
      tableHtml += `<tr style="border-bottom:1px solid rgba(255,255,255,0.03)" class="word-row" data-index="${w.index}">`;
      tableHtml += `<td style="padding:4px 6px;font-weight:500">${esc(w.text)}</td>`;
      tableHtml += `<td style="padding:4px 6px;color:var(--text-dim)">${w.index}</td>`;
      tableHtml += `<td style="padding:4px 6px;text-align:right;font-family:var(--font-mono);color:var(--red)">${escScore(w.suspicion)}</td>`;
      tableHtml += `<td style="padding:4px 6px;text-align:right;color:var(--text-dim)">${w.in_lexicon ? "✓" : "—"}</td>`;
      tableHtml += "</tr>";
    });
    tableHtml += "</tbody></table>";
  } else {
    tableHtml += '<div class="text-xs text-dim">No suspicious words found.</div>';
  }
  tableWrap.innerHTML = tableHtml;

  // Add click-to-highlight for table rows
  tableWrap.querySelectorAll(".word-row").forEach(row => {
    row.style.cursor = "pointer";
    row.addEventListener("click", () => {
      highlightToken(Number(row.dataset.index));
    });
    row.addEventListener("mouseenter", () => { row.style.background = "rgba(255,255,255,0.03)"; });
    row.addEventListener("mouseleave", () => { row.style.background = ""; });
  });

  body.innerHTML = "";
  body.appendChild(stream);
  body.appendChild(legend);
  body.appendChild(tableWrap);

  // Update badge
  $("badge-1").textContent = `${words.length} words, ${flagged.length} suspicious`;
}

function highlightToken(index) {
  // Clear all highlights
  lastWordEls.forEach(el => el.classList.remove("is-highlighted"));
  // Highlight target
  const target = lastWordEls.find(el => Number(el.dataset.index) === index);
  if (target) {
    target.classList.add("is-highlighted");
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function renderStage2(spans) {
  const body = $("body-2");
  if (!spans || !spans.length) {
    body.innerHTML = '<div class="empty-state">No suspicious spans detected.</div>';
    $("badge-2").textContent = "0 spans";
    return;
  }

  const list = document.createElement("div");
  list.className = "spans-list";

  // Mark tokens: span words get highlighted background
  lastWordEls.forEach(el => {
    el.classList.remove("is-flagged");
    const idx = Number(el.dataset.index);
    const inSpan = spans.some(s => idx >= s.start && idx <= s.end);
    if (inSpan) el.classList.add("is-flagged");
  });

  spans.forEach((s, i) => {
    const item = document.createElement("div");
    item.className = "span-item";
    item.innerHTML = `
      <span class="span-text">${esc(s.text)}</span>
      <span class="span-score">${escScore(s.suspicion)}</span>
      <span class="span-badge ${s.reason || 'both'}">${esc(s.reason || 'both')}</span>
      <span class="span-candidates-count">tokens ${s.start}–${s.end}</span>
    `;
    item.addEventListener("click", () => {
      // Highlight the tokens in stage 1
      lastWordEls.forEach(el => el.classList.remove("is-highlighted"));
      for (let j = s.start; j <= s.end; j++) {
        const target = lastWordEls.find(el => Number(el.dataset.index) === j);
        if (target) target.classList.add("is-highlighted");
      }
    });
    list.appendChild(item);
  });

  body.innerHTML = "";
  body.appendChild(list);
  $("badge-2").textContent = `${spans.length} span${spans.length !== 1 ? "s" : ""}`;
}

function renderStage3(candidatesList) {
  const body = $("body-3");
  if (!candidatesList || !candidatesList.length) {
    body.innerHTML = '<div class="empty-state">No candidates retrieved.</div>';
    $("badge-3").textContent = "0 candidates";
    return;
  }

  const total = candidatesList.reduce((acc, g) => acc + (g.candidates || []).length, 0);
  body.innerHTML = "";

  candidatesList.forEach((group, gi) => {
    const wrap = document.createElement("div");
    wrap.className = "span-candidate-group";

    const hdr = document.createElement("div");
    hdr.className = "group-header";
    hdr.innerHTML = `
      <span>Span:</span>
      <span class="span-text-label">${esc(group.span.text)}</span>
      <span class="text-xs text-dim">(${(group.candidates || []).length} candidates)</span>
    `;
    wrap.appendChild(hdr);

    (group.candidates || []).forEach((c, ci) => {
      const isChosen = false; // will be highlighted in stage 4
      const row = document.createElement("div");
      row.className = "candidate-row";

      const score = Number(c.phonetic_score || 0);
      const barClass = score >= 0.85 ? "good" : score >= 0.60 ? "ok" : "low";

      row.innerHTML = `
        <span class="candidate-term">${esc(c.term)}</span>
        <span class="candidate-type">${esc(c.term_type || "")}</span>
        <span class="candidate-source">${esc(c.source || "")}</span>
        <div class="score-bar-wrap">
          <div class="score-bar-bg">
            <div class="score-bar-fill ${barClass}" style="width:${Math.round(score * 100)}%"></div>
          </div>
          <span class="candidate-score">${escPct(score)}%</span>
        </div>
        <span class="candidate-desc" title="${esc(c.description || "")}">${esc((c.description || "").slice(0, 60))}</span>
        <button class="btn btn-sm btn-green teach-btn" data-term="${esc(c.term)}" data-span="${esc(group.span.text)}">Teach</button>
      `;

      const teachBtn = row.querySelector(".teach-btn");
      teachBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        await teachTerm(group.span.text, c.term, teachBtn);
      });

      wrap.appendChild(row);
    });

    body.appendChild(wrap);
  });

  $("badge-3").textContent = `${total} candidate${total !== 1 ? "s" : ""}`;
}

function renderStage4(decisions, spans, candidatesList) {
  const body = $("body-4");
  if (!decisions || !decisions.length) {
    body.innerHTML = '<div class="empty-state">No decisions made.</div>';
    $("badge-4").textContent = "0 decisions";
    return;
  }

  const grid = document.createElement("div");
  grid.className = "decisions-grid";

  let autoCount = 0, llmCount = 0, escalateCount = 0;

  decisions.forEach((d, i) => {
    const path = d.path || "unknown";
    if (path === "auto_fix") autoCount++;
    else if (path === "llm") llmCount++;
    else if (path && path.includes("hitl")) escalateCount++;

    const card = document.createElement("div");
    card.className = `decision-card path-${path.includes("hitl") ? "hitl" : path === "auto_fix" ? "auto" : path === "llm" ? "llm" : "hitl"}`;
    card.innerHTML = `
      <div class="dec-span-text">${esc(d.span.text)}</div>
      <div class="dec-chosen">→ <strong>${d.chosen ? esc(d.chosen) : '<span style="color:var(--text-dim)">no change</span>'}</strong></div>
      <div style="display:flex;align-items:center;gap:6px;margin-top:4px">
        <span class="dec-path-tag">${esc(path.replace(/_/g, " "))}</span>
        <span class="text-xs text-dim">conf ${escScore(d.confidence)}</span>
      </div>
    `;

    card.addEventListener("click", () => {
      // Highlight the span tokens
      const span = spans && spans[i];
      if (span) {
        lastWordEls.forEach(el => el.classList.remove("is-highlighted"));
        for (let j = span.start; j <= span.end; j++) {
          const target = lastWordEls.find(el => Number(el.dataset.index) === j);
          if (target) target.classList.add("is-highlighted");
        }
      }
    });
    card.style.cursor = "pointer";

    grid.appendChild(card);
  });

  body.innerHTML = "";
  body.appendChild(grid);
  $("badge-4").textContent = `${autoCount} auto · ${llmCount} llm · ${escalateCount} escalated`;
}

function renderStage5(originalText, correctedText, decisions, spans) {
  const body = $("body-5");
  const corrections = (decisions || []).filter(d => d.chosen).length;

  // Build a simple diff view
  const origWords = originalText.split(/(\s+)/);
  const corrWords = correctedText.split(/(\s+)/);
  const diffHtml = buildDiffHtml(origWords, corrWords);

  const diffContainer = document.createElement("div");
  diffContainer.className = "output-diff";

  // Original pane
  const origPane = document.createElement("div");
  origPane.className = "output-pane";
  origPane.innerHTML = `
    <div class="pane-header">Original</div>
    <div class="pane-body">${diffHtml.original}</div>
  `;

  // Corrected pane
  const corrPane = document.createElement("div");
  corrPane.className = "output-pane";
  corrPane.innerHTML = `
    <div class="pane-header">Corrected</div>
    <div class="pane-body">${diffHtml.corrected}</div>
  `;

  diffContainer.appendChild(origPane);
  diffContainer.appendChild(corrPane);

  // Correction list at bottom
  const correctionList = document.createElement("div");
  correctionList.style.marginTop = "12px";
  if (corrections > 0) {
    let listHtml = `<div class="text-sm text-muted mb-1">${corrections} correction${corrections !== 1 ? "s" : ""} applied:</div>`;
    (decisions || []).filter(d => d.chosen).forEach(d => {
      const origText = d.span.text;
      const newText = d.chosen;
      listHtml += `<div style="display:flex;align-items:center;gap:8px;padding:3px 0;font-size:12.5px">`;
      listHtml += `<span style="color:var(--red);text-decoration:line-through;min-width:120px">${esc(origText)}</span>`;
      listHtml += `<span style="color:var(--text-dim)">→</span>`;
      listHtml += `<span style="color:var(--green);font-weight:500">${esc(newText)}</span>`;
      listHtml += `<span class="text-xs text-dim">(${esc(d.path || "")})</span>`;
      listHtml += `<button class="btn btn-sm btn-green teach-btn-correction" data-term="${esc(newText)}" data-span="${esc(origText)}">Teach</button>`;
      listHtml += `</div>`;
    });
    correctionList.innerHTML = listHtml;
    correctionList.querySelectorAll(".teach-btn-correction").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        await teachTerm(btn.dataset.span, btn.dataset.term, btn);
      });
    });
  } else {
    correctionList.innerHTML = '<div class="text-xs text-dim">No corrections — all words passed.</div>';
  }

  // HITL review: flagged-but-unchanged words
  const hitlSection = document.createElement("div");
  hitlSection.id = "hitl-review";
  hitlSection.style.marginTop = "16px";
  const unresolved = getUnresolvedSpans(spans, decisions);
  if (unresolved.length > 0) {
    renderHitlReview(hitlSection, unresolved);
  }

  body.innerHTML = "";
  body.appendChild(diffContainer);
  body.appendChild(correctionList);
  body.appendChild(hitlSection);
  $("badge-5").textContent = `${corrections} correction${corrections !== 1 ? "s" : ""}`;
}

function buildDiffHtml(origWords, corrWords) {
  let origHtml = "", corrHtml = "";
  const maxLen = Math.max(origWords.length, corrWords.length);
  for (let i = 0; i < maxLen; i++) {
    const o = origWords[i] || "";
    const c = corrWords[i] || "";
    if (o === c) {
      origHtml += `<span class="diff-eq">${esc(o)}</span>`;
      corrHtml += `<span class="diff-eq">${esc(c)}</span>`;
    } else {
      origHtml += `<span class="diff-del">${esc(o)}</span>`;
      corrHtml += `<span class="diff-add">${esc(c)}</span>`;
    }
  }
  return { original: origHtml, corrected: corrHtml };
}

// ─── HITL Review ─────────────────────────────

function getUnresolvedSpans(spans, decisions) {
  // Return ALL spans for HITL review — every decision is reviewable.
  // Pre-fill the input with the auto-chosen term so the human can accept or override.
  if (!spans || !decisions) return [];
  const unresolved = [];
  for (let i = 0; i < spans.length && i < decisions.length; i++) {
    const span = spans[i];
    const decision = decisions[i];
    unresolved.push({
      text: span.text,
      start: span.start,
      end: span.end,
      suspicion: span.suspicion,
      chosen: decision.chosen || null,
      decisionPath: decision.path || "unchanged",
    });
  }
  return unresolved;
}

function renderHitlReview(container, unresolved) {
  const header = document.createElement("div");
  header.className = "hitl-header";
  header.innerHTML = `
    <strong>🔍 Review All Corrections</strong>
    <span class="text-xs text-muted">${unresolved.length} span${unresolved.length !== 1 ? "s" : ""} — review each correction below and save to the lexicon. Press Save to accept the auto-chosen term, or type a different one.</span>
  `;
  container.appendChild(header);

  const list = document.createElement("div");
  list.className = "hitl-list";

  unresolved.forEach((item) => {
    const row = document.createElement("div");
    row.className = "hitl-row";

    const wordLabel = document.createElement("span");
    wordLabel.className = "hitl-word";
    wordLabel.textContent = item.text;

    const arrow = document.createElement("span");
    arrow.className = "text-dim";
    arrow.textContent = "→";

    const input = document.createElement("input");
    input.className = "hitl-input";
    input.type = "text";
    input.placeholder = `Correct form of "${item.text}"...`;
    input.dataset.wrong = item.text;
    if (item.chosen) {
      input.value = item.chosen;
    }

    const pathTag = document.createElement("span");
    pathTag.className = "dec-path-tag";
    pathTag.textContent = (item.decisionPath || "unchanged").replace(/_/g, " ");

    const saveBtn = document.createElement("button");
    saveBtn.className = "btn btn-sm btn-green";
    saveBtn.textContent = "Save to Lexicon";
    saveBtn.addEventListener("click", async () => {
      const correction = input.value.trim();
      if (!correction) {
        input.style.borderColor = "var(--red)";
        return;
      }
      input.style.borderColor = "";
      saveBtn.disabled = true;
      saveBtn.textContent = "Saving...";
      try {
        const resp = await fetch("/api/v2/correct/teach", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            session_id: lastData?.session_id || "",
            wrong_form: item.text,
            correct_term: correction,
            sentence_context: lastTranscript,
          }),
        });
        if (!resp.ok) {
          const txt = await resp.text();
          throw new Error(txt);
        }
        saveBtn.textContent = "✓ Saved";
        saveBtn.classList.remove("btn-green");
        saveBtn.classList.add("btn-blue");
        input.disabled = true;
        row.style.opacity = "0.5";
      } catch (e) {
        saveBtn.textContent = "✗ Failed";
        console.error("HITL save failed:", e);
        setTimeout(() => {
          saveBtn.textContent = "Save to Lexicon";
          saveBtn.disabled = false;
        }, 2000);
      }
    });

    row.appendChild(wordLabel);
    row.appendChild(arrow);
    row.appendChild(input);
    row.appendChild(pathTag);
    row.appendChild(saveBtn);
    list.appendChild(row);
  });

  container.appendChild(list);
}

// ─── Teaching ─────────────────────────────────

async function teachTerm(spanText, term, btn) {
  btn.disabled = true;
  const originalText = btn.textContent;
  btn.textContent = "Saving...";
  try {
    const resp = await fetch("/api/v2/correct/teach", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        session_id: "",
        wrong_form: spanText,
        correct_term: term,
        sentence_context: lastTranscript,
      }),
    });
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(txt);
    }
    btn.textContent = "✓ Saved";
    btn.classList.remove("btn-green");
    btn.classList.add("btn-blue");
  } catch (e) {
    btn.textContent = "✗ Failed";
    console.error("Teach failed:", e);
    setTimeout(() => { btn.textContent = originalText; btn.disabled = false; }, 1500);
    return;
  }
  setTimeout(() => { btn.disabled = false; }, 2000);
}

// ─── Pipeline run ────────────────────────────

async function runPipeline() {
  const transcript = $("transcript-input").value.trim();
  if (!transcript) return;

  const runBtn = $("btn-run");
  const overlay = $("running-overlay");
  const timingEl = $("run-timing");

  runBtn.disabled = true;
  overlay.classList.add("active");
  $("running-text").textContent = "Running pipeline stages...";
  timingEl.textContent = "";
  lastTranscript = transcript;

  // Reset UI
  resetAllStages();
  lastWordEls = [];

  // Animate stages as active
  setStage(1, "active");

  const t0 = performance.now();

  try {
    const resp = await fetch("/api/v2/correct", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ transcript, interactive: false }),
    });

    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${txt.slice(0, 200)}`);
    }

    const data = await resp.json();
    lastData = data;
    const report = validateReport(data);

    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    timingEl.textContent = `${elapsed}s`;

    // Stage 1 — Word scoring (always available from the report)
    const scoredWords = report.scored_words || report.scoredWords || [];
    renderStage1(scoredWords);
    setStage(1, "done");
    setStage(2, "active");

    // Stage 2 — Flagged spans
    const spans = report.spans || [];
    renderStage2(spans);
    setStage(2, "done");
    setStage(3, "active");

    // Stage 3 — Candidates
    const candidatesList = report.candidates || [];
    renderStage3(candidatesList);
    setStage(3, "done");
    setStage(4, "active");

    // Stage 4 — Decisions
    const decisions = report.decisions || [];
    renderStage4(decisions, spans, candidatesList);
    setStage(4, "done");
    setStage(5, "active");

    // Stage 5 — Corrected output
    const corrected = data.corrected_text || data.correctedText || "";
    renderStage5(transcript, corrected, decisions, spans);
    setStage(5, "done");

    // Show approach tags for each stage
    const approaches = report.approaches || {};
    for (let i = 1; i <= 5; i++) {
      const tag = $(`approach-${i}`);
      const key = ["scoring","flagging","retrieval","decision","correction"][i - 1];
      const info = approaches[key];
      if (info) {
        tag.textContent = info.label || info.mode;
        tag.title = info.description || "";
        tag.className = "approach-tag " + (info.status === "primary" ? "tag-primary" : "tag-fallback");
      } else {
        tag.textContent = "?";
        tag.className = "approach-tag tag-unknown";
      }
    }

    // Timing on badge
    $("run-timing").textContent = `completed in ${elapsed}s`;
    $("running-text").textContent = "Pipeline complete ✓";

  } catch (e) {
    console.error("Pipeline error:", e);
    $("running-text").textContent = `Error: ${e.message}`;
    // Mark the current active stage as error
    for (let i = 1; i <= 5; i++) {
      const dot = $(`dot-${i}`);
      if (dot.classList.contains("running")) {
        setStage(i, "error");
        break;
      }
    }
    timingEl.textContent = "error";
  } finally {
    overlay.classList.remove("active");
    runBtn.disabled = false;
  }
}

// ─── Lexicon panel ────────────────────────────

let allLexicon = [];

async function refreshLexicon() {
  try {
    const r = await fetch("/api/lexicon");
    const data = await r.json();
    allLexicon = data.entries || [];
    $("lexicon-count").textContent = `(${data.count})`;
    renderLexicon($("lexicon-search").value);
  } catch (_) {
    /* ignore */
  }
}

function renderLexicon(query) {
  const q = (query || "").trim().toLowerCase();
  const filtered = q
    ? allLexicon.filter(e =>
        e.term.toLowerCase().includes(q) ||
        (e.aliases || []).some(a => a.toLowerCase().includes(q))
      )
    : allLexicon;
  const list = $("lexicon-list");
  list.innerHTML = "";
  filtered.slice(0, 300).forEach(e => {
    const item = document.createElement("div");
    item.className = "lexicon-item";
    const aliasStr = (e.aliases || []).length
      ? `<span class="lex-aliases" title="${esc(e.aliases.join(", "))}">${esc(e.aliases.slice(0, 3).join(", "))}${e.aliases.length > 3 ? "…" : ""}</span>`
      : "";
    item.innerHTML = `<span class="lex-term">${esc(e.term)}</span><span class="lex-type">${esc(e.type || "")}</span>${aliasStr}`;
    item.addEventListener("click", () => {
      $("transcript-input").value = $("transcript-input").value + " " + e.term;
    });
    item.style.cursor = "pointer";
    list.appendChild(item);
  });
  if (filtered.length > 300) {
    const more = document.createElement("div");
    more.className = "lexicon-item text-dim text-xs";
    more.textContent = `… ${filtered.length - 300} more`;
    list.appendChild(more);
  }
}

// Toggle lexicon panel
$("btn-toggle-lexicon").addEventListener("click", () => {
  const panel = $("lexicon-panel");
  panel.style.display = panel.style.display === "none" ? "flex" : "none";
  if (panel.style.display === "flex") refreshLexicon();
});
$("btn-close-lexicon").addEventListener("click", () => {
  $("lexicon-panel").style.display = "none";
});
$("lexicon-search").addEventListener("input", (e) => renderLexicon(e.target.value));

// ─── Error boundary ────────────────────────────

function validateReport(data) {
  if (!data || typeof data !== "object") throw new Error("Invalid response: not an object");
  const report = data.report;
  if (!report || typeof report !== "object") throw new Error("Invalid response: missing report");
  // Top-level corrected_text should exist
  if (!data.corrected_text && !data.correctedText) {
    console.warn("Pipeline response missing corrected_text", data);
  }
  return report;
}

// ─── Events ──────────────────────────────────

$("btn-run").addEventListener("click", runPipeline);

$("btn-example").addEventListener("click", () => {
  $("transcript-input").value =
    "The patient presents with fever and should take dolly prahn twice daily " +
    "alongside salbu tamol for the wheeze. Blood pressure was measured " +
    "using a sfigmomanometre. The attending physician prescribed " +
    "amoxicilin for the secondary infection.";
  $("word-count").textContent = `${$("transcript-input").value.split(/\s+/).filter(Boolean).length} words`;
});

// Word count on input
$("transcript-input").addEventListener("input", () => {
  const words = $("transcript-input").value.split(/\s+/).filter(Boolean).length;
  $("word-count").textContent = words ? `${words} words` : "";
});

// Keyboard shortcut: Ctrl+Enter to run
$("transcript-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    runPipeline();
  }
});

// Initial word count
$("transcript-input").dispatchEvent(new Event("input"));
