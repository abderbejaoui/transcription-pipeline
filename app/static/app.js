// ---------------------------------------------------------------------------
// Medical Voice — Gulf Arabic client.
//
// Two pipelines are wired here:
//   • DEFAULT (Record button) → /api/transcribe_debug
//       Returns: raw transcript, MMS-aligned word timestamps, phonetic
//       flags with candidates, optional LLM 'likely_term', and an
//       auto-corrected transcript (high-confidence LLM fixes applied).
//   • LEGACY (file upload, paste text) → /api/transcribe_stream
//       The older pipeline with the LLM detect/decide trace.
//
// Both render into the same tabbed result card.
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const escapeHtml = (s) =>
  String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

let lastRawText = "";
let lastSessionId = null;
let lastFlags = [];
let lastAudioUrl = null;

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
function activateTab(name) {
  document.querySelectorAll(".tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    p.classList.toggle("active", p.id === "tab-" + name);
  });
}
document.querySelectorAll(".tab").forEach((b) => {
  b.addEventListener("click", () => activateTab(b.dataset.tab));
});

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------
async function postJson(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.text()) || `HTTP ${r.status}`);
  return r.json();
}

// ---------------------------------------------------------------------------
// Main result renderer — used by both pipelines.
// ---------------------------------------------------------------------------
function showDebugResult(data) {
  if (data.error) {
    alert("Pipeline error: " + data.error);
    return;
  }
  $("result-card").hidden = false;

  // 1) Corrected transcript tab
  const corrected = data.corrected_transcript || data.transcript || "";
  const raw = data.transcript || "";
  lastRawText = raw;
  lastSessionId = data.session_id || null;
  lastFlags = data.flags || [];
  lastAudioUrl = data.audio_url || null;

  $("corrected-text").value = corrected;
  $("raw-text").textContent = raw;
  $("save-status").textContent = "";

  const applied = data.auto_corrections || [];
  if (applied.length) {
    const summary = applied
      .map((a) => `${a.original.trim()} → ${a.corrected}`)
      .join(", ");
    $("correction-meta").innerHTML =
      `<span class="muted">${applied.length} auto-correction${applied.length === 1 ? "" : "s"} applied:</span> ${escapeHtml(summary)}`;
  } else if (lastFlags.length) {
    $("correction-meta").innerHTML =
      `<span class="muted">No auto-corrections applied. ${lastFlags.length} word${lastFlags.length === 1 ? "" : "s"} flagged for review.</span>`;
  } else {
    $("correction-meta").innerHTML =
      `<span class="muted">No suspicious words detected.</span>`;
  }

  // 2) Flags tab — the main attraction
  renderFlags(data);

  // 3) Audio player
  if (lastAudioUrl) {
    $("debug-audio").src = lastAudioUrl;
  }

  // Update tab badge
  const badge = $("flag-badge");
  if (lastFlags.length) {
    badge.hidden = false;
    badge.textContent = lastFlags.length;
  } else {
    badge.hidden = true;
  }

  activateTab("corrected");
}

function renderFlags(data) {
  const flagsEl = $("flags-list");
  flagsEl.innerHTML = "";
  const flags = data.flags || [];

  if (!flags.length) {
    $("flag-summary").textContent = "No suspicious medical words found. 🎉";
  } else {
    $("flag-summary").textContent =
      `${flags.length} suspicious word${flags.length === 1 ? "" : "s"} found. ` +
      `Click ▶ to listen to each flagged span.`;
  }

  for (const f of flags) {
    const top = (f.candidates || [])[0];
    const llm = f.llm_likely_term || "";
    const conf = f.llm_confidence;
    const proposed = llm || (top && top.term) || "?";

    // Determine confidence class for styling
    let confClass = "";
    if (conf != null && conf >= 0.9) confClass = "high-conf";
    else if (conf != null && conf >= 0.6) confClass = "medium-conf";

    const div = document.createElement("div");
    div.className = "flag-row " + confClass;

    const cands = (f.candidates || [])
      .slice(0, 3)
      .map((c, idx) => {
        const cls = idx === 0 ? "candidate-chip top" : "candidate-chip";
        return `<span class="${cls}">${escapeHtml(c.term)} <span class="muted">${(c.phonetic_similarity || 0).toFixed(2)}</span></span>`;
      })
      .join("");

    const t0 = f.start_s != null ? f.start_s.toFixed(2) : null;
    const t1 = f.end_s != null ? f.end_s.toFixed(2) : null;
    const playBtn =
      t0 != null && t1 != null
        ? `<button class="ghost play-slice" data-start="${f.start_s}" data-end="${f.end_s}">▶ ${t0}–${t1}s</button>`
        : `<span class="muted small">no alignment</span>`;

    const llmInfo = conf != null
      ? `LLM: <code>${escapeHtml(llm || "—")}</code> ${(conf * 100).toFixed(0)}%`
      : (llm ? `LLM: <code>${escapeHtml(llm)}</code>` : "");

    div.innerHTML = `
      <div class="flag-header">
        <div>
          <span class="flag-word" dir="auto">${escapeHtml(f.word || "")}</span>
          <span class="flag-arrow">→</span>
          <span class="flag-corrected">${escapeHtml(proposed)}</span>
        </div>
        ${playBtn}
      </div>
      <div class="flag-meta">
        <span>#${f.index}</span>
        <span>${escapeHtml(f.reason || "")}</span>
        ${llmInfo ? `<span>${llmInfo}</span>` : ""}
      </div>
      <div class="flag-candidates">${cands}</div>
    `;
    flagsEl.appendChild(div);
  }

  // Per-word alignment table
  const tbody = $("words-table").querySelector("tbody");
  tbody.innerHTML = "";
  (data.words || []).forEach((w, i) => {
    const tr = document.createElement("tr");
    const s = w.start_s != null ? w.start_s.toFixed(2) : "—";
    const e = w.end_s != null ? w.end_s.toFixed(2) : "—";
    tr.innerHTML = `<td>${i}</td><td dir="auto"><code>${escapeHtml(w.word)}</code></td><td>${s}</td><td>${e}</td><td>${(w.confidence ?? 0).toFixed(2)}</td>`;
    tbody.appendChild(tr);
  });

  // Wire play buttons (slice playback)
  const audioEl = $("debug-audio");
  flagsEl.querySelectorAll(".play-slice").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!audioEl.src && lastAudioUrl) audioEl.src = lastAudioUrl;
      const t0 = parseFloat(btn.dataset.start);
      const t1 = parseFloat(btn.dataset.end);
      audioEl.currentTime = Math.max(0, t0);
      audioEl.play();
      const wait = Math.max(80, (t1 - t0) * 1000 + 100);
      setTimeout(() => audioEl.pause(), wait);
    });
  });
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------
let mediaRecorder = null;
let mediaStream = null;
let recordedChunks = [];
let recordedBlob = null;
let recordedMime = "audio/webm";
let recordTimer = null;
let recordStart = 0;
// Which pipeline the current recording should feed once stopped.
// "debug" = default correction pipeline, "ab" = v2 A/B model test.
let recordMode = "debug";

const PREFERRED_MIMES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/ogg;codecs=opus",
  "",
];

function pickMime() {
  if (typeof MediaRecorder === "undefined") return null;
  for (const m of PREFERRED_MIMES) {
    if (m === "" || MediaRecorder.isTypeSupported(m)) return m;
  }
  return null;
}

function setRecordingUI(isRecording) {
  const ab = recordMode === "ab";
  // While recording, only the active mode's Stop button shows; the other
  // mode's Record button is hidden. When idle, BOTH Record buttons show.
  // Default pipeline buttons
  $("btn-record").hidden = isRecording;
  $("btn-stop").hidden = !(isRecording && !ab);
  $("btn-record").classList.toggle("recording", isRecording && !ab);
  // A/B pipeline buttons
  $("btn-record-ab").hidden = isRecording;
  $("btn-stop-ab").hidden = !(isRecording && ab);
  $("btn-record-ab").classList.toggle("recording", isRecording && ab);
}

function fmtTime(ms) {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

async function startRecording(mode = "debug") {
  // Guard: ignore clicks while a recording is already in progress so the
  // buttons can't kick off a second overlapping recorder.
  if (mediaRecorder && mediaRecorder.state === "recording") return;
  recordMode = mode === "ab" ? "ab" : "debug";
  $("record-status").textContent = "";

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $("record-status").textContent =
      "❌ Microphone API not available. Use Chrome/Edge/Safari over http://localhost or https.";
    return;
  }
  if (typeof MediaRecorder === "undefined") {
    $("record-status").textContent = "❌ MediaRecorder not supported in this browser.";
    return;
  }

  const mime = pickMime();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    if (err.name === "NotAllowedError" || err.name === "SecurityError") {
      $("record-status").textContent = "❌ Microphone permission denied.";
    } else if (err.name === "NotFoundError") {
      $("record-status").textContent = "❌ No microphone found.";
    } else {
      $("record-status").textContent = "❌ " + err.message;
    }
    return;
  }

  mediaStream = stream;
  recordedChunks = [];
  try {
    mediaRecorder = mime
      ? new MediaRecorder(stream, { mimeType: mime })
      : new MediaRecorder(stream);
  } catch (err) {
    $("record-status").textContent = "❌ Recorder failed: " + err.message;
    stream.getTracks().forEach((t) => t.stop());
    return;
  }

  recordedMime = mediaRecorder.mimeType || mime || "audio/webm";

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };

  mediaRecorder.onstop = async () => {
    clearInterval(recordTimer);
    setRecordingUI(false);
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    if (!recordedChunks.length) {
      $("record-status").textContent = "❌ No audio captured.";
      return;
    }
    recordedBlob = new Blob(recordedChunks, { type: recordedMime });
    const playback = $("record-playback");
    playback.src = URL.createObjectURL(recordedBlob);
    playback.hidden = false;
    $("record-status").textContent =
      `Processing ${(recordedBlob.size / 1024).toFixed(0)} KB…`;
    try {
      if (recordMode === "ab") {
        await transcribeAB(recordedBlob);
      } else {
        await transcribeDebug(recordedBlob);
      }
      $("record-status").textContent = "Done.";
    } catch (err) {
      $("record-status").textContent = "❌ " + err.message;
    }
  };

  mediaRecorder.onerror = (e) => {
    $("record-status").textContent =
      "❌ Recorder error: " + (e.error?.message || "unknown");
  };

  mediaRecorder.start(250);
  recordStart = Date.now();
  setRecordingUI(true);
  $("record-status").textContent = "Recording 00:00";
  recordTimer = setInterval(() => {
    $("record-status").textContent = "Recording " + fmtTime(Date.now() - recordStart);
  }, 250);
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  } else {
    setRecordingUI(false);
  }
}

// ---------------------------------------------------------------------------
// Default pipeline — calls /api/transcribe_debug (transcript + flags + align).
// ---------------------------------------------------------------------------
async function transcribeDebug(blob) {
  const form = new FormData();
  const ext = (blob.type.split("/")[1] || "webm").split(";")[0];
  form.append("audio", blob, `recording.${ext}`);
  const lang = $("lang").value;
  if (lang) form.append("language", lang);

  $("flag-summary").textContent = "Running ASR + flag + alignment…";
  $("flags-list").innerHTML = "";
  $("result-card").hidden = false;

  const r = await fetch("/api/transcribe_debug", { method: "POST", body: form });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error || msg; } catch (_) { msg = await r.text(); }
    $("flag-summary").textContent = "❌ " + msg;
    throw new Error(msg);
  }
  const data = await r.json();
  showDebugResult(data);
}

// ---------------------------------------------------------------------------
// A/B pipeline — calls /api/transcribe_ab and shows both v2 arms' output.
// ---------------------------------------------------------------------------
async function transcribeAB(blob) {
  const form = new FormData();
  const ext = (blob.type.split("/")[1] || "webm").split(";")[0];
  form.append("audio", blob, `recording.${ext}`);
  const lang = $("lang").value;
  if (lang) form.append("language", lang);

  $("ab-card").hidden = false;
  $("ab-status").textContent = "Running both arms (first run loads the models, this can take a while)…";
  $("ab-a-text").textContent = "";
  $("ab-b-text").textContent = "";
  $("ab-a-meta").textContent = "";
  $("ab-b-meta").textContent = "";

  const r = await fetch("/api/transcribe_ab", { method: "POST", body: form });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error || msg; } catch (_) { msg = await r.text(); }
    $("ab-status").textContent = "❌ " + msg;
    throw new Error(msg);
  }
  showABResult(await r.json());
}

function renderArm(prefix, arm) {
  if (!arm) return;
  if (arm.label) $(prefix + "-label").textContent = arm.label;
  if (arm.error) {
    $(prefix + "-text").textContent = "";
    $(prefix + "-meta").innerHTML = `<span style="color:#c0392b">❌ ${escapeHtml(arm.error)}</span>`;
    return;
  }

  // `text` is already drug-normalized (Arabic brand names -> Latin).
  $(prefix + "-text").textContent = arm.text || "(empty)";

  const bits = [];
  if (arm.elapsed_s != null) bits.push(`${arm.elapsed_s}s`);

  const fixes = arm.drug_corrections || [];
  if (fixes.length) {
    const summary = fixes
      .map((f) => `${escapeHtml(f.from)} → ${escapeHtml(f.to)}`)
      .join(", ");
    bits.push(
      `<span class="muted">${fixes.length} drug fix${fixes.length === 1 ? "" : "es"}:</span> ${summary}`
    );
  }
  // Show the raw ASR output (pre-normalization) only when it differed.
  if (arm.raw_text && arm.raw_text !== arm.text) {
    bits.push(`<span class="muted">raw:</span> ${escapeHtml(arm.raw_text)}`);
  }
  $(prefix + "-meta").innerHTML = bits.join(" &nbsp;·&nbsp; ");
}

function showABResult(data) {
  $("ab-card").hidden = false;
  $("ab-status").textContent = "Compare the two models below.";
  renderArm("ab-a", data.arm_a);
  renderArm("ab-b", data.arm_b);
  if (data.audio_url) {
    const a = $("ab-audio");
    a.src = data.audio_url;
    a.hidden = false;
  }
  $("ab-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------------------------------------------------------------------------
// Legacy streaming pipeline — used by the file upload + paste-text inputs.
// Streams trace events into the Pipeline tab.
// ---------------------------------------------------------------------------
async function streamTranscribe(form) {
  resetTrace();
  $("result-card").hidden = false;
  $("trace-status").textContent = "Pipeline running…";
  activateTab("trace");

  const r = await fetch("/api/transcribe_stream", { method: "POST", body: form });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error || msg; } catch (_) { msg = await r.text(); }
    $("trace-status").textContent = "❌ " + msg;
    throw new Error(msg);
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";
  let final = null;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      try {
        const event = JSON.parse(line);
        appendTraceEvent(event);
        if (event.stage === "final") final = event.payload;
      } catch (_) {}
    }
  }
  $("trace-status").textContent = "Done.";

  // Adapt the legacy 'final' payload into the same shape showDebugResult expects.
  if (final) {
    const adapted = {
      session_id: final.session_id,
      transcript: final.raw_text,
      corrected_transcript: final.corrected_text,
      auto_corrections: [],
      flags: (final.suspicious || []).map((s, i) => ({
        index: i,
        word: s.span || "",
        start_s: s.start_s,
        end_s: s.end_s,
        reason: s.reason || "legacy",
        candidates: (s.candidates || []).map((c) => ({
          term: c.term,
          phonetic_similarity: c.similarity || 0,
        })),
        llm_likely_term: s.chosen || "",
      })),
      words: (final.asr && final.asr.words) || [],
      audio_url: `/api/session_audio/${final.session_id}`,
    };
    showDebugResult(adapted);
  }
}

function resetTrace() {
  $("trace-events").innerHTML = "";
}

function appendTraceEvent(event) {
  const wrap = document.createElement("div");
  wrap.className = "trace-event";
  wrap.dataset.stage = event.stage;
  const t = event.t != null ? event.t.toFixed(3) + "s" : "";
  const summary = traceSummary(event.stage, event.payload);
  wrap.innerHTML = `
    <div class="head">
      <span class="t">${escapeHtml(t)}</span>
      <span class="stage">${escapeHtml(event.stage)}</span>
      <span class="muted small">${escapeHtml(summary)}</span>
    </div>
    <pre hidden></pre>
  `;
  const pre = wrap.querySelector("pre");
  pre.textContent = JSON.stringify(event.payload, null, 2);
  wrap.querySelector(".head").addEventListener("click", () => {
    pre.hidden = !pre.hidden;
  });
  $("trace-events").appendChild(wrap);
  $("trace-events").scrollTop = $("trace-events").scrollHeight;
}

function traceSummary(stage, payload) {
  if (!payload || typeof payload !== "object") return "";
  if (stage === "asr.start") return `${payload.size_bytes} bytes`;
  if (stage === "asr.done") return (payload.raw_text || "").slice(0, 80);
  if (stage === "detect.spans") return `${(payload.spans || []).length} spans`;
  if (stage === "voice_first.spans") return `${(payload.spans || []).length} spans`;
  if (stage === "spans.merged") return `${(payload.spans || []).length} merged`;
  if (stage === "decide.done") return JSON.stringify(payload.decisions || {});
  if (stage === "final") return `corrected: ${(payload.corrected_text || "").slice(0, 80)}`;
  if (stage.endsWith(".error")) return payload.error || "";
  return "";
}

// ---------------------------------------------------------------------------
// Button wiring
// ---------------------------------------------------------------------------
$("btn-record").addEventListener("click", (e) => { e.preventDefault(); e.currentTarget.blur(); startRecording("debug"); });
$("btn-stop").addEventListener("click", (e) => { e.preventDefault(); e.currentTarget.blur(); stopRecording(); });
$("btn-record-ab").addEventListener("click", (e) => { e.preventDefault(); e.currentTarget.blur(); startRecording("ab"); });
$("btn-stop-ab").addEventListener("click", (e) => { e.preventDefault(); e.currentTarget.blur(); stopRecording(); });

document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat) return;
  const tgt = e.target;
  // Ignore when typing in a field, or when a button is focused (the button's
  // own Space-to-click already handles it — avoids a double trigger).
  if (tgt && /INPUT|TEXTAREA|SELECT|BUTTON/.test(tgt.tagName)) return;
  e.preventDefault();
  if (mediaRecorder?.state === "recording") return;
  startRecording("debug");
});
document.addEventListener("keyup", (e) => {
  if (e.code !== "Space") return;
  const tgt = e.target;
  if (tgt && /INPUT|TEXTAREA|SELECT|BUTTON/.test(tgt.tagName)) return;
  // Only the spacebar push-to-talk (debug mode) auto-stops on key release.
  if (mediaRecorder?.state === "recording" && recordMode === "debug") stopRecording();
});

$("btn-correct").addEventListener("click", async () => {
  const text = $("raw-input").value.trim();
  if (!text) return;
  try {
    const data = await postJson("/api/correct", { text });
    // Adapt to showDebugResult shape.
    showDebugResult({
      session_id: data.session_id,
      transcript: data.raw_text || text,
      corrected_transcript: data.corrected_text || text,
      auto_corrections: [],
      flags: [],
      words: [],
      audio_url: null,
    });
  } catch (e) {
    alert(e.message);
  }
});

$("btn-example").addEventListener("click", () => {
  $("raw-input").value =
    "Patient with myokardial infarction. Start metoprol and atorvasta and clopidogr.";
});

$("btn-upload").addEventListener("click", async () => {
  const f = $("upload-file").files?.[0];
  if (!f) {
    alert("Pick an audio file first.");
    return;
  }
  const form = new FormData();
  form.append("audio", f, f.name);
  const lang = $("lang").value;
  if (lang) form.append("language", lang);
  try {
    // For file uploads use the debug pipeline too — it's the new default.
    const r = await fetch("/api/transcribe_debug", { method: "POST", body: form });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    showDebugResult(data);
  } catch (e) {
    alert(e.message);
  }
});

// ---------------------------------------------------------------------------
// Save user edits → auto-learn (text + voice)
// ---------------------------------------------------------------------------
$("btn-save-edits").addEventListener("click", async () => {
  const corrected = $("corrected-text").value.trim();
  if (!corrected || !lastRawText) return;
  $("btn-save-edits").disabled = true;
  $("save-status").textContent = "Saving…";
  try {
    const data = await postJson("/api/learn_from_edit", {
      raw_text: lastRawText,
      corrected_text: corrected,
      session_id: lastSessionId,
      type: "drug",
    });
    const learnedTerms = (data.learned_text || []).map((l) =>
      l.from_alias ? `${l.entry.term} (was: ${l.from_alias})` : l.entry.term,
    );
    const learnedVoices = (data.learned_voices || []).map((v) => v.voice.term);
    const parts = [];
    if (learnedTerms.length) parts.push(`text: ${learnedTerms.join(", ")}`);
    if (learnedVoices.length) parts.push(`voice: ${learnedVoices.join(", ")}`);
    if (!parts.length) {
      $("save-status").textContent = "✓ No new terms.";
    } else {
      $("save-status").textContent = `✓ Learned ${parts.join(" | ")}`;
    }
    refreshLexicon();
  } catch (e) {
    $("save-status").textContent = "❌ " + e.message;
  } finally {
    $("btn-save-edits").disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Vocabulary list
// ---------------------------------------------------------------------------
let allEntries = [];

async function refreshLexicon() {
  try {
    const r = await fetch("/api/lexicon");
    const data = await r.json();
    allEntries = data.entries || [];
    $("lexicon-count").textContent = `(${data.count})`;
    renderLexicon($("vocab-search").value);
  } catch (_) {}
}

function renderLexicon(query) {
  const q = (query || "").trim().toLowerCase();
  const filtered = q
    ? allEntries.filter(
        (e) =>
          e.term.toLowerCase().includes(q) ||
          (e.aliases || []).some((a) => a.toLowerCase().includes(q)),
      )
    : allEntries;
  const list = $("lexicon-list");
  list.innerHTML = "";
  filtered.slice(0, 200).forEach((e) => {
    const row = document.createElement("div");
    row.className = "row";
    const aliasStr = (e.aliases || []).length
      ? `<span class="aliases">aka ${e.aliases.join(", ")}</span>`
      : "";
    row.innerHTML = `<span class="term">${escapeHtml(e.term)}</span><span class="type">${escapeHtml(e.type)}</span>${aliasStr}`;
    list.appendChild(row);
  });
  if (filtered.length > 200) {
    const more = document.createElement("div");
    more.className = "row muted";
    more.textContent = `… ${filtered.length - 200} more`;
    list.appendChild(more);
  }
}

$("btn-refresh-lexicon").addEventListener("click", refreshLexicon);
$("vocab-search").addEventListener("input", (e) => renderLexicon(e.target.value));
refreshLexicon();
