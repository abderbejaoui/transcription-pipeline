// ---------------------------------------------------------------------------
// Voice-grounded medical corrector — minimal client
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

async function postForm(url, form) {
  const r = await fetch(url, { method: "POST", body: form });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      msg = (await r.json()).error || msg;
    } catch (_) {
      msg = await r.text();
    }
    throw new Error(msg);
  }
  return r.json();
}

function showResult(payload) {
  $("result").hidden = false;
  lastRawText = payload.raw_text || "";
  lastSessionId = payload.session_id || null;
  $("raw-text").textContent = lastRawText;
  $("corrected-text").value = payload.corrected_text || "";
  $("save-status").textContent = "";

  // Dual-ASR breakdown: shown only when the backend ran USE_DUAL_ASR=1
  // and returned both raw transcripts plus the Calme merge reason.
  const dual = payload.asr && payload.asr.dual;
  const dualSection = $("dual-asr-section");
  if (dual && (dual.transcript_a_gulf_lora || dual.transcript_b_base)) {
    dualSection.hidden = false;
    $("dual-text-a").textContent = dual.transcript_a_gulf_lora || "";
    $("dual-text-b").textContent = dual.transcript_b_base || "";
    $("dual-reason").textContent = dual.merge_reason || "";
    dualSection.open = true;  // expand by default so the user can see it
  } else {
    dualSection.hidden = true;
  }

  const tbody = $("spans-table").querySelector("tbody");
  tbody.innerHTML = "";
  // /api/transcribe returns `suspicious[]`; /api/correct returns `suspicious_spans[]`.
  const spans = payload.suspicious || payload.suspicious_spans || [];
  spans.forEach((span) => {
    const orig = span.span || span.original_text || "";
    const chosen = span.chosen || span.possible_correction || "—";
    const conf =
      (span.audio_hits && span.audio_hits[0] && `audio sim ${span.audio_hits[0].similarity}`) ||
      (span.confidence != null ? `conf ${span.confidence}` : "");
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${escapeHtml(orig)}</code></td>
      <td><code>${escapeHtml(chosen)}</code></td>
      <td>${escapeHtml(conf)}</td>`;
    tbody.appendChild(tr);
  });
  $("corrected-text").focus();
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
  $("btn-record").hidden = isRecording;
  $("btn-stop").hidden = !isRecording;
  $("btn-record").classList.toggle("recording", isRecording);
}

function fmtTime(ms) {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

async function startRecording() {
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
      $("record-status").textContent = "❌ Microphone permission denied. Allow it in the address bar.";
    } else if (err.name === "NotFoundError") {
      $("record-status").textContent = "❌ No microphone found on this device.";
    } else {
      $("record-status").textContent = "❌ " + err.message;
    }
    return;
  }

  mediaStream = stream;
  recordedChunks = [];
  try {
    mediaRecorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
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
    $("record-status").textContent = `Transcribing ${(recordedBlob.size / 1024).toFixed(0)} KB...`;
    try {
      await transcribeBlob(recordedBlob);
      $("record-status").textContent = "Done.";
    } catch (err) {
      $("record-status").textContent = "❌ " + err.message;
    }
  };

  mediaRecorder.onerror = (e) => {
    $("record-status").textContent = "❌ Recorder error: " + (e.error?.message || "unknown");
  };

  mediaRecorder.start(250);
  recordStart = Date.now();
  setRecordingUI(true);
  $("record-status").textContent = "🔴 Recording 00:00";
  recordTimer = setInterval(() => {
    $("record-status").textContent = "🔴 Recording " + fmtTime(Date.now() - recordStart);
  }, 250);
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  } else {
    setRecordingUI(false);
  }
}

async function transcribeBlob(blob) {
  const form = new FormData();
  const ext = (blob.type.split("/")[1] || "webm").split(";")[0];
  form.append("audio", blob, `recording.${ext}`);
  const lang = $("lang").value;
  if (lang) form.append("language", lang);
  form.append("model_size", $("model").value);
  await streamTranscribe(form);
}

async function streamTranscribe(form) {
  resetTrace();
  $("trace-card").hidden = false;
  $("trace-status").textContent = "Pipeline running...";
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
      } catch (_) { /* ignore */ }
    }
  }
  $("trace-status").textContent = "Done.";
  if (final) showResult(final);
}

function resetTrace() {
  $("trace-events").innerHTML = "";
}

function appendTraceEvent(event) {
  const wrap = document.createElement("div");
  wrap.className = "trace-event";
  wrap.dataset.stage = event.stage;
  const t = (event.t != null) ? event.t.toFixed(3) + "s" : "";
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
  if (stage === "asr.start") return `${payload.size_bytes} bytes, model ${payload.model}`;
  if (stage === "asr.done")  return `${payload.raw_text}`;
  if (stage === "detect.spans") return `${(payload.spans||[]).length} spans`;
  if (stage === "voice_first.spans") return `${(payload.spans||[]).length} spans`;
  if (stage === "spans.merged") return `${(payload.spans||[]).length} merged`;
  if (stage === "retrieve.span") {
    const auto = payload.auto ? " AUTO" : "";
    const top = (payload.user_hits||[])[0];
    const sim = top ? ` user-top ${top.term}@${top.similarity}` : "";
    return `${payload.span_text}${auto}${sim}`;
  }
  if (stage === "decide.request") return `${(payload.user?.spans||[]).length} spans -> LLM`;
  if (stage === "decide.response") return (payload.raw||"").slice(0, 80);
  if (stage === "decide.done") return JSON.stringify(payload.decisions||{});
  if (stage === "detect.request") return "LLM detect call";
  if (stage === "detect.response") return (payload.raw||"").slice(0, 80);
  if (stage === "final") return `corrected: ${(payload.corrected_text||"").slice(0,80)}`;
  if (stage.endsWith(".error")) return payload.error || "";
  return "";
}

$("btn-record").addEventListener("click", startRecording);
$("btn-stop").addEventListener("click", stopRecording);

document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat) return;
  const t = e.target;
  if (t && /INPUT|TEXTAREA|SELECT/.test(t.tagName)) return;
  e.preventDefault();
  if (mediaRecorder?.state === "recording") return;
  startRecording();
});
document.addEventListener("keyup", (e) => {
  if (e.code !== "Space") return;
  const t = e.target;
  if (t && /INPUT|TEXTAREA|SELECT/.test(t.tagName)) return;
  if (mediaRecorder?.state === "recording") stopRecording();
});

// ---------------------------------------------------------------------------
// Text correction
// ---------------------------------------------------------------------------
$("btn-correct").addEventListener("click", async () => {
  const text = $("raw-input").value.trim();
  if (!text) return;
  try {
    const data = await postJson("/api/correct", { text });
    showResult(data);
  } catch (e) {
    alert(e.message);
  }
});

$("btn-example").addEventListener("click", () => {
  $("raw-input").value =
    "Patient with myokardial infarction. Start metoprol and atorvasta and clopidogr.";
});

// ---------------------------------------------------------------------------
// File upload
// ---------------------------------------------------------------------------
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
  form.append("model_size", $("model").value);
  try {
    await streamTranscribe(form);
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
  $("save-status").textContent = "Saving...";
  try {
    const data = await postJson("/api/learn_from_edit", {
      raw_text: lastRawText,
      corrected_text: corrected,
      session_id: lastSessionId,
      type: "drug",
    });
    const learnedTerms = (data.learned_text || []).map((l) =>
      l.from_alias ? `${l.entry.term} (was: ${l.from_alias})` : l.entry.term
    );
    const learnedVoices = (data.learned_voices || []).map((v) => v.voice.term);
    const parts = [];
    if (learnedTerms.length) parts.push(`text: ${learnedTerms.join(", ")}`);
    if (learnedVoices.length) parts.push(`voice: ${learnedVoices.join(", ")}`);
    if (!parts.length) {
      $("save-status").textContent = "✓ No new terms — your edits already match the database.";
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
  } catch (_) {
    /* ignore */
  }
}

function renderLexicon(query) {
  const q = (query || "").trim().toLowerCase();
  const filtered = q
    ? allEntries.filter(
        (e) =>
          e.term.toLowerCase().includes(q) ||
          (e.aliases || []).some((a) => a.toLowerCase().includes(q))
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
