// ---------------------------------------------------------------------------
// ASR Benchmark UI Logic
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const escapeHtml = (s) =>
  String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

$("btn-upload").onclick = async () => {
    const fileInput = $("upload-file");
    if (!fileInput.files.length) {
      alert("Please select an audio file first.");
      return;
    }
    
    const sessionId = generateUUID();
    $("btn-upload").disabled = true;
    $("benchmark-status").innerHTML = `<span class="pulse"></span> <strong>Starting benchmark...</strong> (Session: ${sessionId.slice(0, 8)})`;
    $("results-card").hidden = true;
    $("results-container").innerHTML = "";
    
    // Start polling for progress
    const progressInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/benchmark_progress/${sessionId}`);
            if (res.ok) {
                const data = await res.json();
                if (data.status && data.status !== "unknown") {
                    let text = `<span class="pulse"></span> <strong>Status:</strong> ${data.status}`;
                    if (data.total > 0) {
                        text += ` <br/><small>Progress: ${data.completed} / ${data.total} models completed.</small>`;
                    }
                    $("benchmark-status").innerHTML = text;
                }
            }
        } catch (e) {
            console.error("Progress poll failed:", e);
        }
    }, 1000);
    
    const form = new FormData();
    form.append("file", fileInput.files[0], fileInput.files[0].name);
    form.append("client_session_id", sessionId);
    // form.append("models", "faster-whisper-large-v3,Qwen3-ASR-1.7B"); // Optional specific test
    
    try {
        const r = await fetch("/api/benchmark_asr", { method: "POST", body: form });
        if (!r.ok) {
            throw new Error(`HTTP ${r.status} - ${(await r.text())}`);
        }
        
        const payload = await r.json();
        
        clearInterval(progressInterval);
        $("benchmark-status").innerHTML = `<strong>✅ Benchmark complete!</strong>`;
        $("results-card").hidden = false;
        $("audio-duration-display").textContent = `(Audio Length: ${payload.audio_duration_s}s)`;
        
        renderResults(payload.results);
        
    } catch (err) {
        clearInterval(progressInterval);
        $("benchmark-status").innerHTML = "❌ " + escapeHtml(err.message);
    } finally {
        $("btn-upload").disabled = false;
    }
};

function renderResults(results) {
    const container = $("results-container");
    container.innerHTML = "";
    
    results.forEach(res => {
        const resultCard = document.createElement("div");
        resultCard.className = "model-result";
        resultCard.style.padding = "10px";
        resultCard.style.border = "1px solid var(--border)";
        resultCard.style.borderRadius = "var(--radius)";
        resultCard.style.margin = "10px 0";

        const title = document.createElement("h3");
        title.style.marginTop = "0";
        title.textContent = res.model_key;
        
        const info = document.createElement("div");
        info.className = "muted small";
        if (res.error) {
             info.innerHTML = `<strong>Error:</strong> <span style="color: var(--danger)">${escapeHtml(res.error)}</span>`;
        } else {
             info.innerHTML = `<strong>Lang:</strong> ${escapeHtml(res.language)} | <strong>Time:</strong> ${res.duration_s}s`;
        }
        
        const transcript = document.createElement("pre");
        transcript.style.whiteSpace = "pre-wrap";
        transcript.style.fontFamily = "var(--font-sans)";
        transcript.style.marginTop = "10px";
        transcript.textContent = res.transcript || "(No transcript output)";
        if (res.error) transcript.style.color = "var(--danger)";

        resultCard.appendChild(title);
        resultCard.appendChild(info);
        resultCard.appendChild(transcript);
        container.appendChild(resultCard);
    });
}
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
