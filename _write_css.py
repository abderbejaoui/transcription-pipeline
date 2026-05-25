"""Write the pipeline UI CSS to app/static/styles.css."""
import pathlib

content = r"""/* Pipeline UI Dark theme */

:root {
  --bg: #0b0d12;
  --surface: #13161e;
  --surface-hover: #1a1e2b;
  --border: #222738;
  --border-light: #2a2f3f;
  --text: #e8eaed;
  --text-muted: #7c8299;
  --text-dim: #4a4f66;
  --accent: #5bd1ff;
  --accent-strong: #2bb3e6;
  --accent-dim: rgba(91, 209, 255, 0.12);
  --green: #34d399;
  --green-bg: rgba(52, 211, 153, 0.12);
  --green-border: rgba(52, 211, 153, 0.25);
  --amber: #fbbf24;
  --amber-bg: rgba(251, 191, 36, 0.12);
  --amber-border: rgba(251, 191, 36, 0.25);
  --red: #f87171;
  --red-bg: rgba(248, 113, 113, 0.12);
  --red-border: rgba(248, 113, 113, 0.25);
  --blue: #60a5fa;
  --blue-bg: rgba(96, 165, 250, 0.10);
  --blue-border: rgba(96, 165, 250, 0.20);
  --radius: 10px;
  --radius-sm: 6px;
  --radius-xs: 4px;
  --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif;
  --font-mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}

* { box-sizing: border-box; }

html, body {
  margin: 0;
  padding: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.6 var(--font-sans);
  -webkit-font-smoothing: antialiased;
}

.app {
  max-width: 900px;
  margin: 0 auto;
  padding: 28px 20px 80px;
  display: flex;
  flex-direction: column;
  gap: 0;
}

.app-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 20px;
}

.app-header h1 {
  font-size: 22px;
  font-weight: 600;
  margin: 0 0 2px;
  letter-spacing: -0.01em;
}

.pipeline-badge {
  display: inline-block;
  vertical-align: middle;
  margin-left: 8px;
  padding: 1px 8px;
  font-size: 11px;
  font-weight: 600;
  border-radius: 20px;
  background: var(--accent-dim);
  color: var(--accent);
  border: 1px solid var(--accent);
  line-height: 1.6;
}

.subtitle {
  margin: 0;
  font-size: 13px;
  color: var(--text-muted);
}

.flex-row { display: flex; align-items: center; }
.gap-sm { gap: 8px; }
.text-xs { font-size: 12px; }
.text-sm { font-size: 13px; }
.text-muted { color: var(--text-muted); }
.text-dim { color: var(--text-dim); }
.mb-1 { margin-bottom: 6px; }

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  display: flex;
  flex-direction: column;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
}

.card-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
}

.card-badge {
  font-size: 11px;
  color: var(--text-muted);
  background: rgba(255,255,255,0.04);
  border-radius: 12px;
  padding: 1px 8px;
  font-weight: 500;
}

.card-body {
  padding: 14px 16px;
  min-height: 20px;
}

.input-card .card-body { padding: 0; }

.input-card textarea {
  width: 100%;
  min-height: 100px;
  background: transparent;
  border: none;
  border-bottom: 1px solid var(--border);
  color: var(--text);
  font: 14px/1.6 var(--font-sans);
  padding: 14px 16px;
  resize: vertical;
  outline: none;
}

.input-card textarea:focus {
  background: rgba(91, 209, 255, 0.02);
}

.input-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 16px;
}

.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: var(--radius-sm);
  padding: 7px 13px;
  font: inherit;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
  user-select: none;
  transition: all 0.15s;
}

.btn:hover:not(:disabled) {
  border-color: var(--accent);
  color: var(--accent);
}

.btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.btn-sm { padding: 4px 10px; font-size: 12px; }

.btn-primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #00131c;
}

.btn-primary:hover:not(:disabled) {
  background: var(--accent-strong);
  border-color: var(--accent-strong);
  color: #fff;
}

.btn-green {
  border-color: var(--green-border);
  color: var(--green);
}

.btn-green:hover:not(:disabled) {
  background: var(--green-bg);
  border-color: var(--green);
}

.btn-blue {
  border-color: var(--blue-border);
  color: var(--blue);
}

.btn-blue:hover:not(:disabled) {
  background: var(--blue-bg);
  border-color: var(--blue);
}

.stage-card {
  margin: 0;
  transition: border-color 0.3s, box-shadow 0.3s;
}

.step-number {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  font-size: 11px;
  font-weight: 700;
  background: var(--border);
  color: var(--text-dim);
  flex-shrink: 0;
  transition: all 0.3s;
}

.step-number.is-active {
  background: var(--accent);
  color: #00131c;
  box-shadow: 0 0 0 3px var(--accent-dim);
}

.step-number.is-done {
  background: var(--green);
  color: #00131c;
  font-size: 0;
}

.step-number.is-done::after {
  content: "\2713";
  font-size: 12px;
}

.step-number.is-error {
  background: var(--red);
  color: #fff;
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
  transition: all 0.3s;
}

.status-dot.pending { background: var(--text-dim); }

.status-dot.running {
  background: var(--accent);
  box-shadow: 0 0 5px var(--accent);
  animation: dot-pulse 1s ease-in-out infinite;
}

.status-dot.done { background: var(--green); }
.status-dot.error { background: var(--red); }

@keyframes dot-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.stage-connector {
  display: flex;
  justify-content: center;
  height: 24px;
  position: relative;
}

.stage-connector .line {
  width: 2px;
  height: 100%;
  background: var(--border);
  transition: background 0.3s;
}

.stage-connector .line.active {
  background: var(--accent-dim);
}

.stage-connector .line.done {
  background: var(--green);
}

.empty-state {
  font-size: 13px;
  color: var(--text-dim);
  text-align: center;
  padding: 24px 0;
  font-style: italic;
}

.running-overlay {
  display: none;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  background: var(--surface);
  border: 1px solid var(--accent);
  border-radius: var(--radius);
  margin-bottom: 16px;
  font-size: 13px;
  color: var(--accent);
  animation: overlay-fade 0.2s ease-out;
}

.running-overlay.active { display: flex; }

@keyframes overlay-fade {
  from { opacity: 0; transform: translateY(-8px); }
  to { opacity: 1; transform: translateY(0); }
}

.spinner {
  display: inline-block;
  width: 14px;
  height: 14px;
  border: 2px solid var(--accent-dim);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.word-token {
  position: relative;
  display: inline;
  padding: 1px 2px;
  border-radius: 3px;
  cursor: pointer;
  transition: background 0.15s;
  white-space: pre-wrap;
}

.word-token:hover { filter: brightness(1.15); }

.word-token.is-highlighted {
  outline: 2px solid var(--accent);
  outline-offset: 1px;
  border-radius: 3px;
  z-index: 2;
}

.word-token.is-flagged {
  text-decoration: underline wavy var(--amber);
  text-underline-offset: 2px;
}

.word-token.score-0  { background: rgba(52, 211, 153, 0.08); }
.word-token.score-1  { background: rgba(110, 231, 183, 0.08); }
.word-token.score-2  { background: rgba(167, 243, 208, 0.08); }
.word-token.score-3  { background: rgba(252, 211, 77, 0.06); }
.word-token.score-4  { background: rgba(251, 191, 36, 0.10); }
.word-token.score-5  { background: rgba(251, 191, 36, 0.18); }
.word-token.score-6  { background: rgba(251, 146, 60, 0.18); }
.word-token.score-7  { background: rgba(251, 146, 60, 0.28); }
.word-token.score-8  { background: rgba(248, 113, 113, 0.20); }
.word-token.score-9  { background: rgba(248, 113, 113, 0.32); }
.word-token.score-10 { background: rgba(239, 68, 68, 0.45); }

.word-token.in-lexicon {
  box-shadow: inset 0 0 0 1px var(--accent-dim);
}

.word-token .tooltip {
  display: none;
  position: absolute;
  bottom: calc(100% + 4px);
  left: 50%;
  transform: translateX(-50%);
  background: #1a1e2b;
  border: 1px solid var(--border-light);
  padding: 3px 8px;
  border-radius: 4px;
  font-size: 11px;
  color: var(--text-muted);
  white-space: nowrap;
  z-index: 10;
  pointer-events: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

.word-token:hover .tooltip { display: block; }

.token-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  margin-top: 10px;
  padding: 8px 10px;
  background: rgba(255,255,255,0.02);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}

.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--text-muted);
}

.legend-swatch {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
}

.spans-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.span-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background 0.15s;
}

.span-item:hover {
  background: var(--surface-hover);
  border-color: var(--border-light);
}

.span-text {
  font-weight: 600;
  font-size: 13px;
  min-width: 100px;
}

.span-score {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--red);
}

.span-badge {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 1px 6px;
  border-radius: 10px;
  background: rgba(255,255,255,0.05);
  color: var(--text-muted);
}

.span-badge.semantic { background: var(--blue-bg); color: var(--blue); }
.span-badge.phonetic { background: var(--amber-bg); color: var(--amber); }
.span-badge.both { background: var(--red-bg); color: var(--red); }

.span-candidates-count {
  margin-left: auto;
  font-size: 11px;
  color: var(--text-dim);
}

.span-candidate-group {
  margin-bottom: 12px;
}

.span-candidate-group:last-child { margin-bottom: 0; }

.group-header {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text-muted);
  padding: 0 0 6px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 6px;
}

.span-text-label {
  font-weight: 600;
  color: var(--text);
  font-size: 13px;
}

.candidate-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 8px;
  border-radius: var(--radius-xs);
  transition: background 0.1s;
  font-size: 12.5px;
}

.candidate-row:hover {
  background: rgba(255,255,255,0.02);
}

.candidate-term {
  font-weight: 600;
  min-width: 110px;
  color: var(--accent);
}

.candidate-type {
  font-size: 11px;
  color: var(--text-dim);
  min-width: 50px;
}

.candidate-source {
  font-size: 11px;
  color: var(--text-dim);
  min-width: 50px;
}

.score-bar-wrap {
  display: flex;
  align-items: center;
  gap: 5px;
  flex: 1;
  max-width: 200px;
}

.score-bar-bg {
  flex: 1;
  height: 6px;
  background: rgba(255,255,255,0.06);
  border-radius: 4px;
  overflow: hidden;
}

.score-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s ease-out;
  background: var(--amber);
}

.score-bar-fill.good { background: var(--green); }
.score-bar-fill.ok   { background: var(--amber); }
.score-bar-fill.low  { background: var(--red); }

.candidate-score {
  font-family: var(--font-mono);
  font-size: 11px;
  min-width: 32px;
  text-align: right;
  color: var(--text-muted);
}

.candidate-desc {
  color: var(--text-dim);
  font-size: 11px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 120px;
}

.teach-btn { flex-shrink: 0; }

.decisions-grid {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.decision-card {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  transition: all 0.15s;
  cursor: pointer;
}

.decision-card:hover {
  border-color: var(--border-light);
  background: var(--surface-hover);
}

.decision-card.path-auto {
  border-left: 3px solid var(--green);
}

.decision-card.path-llm {
  border-left: 3px solid var(--amber);
}

.decision-card.path-hitl {
  border-left: 3px solid var(--red);
}

.dec-span-text {
  font-weight: 600;
  font-size: 13px;
  min-width: 110px;
  text-decoration: line-through;
  color: var(--text-muted);
}

.dec-chosen {
  font-size: 13px;
  flex: 1;
}

.dec-path-tag {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 1px 6px;
  border-radius: 10px;
  background: rgba(255,255,255,0.05);
  color: var(--text-muted);
}

.output-diff {
  display: flex;
  gap: 12px;
}

.output-pane {
  flex: 1;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.pane-header {
  background: rgba(255,255,255,0.03);
  padding: 5px 10px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
}

.pane-body {
  padding: 10px;
  font-size: 13px;
  line-height: 1.7;
  word-wrap: break-word;
}

.diff-eq { color: var(--text); }

.diff-del {
  color: var(--red);
  text-decoration: line-through;
  background: var(--red-bg);
  border-radius: 2px;
  padding: 0 1px;
}

.diff-add {
  color: var(--green);
  font-weight: 500;
  background: var(--green-bg);
  border-radius: 2px;
  padding: 0 1px;
}

.lexicon-panel {
  display: none;
  flex-direction: column;
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: 380px;
  max-height: 500px;
  z-index: 100;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}

.lexicon-search-wrap {
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
}

.lexicon-search {
  width: 100%;
  background: rgba(0,0,0,0.3);
  border: 1px solid var(--border);
  border-radius: var(--radius-xs);
  padding: 5px 8px;
  color: var(--text);
  font-size: 12px;
  outline: none;
}

.lexicon-search:focus {
  border-color: var(--accent);
}

#lexicon-list {
  overflow-y: auto;
  max-height: 360px;
  padding: 6px 0;
}

.lexicon-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  font-size: 12px;
  cursor: pointer;
}

.lexicon-item:hover {
  background: var(--surface-hover);
}

.lex-term {
  font-weight: 600;
  color: var(--accent);
  min-width: 100px;
}

.lex-type {
  font-size: 10px;
  color: var(--text-dim);
  min-width: 60px;
}

.lex-aliases {
  color: var(--text-muted);
  font-size: 11px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

@media (max-width: 600px) {
  .app-header { flex-direction: column; }
  .output-diff { flex-direction: column; }
  .candidate-row { flex-wrap: wrap; }
  .score-bar-wrap { max-width: none; }
  .lexicon-panel { width: calc(100vw - 40px); right: 10px; }
}
"""

p = pathlib.Path("app/static/styles.css")
p.write_text(content, encoding="utf-8")
print(f"Written {len(content)} chars to {p}")
