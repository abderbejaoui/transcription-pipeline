# API Key Test Results

## Tested on: 2026-05-25

| # | Provider | Env Variable | Result | Notes |
|---|----------|-------------|--------|-------|
| 1 | **GitHub** | `GITHUB_API_KEY` | **WORKS** | HTTP 200, user "Abbes-Younes" |
| 2 | **Groq** | `GROQ_API_KEY` | **WORKS** | HTTP 200, model `llama-3.3-70b-versatile` |
| 3 | **NVIDIA** | `NVIDIA_API_KEY` | **WORKS** | HTTP 200, model `meta/llama-3.1-8b-instruct` |
| 4 | **OpenRouter** | `OPENROUTER_API_KEY` | **WORKS** | HTTP 200, model `openrouter/owl-alpha` |
| 5 | **Google AI Studio** | `GOOGLE_AI_STUDIO_API_KEY` | **FAILED** | HTTP 429 (rate limited) on gemini-2.0-flash, HTTP 404 on gemini-1.5 models |
| 6 | **Ollama** | `OLLAMA_API_KEY` | **FAILED** | HTTP 403, model `glm-5.1:cloud` requires paid subscription |
| 7 | **Cloudflare** | `CLOUDFLARE_API_KEY` | **FAILED** | Missing `CLOUDFLARE_ACCOUNT_ID` - need your Account ID from dash.cloudflare.com |

---

## Detailed Results

### 1. GitHub — WORKS
- Endpoint: `GET https://api.github.com/user`
- Status: 200 OK
- Account: Abbes-Younes (Younes Abbes)
- **Can be used for:** Repository access, code search, automation

### 2. Groq — WORKS
- Endpoint: `POST https://api.groq.com/openai/v1/chat/completions`
- Status: 200 OK with `llama-3.3-70b-versatile`
- Format: OpenAI-compatible
- **Can be used for:** Fast LLM inference (could replace Gemini in the pipeline!)

### 3. NVIDIA — WORKS
- Endpoint: `POST https://integrate.api.nvidia.com/v1/chat/completions`
- Status: 200 OK with `meta/llama-3.1-8b-instruct`
- Format: OpenAI-compatible
- **Can be used for:** LLM inference via NVIDIA's hosted API

### 4. OpenRouter — WORKS
- Endpoint: `POST https://openrouter.ai/api/v1/chat/completions`
- Status: 200 OK with `openrouter/owl-alpha`
- Format: OpenAI-compatible
- **Already integrated in the pipeline** (currently used for Stage 4 decisions)

### 5. Google AI Studio — FAILED
- **Problem 1:** The env variable is `GOOGLE_AI_STUDIO_API_KEY` but our pipeline code looks for `GEMINI_API_KEY` — name mismatch!
- **Problem 2:** `gemini-2.0-flash` returned HTTP 429 (rate limited) — key may be valid but hitting daily quota
- **Problem 3:** `gemini-1.5-flash` and `gemini-1.5-pro` returned HTTP 404 — these models may have been deprecated
- **Fix:** Try using `gemini-2.0-flash-lite` or check usage at aistudio.google.com

### 6. Ollama Cloud — FAILED
- HTTP 403: Model `glm-5.1:cloud` requires a paid subscription (upgrade at ollama.com/upgrade)
- The API key itself may be valid but the model needs a higher-tier plan

### 7. Cloudflare Workers AI — FAILED
- Missing `CLOUDFLARE_ACCOUNT_ID` in .env
- The API key was detected but the test couldn't discover the Account ID (token may lack account list permissions)
- **Fix:** Add `CLOUDFLARE_ACCOUNT_ID=<your-account-id>` to .env (find it in the Cloudflare dashboard sidebar)

---

## Which Keys Fit Best in the Pipeline

| Use Case | Best Key | Why |
|----------|---------|-----|
| **Stage 4 Decision** (current) | OpenRouter | Already works, gpt-4o-mini is reliable |
| **Stage 4 Decision** (alternative) | Groq | Fast inference, supports Llama-3.3-70B, OpenAI-compatible |
| **Stage 4 Decision** (alternative) | NVIDIA | Works with Llama 3.1 8B, OpenAI-compatible |
| **Stage 1 Scoring** (if needed) | Groq or NVIDIA | Both could replace the local LLM with an API call |
