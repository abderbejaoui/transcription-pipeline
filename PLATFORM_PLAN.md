# Self-Improving Voice Platform — Plan

> A self-hosted system that listens to people speak, writes down what they
> said, fixes mistakes, and **learns from every correction** so the same
> mistake never happens again. Medical is the first use case, but the same
> engine works for legal, finance, or any other field — by swapping the
> vocabulary database.

---

## 1. Glossary (Read This First)

Plain-language definitions for every short word used later.

| Word | Meaning |
|------|---------|
| **Vocabulary database** (also written *KB*, knowledge base) | The list of all real words the system knows: drug names, disease names, brand names, plus how they sound and how they are usually misspelled by the speech-to-text. This is the most important part of the platform. |
| **Speech-to-text** (also *ASR*) | The model that listens to audio and writes the words. Example: Whisper. |
| **AI text model** (also *LLM*) | A large language model used to pick the right correction. Example: Qwen2.5, Calme. |
| **Human review** (also *HITL*, human-in-the-loop) | A real person confirms or fixes a correction. This feedback is what the system learns from. |
| **On-prem** | Everything runs on your own computers. No data leaves the building. |
| **Forced alignment** | A technique that lines up each spoken sound with the exact moment in the audio where it was said. |
| **Phonetic / IPA** | The way a word sounds, written in a special phonetic alphabet. Useful for matching words that sound alike but are spelled differently. |
| **Vector database** | A special database that stores items by their meaning or sound, not just by exact text. Example: Qdrant. |
| **Embedding** | A list of numbers that represents the meaning or sound of a word, sentence, or audio clip. Two similar items have similar embeddings. |
| **MVP** | First working version. Small but already useful. |
| **Tier / Scope** | Which level of the vocabulary the entry belongs to: just one user, one organization (e.g. one hospital), or shared globally across all users. |
| **Code-switching** | When a speaker mixes two languages in the same sentence (very common in Gulf Arabic medical conversations). |
| **OOV (out-of-vocabulary)** | A word the system has never seen before. |
| **N-best** | The top several alternative guesses the speech-to-text gave for the same audio. |
| **VAD (voice activity detection)** | Detects which parts of the audio actually contain speech. |
| **Diarization** | Figuring out *who* spoke *when*, when there are multiple speakers. |
| **LoRA / adapter fine-tune** | A cheap and reversible way to teach a model new things without retraining it from scratch. |

---

## 2. One-Line Summary

Listen → write → spot suspicious words → look them up in our growing vocabulary database → let an AI text model pick the right correction → ask a human if uncertain → save the confirmed answer back into the vocabulary database forever.

---

## 3. Locked Decisions

Based on your answers to clarifying questions:

| Topic | Choice |
|-------|--------|
| Speed | Accuracy matters most. The system does not need to be real-time. |
| Languages | Gulf Arabic and English. Mixing them in one sentence is allowed. |
| Vocabulary scoping | Three levels: per user, per organization, and global (shared). |
| Where it runs | Everything on the customer's own servers. Nothing leaves. |
| Human review | Two ways: quick inline accept/reject, plus a review queue for unclear cases. |
| Timeline | A small first version in about 4 weeks; the full system in about 3 months. |

---

## 4. Why This Architecture (Plain English)

1. **The vocabulary database is the moat, not the model.** Anyone can run Whisper. But only a system that captures *how this hospital actually pronounces "Doliprane"* keeps getting better with use. That is what makes it valuable and hard to copy.
2. **Keep the corrector separate from the speech-to-text.** Speech models keep changing. If we mix them, we have to rebuild. Keeping them separate lets us swap one without breaking the other.
3. **Never let the AI text model invent corrections on its own.** Your earlier tests proved this fails: the model made up changes on already-correct text. Instead, we give it a short list of candidate words and only let it pick one (or say "no change").
4. **Use the actual audio as a tie-breaker, not the main signal.** It is more reliable to compare the real audio to a candidate word's pronunciation than to generate fake audio with text-to-speech.
5. **Human review must be two-tier.** Quick inline confirmations cover ~95% of cases. A small review queue handles the hard 5% where most of the learning value lives.
6. **Vocabulary must be hierarchical.** A new doctor joining a hospital should immediately know the hospital's words. But their own personal corrections should not leak into the global database without curator approval.

---

## 5. Final Technical Choices

### 5.1 Models

| Component | Choice | Why |
|-----------|--------|-----|
| Speech-to-text (main) | Whisper-large-v3-turbo via faster-whisper | Supports Arabic, English, and 99 other languages. Returns word-level timestamps and confidence scores. |
| Backup speech-to-text for Arabic | MMS-1B, then later a fine-tuned Whisper for Gulf Arabic | Better dialect coverage if needed. |
| Forced alignment | ctc-forced-aligner with per-language wav2vec2 | English: facebook 960h-self. Arabic: jonatasgrosman xlsr-53-arabic. |
| Universal phonetic recognition | Allosaurus | Covers 2000+ languages including Gulf Arabic dialects. |
| Phonetic spelling | espeak-ng | Generates IPA for English and Arabic. |
| Voice activity detection | Silero-VAD | Reduces hallucination on silence. |
| Speaker identification | pyannote-audio (community model) | Tells us *who* taught a new word. |
| Text embeddings | BGE-M3 multilingual (1024-d) | Best open-source embedding for Arabic + English. |
| Audio embeddings | Wav2Vec2-XLS-R-300m | Cross-lingual; current pipeline can also produce these. |
| AI text model (corrector) | Qwen2.5-72B-Instruct or your existing Calme-3.2-78B | Forced to return strict JSON via Outlines. |

### 5.2 Infrastructure

| Layer | Choice | Why |
|-------|--------|-----|
| Vocabulary database | Qdrant | Apache-2.0 license. Stores meaning-based and keyword-based search together. Fast. |
| Keyword search | BM25 / SPLADE inside Qdrant | Co-located with meaning search. |
| Human review interface | Argilla (Apache-2.0, runs in Docker) | Built for AI/NLP labeling tasks. |
| Job orchestrator | FastAPI + Celery + Redis | Async jobs, retries, queues. |
| Metadata store | Postgres | Audit log, permissions, history. |
| File store | MinIO (S3-compatible, on-prem) | Stores audio clips of confirmed words. |
| Training | NeMo adapters (speech), PEFT/LoRA (text model) | Cheap, reversible, can be rolled back. |
| Monitoring | OpenTelemetry + Prometheus + Loki | All open-source, all on-prem. |

### 5.3 Phonetic / Edit-Distance

| Need | Choice |
|------|--------|
| English sound-alike matching | jellyfish (already in repo). |
| Arabic phonetic distance | Custom IPA edit distance using espeak-ng output. |
| Cross-lingual phonetic | Allosaurus IPA strings + edit distance. |

---

## 6. The 9 Services

> Items marked **MVP** are part of the first 4-week version. The rest come in phases B and C.

```
                ┌──────────────────────────────────────────────────────────────┐
                │              INPUT  (audio + speaker info)                   │
                └────┬─────────────────────────────────────────────────────────┘
                     │
       ┌─────────────▼──────────────┐    ┌────────────────────────────────────┐
       │ 1. Speech-to-text [MVP]    │    │ 2. Audio cleaner [MVP]             │
       │    Whisper-large-v3-turbo  │    │    Silero VAD, denoise, diarize    │
       │    (faster-whisper)        │◄──►│    Output: clean segments + who   │
       │    + alternatives          │    │              spoke each one        │
       │    + word timestamps       │    └────────────────────────────────────┘
       │    + word confidence       │
       └─────────────┬──────────────┘
                     │
       ┌─────────────▼──────────────┐
       │ 3. Mistake spotter [MVP]   │   triggers when:
       │   - low confidence word    │   • word_confidence is too low
       │   - word not in vocabulary │   • word is unknown to the vocab DB
       │   - rare suffix/prefix     │   • span looks like a split medical
       │   - alternatives disagree  │     term joined by "and"
       └─────────────┬──────────────┘
                     │  list of suspicious spans
                     │
       ┌─────────────▼──────────────────────────────────────────────────────┐
       │ 4. Vocabulary DB lookup [MVP]                                      │
       │                                                                    │
       │   Qdrant collections per scope:                                    │
       │     global_terms (curated, shared by everyone)                     │
       │     org_{org_id}_terms (one hospital, one law firm, etc.)          │
       │     user_{user_id}_terms (one specific person)                     │
       │                                                                    │
       │   Each entry stores 4 different signals:                           │
       │     • text-meaning vector (1024 numbers)                           │
       │     • keyword vector (BM25/SPLADE)                                 │
       │     • how it sounds (IPA + sound bigrams)                          │
       │     • real-audio fingerprint (1024 numbers, average of recordings) │
       │                                                                    │
       │   Search combines: spelling + sound + audio match + alias          │
       │   Returns: best candidates (user > org > global, all weighted)     │
       └─────────────┬──────────────────────────────────────────────────────┘
                     │  list of candidate corrections
                     │
       ┌─────────────▼──────────────────┐    ┌──────────────────────────────┐
       │ 5. Audio matcher (week 6)      │    │ 6. AI text picker [MVP]      │
       │  - ctc-forced-aligner          │◄──►│  Qwen2.5-72B / Calme-78B     │
       │  - per-language wav2vec2       │    │  via vLLM + Outlines JSON    │
       │  - Allosaurus phonemes         │    │  Input: span + sentence      │
       │  - score(audio vs candidate)   │    │         + candidates+scores  │
       │  - boosts/lowers candidates    │    │  Output: choice or NO_CHANGE │
       └─────────────┬──────────────────┘    └──────────────┬───────────────┘
                     │                                       │
                     └───────────────┬───────────────────────┘
                                     │
                     ┌───────────────▼────────────────┐
                     │ 7. Confidence sorter [MVP]     │
                     │   AUTO-FIX  (>= 0.92 + KB hit) │
                     │   SUGGEST   (0.70 - 0.92)      │
                     │   ASK HUMAN (< 0.70 or new)    │
                     └───────────────┬────────────────┘
                                     │
       ┌─────────────────────────────▼──────────────────────────────────────┐
       │ 8. Human review service [MVP]                                      │
       │                                                                    │
       │  Inline UI (FastAPI + simple web):                                 │
       │   - shows suggestions in client app                                │
       │   - accept / reject / edit -> emits CorrectionEvent                │
       │                                                                    │
       │  Argilla review queue:                                             │
       │   - ASK-HUMAN cases land here                                      │
       │   - reviewer can promote span to org or global vocab DB            │
       │   - supports new-term capture ("teach the system")                 │
       └─────────────┬──────────────────────────────────────────────────────┘
                     │  CorrectionEvent (signed, attributed)
                     │
       ┌─────────────▼──────────────────────────────────────────────────────┐
       │ 9. Learning loop (Tier 1 in MVP; Tiers 2 and 3 later)              │
       │                                                                    │
       │  Tier 1 (instant, MVP):                                            │
       │    save into vocab DB:                                             │
       │      term, aliases, IPA, audio examples,                           │
       │      domain, scope (user|org|global), confidence, source           │
       │    + add to speech-to-text "boost list" for that scope             │
       │                                                                    │
       │  Tier 2 (nightly, week 8):                                         │
       │    LoRA fine-tune of AI text picker on confirmed corrections       │
       │    PEFT on Qwen2.5 with 4-bit quant; promote if eval gain >= 1%    │
       │                                                                    │
       │  Tier 3 (weekly, week 12):                                         │
       │    NeMo / HF adapter fine-tune of speech-to-text on (audio, gold)  │
       │    A/B versus frozen model; promote only if regression test passes │
       │                                                                    │
       │  Promotion rules:                                                  │
       │    user vocab:    1 confirm  -> save to that user's scope          │
       │    org vocab:     3 confirms across 2 different users + reviewer   │
       │    global vocab:  curator promotion only (never automatic)         │
       └─────────────────────────────────────────────────────────────────────┘
```

### 6.1 Service-by-service summary

| # | Service | In MVP? | What it does |
|---|---------|---------|--------------|
| 1 | Speech-to-text | yes | Whisper-large-v3-turbo. Returns alternatives, word timestamps, word confidence. |
| 2 | Audio cleaner | yes | Removes silence, denoises, identifies speakers. |
| 3 | Mistake spotter | yes | Flags low-confidence words, unknown words, suspicious spans. |
| 4 | Vocabulary DB lookup | yes | Looks up candidate corrections across user / org / global scopes. |
| 5 | Audio matcher | week 6 | Uses real audio to confirm or reject candidates. |
| 6 | AI text picker | yes | Forced JSON output. Picks one candidate or "no change". |
| 7 | Confidence sorter | yes | Routes to auto-fix, suggest, or human-review. |
| 8 | Human review | yes | Inline UI + Argilla queue. Saves confirmed corrections. |
| 9 | Learning loop | Tier 1 | Updates vocab DB; nightly model tuning later. |

---

## 7. Vocabulary Database Entry — What It Looks Like

> This is the heart of the system. Every confirmed correction lives here.

```text
┌──────────────────── Vocabulary Entry (one Qdrant point) ───────────────┐
│ id: uuid                                                                │
│ scope: global | org:{org_id} | user:{user_id}                           │
│ domain: medical | legal | finance | custom_{x}                          │
│ language: ar | en | code-switch                                         │
│ term: "Doliprane"                                                       │
│ canonical_form: "doliprane"                                             │
│ term_type: drug_brand | disease | symptom | procedure | ...             │
│ maps_to: "acetaminophen"   <- cross-link to generic name                │
│ aliases: ["doliprane", "دوليبران"]                                      │
│ wrong_forms: ["dolly brain", "dole preen", "دولي بران"]                 │
│ ipa: "/ˌdɒlɪˈpreɪn/"     <- phonetic spelling                           │
│ audio_examples: [s3://.../doliprane_001.wav, ...]                       │
│ vectors:                                                                │
│   text_meaning   (1024 numbers)                                         │
│   keyword        (BM25 / SPLADE)                                        │
│   sound          (IPA string + bigrams)                                 │
│   audio_average  (1024 numbers, average of recordings)                  │
│ provenance:                                                             │
│   source: human_confirmed | imported_lexicon | curator                  │
│   confirmations: [{user_id, ts, audio_ref}]                             │
│   confidence: 0.97                                                      │
│ created_at, updated_at, version                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

| Field | Why we keep it |
|-------|----------------|
| `scope` | Drives privacy and search ordering. |
| `aliases` / `wrong_forms` | Direct match against past speech-to-text mistakes. |
| `ipa` | Sound-alike matching. |
| `audio_examples` + `audio_average` | Real-audio confirmation. |
| `provenance.confirmations` | Decides when to promote from user → org → global. |
| `version` | Roll back if a promotion turns out wrong. |

---

## 8. Walkthrough: How "Doliprane" Gets Learned

```
1. Doctor says:   "Take Doliprane twice a day."
2. Speech-to-text writes:  "Take dolly brain twice a day."

3. Mistake spotter flags "dolly brain" (low confidence + unknown word).

4. Vocabulary DB lookup:
     - user vocab:  nothing
     - org vocab:   nothing
     - global vocab: weak match -> confidence 0.55
   Tier = ASK HUMAN.

5. Argilla shows reviewer:
     - audio segment
     - transcript
     - candidates suggested by the AI text picker
   Reviewer types: "Doliprane".

6. Learning loop fires:
     - extracts the audio segment
     - turns it into an audio fingerprint (Wav2Vec2-XLS-R)
     - turns "Doliprane" into IPA (espeak-ng)
     - saves a new vocabulary entry into org_{this_hospital}
     - adds "dolly brain" to wrong_forms
     - adds "Doliprane" to the speech-to-text boost list for that org

7. Next time "dolly brain" appears for ANY user in that hospital:
     -> vocab DB returns Doliprane with confidence > 0.95
     -> AUTO-FIX. No human needed.

8. After 3 confirmations across different users:
     -> term becomes a candidate for global promotion
     -> requires curator approval (never automatic).
```

---

## 9. Safety Rules (Anti-Poisoning)

| Rule | Why |
|------|-----|
| Vocabulary DB never learns from its own guess. Only from confirmed human action. | Stops the system from training on its own mistakes. |
| One-person corrections stay in user scope until other people confirm them. | Prevents cross-contamination. |
| Every correction is signed: user id, timestamp, audio hash. | Full audit trail; rollback possible. |
| Every promotion runs against a locked test set. Auto-revert if it fails. | Prevents silent quality drops. |
| Per-user "do not touch" list. If a user rejects a correction twice, the system stops suggesting it for them. | Respects user intent. |
| Outbound network monitor verifies zero egress on isolated runs. | On-prem guarantee. |

---

## 10. Staged Delivery

### 10.1 MVP — weeks 1-4 (already self-improving)

| Item | Notes |
|------|-------|
| Services 1, 2, 3, 4, 6, 7, 8 (inline only), 9 Tier 1 | English first, then Arabic. |
| Single-machine Qdrant + Postgres + MinIO | All on one GPU box. |
| AI text picker uses your existing Ollama endpoint | Add Outlines for strict JSON. |
| Test set | Reuse [eval_corrector.py](eval_corrector.py); expand to 200+ examples. |

### 10.2 Hardening — weeks 5-8

| Item |
|------|
| Service 5 (audio matcher): ctc-forced-aligner + Allosaurus. |
| Argilla review queue with reviewer roles + promotion API. |
| Arabic stack: Whisper Arabic prompts, Arabic phonemizer, Arabic test set. |
| Speaker identification (pyannote) so the system knows *who* taught each term. |
| Monitoring: per-tenant accuracy dashboards, vocab DB growth metrics. |

### 10.3 Full target — weeks 9-12

| Item |
|------|
| Tier 2: nightly LoRA fine-tune of AI text picker on confirmed corrections. |
| Tier 3: weekly NeMo adapter fine-tune of speech-to-text. |
| Multi-domain templates: legal, finance, etc. |
| Permissions, signed audit log, high-availability Qdrant. |
| Optional real-time streaming on the same backend. |

---

## 11. Implementation Steps

### 11.1 Phase A — MVP (weeks 1-4)

1. Stand up infra: Qdrant + Postgres + MinIO + Redis + FastAPI skeleton on a single GPU machine. *In parallel: Argilla in Docker.*
2. Wrap faster-whisper (large-v3-turbo) as the speech-to-text service. Returns alternatives, word timestamps, per-word confidence, language hint. Replaces the current text-to-speech round-trip on the hot path.
3. Migrate [data/medical_lexicon.jsonl](data/medical_lexicon.jsonl) into Qdrant with the 4-vector schema (text-meaning via BGE-M3, keyword via BM25, IPA via espeak-ng, audio fingerprint starts empty). Define hierarchical scopes.
4. Port [medical_corrector.py](medical_corrector.py) span generator and lookup logic to query Qdrant hierarchically (user > org > global) with score fusion. *Can run in parallel with step 3.*
5. Make the AI text picker use Outlines + JSON schema against your existing Ollama endpoint. Keep `NO_CHANGE` as a first-class output.
6. Implement confidence tiers and inline human-review endpoints (accept / reject / edit). Wire the CorrectionEvent producer.
7. Tier-1 learning loop: on accept, save the vocabulary entry into the correct scope, embed the audio segment as an example, append the term to the speech-to-text boost list.
8. Expand the test set to 200+ examples (real Whisper output on Gulf Arabic + English medical audio); rerun benchmark; lock as regression suite.

### 11.2 Phase B — Hardening (weeks 5-8)

9. Audio matcher service: ctc-forced-aligner per language + Allosaurus phonemes. Score each candidate against the real audio segment.
10. Argilla queue for ASK-HUMAN cases. Reviewer roles: user, org curator, global curator. Promotion API.
11. Arabic stack: Whisper-large-v3 with Arabic prompts, Arabic phonemizer settings, Arabic alignment model, Arabic test set.
12. Speaker identification (pyannote) so the system attributes new terms to the right person.
13. Monitoring: span-level traces, per-tenant accuracy dashboards, vocab DB growth and quality metrics.

### 11.3 Phase C — Full target (weeks 9-12)

14. Tier 2: nightly LoRA fine-tune of the AI text picker on confirmed corrections. A/B against frozen.
15. Tier 3: weekly NeMo adapter fine-tune of the speech-to-text on (audio, gold transcript) pairs. Regression-gated promotion.
16. Multi-domain templates: legal, finance starter vocabularies.
17. Permissions + audit log + signed corrections; high-availability Qdrant.
18. Optional streaming path reusing the same backend.

---

## 12. Files in the Repo and What Happens to Them

| Path | What changes |
|------|--------------|
| [pipeline.py](pipeline.py) | Keep `SoundEmbedder` for batch audio fingerprint creation. No longer on the hot path. |
| [medical_corrector.py](medical_corrector.py) | Refactor `_generate_spans`, `_retrieve_candidates`, `_already_valid` into a `correction_engine/` package. Replace in-memory lexicon with Qdrant client. |
| [audio_grounding.py](audio_grounding.py) | Switch to ctc-forced-aligner + Allosaurus instead of text-to-speech round-trip. |
| [data/medical_lexicon.jsonl](data/medical_lexicon.jsonl) | Becomes a *seed migration* into Qdrant `global_medical` collection. Not the runtime store anymore. |
| [eval_corrector.py](eval_corrector.py) | Generalize to per-tenant tests, regression tests, and promotion gating. |
| `services/asr/` (new) | faster-whisper wrapper. |
| `services/kb/` (new) | Qdrant client + scope-aware lookup. |
| `services/orchestrator/` (new) | FastAPI + Celery routing. |
| `services/hitl/` (new) | Inline endpoints + Argilla integration. |
| `services/learning/` (new) | Tier 1 / 2 / 3 jobs. |
| `infra/docker-compose.yml` (new) | Qdrant + Postgres + MinIO + Redis + Argilla + vLLM. |
| `schemas/correction_event.json` (new) | Signed event contract. |

---

## 13. Verification Gates

| # | Gate | Pass criteria |
|---|------|---------------|
| 1 | MVP correctness | All 22 current eval cases still 100%. New 200+ case Whisper-output benchmark: detection >= 85%, exact correction >= 75%, zero false positives on negatives. |
| 2 | Self-learning | "Doliprane scenario" smoke test: first occurrence enters ASK-HUMAN; after one accept the second occurrence is AUTO-FIX with confidence > 0.9. |
| 3 | Hierarchy | A term taught in `org_A` does NOT appear in `org_B`. A term taught for `user_X` only appears for `user_Y` in the same org after the configured number of confirmations. |
| 4 | Negatives | Regression set of 50 already-correct medical sentences -> 0 unwanted edits. |
| 5 | Arabic | Parallel benchmark of 50 Gulf Arabic medical utterances reaches the same thresholds as English. |
| 6 | Promotion rollback | A planted bad correction triggers regression failure -> auto-revert. |
| 7 | Privacy | Network egress monitor confirms zero outbound calls during an isolated run. |

---

## 14. Decisions Recorded

| Decision | Reason |
|----------|--------|
| Whisper-large-v3-turbo over Parakeet | Gulf Arabic needs more than the 25 European languages Parakeet covers. |
| Qdrant over LanceDB / Milvus | Combines meaning search and keyword search in one engine. Apache-2.0. Fastest in independent benchmarks. |
| AI text picker with strict JSON over free generation | Directly fixes the false-positive problem you observed in earlier tests. |
| Allosaurus + ctc-forced-aligner over text-to-speech round-trip | Real-audio matching is more reliable than synthesizing fake audio. |
| Argilla over Label Studio | Built specifically for AI/NLP labeling. Easier to deploy on-prem. |
| BGE-M3 over OpenAI embeddings | Best multilingual open-source model. Runs on-prem. |
| Hierarchical scoping (user / org / global) | Only structure that gives both privacy and fast cold-start for new users. |
| LoRA / adapter fine-tunes for both models | Cheap, reversible, can be rolled back if a new tune hurts quality. |

---

## 15. Business Lens (CEO View)

| Aspect | Position |
|--------|----------|
| Moat | Per-organization vocabulary DB grows non-portably with usage. Switching cost compounds every day. |
| Sales motion | Niche down to medical with 3 design partners. Pitch: "your hospital's vocabulary, not someone else's." |
| Pricing | Per active speaker per month + storage tier for vocab DB size. Margins improve as the DB grows. |
| Expansion | Same engine, new vertical = new global lexicon import + ~2 weeks of human-review bootstrapping. |
| Defensibility vs cloud speech APIs | Cloud APIs cannot match per-tenant vocabulary without violating privacy. On-prem is the wedge for healthcare and legal. |
| Risks | Bad data poisoning the DB; reviewer fatigue. Mitigated by promotion gates and active-learning sampling. |

---

## 16. Things to Think About Later

| # | Thing | Recommendation |
|---|-------|----------------|
| 1 | Code-switching between Arabic and English in the same sentence (very common in Gulf medical settings). | Use a single Whisper-large-v3-turbo with `language=None` and a code-switch-aware tokenizer for span generation. Alternative: run an Arabic and an English pass separately, then merge. Recommend single-pass first. |
| 2 | The review queue can grow large as the system scales. | Use active-learning sampling: only review the spans where models disagree the most. Alternative: review everything (safest, slowest). Recommend active-learning with a safety floor. |
| 3 | Whisper sometimes hallucinates on silence. | Already handled by Silero VAD plus `condition_on_prev_text=False`. Worth a dedicated test in the MVP gate. |

---

## 17. Known Limitations and Mitigations

> An honest list of where the design in sections 1-16 will fail or degrade,
> and a concrete mitigation for each. Read this before committing to the plan.

### 17.1 Coverage Gaps — Mistakes We Will Not Catch by Default

| # | Limitation | Why it matters | Mitigation |
|---|------------|----------------|------------|
| 1 | **Confident-but-wrong words.** The mistake spotter only fires on low confidence or unknown words. If Whisper writes the wrong *real* word with high confidence (e.g. "metformin" instead of "metoprolol"), nothing flags it. | These are the most dangerous errors because they look correct. | Add a second detection path: for every medical-looking token, run a *context plausibility check* against the org vocabulary. If the token is valid in general but rare/inconsistent in this clinical context (e.g. a diabetes drug appearing in a hypertension note), flag it for review even at high confidence. The AI text picker is used only as a binary plausibility check, never free generation. |
| 2 | **Multi-word medical concepts.** "Left ventricular ejection fraction" is one concept of four words; "non-Hodgkin lymphoma" spans a hyphen. Single-word and merge logic can chop these. | Partially correcting one of these breaks meaning. | Span generator must include up to 6-word spans **and** look up multi-word entries in the vocab DB *before* single-word correction. When both score above threshold, prefer the multi-word match. |
| 3 | **Whisper hallucination on noise.** Even with VAD, Whisper produces plausible fabricated text in low-SNR segments. | Hallucinated text passes the spotter because individual words look fine. | (a) Combine word-confidence with VAD energy: low VAD energy + high Whisper confidence in the same window = suspicious by construction. (b) Run a second decoding pass with a smaller model (Whisper-medium) and flag spans that disagree across passes. |
| 4 | **Code-switching reality.** Gulf Arabic medical conversations switch mid-sentence. Whisper's `language=None` works on long form but struggles on tight switches. | Sub-spans get transcribed in the wrong script and never line up with the right vocab entry. | Per-segment language detection (not just file-level), plus a fallback dual-pass mode (Arabic + English) that merges hypotheses where the primary pass has low confidence. Acceptance gate: a dedicated Arabic-English code-switch test set during MVP. |

### 17.2 Data Quality and Poisoning

| # | Limitation | Why it matters | Mitigation |
|---|------------|----------------|------------|
| 5 | **Cold start: empty vocabulary on day 1.** A new organization starts with only the global lexicon and zero audio examples. Accuracy is poor for the first weeks. | Customers churn before the system gets good. | (a) Seed every new tenant with curated medical lexicons (RxNorm, SNOMED-CT subsets, regional drug brand lists). (b) Provide an "import" feature for the customer's pharmacy formulary, internal procedure list, doctor roster. (c) One-time bootstrap session: customer reads a 5-minute calibration script that captures audio examples for the most common terms. |
| 6 | **Lazy or wrong user accepts.** A user clicks "accept" without reading, and a wrong correction enters their personal vocab. | Each lazy click teaches the system a wrong fact. | Require a *second signal* before any term is saved durably: (a) a typed confirmation for brand-new terms, or (b) the same correction accepted across two distinct utterances (intra-user agreement). A single click only updates a transient cache, not the durable vocab DB. |
| 7 | **Single-user poisoning of own scope.** A careless or malicious user teaches their own scope wrong things. | Their transcripts get worse and worse. | (a) Anomaly detection on per-user accept patterns: if a user's accept rate is >3σ from peers, flag for review. (b) Periodic "calibration audits" sampling their personal vocab against the org gold list. (c) Allow users to reset personal scope. |
| 8 | **PII inside audio exemplars.** Doctor says "give Doliprane to patient Maria Rashid." If we save that whole utterance as an exemplar, we just stored PHI. | Major regulatory liability (HIPAA / GDPR) even on-prem. | (a) Always extract a *tight word-level audio crop* using forced alignment, never a full utterance. (b) Run a PII detector (NER) on the transcript window before storing the crop; if PII is within 0.5 s, refuse to store. (c) Hash speaker identity in metadata; never store names. (d) Document this in the customer DPA. |
| 9 | **GDPR right to be forgotten.** A user or patient asks for deletion. We must remove their corrections, audio exemplars, their share of audio averages, and their influence on fine-tuned weights. | Non-deletion is a legal blocker. | (a) Every Qdrant point carries `contributor_user_id`. (b) Store audio averages as **sum + count**, not just average — so we can decrement on deletion without recomputing. (c) Tier-2 / Tier-3 fine-tunes use deletion-aware training: keep a contribution log and re-train, or roll back to the pre-contribution checkpoint, on deletion. (d) Run a quarterly deletion drill. |

### 17.3 Scale and Operations

| # | Limitation | Why it matters | Mitigation |
|---|------------|----------------|------------|
| 10 | **Reviewer bottleneck.** As volume grows, the ASK-HUMAN queue grows linearly. Reviewers fatigue, accuracy drops. | The whole self-improvement loop stalls. | (a) Active learning with a quantified uncertainty score: only the top-N highest-disagreement spans per day reach the queue; the rest are auto-decided on retrieval-only confidence. (b) Track Cohen's kappa across reviewers to detect fatigue. (c) Tiered reviewers: clinicians for clinical correctness, ops staff for obvious typos. (d) Cap each reviewer's queue size per session. |
| 11 | **Whisper boost list token limit.** Whisper's `initial_prompt` is capped at about 224 tokens. A growing org vocabulary cannot all fit. | Most rare terms never receive the ASR boost they need. | (a) Per-utterance *dynamic* boost list: predict the topic of the audio from the previous transcript and pick top-K most relevant terms. (b) Use decoder-level word biasing (faster-whisper hotwords / CTC biasing) instead of stuffing the prompt. (c) Maintain a per-user "frequently confused" shortlist that always wins a slot. |
| 12 | **Audio exemplar storage growth.** Storing every accepted audio crop forever balloons MinIO. | Cost spikes; backups slow down. | (a) Cap to N exemplars per term per scope (e.g. 20 per user, 200 per org) with a quality-based replacement policy: keep highest-SNR, most-diverse-speaker examples. (b) Store sub-second word crops, never full utterances. (c) Keep the audio average even after deleting raw clips. (d) Archive cold (rarely matched) terms to compressed cold storage. |

### 17.4 Quality Control Over Time

| # | Limitation | Why it matters | Mitigation |
|---|------------|----------------|------------|
| 13 | **Silent regression after fine-tune.** A nightly LoRA on the AI text picker may improve confirmed corrections but hurt edge cases not in the eval set. | Quality silently degrades; nobody notices for weeks. | (a) Multi-set evaluation: locked regression set **+** rotating "wild" set sampled from production **+** adversarial negatives. Promote only if *all three* pass. (b) Shadow mode: new model runs in parallel for 24 h; promote only if disagreement rate is bounded. (c) Per-tenant weekly accuracy dashboard with alerting on drop. |
| 14 | **Concept drift and retired terms.** Drugs get withdrawn, guidelines change, terminology evolves. The vocab DB has no expiry. | Stale suggestions become wrong. | (a) Every entry has `last_clinical_review_at`; entries older than 12 months get re-validated against an updated source lexicon. (b) Curators can mark a term `deprecated → use X` — alias mapping is kept but auto-fix stops. (c) Subscribe to RxNorm / SNOMED update feeds; run a monthly diff job. |
| 15 | **Disagreement between users (no consensus mechanism).** In Gulf Arabic, dialects vary; two doctors may legitimately give different "correct" forms. | Promotion to org/global produces an arbitrary winner. | (a) Promotion records *all* variants as `aliases`, not just the most frequent. (b) When variants conflict semantically (different drugs, not different spellings), block promotion and require curator decision. (c) Per-dialect tagging on entries; retrieval prefers the dialect of the speaker as detected by pyannote-language. |

### 17.5 Cross-Cutting Mitigations

Three principles fix many items above at once. Treat them as load-bearing.

1. **Every promotion is gated by a regression test.** Tier-1 user upserts, Tier-2 LLM LoRA, Tier-3 ASR adapters — every one runs against the locked test set + a rotating production sample before being applied. Failure → automatic rollback. Reuses and extends [eval_corrector.py](eval_corrector.py).
2. **Every artifact is contribution-traced.** Every Qdrant point, every audio exemplar, every fine-tuned weight checkpoint records *who contributed to it*. This is what makes deletion (#9), poisoning detection (#7), and rollback (#13) actually possible.
3. **Two-of-N rule for any auto-fix.** No correction is auto-applied unless at least two independent signals agree (e.g. retrieval score + audio match, or retrieval + LLM choice). Single-signal results always drop to SUGGEST or ASK-HUMAN. This kills most poisoning and false-positive scenarios cheaply.

### 17.6 Limitations We Accept in v1 (Not Mitigated)

| Limitation | Why we accept it |
|------------|------------------|
| 100% accuracy is impossible — some audio is genuinely ambiguous (mumbling, speaker overlap, heavy noise). | Forcing a fix here is *more* dangerous than abstaining. The system flags and asks. |
| New rare molecules and novel diseases at launch will be missed until manually added. | Acceptable as long as the human review queue captures them on first occurrence. |
| Real-time / streaming / live-caption use cases. | Locked decision: accuracy > latency. Re-evaluated in Phase D. |
| Per-tenant on-prem backup and disaster recovery. | Customer responsibility on-prem; we provide tooling and documentation, not the actual backup site. |

---

If anything still uses unclear words, point at the section and I will rewrite it even more plainly.
