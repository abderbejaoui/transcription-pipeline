# Datasets, Data Sources & Fine-Tuning Approach

This document is the **single source of truth** for everything we know
about freely-available and commercially-available speech data relevant
to building a Gulf-Arabic medical ASR system, plus the fine-tuning
approach that uses it.

> Last research pass: November 2026. URLs and exact hour counts may
> shift over time — re-verify before starting a fine-tune.

---

## 1. The data problem in one paragraph

Off-the-shelf Whisper has two failure modes on Gulf clinic audio:
1. It hallucinates Arabic-accented English drug names ("Doliprane",
   "Acitrom") into the closest English-sounding tokens.
2. It cannot transcribe Arabic↔English code-switching (a doctor saying
   "the patient has حبوب الضغط and we'll switch to Amlodipine") without
   chunking the language identification mid-utterance.

To fix this we need:
- Acoustic adaptation to **Khaleeji-accented speech** (Saudi, Emirati,
  Omani, etc.).
- Vocabulary coverage for **medical terminology** (drug brand names,
  diagnoses, procedures).
- Multi-language decoder behavior for **AR↔EN code-switching**.

Items 1 and 3 have free public datasets (listed below). Item 2 — Arabic
medical conversational speech — has **no free corpus**; we have to
build it ourselves through pilots and TTS augmentation.

---

## 2. Free Gulf Arabic speech corpora (verified)

These are real, downloadable, and free. Hour counts come from the
papers/dataset cards I read directly.

### SADA — Saudi Audio Dataset for Arabic ⭐
- **Hours:** 668
- **Dialect:** Saudi multi-dialect (the official SDAIA dataset)
- **License:** Free for research and commercial use under their terms
- **Where:** [Kaggle — SADA 2022](https://www.kaggle.com/datasets/sdaiancai/sada2022)
- **Paper:** [SADA: Saudi Audio Dataset for Arabic](https://www.semanticscholar.org/paper/SADA%3A-Saudi-Audio-Dataset-for-Arabic-Alharbi-Alowisheq/de2508f2d48ea42653fe11011f24f9f227d38e71) (ICASSP 2024)
- **Citation count:** 26 (well-validated)
- **Why it matters:** The single biggest free Saudi Arabic speech
  corpus. Larger than the entire LDC Arabic catalog combined. This is
  the foundation for any Khaleeji ASR work.
- **Notes:** Best fine-tuning result reported is MMS-1B with WER 40.9%
  / CER 17.6% on SADA test-clean — i.e. plenty of room to improve.

### Saudilang Code-Switch Corpus (SCC)
- **Hours:** undisclosed in dataset card; small but targeted
- **Dialect:** Saudi Arabic ↔ English code-switching
- **License:** Open on Kaggle
- **Where:** [Kaggle — Saudilang Code-Switch Corpus](https://www.kaggle.com/datasets/sdaiancai/saudilang-code-switch-corpus-scc)
- **Why it matters:** Code-switching is exactly what Gulf doctors do
  with drug names. Same publisher as SADA (SDAIA).

### OMAN-SPEECH
- **Hours:** ~40
- **Dialect:** Omani Arabic across 11 Wilayats (provinces), 32 speakers
- **License:** Open
- **Where:** [aclanthology.org/2026.abjadnlp-1.31](https://aclanthology.org/2026.abjadnlp-1.31.pdf)
- **Why it matters:** Sociolinguistically stratified Omani / Gulf
  Arabic. Useful for evaluating Gulf accent coverage beyond Saudi.

### Ramsa — Emirati Arabic Speech Corpus
- **Coverage:** Large; 10% subset used as ASR/TTS baseline
- **Dialect:** Emirati Arabic
- **License:** Open
- **Where:** [arXiv:2603.08125](https://arxiv.org/pdf/2603.08125.pdf)
- **Why it matters:** UAE-specific corpus from a sociolinguistic lens.

### Traditional Emirati Arabic (heritage broadcasts)
- **Hours:** curated, undisclosed exact count
- **Dialect:** Traditional UAE
- **Where:** [aclanthology.org/2025.icnlsp-1.5](https://aclanthology.org/2025.icnlsp-1.5/)
- **Why it matters:** Heritage and literary register; complements
  modern conversational data.

### ZAEBUC-Spoken
- **Coverage:** Multi-dialect Arabic + English code-switching
- **Where:** [aclanthology — LREC 2024](https://www.aclanthology.org/2024.lrec-main.1546.pdf)
- **Why it matters:** Code-switching guidelines + multi-dialect
  coverage. Cited 11 times.

### ArSyra Gulf Arabic (Khaliji)
- **Where:** [Kaggle — ArSyra Gulf](https://www.kaggle.com/datasets/aqlomate/arsyra-gulf)
- **Companion:** [ArSyra Complete Multi-Dialect](https://www.kaggle.com/datasets/aqlomate/arsyra-complete)
- **Why it matters:** Direct Gulf-labeled dataset.

### ADI-17 Dataset
- **Coverage:** 17 Arabic dialect classifications
- **Where:** [Kaggle — ADI-17](https://www.kaggle.com/datasets/basselabdelmonem/adi-17-dataset)
- **Why it matters:** Useful for dialect ID; not for ASR fine-tune
  directly but helps preprocess multi-dialect data.

---

## 3. Pan-Arabic free corpora (broader baseline)

These are not Gulf-specific but provide acoustic priors and broad MSA
coverage. Useful for pre-training before Gulf-specific fine-tuning.

### MASC (Massive Arabic Speech Corpus)
- **Hours:** ~1,000
- **Coverage:** Pan-Arabic (MSA + various dialects)
- **License:** Free for research
- **Where:** HuggingFace — search `Sigma-AI/MASC` (gated; need account)

### MGB-2 / MGB-3 Aljazeera Arabic
- **Hours:** ~1,200 (MGB-2 broadcast Arabic)
- **License:** Free for research from QCRI
- **Where:** [arabicspeech.org](https://arabicspeech.org/)
- **Why it matters:** Largest free pan-Arabic broadcast corpus. Mostly
  MSA but includes some Gulf speakers in interviews.

### Mozilla Common Voice — Arabic
- **Hours:** ~88 (Common Voice 17.0)
- **Coverage:** Read speech, mostly MSA, multi-speaker
- **License:** CC-0 (public domain)
- **Where:** [commonvoice.mozilla.org](https://commonvoice.mozilla.org/) →
  Datasets → Arabic
- **HF mirror:** [`fsicoli/common_voice_17_0`](https://huggingface.co/datasets/fsicoli/common_voice_17_0)

### Google FLEURS — Arabic
- **Hours:** ~10
- **Coverage:** MSA read-speech, parallel sentences across 102 languages
- **License:** CC-BY 4.0
- **Where:** [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) (config: `ar_eg`)
- **Why it matters:** Clean held-out test set. Perfect for benchmarking
  before/after fine-tuning.

### CoVoST-2 EN↔AR
- **Where:** [`ymoslem/CoVoST2-EN-AR`](https://huggingface.co/datasets/ymoslem/CoVoST2-EN-AR)
- **Why it matters:** Speech translation pairs for AR↔EN.

### MGB-3 Egyptian (HuggingFace)
- **Hours:** ~16
- **Where:** [`MightyStudent/Egyptian-ASR-MGB-3`](https://huggingface.co/datasets/MightyStudent/Egyptian-ASR-MGB-3)
- **Note:** Egyptian, **not Gulf**. Listed only because it's the most
  popular HF Arabic ASR dataset and gets confused with relevant data.

### Other smaller Arabic speech sets on HuggingFace
- [`tunis-ai/arabic_speech_corpus`](https://huggingface.co/datasets/tunis-ai/arabic_speech_corpus)
- [`RetaSy/quranic_audio_dataset`](https://huggingface.co/datasets/RetaSy/quranic_audio_dataset) — Quranic recitation
- [`OmarAhmedSobhy/egyption-with-emotion-dataset`](https://huggingface.co/datasets/OmarAhmedSobhy/egyption-with-emotion-dataset)

### TuniSpeech-21h
- **Hours:** 21
- **Dialect:** Tunisian (not Gulf, included for completeness)
- **Where:** [scitepress.org 2026](https://www.scitepress.org/Papers/2026/144577/144577.pdf)

---

## 4. OpenSLR Arabic-related entries

OpenSLR is the canonical mirror for many ASR datasets. Arabic-specific
slots that exist:

| ID | Name | Notes |
|---|---|---|
| SLR46 | Tunisian_MSA | Tunisian Modern Standard Arabic |
| SLR132 | Mohammed | Quranic Arabic speech-to-text |

There is **no Gulf-specific SLR entry** as of this writing. Browse the
full catalog at [openslr.org/resources](https://www.openslr.org/resources.php).

---

## 5. LDC (paid) — Gulf and Arabic medical-adjacent

For completeness, the LDC catalog has these (not free, but listed if
you have university access):

| ID | Name | Hours | Cost |
|---|---|---:|---|
| LDC2006S43 | Gulf Arabic Conversational Telephone Speech | ~70 | ~$3-5k member rate |
| LDC2006T15 | Gulf Arabic CTS Transcripts | matches above | bundled |
| LDC2006S45 | Iraqi Arabic CTS | ~24 | ~$1.5k |
| LDC2025L01 | Iraqi Arabic - English Lexical DB | text only | — |
| LDC2017L01 | Arabic Speech Recognition Pronunciation Dictionary | text only | — |
| LDC2025S03 | Comprehensive Arabic Phonetic DB | text only | — |
| LDC2014S02 | King Saud University Arabic Speech DB | small | — |
| LDC2017S12 | KSUEmotions | small, emotional | — |

**ELRA has zero matches for "Arabic medical"** — confirmed via search at
[catalog.elra.info](https://catalog.elra.info/en-us/repository/search/?q=Arabic+medical).
Their Gulf Arabic search returns one €185k commercial-license lexical
resource — not useful for ASR fine-tuning.

---

## 6. Arabic medical speech — what does NOT exist

I did a thorough Semantic Scholar search ("arabic medical speech
recognition dataset" — 746 results; "arabic medical conversation
speech ASR corpus" — 363 results; "clinical arabic speech recognition
saudi" — 81 results).

**No free Arabic medical conversational speech corpus exists** in any
catalog: LDC, ELRA, OpenSLR, HuggingFace, Kaggle, or academic papers.

The closest matches I found:

- **MSA Speech Disorders Corpus** (Alqudah 2024) — 40 Jordanian speakers
  with articulation disorders. Public.
  [International Journal of Speech Technology](https://doi.org/10.1007/s10772-024-10086-9)
- **Saudi Dialect SER corpus** (Aljuhani 2021) — Saudi emotional
  speech, small. [IEEE Access](https://ieeexplore.ieee.org/ielx7/6287639/6514899/09530700.pdf)
- **Various Arabic speech disorder corpora** for hearing/speech
  pathology research — not what we want for ASR.

This data scarcity is **the moat for Gulf medical ASR**. Whoever
collects ~500+ hours of Gulf clinical conversational speech first will
have a defensible position that AWS, Nuance, Google won't quickly
replicate.

---

## 7. Building medical vocabulary data ourselves

Since no free medical Arabic conversational corpus exists, we generate
the medical vocabulary coverage in three layers:

### Layer A — TTS-augmented medical readings (~$1k, ~50 hrs)

Use Arabic TTS APIs to synthesize prepared sentences containing your
top medical terms in Gulf dialect:

| Provider | Voices | Cost | Notes |
|---|---|---|---|
| **Azure ar-XA** | Hamed, Zariyah, etc. | $16/1M chars | Best quality Gulf voices |
| **Google Cloud TTS ar-AE** | Wavenet voices | $16/1M chars | Strong Gulf accents |
| **ElevenLabs Arabic** | Custom-clonable | $99/mo Pro | Best naturalness |
| **OpenAI TTS** | Limited Arabic | per-token | Mostly MSA |

Recipe:
1. Take your top ~1,000 medical terms from `data/medical_lexicon.jsonl`.
2. For each term, generate ~10 carrier sentences using GPT/Claude
   ("اعطي المريض dose من <term>", etc.).
3. Render each sentence in 5–10 different Gulf-accent voices.
4. Mix with real ambient clinic noise (recordings of empty waiting
   rooms, AC hum, dictaphone background) at SNR 15–25 dB.
5. ~10,000 utterances × ~5 s each = ~14 hours raw, padded to ~50 hrs
   with carrier-sentence repetition.

This gives medical vocabulary coverage that doesn't exist in any
public corpus.

### Layer B — Hire local Gulf medical professionals to record

Realistic providers:

| Provider | Coverage | Rate |
|---|---|---|
| **Anolytics** | Has Gulf Arabic teams | $8–20/hr |
| **iMerit** | Medical-domain Arabic teams | $10–25/hr |
| **Alpha-CRC** | Saudi-based | varies |
| **Defined.ai** | Custom Gulf medical | $25–80/hr |
| **Appen / Sama** | Scripted Gulf readings | $15–40/hr |
| **Upwork / Bayt** | Local recruitment in Riyadh / Dubai / Doha | $5–15/hr |

For ~$5k you can get ~80 hours of scripted Gulf medical readings from
10 medical students/residents. Read-speech only, but vocabulary-rich.

### Layer C — Pilot data from real clinics (the long-term moat)

Sign up 5–10 clinics for free 60-day pilots. Their corrections through
your UI become aligned (audio, text) training pairs at zero data cost.

Realistic yield:
- 5 clinics × 50 hours/month × 2 months = **500 hours** of real,
  properly labeled, Gulf medical conversational speech.

This is the dataset that makes the company defensible. Track it
carefully and own its IP.

---

## 8. Fine-tuning approach

The plan, sequenced:

### Phase 1 — Bootstrap (Months 1–2, ~$2k total)
1. Download SADA from Kaggle (668 hrs, free).
2. Download OMAN-SPEECH + Ramsa + ZAEBUC-Spoken + ArSyra Gulf (free).
3. Pull MGB-2 Arabic (1,200 hrs, free for research).
4. Generate Layer A TTS-augmented medical readings (~50 hrs, ~$1k).
5. **First LoRA fine-tune** of `whisper-large-v3-turbo` on the combined
   ~2,000 hours.

Expected gain: **+3–5% absolute WER** on Gulf medical audio versus
vanilla Whisper. Day-1 accuracy moves from ~75% to ~88–92%.

### Phase 2 — Real clinical data accumulation (Months 3–6)
1. Run free pilots in 3–5 Gulf clinics.
2. Every user correction → aligned audio+text pair stored.
3. After 3 months: ~150 hours of real Gulf clinical audio.

### Phase 3 — Production fine-tune (Month 7)
1. Re-fine-tune on Phase 1 data + Phase 2 real clinical audio.
2. Validate on a held-out test set of 50 clinical recordings.
3. Expected gain: another **+2–4% WER** improvement.

### Phase 4 — Ministry partnerships (parallel, 6–12 months)
- Saudi MOH **Sehha** digital health initiative — actively partnering
  with AI startups
- UAE Ministry of Health — has open AI grant programs
- **Hamad Medical Corporation** (Qatar) — runs research collaborations
- **King Faisal Specialist Hospital** — has AI research arm
- **Dubai Health Authority** Open Data

These typically yield 1,000+ hours of clinical Gulf Arabic but take
6–12 months to land contractually.

---

## 9. Compute & engineering for the fine-tune

### LoRA fine-tune of Whisper-large-v3-turbo
- **Model size:** 809M parameters (just the encoder + decoder)
- **LoRA rank:** 32 (~12M trainable parameters)
- **Recommended training**: HuggingFace `transformers` + `peft`
- **GPU:** Single A100 40GB or 2× RTX 4090 24GB
- **Dataset size at this stage:** ~2,000 hours mixed
- **Training time:** ~8–16 hours per epoch, 3 epochs typical
- **Cost on Lambda Labs / RunPod:** ~$300–800 per fine-tune round
- **Output:** ~50 MB LoRA adapter file, loadable on any
  Whisper-large-v3-turbo instance

### Recommended starter scripts
- HuggingFace's `examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py`
- Adapter loading: [`peft` library](https://github.com/huggingface/peft)
- Tutorial:
  [Fine-tune Whisper for Multilingual ASR — HuggingFace](https://huggingface.co/blog/fine-tune-whisper)

---

## 10. Evaluation

For honest evaluation we need a **held-out Gulf medical test set**
that none of the training data has seen.

### Recommended evaluation set composition
- 50 clinical recordings, ~10 minutes each = ~8 hours
- Stratified by:
  - Specialty (cardiology, pediatrics, emergency, OB/GYN, internal med)
  - Speaker (≥10 distinct doctors, balanced gender)
  - Recording condition (office mic, dictaphone, phone)
- Each recording manually transcribed and reviewed twice
- Cost: ~$2–5k labor

### Metrics to track
- Overall WER (word error rate)
- **Medical-term WER** (only count errors on medical vocabulary —
  this is the metric that matters for the product)
- Code-switching CER (where applicable)
- Per-speaker WER (catch speaker bias)
- Latency (end-to-end p50/p95)

### Baseline to beat
- Whisper-large-v3-turbo (no fine-tune): expect ~45% WER on Gulf
  conversational audio per the SADA paper baselines
- After Phase 1 fine-tune: target ~25% overall WER
- After Phase 3 fine-tune: target ~12–15% WER on medical-term subset

---

## 11. Quick links — start here

1. **[Kaggle SADA](https://www.kaggle.com/datasets/sdaiancai/sada2022)** — start downloading today, 668 hrs Saudi Arabic, free.
2. **[Kaggle Saudilang Code-Switch](https://www.kaggle.com/datasets/sdaiancai/saudilang-code-switch-corpus-scc)** — same publisher, AR↔EN code-switching.
3. **[Common Voice ar](https://commonvoice.mozilla.org/)** — 88 hrs, CC-0.
4. **[FLEURS Arabic](https://huggingface.co/datasets/google/fleurs)** — 10 hrs, eval-quality MSA.
5. **[OMAN-SPEECH paper](https://aclanthology.org/2026.abjadnlp-1.31.pdf)** — Omani Gulf Arabic.
6. **[SADA paper](https://www.semanticscholar.org/paper/SADA%3A-Saudi-Audio-Dataset-for-Arabic-Alharbi-Alowisheq/de2508f2d48ea42653fe11011f24f9f227d38e71)** — read this before fine-tuning.
7. **[Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper)** — official HuggingFace guide.
8. **[PEFT / LoRA library](https://github.com/huggingface/peft)** — the actual fine-tuning machinery.

---

## 12. Where this leaves us

**For Khaleeji acoustic adaptation:** ~700 hrs Gulf-specific + ~3,000
hrs pan-Arabic free data is more than sufficient. SADA alone fixes
the core acoustic problem.

**For medical vocabulary:** $1–2k of TTS augmentation + pilot data
collection over 3–6 months bridges the gap.

**For production-grade 99% accuracy:** Need ministry-level access to
1,000+ hours of real clinical recordings. Achievable in year 2 if
pilots go well.

**Realistic budget to a working fine-tuned model:**

| Stage | Time | Cost |
|---|---|---:|
| Data downloads + setup | 1 week | $0 |
| TTS medical augmentation | 1 week | ~$1k |
| First LoRA fine-tune + eval | 2 weeks | ~$500 GPU + $2k eval set |
| **Total to first useful model** | **~4 weeks** | **~$3.5k** |

That's the honest answer. The data is there, the path is clear, the
moat is the relationships you build with the first 5 clinics — not
the model itself.
