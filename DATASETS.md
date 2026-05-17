# Datasets, Data Sources & Fine-Tuning Approach

This document is the **single source of truth** for everything we know
about freely-available and commercially-available speech data relevant
to building a Saudi/Emirati medical ASR system, plus the fine-tuning
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
- Acoustic adaptation to **Saudi and Emirati-accented speech**.
- Vocabulary coverage for **medical terminology** (drug brand names,
  diagnoses, procedures).
- Multi-language decoder behavior for **AR↔EN code-switching**.

Items 1 and 3 have free public datasets (listed below). Item 2 — Arabic
medical conversational speech — has **no free corpus**; we have to
build it ourselves through pilots and TTS augmentation.

---

## 2. Free Saudi and Emirati speech corpora

This project now keeps only Saudi and Emirati/UAE dialect sources in the
dataset download workflow. Any source outside those two target dialects is
excluded unless it can be narrowed to Saudi or Emirati clips.

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

### WorldSpeech — Saudi Arabic split
- **Hours:** ~6.1
- **Dialect:** Saudi Arabic
- **License:** CC BY-NC 4.0 / source-specific public-record terms
- **Where:** HuggingFace [`disco-eth/WorldSpeech`](https://huggingface.co/datasets/disco-eth/WorldSpeech), config `ar_sa`
- **Why it matters:** Small but directly Saudi, with clean transcripts and
  useful metadata.

### UAE Arabic-English Bilingual Dataset 40k
- **Hours:** ~120 estimated for train split
- **Dialect:** UAE / Emirati Arabic with English code-switching
- **License:** Free/gated on HuggingFace; accept terms before use
- **Where:** HuggingFace [`vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k`](https://huggingface.co/datasets/vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k)
- **Why it matters:** The most directly useful free UAE code-switching
  source for Emirati clinic-style speech.

### Nexdata UAE Arabic Spontaneous Speech sample
- **Hours:** <1 in the free sample
- **Dialect:** UAE / Emirati Arabic
- **License:** Free sample of a commercial dataset
- **Where:** HuggingFace [`Nexdata/UAE_Arabic_Spontaneous_Speech_Data`](https://huggingface.co/datasets/Nexdata/UAE_Arabic_Spontaneous_Speech_Data)
- **Why it matters:** Tiny, but authentic spontaneous UAE speech.

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

---

## 3. Quick 10-sample download scripts

Use these scripts to inspect the Saudi and Emirati/UAE datasets one by one.
Each script defaults to **10 samples** and writes into its own folder under
`data/dataset_samples/<dataset>/`:

```bash
# Hugging Face sources. Some are gated: run `huggingface-cli login` and
# accept the dataset terms on Hugging Face before downloading.
python scripts/download_worldspeech_saudi_samples.py
python scripts/download_saudilang_scc_samples.py
python scripts/download_uae_bilingual_samples.py
python scripts/download_nexdata_uae_sample.py

# Kaggle sources. Requires `pip install kaggle` and ~/.kaggle/kaggle.json.
python scripts/download_sada2022_samples.py
```

Or run all target datasets and produce a combined output:

```bash
python scripts/download_all_target_samples.py --limit 10
```

The global runner writes:

- `data/dataset_samples/download_summary.json`
- `data/dataset_samples/combined_manifest.jsonl`

Safety defaults: the global script will not download full Kaggle archives
or full YouTube episode audio just to preview 10 samples. SADA on Kaggle is
therefore skipped unless you explicitly allow the full archive, and
Saudilang SCC defaults to metadata rows only.

Saudilang SCC is different from the other datasets: Hugging Face provides
CSV segment annotations with YouTube links, not bundled audio files. To cut
those referenced segments into WAV files, explicitly enable YouTube audio:

```bash
python scripts/download_all_target_samples.py --limit 10 --download-saudilang-audio
```

To allow full Kaggle archives before sampling:

```bash
python scripts/download_all_target_samples.py --limit 10 --allow-full-kaggle-archives
```

All scripts accept the same basic options:

```bash
python scripts/download_sada2022_samples.py --limit 10
python scripts/download_worldspeech_saudi_samples.py --limit 50 --out data/my_check/worldspeech_saudi
python scripts/download_uae_bilingual_samples.py --split validation
```

Each output folder contains:

- `audio/` — sampled audio files
- `manifest.jsonl` — one row per saved sample, with transcript text when found
- `README.md` — source and sample summary

Saudi or Emirati datasets that are listed only as papers or catalogue
pages, such as Ramsa and Traditional Emirati Arabic, are not scripted here
because this file does not currently include a direct public machine-download
URL for them. Add the actual repository URL once verified and a matching
wrapper can be added.

---

## 4. Preprocessing pipeline

Run all Saudi base data and the medical layer through the same cleaner so
the model never sees two transcript formats.

```bash
python scripts/preprocess_code_switch_asr.py \
  --manifest data/dataset_samples/sada2022/manifest.jsonl \
  --manifest data/dataset_samples/saudilang_scc/manifest.jsonl \
  --manifest data/dataset_samples/worldspeech_saudi/manifest.jsonl \
  --manifest data/dataset_samples/uae_bilingual/manifest.jsonl \
  --manifest data/dataset_samples/nexdata_uae_sample/manifest.jsonl \
  --out data/preprocessed/saudi_uae_asr
```

The script writes:

- `audio/` — 16 kHz mono 16-bit PCM WAV files
- `manifest.jsonl` — clean supervised ASR rows
- `rejected.jsonl` — dropped rows with reasons
- `vocab.txt` — unique cleaned word list for artifact/OOV inspection
- `summary.json` — kept/rejected counts and total hours

The text function is `clean_asr_text(text)` in
`scripts/preprocess_code_switch_asr.py`. It applies Arabic normalization,
English lowercasing, Arabic/Latin boundary splitting, digit/unit
verbalization, punctuation removal, HTML/URL/control cleanup, and final
duration-to-text filtering. Audio longer than 30 seconds is rejected unless
you first create aligned chunks; splitting long audio without word alignment
would corrupt the transcript/audio pair.

---

## 5. Arabic medical speech — what does NOT exist

I did a thorough Semantic Scholar search ("arabic medical speech
recognition dataset" — 746 results; "arabic medical conversation
speech ASR corpus" — 363 results; "clinical arabic speech recognition
saudi" — 81 results).

**No free Saudi/Emirati Arabic medical conversational speech corpus exists** in any
catalog: LDC, ELRA, OpenSLR, HuggingFace, Kaggle, or academic papers.

The closest target-relevant match I found:

- **Saudi Dialect SER corpus** (Aljuhani 2021) — Saudi emotional
  speech, small. [IEEE Access](https://ieeexplore.ieee.org/ielx7/6287639/6514899/09530700.pdf)

This data scarcity is **the moat for Saudi/Emirati medical ASR**. Whoever
collects ~500+ hours of Saudi/UAE clinical conversational speech first will
have a defensible position that AWS, Nuance, Google won't quickly
replicate.

---

## 6. Building medical vocabulary data ourselves

Since no free medical Arabic conversational corpus exists, we generate
the medical vocabulary coverage in three layers:

### Layer A — TTS-augmented medical readings (~$1k, ~50 hrs)

Use Arabic TTS APIs to synthesize prepared sentences containing your
top medical terms in Saudi and Emirati dialects:

| Provider | Voices | Cost | Notes |
|---|---|---|---|
| **Azure ar-XA** | Hamed, Zariyah, etc. | $16/1M chars | Gulf-style Arabic voices |
| **Google Cloud TTS ar-AE** | Wavenet voices | $16/1M chars | UAE Arabic locale |
| **ElevenLabs Arabic** | Custom-clonable | $99/mo Pro | Best naturalness |
| **OpenAI TTS** | Limited Arabic | per-token | Mostly MSA |

Recipe:
1. Take your top ~1,000 medical terms from `data/medical_lexicon.jsonl`.
2. For each term, generate ~10 carrier sentences using GPT/Claude
   ("اعطي المريض dose من <term>", etc.).
3. Render each sentence in 5–10 different Saudi/UAE-accent voices.
4. Mix with real ambient clinic noise (recordings of empty waiting
   rooms, AC hum, dictaphone background) at SNR 15–25 dB.
5. ~10,000 utterances × ~5 s each = ~14 hours raw, padded to ~50 hrs
   with carrier-sentence repetition.

This gives medical vocabulary coverage that doesn't exist in any
public corpus.

### Layer B — Hire local Saudi/UAE medical professionals to record

Realistic providers:

| Provider | Coverage | Rate |
|---|---|---|
| **Anolytics** | Has Saudi/UAE Arabic teams | $8–20/hr |
| **iMerit** | Medical-domain Arabic teams | $10–25/hr |
| **Alpha-CRC** | Saudi-based | varies |
| **Defined.ai** | Custom Saudi/UAE medical | $25–80/hr |
| **Appen / Sama** | Scripted Saudi/UAE readings | $15–40/hr |
| **Upwork / Bayt** | Local recruitment in Riyadh / Dubai / Abu Dhabi | $5–15/hr |

For ~$5k you can get ~80 hours of scripted Saudi/Emirati medical readings
from 10 medical students/residents. Read-speech only, but vocabulary-rich.

### Layer C — Pilot data from real clinics (the long-term moat)

Sign up 5–10 clinics for free 60-day pilots. Their corrections through
your UI become aligned (audio, text) training pairs at zero data cost.

Realistic yield:
- 5 clinics × 50 hours/month × 2 months = **500 hours** of real,
  properly labeled, Saudi/Emirati medical conversational speech.

This is the dataset that makes the company defensible. Track it
carefully and own its IP.

---

## 7. Fine-tuning approach

The plan, sequenced:

### Phase 1 — Bootstrap (Months 1–2, ~$2k total)
1. Download SADA from Kaggle (668 hrs, free).
2. Download Saudilang SCC from Kaggle (Saudi AR↔EN code-switching, free).
3. Download WorldSpeech `ar_sa` (Saudi split, free/gated on HuggingFace).
4. Download UAE bilingual + Nexdata UAE sample (Emirati/UAE, free/gated).
5. Add Ramsa / Traditional Emirati Arabic only after a direct machine-download
  URL is verified.
6. Generate Layer A TTS-augmented Saudi/Emirati medical readings (~50 hrs).
7. **First LoRA fine-tune** of `whisper-large-v3-turbo` on the combined
  Saudi/UAE mix.

Expected gain: **+3–5% absolute WER** on Saudi/UAE medical audio versus
vanilla Whisper. Day-1 accuracy moves from ~75% to ~88–92%.

### Phase 2 — Real clinical data accumulation (Months 3–6)
1. Run free pilots in 3–5 Saudi/UAE clinics.
2. Every user correction → aligned audio+text pair stored.
3. After 3 months: ~150 hours of real Saudi/UAE clinical audio.

### Phase 3 — Production fine-tune (Month 7)
1. Re-fine-tune on Phase 1 data + Phase 2 real clinical audio.
2. Validate on a held-out test set of 50 clinical recordings.
3. Expected gain: another **+2–4% WER** improvement.

### Phase 4 — Ministry partnerships (parallel, 6–12 months)
- Saudi MOH **Sehha** digital health initiative — actively partnering
  with AI startups
- UAE Ministry of Health — has open AI grant programs
- **King Faisal Specialist Hospital** — has AI research arm
- **Dubai Health Authority** Open Data

These typically yield 1,000+ hours of clinical Saudi/UAE Arabic but take
6–12 months to land contractually.

---

## 8. Compute & engineering for the fine-tune

### LoRA fine-tune of Whisper-large-v3-turbo
- **Model size:** 809M parameters (just the encoder + decoder)
- **LoRA rank:** 32 (~12M trainable parameters)
- **Recommended training**: HuggingFace `transformers` + `peft`
- **GPU:** Single A100 40GB or 2× RTX 4090 24GB
- **Dataset size at this stage:** ~800–1,000 hours mixed Saudi/UAE data
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

## 9. Evaluation

For honest evaluation we need a **held-out Saudi/UAE medical test set**
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
- Whisper-large-v3-turbo (no fine-tune): expect ~45% WER on Saudi/UAE
  conversational audio per the SADA paper baselines
- After Phase 1 fine-tune: target ~25% overall WER
- After Phase 3 fine-tune: target ~12–15% WER on medical-term subset

---

## 10. Quick links — start here

1. **[Kaggle SADA](https://www.kaggle.com/datasets/sdaiancai/sada2022)** — start downloading today, 668 hrs Saudi Arabic, free.
2. **[Kaggle Saudilang Code-Switch](https://www.kaggle.com/datasets/sdaiancai/saudilang-code-switch-corpus-scc)** — same publisher, AR↔EN code-switching.
3. **[WorldSpeech](https://huggingface.co/datasets/disco-eth/WorldSpeech)** — use config `ar_sa` only.
4. **[UAE Arabic-English Bilingual Dataset 40k](https://huggingface.co/datasets/vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k)** — Emirati/UAE code-switching.
5. **[Nexdata UAE Arabic sample](https://huggingface.co/datasets/Nexdata/UAE_Arabic_Spontaneous_Speech_Data)** — tiny but authentic UAE sample.
6. **[SADA paper](https://www.semanticscholar.org/paper/SADA%3A-Saudi-Audio-Dataset-for-Arabic-Alharbi-Alowisheq/de2508f2d48ea42653fe11011f24f9f227d38e71)** — read this before fine-tuning.
7. **[Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper)** — official HuggingFace guide.
8. **[PEFT / LoRA library](https://github.com/huggingface/peft)** — the actual fine-tuning machinery.

---

## 11. Where this leaves us

**For Saudi/Emirati acoustic adaptation:** SADA is the Saudi backbone;
WorldSpeech `ar_sa` and Saudilang add Saudi variety/code-switching;
UAE bilingual/Nexdata/Ramsa-style sources supply the Emirati side once
access is confirmed. Non-target Arabic corpora are intentionally excluded.

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
