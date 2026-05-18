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

### MixAT / PolyWER MixAT-Tri
- **Hours:** 15
- **Dialect:** Emirati Arabic ↔ English code-switching
- **License:** CC BY-NC-SA 4.0
- **Where:** Original repo [`mbzuai-nlp/mixat`](https://github.com/mbzuai-nlp/mixat), updated HF dataset [`sqrk/mixat-tri`](https://huggingface.co/datasets/sqrk/mixat-tri)
- **Why it matters:** The strongest currently available Emirati-English
  code-switch source. Use `transcript` for ASR training. Preserve
  `transliteration` and `translation` as metadata only; they are useful for
  PolyWER-style evaluation but conflict with our strict-Latin training target.

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

### Contributor setup

Install dependencies and make sure `ffmpeg` is available:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ffmpeg -version | head -1
```

Authenticate with Kaggle before using SADA:

```bash
.venv/bin/kaggle auth login
```

Hugging Face access is optional for the currently working public sources, but
some mirrors are gated. The new CLI is `hf`; `huggingface-cli` may print a
deprecation warning:

```bash
.venv/bin/hf auth login
```

Never paste tokens into chat. Enter secrets directly in the terminal.

Use these scripts to inspect the Saudi and Emirati/UAE datasets one by one.
Each script defaults to **10 samples** and writes into its own folder under
`data/dataset_samples/<dataset>/`:

```bash
# Hugging Face sources. Some are gated: run `.venv/bin/hf auth login` and
# accept the dataset terms on Hugging Face before downloading.
python scripts/download_worldspeech_saudi_samples.py
python scripts/download_saudilang_scc_samples.py
python scripts/download_uae_bilingual_samples.py
python scripts/download_nexdata_uae_sample.py
python scripts/download_mixat_emirati_samples.py

# Kaggle sources. Requires `pip install kaggle` and ~/.kaggle/kaggle.json.
python scripts/download_sada2022_samples.py
```

Or run all target datasets and produce a combined output:

```bash
.venv/bin/python scripts/download_all_target_samples.py --limit 10
```

The global runner writes:

- `data/dataset_samples/download_summary.json`
- `data/dataset_samples/combined_manifest.jsonl`

Safety defaults: the global script will not download full YouTube episode
audio just to preview 10 samples. SADA is sampled by targeted Kaggle per-file
downloads from `batch_1`, so it does **not** require downloading the full SADA
archive for previews. Saudilang SCC defaults to metadata rows only.

Saudilang SCC is different from the other datasets: Hugging Face provides
CSV segment annotations with YouTube links, not bundled audio files. To cut
those referenced segments into WAV files, explicitly enable YouTube audio:

```bash
.venv/bin/python scripts/download_all_target_samples.py --limit 10 --download-saudilang-audio
```

To force SADA full-archive mode instead of targeted per-file sampling:

```bash
.venv/bin/python scripts/download_all_target_samples.py --limit 10 --allow-full-kaggle-archives
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

Metadata layout after the sample run:

| Dataset | Metadata location |
|---|---|
| `sada2022` | `raw/train.csv`, `raw/valid.csv`, `raw/test.csv`, and per-sample `segments/*.segments.jsonl` |
| `saudilang_scc` | `raw/scc_testset.csv`; manifest rows contain transcript + YouTube segment timestamps |
| `worldspeech_saudi` | manifest row `metadata` contains source URL, segment times, CER/SNR, and transcripts; raw README copied to `raw/README.md` |
| `nexdata_uae_sample` | `metadata/*.txt`, `metadata/*.metadata`, and `segments/*.segments.jsonl` |
| `mixat_emirati` | manifest row `text` uses HF `transcript`; row `metadata` preserves `transliteration`, `translation`, `language`, and `duration_ms` |

Current blocked source:

- `uae_bilingual` is not downloadable with the current repo id
  `vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k`. Hugging Face login
  was successful, but the Hub still returns "doesn't exist or cannot be
  accessed". Update `scripts/download_uae_bilingual_samples.py` once the exact
  dataset URL is verified.

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
.venv/bin/python scripts/preprocess_code_switch_asr.py \
  --manifest data/dataset_samples/sada2022/manifest.jsonl \
  --manifest data/dataset_samples/saudilang_scc/manifest.jsonl \
  --manifest data/dataset_samples/worldspeech_saudi/manifest.jsonl \
  --manifest data/dataset_samples/nexdata_uae_sample/manifest.jsonl \
  --manifest data/dataset_samples/mixat_emirati/manifest.jsonl \
  --out data/preprocessed_audios
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

For rows with `segments_path`, the preprocessor expands long recordings into
aligned short clips before applying duration and text filters. This is what
makes SADA and Nexdata usable: their downloaded WAV files are long recordings,
but their metadata gives timestamped transcript segments.

Sample-run output as of the latest audit:

```text
data/preprocessed_audios/
  audio/                 # 16 kHz mono PCM16 WAV clips
  manifest.jsonl         # clean ASR training rows
  rejected.jsonl         # rejected rows/segments with reasons
  vocab.txt              # final cleaned vocabulary
  summary.json           # preprocessing summary
  audit_report.json      # mechanical audit report
```

Latest sample audit:

```text
clips: 1154
hours: 1.4653
sample_rates: {16000: 1154}
channels: {1: 1154}
formats: {'WAV': 1154}
subtypes: {'PCM_16': 1154}
duration min/max/mean/p50/p95: 1.0 / 29.32 / 4.571 / 3.625 / 9.53 sec
audio failures: none
text failures: none
```

The audit checks:

- 16 kHz sample rate
- mono audio
- WAV / PCM16 format
- 1 to 30 second duration
- <=0.5 second edge silence
- no diacritics, tatweel, punctuation, digits, uppercase English, HTML, emoji,
  Cyrillic, or Arabic/Latin attached tokens
- duration-to-text ratio between 1 and 25 characters per second

The only semantic item not fully machine-verifiable is whether Arabic-script
drug transliterations such as `باراسيتامول` should be converted to Latin
`paracetamol`. That needs a medical lexicon or manual review layer.

### DGX full-data run

On the DGX Spark / 128 GB VRAM machine, use the full pipeline script. It
downloads the three currently available audio sources in full:

- `sada2022`
- `worldspeech_saudi`
- `nexdata_uae_sample`
- `mixat_emirati`

Then it preprocesses all aligned clips and writes grouped train/validation/test
splits. Segments from the same source recording stay in the same split to avoid
train/eval leakage.

```bash
.venv/bin/python scripts/prepare_dgx_full_asr_dataset.py \
  --work-dir data/dgx_full \
  --confirm-full-download
```

The script refuses to run full downloads without `--confirm-full-download`.
This is intentional because full SADA is large. It uses targeted per-file
Kaggle downloads across all SADA batch folders rather than pulling one giant
archive.

Outputs:

- `data/dgx_full/raw_datasets/<dataset>/`
- `data/dgx_full/preprocessed_audios/manifest.jsonl`
- `data/dgx_full/preprocessed_audios/rejected.jsonl`
- `data/dgx_full/preprocessed_audios/vocab.txt`
- `data/dgx_full/preprocessed_audios/splits/train.jsonl`
- `data/dgx_full/preprocessed_audios/splits/validation.jsonl`
- `data/dgx_full/preprocessed_audios/splits/test.jsonl`
- `data/dgx_full/preprocessed_audios/splits/split_summary.json`

Default split ratio is `90/5/5`. You can override it:

```bash
.venv/bin/python scripts/prepare_dgx_full_asr_dataset.py \
  --work-dir data/dgx_full \
  --train-ratio 0.9 \
  --validation-ratio 0.05 \
  --test-ratio 0.05 \
  --confirm-full-download
```

Splitting is grouped by source recording: all segments cut from one original
SADA/Nexdata recording stay in the same train/validation/test split. This
prevents leakage where near-identical context from the same 10-minute source
recording appears in both training and evaluation.

MixAT already ships utterance-level clips, so it is not segment-expanded. Its
official HF splits are `train` and `test`; the DGX script downloads both by
calling `download_mixat_emirati_samples.py --limit 0 --split all`, then uses
grouped duration balancing with the rest of the corpus for one unified set of
manifests.

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
4. Download MixAT (`sqrk/mixat-tri`) + Nexdata UAE sample (Emirati/UAE).
5. Add UAE bilingual, Ramsa, or Traditional Emirati Arabic only after a direct machine-download
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
4. **[MixAT / PolyWER MixAT-Tri](https://huggingface.co/datasets/sqrk/mixat-tri)** — Emirati-English code-switching, 15 hrs.
5. **[Nexdata UAE Arabic sample](https://huggingface.co/datasets/Nexdata/UAE_Arabic_Spontaneous_Speech_Data)** — tiny but authentic UAE sample.
6. **[UAE Arabic-English Bilingual Dataset 40k](https://huggingface.co/datasets/vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k)** — currently blocked until exact repo/access is verified.
7. **[SADA paper](https://www.semanticscholar.org/paper/SADA%3A-Saudi-Audio-Dataset-for-Arabic-Alharbi-Alowisheq/de2508f2d48ea42653fe11011f24f9f227d38e71)** — read this before fine-tuning.
8. **[Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper)** — official HuggingFace guide.
9. **[PEFT / LoRA library](https://github.com/huggingface/peft)** — the actual fine-tuning machinery.

---

## 11. Where this leaves us

**For Saudi/Emirati acoustic adaptation:** SADA is the Saudi backbone;
WorldSpeech `ar_sa` and Saudilang add Saudi variety/code-switching;
MixAT supplies the strongest currently available Emirati-English code-switch
signal, with Nexdata as a tiny spontaneous UAE sample. Non-target Arabic corpora
are intentionally excluded.

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
