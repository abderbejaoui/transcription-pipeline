# DGX Data Pipeline Runbook

This runbook explains how to reproduce the Saudi/UAE ASR data workflow on a
DGX / GPU machine. It covers authentication, sample checks, full downloads,
preprocessing, train/validation/test split generation, and troubleshooting.

The main script is:

```bash
.venv/bin/python scripts/prepare_dgx_full_asr_dataset.py
```

It downloads the available audio datasets, preprocesses them into clean ASR
clips, then writes grouped train/validation/test manifests.

---

## 1. Clone And Set Up The Environment

```bash
git clone <repo-url>
cd transcription-pipeline

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check required command-line tools:

```bash
ffmpeg -version | head -1
.venv/bin/python -V
```

`ffmpeg` is required for audio conversion and clipping.

---

## 2. Authenticate External Data Sources

Credentials are machine-local. You must authenticate on the DGX itself, even if
you already authenticated on another machine.

### Kaggle

Preferred OAuth flow:

```bash
source .venv/bin/activate
kaggle auth login
```

Test SADA access:

```bash
kaggle datasets files sdaiancai/sada2022 --page-size 5
```

You should see files such as:

```text
README.md
batch_1/6k_SBA_100_0.wav
```

Alternative API-token file method:

```bash
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

`~/.kaggle/kaggle.json` must contain:

```json
{
  "username": "YOUR_KAGGLE_USERNAME",
  "key": "YOUR_KAGGLE_API_KEY"
}
```

Do not commit this file. Do not paste secrets into issue threads or chat logs.

### Hugging Face

The current Hugging Face CLI is `hf`:

```bash
.venv/bin/hf auth login
.venv/bin/hf auth whoami
```

Some datasets are public and do not require auth, but login is useful for gated
repos and higher request limits.

---

## 3. Optional Local Smoke Test

Before the full DGX run, you can download 10-sample previews:

```bash
.venv/bin/python scripts/download_all_target_samples.py --limit 10
```

Include optional neighbor-Gulf previews:

```bash
.venv/bin/python scripts/download_all_target_samples.py \
  --limit 10 \
  --include-neighbor-gulf
```

Download/cut Saudilang YouTube audio during preview:

```bash
.venv/bin/python scripts/download_all_target_samples.py \
  --limit 10 \
  --download-saudilang-audio
```

Preview output:

```text
data/dataset_samples/<dataset>/
data/dataset_samples/download_summary.json
data/dataset_samples/combined_manifest.jsonl
```

---

## 4. Run The Core Full DGX Pipeline

Core full run:

```bash
.venv/bin/python scripts/prepare_dgx_full_asr_dataset.py \
  --work-dir data/dgx_full \
  --confirm-full-download
```

The confirmation flag is required because the full download is large.

Core datasets:

- `sada2022`
- `worldspeech_saudi`
- `nexdata_uae_sample`
- `mixat_emirati`

The script downloads, preprocesses, and splits the data.

---

## 5. Optional Full-Run Flags

Add Kuwait/Bahrain neighbor-Gulf augmentation:

```bash
.venv/bin/python scripts/prepare_dgx_full_asr_dataset.py \
  --work-dir data/dgx_full \
  --confirm-full-download \
  --include-neighbor-gulf
```

Add Saudilang audio by cutting YouTube-linked segments:

```bash
.venv/bin/python scripts/prepare_dgx_full_asr_dataset.py \
  --work-dir data/dgx_full \
  --confirm-full-download \
  --include-saudilang-audio
```

Add Mansour Emirati cartoon data after confirming rights:

```bash
.venv/bin/python scripts/prepare_dgx_full_asr_dataset.py \
  --work-dir data/dgx_full \
  --confirm-full-download \
  --include-mansour
```

Run everything optional:

```bash
.venv/bin/python scripts/prepare_dgx_full_asr_dataset.py \
  --work-dir data/dgx_full \
  --confirm-full-download \
  --include-neighbor-gulf \
  --include-saudilang-audio \
  --include-mansour
```

---

## 6. Outputs

After the download phase, the script prints total hours per dataset and writes:

```text
data/dgx_full/download_hours_summary.json
```

This file contains per-dataset:

- rows
- audio rows
- text rows
- seconds
- minutes
- hours

Main preprocessed output:

```text
data/dgx_full/preprocessed_audios/audio/
data/dgx_full/preprocessed_audios/manifest.jsonl
data/dgx_full/preprocessed_audios/rejected.jsonl
data/dgx_full/preprocessed_audios/vocab.txt
data/dgx_full/preprocessed_audios/summary.json
```

Split output:

```text
data/dgx_full/preprocessed_audios/splits/train.jsonl
data/dgx_full/preprocessed_audios/splits/validation.jsonl
data/dgx_full/preprocessed_audios/splits/test.jsonl
data/dgx_full/preprocessed_audios/splits/split_summary.json
```

The split logic keeps all segments from the same source recording in the same
split to prevent train/eval leakage.

---

## 7. What Preprocessing Guarantees

The preprocessing script converts accepted clips to:

- 16 kHz
- mono
- 16-bit PCM WAV
- 1 to 30 seconds
- edge-trimmed silence
- normalized Arabic-English transcript text

Text normalization removes or normalizes:

- Arabic diacritics / tashkeel
- tatweel
- Alif variants
- Hamza carriers
- Alif maksura
- digits, by verbalizing them into Arabic words
- punctuation
- uppercase English
- attached Arabic/Latin boundaries
- HTML/control/weird symbols

The training target is always the normalized `text` field in the final
manifest.

---

## 8. Dataset Notes

### SADA

SADA is downloaded from Kaggle by targeted per-file downloads across all batch
folders. The script also downloads `train.csv`, `valid.csv`, and `test.csv`,
then uses segment timestamps to cut aligned clips.

### WorldSpeech Saudi/Kuwait/Bahrain

WorldSpeech provides embedded audio and `human_transcript`. Kuwait and Bahrain
are optional neighbor-Gulf augmentation, not Emirati replacements.

### MixAT

MixAT is downloaded from `sqrk/mixat-tri`. Use `transcript` as ASR training
text. `transliteration` and `translation` remain metadata only.

### Nexdata UAE Sample

The free sample is small. The script downloads `.wav`, `.txt`, and `.metadata`
files and uses timestamped `.txt` segments.

### Saudilang SCC

Hugging Face provides annotations and YouTube segment timestamps. Audio is not
bundled. Use `--include-saudilang-audio` only when you want to download and cut
YouTube audio.

### Mansour

Mansour is optional and requires explicit rights/permission. The script parses
PDF transcripts, extracts timestamps, downloads linked YouTube audio when
enabled, and rejects spans outside 1-30 seconds.

---

## 9. Troubleshooting

### Kaggle auth fails

Run:

```bash
kaggle auth login
kaggle datasets files sdaiancai/sada2022 --page-size 5
```

Or configure `~/.kaggle/kaggle.json` with `chmod 600`.

### Hugging Face login command fails

Use the current CLI:

```bash
.venv/bin/hf auth login
```

`huggingface-cli` may be deprecated.

### Full run refuses to start

Add the required explicit flag:

```bash
--confirm-full-download
```

### Downloaded hours look wrong

Check:

```bash
cat data/dgx_full/download_hours_summary.json
```

### Preprocessed clips are rejected

Check:

```bash
head data/dgx_full/preprocessed_audios/rejected.jsonl
cat data/dgx_full/preprocessed_audios/summary.json
```

Common rejection reasons:

- `too_short`
- `too_long_needs_alignment_chunking`
- `empty_transcript`
- `text_too_short_for_duration`
- `text_too_long_for_duration`
- `missing_audio`
