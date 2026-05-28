# 04 — Preprocessing

This chapter describes the transforms applied to the raw datasets in
[03_data_collection.md](03_data_collection.md) to produce the training
manifests that the fine-tuning script consumes.

## 4.1 Audio preprocessing

For every input audio file we run:

1. **Decode**: parquet rows or .wav/.mp3/.flac files are decoded to
   `numpy.float32` arrays at the original sample rate.
2. **Resample to 16 kHz**: `soxr` high-quality resampler. 16 kHz is the
   sample rate Qwen3-ASR's audio encoder expects.
3. **Downmix to mono**: average of channels.
4. **Loudness normalize**: peak-normalize to -1 dBFS so quiet broadcast
   audio and loud studio audio enter the model with comparable levels.
5. **Duration filter**: drop clips with `duration_s < 0.5` (mostly noise
   tokens) and `duration_s > 30.0` (training window cap; longer clips
   are split at silence).
6. **Silence trim**: leading and trailing silence trimmed at -40 dBFS.
   Internal silences are preserved.

Output is written as 16 kHz mono `.wav` (PCM_16) to local disk for
training-time random access. Streaming directly from the original
parquet/tar files was tried first and rejected because it caused GPU
starvation on the DGX during training (parquet decode + resample is
~3× slower than reading a pre-baked wav).

## 4.2 Text preprocessing

For every reference transcript we run:

1. **NFKC Unicode normalization**.
2. **Strip control characters** (U+200B – U+200F, U+202A – U+202E).
3. **Collapse whitespace** to single spaces.
4. **Strip leading / trailing whitespace and stray punctuation** at
   the ends of utterances.

We do **not** at preprocessing time:
- Strip Tashkeel (diacritics)
- Fold hamza
- Convert Eastern digits to Western digits
- Apply any of the leaderboard normalizations

Those happen only at evaluation time inside the leaderboard normalizer
(see [06_evaluation.md](06_evaluation.md)). The reasoning: the LLM
decoder should learn to produce natural-looking text. Folding diacritics
at training time would teach the model to ignore them in output too,
which would hurt downstream usability.

## 4.3 Manifest schema

The training manifest is a JSONL file with one entry per line:

```json
{
  "audio_filepath": "data/dgx_full/preprocessed_audios/sada_000123.wav",
  "text": "النص العربي الخليجي مثال",
  "duration": 4.52
}
```

The fine-tuning code (`scripts/finetune_qwen3_lora.py`) reads this format
directly. The same schema is used for `train.jsonl`, `validation.jsonl`
and `test.jsonl`.

For the synthesis pipeline (v2 — see chapter 10) the schema is extended
with two extra fields:

```json
{
  "audio_filepath": "...",
  "text": "...",
  "duration_s": 4.52,
  "source": "sada22",
  "tier": 1
}
```

`source` is used by the rehearsal sampler for stratified subsampling and
`tier` is used by the synthesis tier-weighted sampler.

## 4.4 Train / val / test split

Stratified split by source dataset so each split sees a representative
mix of every corpus, not a random split that could leave one whole source
in test:

```
train.jsonl       ~280,000 clips  ~810h
validation.jsonl    ~3,000 clips    ~7h
test.jsonl          ~1,000 clips    ~3h
```

The same speaker never appears in more than one split where speaker IDs
are available (SADA22, OMAN-SPEECH). For the WorldSpeech parliamentary
data, speaker IDs are not reliable so episodes are used as the
speaker proxy — same episode never crosses splits.

## 4.5 Oversampling

The training loader applies fixed oversampling weights at batch
construction time, not at preprocessing time. The weights:

| Source group         | Hours | Weight |
|---|---:|---:|
| SADA22               | 667   | 1.0    |
| WorldSpeech Gulf     | 454   | 1.0    |
| OMAN-SPEECH / Ramsa  | ~100  | 1.0    |
| UAE-specific (vadimbelsky + Nexdata + Mixat) | ~150 | 3.0 |
| FLEURS ar anchor     | 10    | 0.2    |

UAE-specific data is oversampled 3× because the eval set is UAE-only.
FLEURS ar is undersampled to 0.2× so it acts as a regularization
anchor without dominating any batch.

These multipliers are applied as sampling probabilities, not as
duplication — the preprocessed file count on disk is unchanged.

## 4.6 What lives where

| Item                                | Location |
|---|---|
| Original raw datasets (parquet/tar) | `~/abder/transcription/datasets_cache/` on DGX |
| Preprocessed wav files              | `data/dgx_full/preprocessed_audios/*.wav` on DGX |
| Train manifest                      | `data/dgx_full/preprocessed_audios/splits/train.jsonl` |
| Val manifest                        | `data/dgx_full/preprocessed_audios/splits/validation.jsonl` |
| Test manifest                       | `data/dgx_full/preprocessed_audios/splits/test.jsonl` |
| Build script                        | `scripts/build_train_corpus.py` |
| Inspection summary                  | `data/dgx_full/preprocessed_audios/SUMMARY.json` |

The preprocessing script is idempotent and tracks per-source progress in
`SUMMARY.json` so re-running it after a partial failure resumes cleanly.

## 4.7 Common preprocessing pitfalls we hit and fixed

- **Stereo audio decoded as mono junk**: early runs averaged the two
  channels but skipped the dtype conversion, causing 32-bit int overflow
  on some sources. Fixed by explicit `.astype(np.float32) / 32768.0`.
- **Resampler caching**: `soxr` was creating new filter states per call,
  which dominated CPU. Fixed by reusing a single resampler instance per
  source rate.
- **WorldSpeech reference CER filter**: WorldSpeech ships per-clip CER
  between its reference text and its own ASR alignment. Setting a 0.25
  threshold dropped ~8% of clips that had clearly bad transcripts.
  Without this filter, the model picks up the misalignments.
