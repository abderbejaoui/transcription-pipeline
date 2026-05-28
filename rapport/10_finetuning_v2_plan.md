# 10 — Fine-Tuning v2 (Synthetic Medical + Mixed Corpus)

This chapter documents the v2 retraining plan: how the data is built,
what the lexicon contains, why the proportions are what they are, and
the training command that will be issued once the data is ready. The
implementation scripts are committed on branch `docs/finetuning-doc`.

## 10.1 What v1 missed and what v2 must add

From [09_failure_analysis.md](09_failure_analysis.md), the two failure
modes that survived v1:

- **Medical vocabulary**: drug brand names and clinical diagnoses are
  absent from the 900h general-Gulf training corpus. The LoRA never
  learned them.
- **Code-switching boundaries**: most of the v1 corpus is monolingual
  Arabic (parliamentary, TV, conversational); explicit AR↔EN switching
  audio is rare.

v2 adds both signals through a 150-hour mixed corpus.

## 10.2 The 150-hour data composition

| Bucket | Hours | Fraction | Source | Status |
|---|---:|---:|---|---|
| Synthetic medical Gulf | 60 | 40% | LLM-generated sentences + VoxCPM2 TTS, 10k-term lexicon, tier-weighted | Script ready, generation pending |
| Real Gulf rehearsal | 45 | 30% | Stratified sample of v1's 900h | Script ready (`scripts/sample_rehearsal.py`) |
| Real Arabic-English code-switched | 30 | 20% | MASC (HF: `pain/MASC`), filtered for code-switched segments | Download script in chapter 10.7 |
| Real English medical | 15 | 10% | PriMock57 (~9h) + Common Voice English filtered by medical vocab (~6h) | Download script in chapter 10.7 |
| **Total** | **150** | | | |

The proportions are chosen as follows:

- **40% synthetic medical** to drown the medical-vocabulary signal in
  the LoRA's training distribution. This is the primary signal we are
  adding.
- **30% rehearsal** keeps general Gulf performance intact and prevents
  catastrophic forgetting. Empirical rule of thumb for continual
  fine-tuning: rehearsal fraction should be 20–40% of the new mix.
- **20% code-switched** addresses the second failure mode (boundary
  garbage at AR↔EN switches). Real audio is preferred to synthetic
  here because code-switch prosody is hard to fake.
- **10% English medical** ensures the model still produces clean
  English drug names even when the surrounding context is English
  rather than Arabic — important because some pharmacist scripts are
  entirely in English.

Total 150h is approximately a 16% increase over v1's 900h, with the new
signal concentrated in the failing categories.

## 10.3 The 10,000-term lexicon

Built by [scripts/build_full_lexicon.py](../scripts/build_full_lexicon.py).
Outputs to `data/full_lexicon.jsonl`.

### Sources merged

| Source | Type | Count | Tier |
|---|---|---:|---:|
| Hand-curated Gulf clinic drugs | brand + generic | 355 | 1 |
| Hand-curated Gulf clinic diseases | ICD-style | 188 | 1 |
| RxNorm ingredients (TTY=IN, NLM RxNav API) | generic | ~14,000 | 2 (short names) / 3 (rest) |
| RxNorm brand names (TTY=BN, NLM RxNav API) | brand | ~5,000 | 2 / 3 |
| ICD-10-CM diagnoses (NLM clinical tables) | clinical | ~3,000 | 3 |
| Calme-augmented Gulf brand variants | brand | + a few hundred | 2 |

Total after dedup and trimming to 10,000: ~10,000 unique English terms.

### Why English-only, no Arabic aliases

The ASR's correct output for a drug name is the **English spelling**,
inside an otherwise Arabic sentence. Training on Arabic-transliterated
drug names ("بنادول" for panadol) would teach the model the wrong
behaviour. The v1 lexicon mixed Arabic aliases in; the v2 lexicon
purges them.

The trade-off: the LoRA needs to learn to *hear* the Arabic-accented
pronunciation of an English drug and produce the English spelling. The
text side stays English; the audio side is the Gulf TTS pronouncing
the English word. This is exactly the mapping we want the model to
internalize.

### Tiered sampling

Each lexicon entry carries a `tier` field. The synthesis script uses
the tier to choose how many sentences to generate for that term:

| Tier | Count | Sentences each | Audio per term (approx) | Total audio |
|---|---:|---:|---:|---:|
| 1 (hand-picked Gulf-clinic) | 543 | 60 | ~ 8 min | ~ 30h |
| 2 (common RxNorm + Calme brand augment) | ~1,500 | 12 | ~ 1.5 min | ~ 25h |
| 3 (long-tail RxNorm + ICD-10) | ~7,957 | 2 | ~ 16 s | ~ 15h |
| | | | | ~ 70h |

This is intentionally over-provisioned compared to the 60h target so
that the TTS phase has slack to drop bad outputs and still hit 60h.

## 10.4 Synthesis prompt design

In [scripts/generate_medical_training_data.py](../scripts/generate_medical_training_data.py),
the system prompt to Calme 78B forbids Arabic-script drug names with
explicit examples of correct vs wrong outputs. After the LLM returns,
the script **drops** any sentence that does not contain the English term
verbatim. Bad LLM outputs never enter the training set.

Voice prompts for VoxCPM2 cover four Gulf clinical personas:

```
Gulf Arabic male doctor, calm professional tone
Gulf Arabic male patient, casual conversational tone
Gulf Arabic female doctor, professional tone
Gulf Arabic female patient, casual worried tone
Gulf Arabic young male, nervous speaking to doctor
Gulf Arabic elderly male, slow calm speech
```

VoxCPM2 is voice-design-aware: prefixing the prompt with a parenthetical
voice description steers prosody without needing reference audio.

## 10.5 Why we do not just inflate the synthetic share to 100%

A 100% synthetic corpus would teach the model the **TTS speaker
acoustics**, not the underlying Gulf vocabulary. Real audio is required
to keep the model's encoder honest about real prosody, microphone
characteristics, and background noise. 60% real (rehearsal + code-
switched + English medical) is the floor.

## 10.6 The training command (v2)

To be run after the synthesis is complete:

```bash
python scripts/finetune_qwen3_lora.py \
    --base-model runs/qwen3_gulf_merged \         # v1 LoRA merged into base
    --train-manifest data/training/master_v2/manifest.jsonl \
    --eval-manifests \
        data/dgx_full/preprocessed_audios/splits/validation.jsonl \
        eval/casablanca_emirati_full/manifest.jsonl \
        eval/medical_transcript_eval.jsonl \
    --lora-rank 64 --lora-alpha 128 --use-rslora \
    --lora-dropout 0.05 \
    --num-epochs 3 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 16 \
    --learning-rate 1e-4 \
    --warmup-steps 500 \
    --lr-scheduler cosine \
    --eval-steps 500 \
    --output-dir runs/qwen3_gulf_med_v2 \
    --report-to tensorboard
```

Key differences from v1:
- `--base-model` points at the merged v1 LoRA (we re-LoRA on top of v1,
  not on the original base). This preserves the dialect adaptation v1
  already achieved.
- `--lora-rank 64` again — same capacity, narrower domain shift.
- `--num-epochs 3` instead of 2 — smaller corpus, more passes needed.
- Eval manifests include the medical eval set so we can watch medical
  WER specifically.

## 10.7 Where the non-synthetic data comes from

The user-facing instructions, with HuggingFace dataset names:

### Code-switched (MASC, ~30h subsample)

```python
from datasets import load_dataset
ds = load_dataset("pain/MASC", split="train", streaming=True)
# Filter for rows whose text contains ASCII alpha (a code-switch
# heuristic) and stop when 30h are collected.
```

### English medical (PriMock57 + Common Voice filtered)

```bash
git clone https://github.com/babylonhealth/primock57.git data/raw/primock57
# ~9h of mock primary-care consultations.
```

```python
from datasets import load_dataset
ds = load_dataset("mozilla-foundation/common_voice_17_0", "en",
                  split="train", streaming=True)
# Filter sentences containing any word from medical_terms.txt
# until 6h collected. Combined with PriMock57 -> ~15h.
```

### Rehearsal (already in your 900h)

```bash
python scripts/sample_rehearsal.py \
    --manifest data/dgx_full/preprocessed_audios/splits/train.jsonl \
    --out data/training/gulf_rehearsal/manifest.jsonl \
    --target-hours 45
```

The sampler is stratified by source dataset so all of SADA / WorldSpeech
/ OMAN-SPEECH / Mixat are proportionally represented.

### Master manifest

```bash
python scripts/build_master_manifest.py \
    --synthetic   data/training/medical_gulf_v2/manifest.jsonl \
    --rehearsal   data/training/gulf_rehearsal/manifest.jsonl \
    --codeswitch  data/raw/masc/manifest.jsonl \
    --english-med data/raw/primock57/manifest.jsonl \
                  data/raw/cv_medical/manifest.jsonl \
    --out         data/training/master_v2/manifest.jsonl \
    --target-hours 150 \
    --ratios 0.40 0.30 0.20 0.10
```

## 10.8 Expected outcome

We are not promising a number — the prior round (v1) had to break the
eval pipeline before we knew the real WER, and synthetic data carries
its own risks. The qualitative expectations:

- **Medical drug names**: drop from "occasional mangle into Arabic
  noun phrases" to "occasional minor misspelling".
- **General Gulf WER**: held within ±2 points of v1 (rehearsal anchor
  is sized to make this almost guaranteed).
- **Code-switching boundaries**: improvement is likely but bounded by
  what 30h of MASC can teach.
- **English medical**: same English-medical accuracy as Whisper-
  large-v3 baseline (PriMock57 published numbers).

A regression on general Gulf WER would mean the rehearsal share was
too small or the synthetic share too noisy. The mitigation is to
increase rehearsal to 50% in v2.1.

## 10.9 Current implementation status

| Component | Status |
|---|---|
| 10k-term lexicon builder | implemented (`scripts/build_full_lexicon.py`) |
| English-only synthesis prompt | implemented and committed |
| Tier-weighted sentence generation | implemented |
| TTS budget cap (70h) | implemented |
| Rehearsal sampler | implemented (`scripts/sample_rehearsal.py`) |
| Master manifest builder | implemented (`scripts/build_master_manifest.py`) |
| MASC download script | snippet in this doc, not committed |
| PriMock57 + CV medical download | snippet in this doc, not committed |
| Synthesis run on DGX | running at time of writing |
| v2 training launch | pending synthesis completion |

The branch `docs/finetuning-doc` has all the code that is implemented.
