# Qwen3-ASR-1.7B LoRA Fine-Tune — DGX Spark Runbook

Run order, exact commands, expected timings, kill criteria. Read top to bottom.

## 0. Preconditions (verify once)

```bash
cd /workspace/test_sound_pipeline

# 1. Already-validated eval works
python -m scripts.eval_arabic --hyp eval/casablanca_UAE/predictions/qwen3.jsonl \
    --ref  eval/casablanca_UAE/manifest.jsonl
# Expected: WER ~67.67%  CER ~22.29%

# 2. Install the OFFICIAL Qwen3-ASR wrapper + PEFT + I/O deps
pip install -U qwen-asr datasets
pip install 'peft>=0.13.0' 'soxr>=0.4.0' 'soundfile>=0.12.1' \
            'pyarrow>=17.0.0' 'librosa>=0.10.0'
# FlashAttention 2 (optional, but speeds training ~30%)
pip install -U flash-attn --no-build-isolation

# 3. HF token for gated datasets
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx
# (accept terms on the dataset pages first — see SOURCES in build_train_corpus.py)

# 4. SANITY: inspect Qwen3-ASR module layout so the LoRA target list is correct
python -m scripts.inspect_qwen3_modules
# Expected output:
#   audio_tower: ~96 Linear modules   <-- frozen, NOT touched
#   language_model: ~200 Linear modules
#   [summary] LoRA targets ... ~196 modules
# If language_model count is 0, STOP — model layout changed upstream and
# you must update DEFAULT_LORA_TARGET_SUFFIXES in finetune_qwen3_lora.py.
```

## 1. Build the training corpus

Smoke test first to make sure auth + decoding work:

```bash
python -m scripts.build_train_corpus \
    --sources SADA22 \
    --max-clips 1000
```

Then the real run:

```bash
python -m scripts.build_train_corpus 2>&1 | tee logs/build_corpus.log
```

Expected: ~1.4M clips, ~1,480 hours, ~180 GB. Watch `SUMMARY.json` for hours/source.

**Kill criteria**: any source returns < 50% of the planned hours → check
its access (gated terms, parquet path).

## 2. Round-1 LoRA — full corpus

```bash
mkdir -p runs/qwen3_lora_r1
python -m scripts.finetune_qwen3_lora \
    --model-path Qwen/Qwen3-ASR-1.7B \
    --train-manifest data/train_corpus/manifest.jsonl \
    --eval-manifests eval/casablanca_UAE/manifest.jsonl eval/bakeoff_30min/manifest.jsonl \
    --output-dir runs/qwen3_lora_r1 \
    --num-epochs 3 \
    --learning-rate 1e-4 \
    --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 16 \
    --eval-every-steps 2000 \
    2>&1 | tee logs/round1.log
```

**Watch**: the `[eval-cb step=...]` lines. Casablanca-UAE WER should drop
from 67.67% → expect mid-50s by step 10 000, mid-40s by end of epoch 3.

**Kill criteria**:
- Loss is NaN → drop LR to 5e-5
- WER goes *up* between two consecutive evals → reduce LR by 2× and restart
  from the last checkpoint
- VRAM OOM → drop `per-device-train-batch-size` to 2, raise
  `gradient-accumulation-steps` to 32

## 3. Mine hard examples for Round 2

```bash
python -m scripts.mine_hard_examples \
    --adapter runs/qwen3_lora_r1/final_adapter \
    --train-manifest data/train_corpus/manifest.jsonl \
    --output-manifest data/train_corpus/hard_manifest.jsonl \
    --min-wer 0.30 \
    --max-keep 150000 \
    2>&1 | tee logs/mine_hard.log
```

Expected: ~10–20% of training clips retained.

## 4. Round-2 LoRA — hard examples only

```bash
mkdir -p runs/qwen3_lora_r2
python -m scripts.finetune_qwen3_lora \
    --model-path Qwen/Qwen3-ASR-1.7B \
    --train-manifest data/train_corpus/hard_manifest.jsonl \
    --eval-manifests eval/casablanca_UAE/manifest.jsonl eval/bakeoff_30min/manifest.jsonl \
    --output-dir runs/qwen3_lora_r2 \
    --num-epochs 2 \
    --learning-rate 5e-5 \
    --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 16 \
    --eval-every-steps 1000 \
    2>&1 | tee logs/round2.log
```

Note: LR is half (5e-5) — we're refining, not exploring.

## 5. Merge Round 1 + Round 2

```bash
python -m scripts.merge_adapters \
    --adapters runs/qwen3_lora_r1/final_adapter runs/qwen3_lora_r2/final_adapter \
    --weights 0.7 0.3 \
    --output runs/qwen3_lora_ilt \
    2>&1 | tee logs/merge.log
```

## 6. Final eval on all test sets

```bash
python -m scripts.compare_models \
    --backends qwen3 qwen3_ilt \
    --manifests eval/casablanca_UAE/manifest.jsonl eval/bakeoff_30min/manifest.jsonl \
    --score-only
```

(After registering the merged adapter as `qwen3_ilt` in `scripts/bakeoff.py`.)

---

# Phase 2: Medical domain adaptation

After Phase 1 lands, the model speaks Gulf Arabic. Phase 2 makes it speak
*clinic* Gulf Arabic. We do this with **text-only decoder adaptation** —
no audio needed. The medical LoRA stacks on top of Phase 1 by adapter merge.

## 7. Build the Arabic medical text corpus

The corpus is built in **two steps**, in order:

### 7a. Scrape public Arabic medical sources

```bash
# Full run: ~30-50 MB of Arabic Wikipedia medical-category articles
# (~10-15k articles) + ~3-5 MB of Wikidata Arabic labels for
# drugs/diseases/symptoms/anatomy. Official APIs only, polite throttling,
# takes ~30-60 min.
python -m scripts.scrape_medical_text \
    --arwiki-depth 3 \
    --arwiki-max-articles 15000 \
    --wikidata-limit-per-class 15000 \
    2>&1 | tee logs/scrape_medical.log

# Verify what landed
ls -la data/medical_text/external/
cat data/medical_text/external/SCRAPE_SUMMARY.json
```

For a fast smoke test first, run with tiny caps:

```bash
python -m scripts.scrape_medical_text \
    --arwiki-depth 1 \
    --arwiki-max-articles 50 \
    --wikidata-limit-per-class 200
```

### 7b. Mix scraped data + curated seeds + synthetic templates

```bash
python -m scripts.build_medical_text \
    --n-templated 200000 \
    --external-dirs data/medical_text/external \
    --output-dir data/medical_text \
    2>&1 | tee logs/build_medical_text.log
```

Final `data/medical_text/corpus.jsonl` will contain:
- ~75 UAE drugs × brand names × dose templates (curated seeds)
- ~90 ICD-coded diseases + ~60 symptoms (curated seeds)
- ~200k clinical-Arabic templated sentences
- ~10-15k Wikipedia paragraphs (scraped)
- ~30-50k Wikidata Arabic medical entity labels (scraped)
- **Total: ~250k sentences ≈ 10-15M tokens ≈ 50-80 MB**

Inspect `SUMMARY.json` for source breakdown before training.

**Honest expectations by corpus size**:

| Corpus tokens | What the decoder learns | Medical WER impact |
|--------------:|-------------------------|-------------------:|
| <1M | Almost nothing | negligible |
| 5-10M | Common drug names spelled right | ~3-5% relative |
| 10-15M (this setup) | Drug + disease + clinical syntax | **~8-15% relative** |
| 50-100M | Strong memorization | ~20-30% relative |
| 200M+ | Diminishing returns | saturates |

To go beyond ~15M tokens later, drop more `.jsonl` (with a `text` field) or
`.txt` files into `data/medical_text/external/` and re-run step 7b. The
builder will auto-split long paragraphs into sentences.

## 8. Text-only decoder LoRA

```bash
mkdir -p runs/qwen3_medical_text
python -m scripts.finetune_decoder_text \
    --model-path Qwen/Qwen3-ASR-1.7B \
    --train-corpus data/medical_text/corpus.jsonl \
    --output-dir runs/qwen3_medical_text \
    --num-epochs 2 \
    --learning-rate 1e-4 \
    --block-size 1024 \
    --per-device-train-batch-size 8 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee logs/medical_text.log
```

This is fast — text-only, no audio I/O, ~2-6 hours on the DGX depending
on corpus size.

**Watch**: training loss should drop ~0.5-1.0 nats. If it stays flat the
corpus is too small or the LR too low.

## 9. Merge Phase-1 ASR adapter + medical decoder adapter

```bash
python -m scripts.merge_adapters \
    --adapters runs/qwen3_lora_ilt runs/qwen3_medical_text/final_adapter \
    --weights 0.6 0.4 \
    --output runs/qwen3_gulf_medical \
    2>&1 | tee logs/merge_medical.log
```

Weights tuning:
- `0.7 / 0.3` → safer: keeps Phase-1 dialect strength, milder medical shift
- `0.6 / 0.4` → **default**: balanced
- `0.5 / 0.5` → aggressive: stronger medical bias, may hurt general WER

## 10. Eval the medical-adapted model

Build a small Gulf medical eval set first (50-200 clips of clinical audio
— from your customer/partner). Drop it at
`eval/gulf_medical/manifest.jsonl`. Then:

```bash
python -m scripts.compare_models \
    --backends qwen3 qwen3_ilt qwen3_gulf_medical \
    --manifests eval/casablanca_UAE/manifest.jsonl eval/gulf_medical/manifest.jsonl \
    --score-only
```

You want to see:
- `qwen3_gulf_medical` < `qwen3_ilt` < `qwen3` on `eval/gulf_medical`
- `qwen3_gulf_medical` ≈ `qwen3_ilt` on `eval/casablanca_UAE`
  (no significant regression on general Gulf speech)

If the medical adapter *hurts* general Gulf WER by >2 pts, drop merge
weight to 0.7/0.3 or shrink the medical corpus.

## Realistic outcome bands

| Stage                                        | General Gulf WER | Medical Gulf WER |
|----------------------------------------------|-----------------:|-----------------:|
| Base Qwen3-ASR-1.7B (measured)               |           67.67% |          ~70–80% |
| After Round-1 LoRA                           |           50–58% |          ~60–70% |
| After Round-2 LoRA + merge (ILT)             |           46–54% |          ~55–65% |
| + Contextual biasing at inference            |           43–52% |          ~45–55% |
| + Text-only medical decoder adapt (Phase 2)  |           42–50% |          **35–50%** |
| + Real Gulf medical audio (5-20h, future)    |           42–48% |          **22–35%** |

If you see < 40% you are in unprecedented territory — verify there's no
test leak.

---

# Appendix A: Gulf code-switch data pipeline (real audio only)

These four scripts replace the legacy `build_train_corpus` flow when you
want a curated, **leakage-safe**, real-audio-only Gulf code-switch corpus.
No synthetic data. See `DATASETS.md` for the full inventory, licenses, and
the two-stage curriculum rationale.

## A1. Prepare datasets from Hugging Face

```bash
# List the registered (ungated, loadable) Gulf datasets:
python -m scripts.prepare_datasets --list

# Prepare one, capped for a smoke test:
python -m scripts.prepare_datasets --dataset mixat --max-clips 200

# Prepare everything for a stage:
python -m scripts.prepare_datasets --stage 1 --all      # broad Gulf base
python -m scripts.prepare_datasets --stage 2 --all      # code-switch focus
```

Each dataset lands at `data/preprocessed/<slug>/manifest.jsonl` with 16 kHz
mono WAVs. Manifest schema:
`{audio_path, text, source, dialect, code_switch, weight, stage}`.

The script **refuses** anything in `SYNTHETIC_BLOCKLIST`
(e.g. `vadimbelsky/uae_arabic_english_bilingual_dataset_40k`).

## A2. Mine extra code-switch rows from existing manifests

```bash
python -m scripts.mine_code_switch \
    --in data/preprocessed/*/manifest.jsonl \
    --out data/splits/mined_cs.jsonl \
    --weight 3.0 --min-latin-tokens 1 --stage 2
```

Emits a Stage-2, up-weighted, CS-only manifest. Stage-2 up-weighting needs
**no code change** — it is just the `weight` field consumed by the weighted
sampler in `finetune_qwen3_lora.py`.

## A3. Split into disjoint train / val (leakage guard)

`finetune_qwen3_lora.py` takes explicit `--train-manifest` and
`--eval-manifests`; it does NOT split internally. Use this to carve a
held-out val set deterministically:

```bash
python -m scripts.split_manifest \
    --in data/preprocessed/*/manifest.jsonl \
    --out-prefix data/splits/gulf \
    --val-frac 0.05 --stratify-by dialect --dedup-text
# -> data/splits/gulf.train.jsonl  data/splits/gulf.val.jsonl
```

It is seeded (reproducible), stratifies by a field so every bucket appears
on both sides, can dedup by transcript text, and **aborts** if the two sides
share any clip (prints `leakage=0` on success).

## A4. Two-stage curriculum run

Stage 1 (broad Gulf), then resume into Stage 2 (code-switch up-weighted):

```bash
# Stage 1
python -m scripts.finetune_qwen3_lora \
    --model-path Qwen/Qwen3-ASR-1.7B \
    --train-manifest data/splits/stage1.train.jsonl \
    --eval-manifests data/splits/stage1.val.jsonl eval/casablanca_UAE/manifest.jsonl \
    --output-dir runs/qwen3_stage1 \
    --lora-r 64 --lora-alpha 128 --use-rslora \
    --lr-scheduler-type cosine --warmup-steps 200 \
    2>&1 | tee logs/stage1.log

# Stage 2 — resume from Stage 1, code-switch manifest is up-weighted
python -m scripts.finetune_qwen3_lora \
    --model-path Qwen/Qwen3-ASR-1.7B \
    --train-manifest data/splits/stage2.train.jsonl \
    --eval-manifests data/splits/stage2.val.jsonl eval/casablanca_UAE/manifest.jsonl \
    --output-dir runs/qwen3_stage2 \
    --resume-from-checkpoint runs/qwen3_stage1/final_adapter \
    --lora-r 64 --lora-alpha 128 --use-rslora \
    --learning-rate 5e-5 --lr-scheduler-type cosine --warmup-steps 100 \
    2>&1 | tee logs/stage2.log
```

## A5. Held-out evaluation

```bash
python -m scripts.test_asr \
    --model-path Qwen/Qwen3-ASR-1.7B \
    --adapter runs/qwen3_stage2/final_adapter \
    --manifest data/splits/stage2.val.jsonl eval/casablanca_UAE/manifest.jsonl \
    --breakdown
```

`test_asr.py` imports the *exact* inference path used during training
(`_run_eval`, `_build_prefix_messages`), so its WER/CER match the
in-training eval numbers. `--breakdown` buckets by source / dialect /
code_switch. Add `--fast` for a quick comparable single number.

## New finetune CLI flags (teacher recommendations)

| Flag | Default | Purpose |
|---|---|---|
| `--use-rslora` | off | rank-stabilized LoRA scaling (recommended at r=64) |
| `--use-dora` | off | DoRA (weight-decomposed LoRA); ~+39% train time |
| `--unfreeze-encoder-layers N` | 0 | LoRA-adapt the last N audio-tower blocks |
| `--encoder-lora-lr LR` | `learning_rate*0.1` | separate (lower) LR for encoder LoRA params |
| `--lr-scheduler-type` | linear | e.g. `cosine` |
| `--warmup-steps N` | 0 | absolute warmup; overrides `--warmup-ratio` when >0 |

Encoder LoRA params (names containing `audio_tower`) are routed to their own
lower-LR optimizer group automatically when `--unfreeze-encoder-layers > 0`.
The audio-tower *base* weights stay frozen — only the adapters train.

## What is intentionally NOT in this runbook

- Synthetic medical TTS training data — vadimbelsky proved this fails
  (25.58% CER on real Casablanca-UAE). We use text-only decoder adaptation
  + hotword biasing instead. That's Phase 2, after Round-1 results land.
- MAS-LoRA dialect experts (SA vs UAE separate adapters) — useful only if
  Round 1 shows the model is forgetting one dialect to learn the other.
  Decide post-Round-1 from the per-source eval breakdown.
- Encoder fine-tuning — frozen by design.
