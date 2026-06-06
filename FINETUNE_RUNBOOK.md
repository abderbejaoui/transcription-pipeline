# Qwen3-ASR-1.7B LoRA Fine-Tune — DGX Spark Runbook

Run order, exact commands, expected timings, kill criteria. Read top to bottom.

> **⚠️ AUTHORITATIVE SECTION BELOW (`A`).** Sections 0–7 further down are the
> OLD `build_train_corpus` plan and are kept only for the Phase-2 *medical-text*
> recipe. For the current dataset pipeline (prepare → split → smoke → train →
> test) follow **Part A** start to finish. The `*_corpus` commands in §0–§2 are
> SUPERSEDED by Part A.

---

# Part A — Dataset pipeline (CURRENT, verified 2026-06-06)

Host: DGX Spark `spark-a6f4`. Repo: `/home/abder/abder/transcription/transcription-pipeline`.
Python: `.venv` (python3.12). Run long jobs inside the `qwen3` tmux session.
`transformers` is PINNED to 4.57.6 — **never** `pip install -U transformers`.

**Final dataset decision (V2 "max data", verified 2026-06-06):**

Phase 1 = **maximum real Gulf/Arabic acoustic** from Qwen3 BASE. Phase 2 = 20h
synthetic medical CS + ALL code-switch sets + ~100h rehearsal **carved out of**
the Phase-1 pool (so the same audio is never trained twice across phases).

| Dataset | Slug(s) | Hrs | Role | In training pool? |
|---|---|---|---|---|
| your existing 804h corpus (already **contains SADA**) | — | ~804 | Phase-1 base | ✅ yes |
| WorldSpeech Bahrain | `worldspeech_bh` | 272.5 | Phase-1 base | ✅ NEW |
| WorldSpeech Kuwait | `worldspeech_kw` | 175.5 | Phase-1 base | ✅ NEW |
| WorldSpeech Saudi | `worldspeech_sa` | 6.1 | Phase-1 base | ✅ NEW |
| WorldSpeech UN (MSA anchor) | `worldspeech_un` | 11.1 | Phase-1 (weight 0.3) | ✅ NEW |
| MASC (clean, `type='c'`) | `masc` | ~1000* | Phase-1 base (weight 0.7) | ✅ NEW |
| `mixat` (15h Emirati-Eng CS) | `mixat` | 15 | Phase-2 code-switch | ✅ yes |
| `sada22` | `sada22` | — | re-prep/verify only — **already in 804h, do NOT add hours** | optional |
| `scc22` | `scc22` | ~5 | held-out Saudi-Eng CS benchmark | ❌ eval-only |
| `casablanca` (Emirati) | — | — | held-out benchmark | ❌ eval-only |
| `sawtarabi` | `sawtarabi` | — | **ELIMINATED** (no card) | ❌ disabled |
| `emirati_shows` | `emirati_shows` | — | **ELIMINATED** (~0.5h) | ❌ disabled |
| OMAN-SPEECH | `oman_speech` | ~40 | paper-only, **NOT on HF** — needs a local loader | ❌ disabled stub |
| `ADI17` | — | — | **ELIMINATED** (no transcripts) | ❌ never |

\* MASC is multi-dialect pan-Arabic, gated by `type='c'` to clean clips only and
down-weighted to 0.7 as an MSA/pan-Arabic anchor (not Gulf-specific).

WorldSpeech and MASC are **gated** — accept the terms on each dataset page (and
set `HF_TOKEN`) before A2. WorldSpeech rows are quality-gated by `cer ≤ 0.25`;
MASC by `type == 'c'`. Both filters are built into `prepare_datasets.py`.

`scc22`/`casablanca` are tagged `eval_only:true` in their manifests, and
`split_manifest.py` drops any `eval_only` row from the train/val split, so they
**cannot** leak into training.

## A0. One-time preflight (cheap, do this first)

```bash
cd /home/abder/abder/transcription/transcription-pipeline
git pull --no-rebase --no-edit origin main
source .venv/bin/activate

# 0a. Confirm the pin is intact (MUST print 4.57.6)
python -c "import transformers; print('transformers', transformers.__version__)"

# 0b. Confirm the registry reflects the V2 audit. ACTIVE: mixat, scc22, sada22,
#     worldspeech_bh/kw/sa/un, masc. DISABLED: sawtarabi, emirati_shows,
#     oman_speech.
python scripts/prepare_datasets.py --list

# 0b2. Accept gated-dataset terms ONCE in a browser, then export your token:
#        - https://huggingface.co/datasets/disco-eth/WorldSpeech  (click Agree)
#        - https://huggingface.co/datasets/pain/MASC              (click Agree)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx

# 0c. Confirm the LoRA target layout is intact.
#     Decoder lives at thinker.model.layers.* (there is NO `language_model`
#     name in this build). Expect:  decoder: ~197 Linear ... LoRA targets ~196.
#     If [summary] prints 0 modules, STOP — the layout changed upstream.
python -m scripts.inspect_qwen3_modules
```

## A1. Smoke prep (download + preprocess, capped) — proves decoding works

```bash
# CUDA_VISIBLE_DEVICES="" avoids the harmless torchcodec/CUDA teardown
# core-dump after "[prep] done". HF_HUB_DOWNLOAD_TIMEOUT guards flaky pulls.
HF_HUB_DOWNLOAD_TIMEOUT=30 CUDA_VISIBLE_DEVICES="" \
  python -m scripts.prepare_datasets --all --max-clips 200
```

**PASS criteria:** for each active set you see
`wrote N clips ... decode_fail=0` with N>0, and
`SKIP emirati_shows / sawtarabi / oman_speech`. If any active set writes 0
clips, read the `first-row keys:` line it printed and fix `text_keys`/`audio_key`
before going further. **Do not proceed past a 0-clip set.**

Extra checks for the new gated sets:
- `worldspeech_*`: the skip line shows `cer=K` (rows dropped by the `cer≤0.25`
  gate). Some drops are expected; if `wrote=0` and `cer=` is huge, the `cer`
  field name changed — inspect the dumped first-row keys.
- `masc`: the skip line shows `type=K` (noisy `type='n'` rows dropped). If it
  errors with a `trust_remote_code`/script message, the loader script changed;
  fall back to a parquet/raw download.

## A2. Real prep (full, no cap) — run in tmux

```bash
tmux new -s prep    # or: tmux attach -t qwen3
HF_HUB_DOWNLOAD_TIMEOUT=30 CUDA_VISIBLE_DEVICES="" \
  python -m scripts.prepare_datasets --all 2>&1 | tee logs/prep_all.log
# detach: Ctrl-b d   ;   reattach: tmux attach -t prep
```

Check each `data/preprocessed/<slug>/summary.json` for `clips` and
`decode_fail`. `decode_fail` should be ~0.

## A3. Build the disjoint Phase-1 split AND carve the Phase-2 rehearsal pool

One command does the whole split: it (1) drops `eval_only` rows, (2) makes a
disjoint train/val split, and (3) **carves ~100h out of train** into a separate
Phase-2 rehearsal manifest (so those clips are removed from Phase-1 train). The
carve is proportional across `--stratify-by` so the rehearsal pool stays as
diverse as the corpus. Only `worldspeech_*`, `masc`, `sada22` and your 804h
manifest are real Phase-1 acoustic — point the glob at those, **not** `mixat`
(which is Phase-2 CS) and **not** the eval-only sets.

```bash
mkdir -p data/splits
python scripts/split_manifest.py \
    --in <PATH_TO_YOUR_804h_MANIFEST> \
         data/preprocessed/worldspeech_bh/manifest.jsonl \
         data/preprocessed/worldspeech_kw/manifest.jsonl \
         data/preprocessed/worldspeech_sa/manifest.jsonl \
         data/preprocessed/worldspeech_un/manifest.jsonl \
         data/preprocessed/masc/manifest.jsonl \
    --out-prefix data/splits/phase1 \
    --val-frac 0.02 \
    --stratify-by source \
    --dedup-text \
    --carve-hours 100 \
    --carve-out data/splits/phase2_rehearsal.jsonl
```

**PASS criteria — the last lines MUST read:**
```
[split] excluded <K> eval_only (held-out benchmark) row(s) from the train/val split
[split] carved <C> clips (~100.0h, target 100.0h) into Phase-2 rehearsal -> data/splits/phase2_rehearsal.jsonl
[split] train=<N>  val=<M>  carved=<C>  (val_frac=0.0x of train+val, leakage=0)
```
`leakage=0` is non-negotiable. The carved clips are tagged `rehearsal:true` /
`stage:2` and are **not** in `phase1.train.jsonl`. Writes:
`data/splits/phase1.train.jsonl`, `data/splits/phase1.val.jsonl`,
`data/splits/phase2_rehearsal.jsonl`.

> Hour accounting uses the per-clip `duration` written by `prepare_datasets.py`.
> If your 804h manifest predates the `duration` field, rows without it fall back
> to `--default-clip-sec` (8.0s); re-prep or pass a closer estimate so the 100h
> carve is accurate.

## A4. ⭐ Validation smoke test — PROVE eval works BEFORE the long run

This is the step that was broken in your previous fine-tune. `--eval-at-start`
runs the held-out eval at **step 0**, and `--max-steps 6` exercises the full
train→eval→save path in a couple of minutes.

```bash
python -m scripts.finetune_qwen3_lora \
    --model-path Qwen/Qwen3-ASR-1.7B \
    --train-manifest data/splits/phase1.train.jsonl \
    --eval-manifests data/splits/phase1.val.jsonl \
    --output-dir runs/smoke \
    --max-steps 6 \
    --eval-at-start \
    --eval-max-samples 8 \
    --early-stopping-patience 0 \
    2>&1 | tee logs/smoke.log
```

**PASS criteria — you MUST see, at step 0, a line like:**
```
[eval-cb] eval-at-start: baseline held-out eval (step 0)
[eval-cb step=0] phase1.val.jsonl: WER=XX.XX%  CER=YY.YY%  n=8
```
- `n=8` (NOT `n=0`) and **WER is a real number, not `nan`**. If WER is `nan`
  or `n=0`, the eval path is broken — STOP and fix before any long run.
- Training then runs 6 steps and saves. If this whole block is green, the
  validation logic is proven and you can launch the real run with confidence.

## A5. Phase 1 — base acoustic LoRA (real-only, from Qwen3 BASE fresh)

Phase 1 trains MAX real Gulf/Arabic acoustic from the **base** model (not your
old 804h checkpoint) on the **carved** train manifest from A3 (the 100h
rehearsal has already been removed for Phase 2).

**Teacher recommendations baked in:** DoRA first (`--use-dora`); rank-stabilised
scaling (`--use-rslora`); a short warmup + cosine schedule; `q/k/v/o/gate/up/down`
LoRA targets (default); the audio tower stays **frozen** in Phase 1.

```bash
mkdir -p runs/phase1
python -m scripts.finetune_qwen3_lora \
    --model-path Qwen/Qwen3-ASR-1.7B \
    --train-manifest data/splits/phase1.train.jsonl \
    --eval-manifests data/splits/phase1.val.jsonl \
                     data/preprocessed/scc22/manifest.jsonl \
    --output-dir runs/phase1 \
    --num-epochs 3 \
    --learning-rate 1e-4 \
    --lr-scheduler-type cosine --warmup-ratio 0.02 --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
    --use-dora --use-rslora \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 16 \
    --eval-every-steps 2000 \
    --eval-at-start \
    --early-stopping-patience 3 --early-stopping-metric wer \
    --gradient-checkpointing \
    --save-total-limit 5 \
    2>&1 | tee logs/phase1.log
```

> **Teacher's higher-capacity variant (optional, if Phase-1 WER plateaus):**
> bump capacity and LR — `--lora-r 64 --lora-alpha 128 --learning-rate 3e-4`.
> Try the conservative `r32/α64/1e-4` settings first; only escalate to
> `r64/α128/3e-4` if the held-out WER stalls. Do **not** change rank mid-run —
> start a fresh Phase 1.

> **Encoder unfreeze (advanced, only after a good frozen-encoder Phase 1):**
> add `--unfreeze-encoder-layers 4 --encoder-lora-lr 1e-5` to adapt the top of
> the audio tower. This is the teacher's "stage 2 of curriculum within Phase 1"
> — keep the encoder LR ~10× lower than the decoder LR.

The FIRST `--eval-manifests` entry (`phase1.val`) drives early stopping; the
second (`scc22`, eval-only) is a held-out generalisation read.
Best adapter is saved to `runs/phase1/best_adapter` on every improvement.

**Kill criteria:** loss `NaN` → LR 5e-5; WER rises 2 evals running → halve LR,
resume from last checkpoint; OOM → batch-size 2 / grad-accum 32.

## A6. Phase 2 — medical CS mixed with real rehearsal (resume from Phase 1)

Phase 2 = **synthetic medical CS MIXED with real data** (never synthetic-only).
The mix is:

| Component | Source | ~Hours |
|---|---|---|
| synthetic medical code-switch | your synthetic generator | ~20 |
| ALL real code-switch sets | `mixat` (+ any mined CS) | ~15–25 |
| Gulf rehearsal **carved from Phase 1** | `data/splits/phase2_rehearsal.jsonl` | ~100 |

Resume from the **NEW** Phase-1 checkpoint (`runs/phase1/best_adapter`).

```bash
# 1) Build the Phase-2 mixed manifest = synthetic medical CS + all CS sets +
#    the carved 100h rehearsal (from A3). Just concatenate the manifests:
mkdir -p data/splits
cat data/preprocessed/synthetic_medical_cs/manifest.jsonl \
    data/preprocessed/mixat/manifest.jsonl \
    data/splits/phase2_rehearsal.jsonl \
    > data/splits/phase2_mixed.train.jsonl
wc -l data/splits/phase2_mixed.train.jsonl   # sanity: sum of the three

# 2) Train (resume from Phase-1 best adapter; lower LR; DoRA kept on).
mkdir -p runs/phase2
python -m scripts.finetune_qwen3_lora \
    --model-path Qwen/Qwen3-ASR-1.7B \
    --resume-from-checkpoint runs/phase1/best_adapter \
    --train-manifest data/splits/phase2_mixed.train.jsonl \
    --eval-manifests data/splits/phase1.val.jsonl \
                     data/preprocessed/scc22/manifest.jsonl \
    --output-dir runs/phase2 \
    --num-epochs 2 \
    --learning-rate 5e-5 \
    --lr-scheduler-type cosine --warmup-ratio 0.03 --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 --use-dora --use-rslora \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 16 \
    --eval-every-steps 1000 \
    --eval-at-start \
    --early-stopping-patience 3 --early-stopping-metric wer \
    --gradient-checkpointing \
    2>&1 | tee logs/phase2.log
```

> The ~100h rehearsal is what prevents catastrophic forgetting of Phase-1 Gulf
> acoustic while the model learns medical code-switch. Because those clips were
> **removed** from `phase1.train.jsonl` in A3, no audio is trained in both
> phases. If you instead want CS clips up-weighted in the sampler, raise their
> `weight`/`cs_weight` in `prepare_datasets.py` before re-prepping `mixat`.

## A7. Final held-out test (the real WER/CER numbers)

```bash
# Full-set eval (no sampling cap) on the held-out benchmarks.
python scripts/test_asr.py \
    --model-path Qwen/Qwen3-ASR-1.7B \
    --adapter runs/phase2/best_adapter \
    --manifest data/preprocessed/scc22/manifest.jsonl \
    --breakdown \
    --out eval_results/phase2_scc22.json

# (Repeat for any casablanca Emirati eval manifest once prepared.)
```

`--breakdown` prints per-source / per-dialect / CS-vs-non-CS WER+CER.

---

# Part B — Legacy plan (medical-text Phase 2 reference only)

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

## Phase 1 vs Phase 2 (what trains on what)

- **Phase 1 (this runbook):** real-audio acoustic finetune (~1,900h Gulf +
  code-switch, Stage 1 then Stage 2). NO synthetic. Dominant-impact run.
- **Phase 2 (after Phase 1 WER lands):** medical-vocabulary stage. Keeps the
  21h synthetic medical Gulf data but **mixed, never synthetic-only** — Arm B:
  `merge_and_unload` Phase 1 into the base, then a fresh medical LoRA on a
  shuffled, rehearsal-heavy manifest (synthetic 21h + Gulf rehearsal +
  code-switch + english-medical) at low LR, plus a stock-base control arm.
  See `V2_TRAINING_PLAN.md` and `DATASETS.md` for the mix ratios.

## What is intentionally NOT in this runbook

- Synthetic-**only** medical TTS acoustic training — tried, regressed
  (25.58% CER on real Casablanca-UAE). Phase 2 instead **mixes** the 21h
  synthetic with real rehearsal (Arm B, above) and leans on hotword biasing
  for the long-tail drug names.
- MAS-LoRA dialect experts (SA vs UAE separate adapters) — useful only if
  Round 1 shows the model is forgetting one dialect to learn the other.
  Decide post-Round-1 from the per-source eval breakdown.
- Encoder fine-tuning — frozen by design.
