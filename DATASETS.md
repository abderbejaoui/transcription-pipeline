# Datasets

End-to-end Gulf Arabic ASR fine-tuning of **Qwen3-ASR-1.7B**.

**Hard constraints**
- **Phase 1 (acoustic) is real recorded audio only — NO synthetic / TTS / spliced data.**
- Research-only: non-commercial licenses (CC-BY-NC-SA / OpenRAIL-M) are acceptable.
- Gulf-focused (Emirati / Saudi / Khaliji). Non-Gulf Arabic only as robustness padding.

## Two-phase training plan (Phase 1 vs Phase 2)

**These are separate from the Stage 1 / Stage 2 *curriculum* inside Phase 1.**
A *phase* is a whole training run; a *stage* is the curriculum order within Phase 1.

| | **Phase 1 — real acoustic finetune** | **Phase 2 — medical vocabulary** |
|---|---|---|
| Data | ~1,900h **real** Gulf/Arabic (Stage 1 base + Stage 2 code-switch) | 21h **synthetic** medical Gulf **+** real pure-Gulf rehearsal (sampled from Phase 1: SADA22 etc.) + code-switch + english-medical |
| Teaches | How the model *hears*: dialect, accent, ar↔en code-switch | Medical **spellings / vocabulary** (tier-1 drugs) |
| Impact | **Dominant** — shapes core capability, broad WER gain | Narrow vocab slice on top |
| Synthetic? | **No** | **Yes — but mixed, never synthetic-only** |
| Order | Run first | Run after Phase 1 WER numbers land |

**Why Phase 2 keeps the synthetic data but mixes it (Arm B):** training a LoRA on
synthetic-only audio was tried and regressed badly (≈25.58% CER on real Casablanca–UAE)
because the adapter over-fits the TTS acoustic fingerprint. The fix is **not** to drop
the data but to (a) **merge Phase 1 → base (`merge_and_unload`)** so Gulf skill can't be
forgotten, then train a **fresh** medical LoRA on a **shuffled, rehearsal-heavy** manifest
where the 21h synthetic is a *minority* of acoustic hours, at low LR. Keep a control arm
(fresh medical LoRA on stock base) for comparison. The ~9,700-name drug tail the 21h can't
cover is handled by **inference-time hotword biasing + `MedicalCorrector`**, not training.

---

The data feeds a **two-stage curriculum** (teacher's *Change Five — Staged Training*):

- **Stage 1 — general / easier:** pooled MSA + high-resource Gulf acoustic. Teaches
  broad Gulf phonetics and dialect coverage.
- **Stage 2 — harder / specific:** pure dialect + English↔Arabic code-switch,
  **up-weighted**. Specializes the Stage-1 checkpoint.

Run as two passes of [`scripts/finetune_qwen3_lora.py`](scripts/finetune_qwen3_lora.py)
with `--resume-from-checkpoint`. Up-weighting needs **no code change** — set a higher
`weight` per record and the existing `build_weighted_sampler` handles it.
Expected gain from staging: **~3-8% relative WER**.

---

## 1. Existing corpus - First fine-tuning (real, no synthetic)

The May 22 2026 run trained on **~314,000 clips / 804 hours** of real Gulf Arabic
(r=64, alpha=128, rsLoRA, frozen encoder, 2-epoch budget + WER early stopping).

| Source | What it is | Dialect | Transcripts | Stage |
|---|---|---|---|---|
| SADA22 | Saudi broadcast / read speech | Saudi Khaliji | yes | 1 |
| WorldSpeech corpus | Multi-source Arabic speech | Mixed Gulf | yes | 1 |
| MixAT (`sqrk/mixat-tri`) | Emirati-English **code-switch**, 15h | Emirati | yes | 2 |
| Custom dialect collections | Own mined Gulf clips | Emirati / Saudi / Kuwaiti | yes | 1 |

**Total existing: ~804h** (of which MixAT ~15h is dedicated code-switch).

---

## 2. New findings to add (all real audio, free / un-gated unless noted)

### 2a. Code-switch - priority gap (Stage 2)

| Dataset | Hours | Dialect | License / Access | Status |
|---|---|---|---|---|
| MixAT | 15h | Emirati CS | CC-BY-NC-SA | already in 804h |
| **ZAEBUC-Spoken** | 12h | Gulf AR/EN/MSA CS | CC-BY-NC-SA, gated form | add - best new CS |
| **Casablanca (UAE)** | ~6h | Emirati CS | OpenRAIL-M, `UBC-NLP/Casablanca` | add (also eval baseline) |
| **Saudilang SCC** (`MohamedRashad/SCC22`) | 5h | Saudi/Gulf CS | CC-BY-NC-SA, **ungated** | add - move test to train |
| Own-mined CS | TBD | Gulf | - | mine from 804h (Latin-token regex) |

**Real Gulf code-switch after additions: ~38h** (+ own-mined).
Optional non-Gulf CS padding: ArzEn ~12h (Egyptian), ESCWA ~2.8h (MSA).

### 2b. Plain Gulf / Arabic acoustic (Stage 1 base)

| Dataset | Hours | Dialect | Transcripts | License / Access | Status |
|---|---|---|---|---|---|
| **MASC** | ~1000h | Multi (incl. Gulf) | yes | CC-BY-4.0, **ungated** | add - largest open base pool |
| SADA22 (`MohamedRashad/SADA22`) | 668h | Saudi Khaliji | yes | CC-BY-NC-SA, ungated | partly in 804h; expand |
| **EmiratiDialectShows** (`eabayed/EmiratiDialictShowsAudioTranscription`) | ~0.4h (467 clips) | Emirati | yes | AFL-3.0, **ungated** | add - pure Emirati, ungated |
| **sawtarabi** (`ArabicSpeech/sawtarabi`) | small (3.3k rows) | Arabic | yes | public | add - small base |
| Ramsa | 41h | Emirati (157 spk) | yes | gated (email author) | if access |
| saudi_dialect_asrv1.0 (`musabalosimi`) | ~8.25k clips | Saudi | yes | free | add |
| Alsanaa (`MahaAlBlooki/alsanaa-emirati-dataset`) | 4h | Emirati (1 spk) | yes | open, GitHub | small |
| Common Voice Arabic | ~157h | MSA / mixed | yes | CC0 | optional MSA pool |
| ADI17 (UAE track) | ~112h | Emirati | **labels only** | CC-BY-SA, `ArabicSpeech/ADI17` | SSL pretrain only |

---

## 3. Stage -> dataset mapping

| Stage | Purpose | Datasets | Sampling |
|---|---|---|---|
| **Stage 1** | General Gulf + MSA acoustic base | 804h (SADA + WorldSpeech + custom) + **MASC** + EmiratiShows + sawtarabi + Ramsa + saudi_asrv1 (+ Common Voice MSA) | uniform / dialect-balanced |
| **Stage 2** | Dialect + code-switch specialization | MixAT + **ZAEBUC** + Casablanca-UAE + **SCC22** + own-mined CS | **up-weighted** CS |
| (optional) | SSL acoustic pretrain | ADI17 (no transcripts) | - |

---

## 4. Excluded

- **No synthetic data** - explicitly rejects `vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k`
  (synthetic) and any TTS / spliced corpora.
- **QASR** - dropped (QCRI request overhead).
- **Paid:** LDC2006 Gulf phone, Appen Gulf CTS (~46h), DataOcean King-ASR-109 (451h), Macgence (100h).
- **ArEnAV (765h)** - deepfake AV corpus, not clean ASR.
- **ARCADE / ArPod** - optional only (dialect-ID weak transcripts / GPL author-contact).

---

## 5. Hour totals (real, free / research)

- **Code-switch (Stage 2, transcribed):** ~38h Gulf (~53h incl. Egyptian/MSA) + own-mined.
- **Gulf/Arabic acoustic (Stage 1, transcribed):** 804h existing + MASC ~1000h + SADA expansion
  + Ramsa 41h + saudi_asrv1 + EmiratiShows + sawtarabi + Alsanaa 4h ~= **~1900h+**.
- **Audio-only (SSL, optional):** ADI17 ~112h.

---

## 6. How datasets are wired in

1. [`scripts/prepare_datasets.py`](scripts/prepare_datasets.py) - downloads each HF dataset and
   writes a unified manifest (`audio_path`, `text`, `source`, `dialect`, `code_switch`, `weight`,
   `stage`). Synthetic sources are refused.
2. [`scripts/mine_code_switch.py`](scripts/mine_code_switch.py) - scans the existing 804h
   transcripts for Latin/English tokens and emits a `code_switch=True` subset for Stage 2.
3. [`scripts/finetune_qwen3_lora.py`](scripts/finetune_qwen3_lora.py) - consumes manifests;
   per-record `weight` drives the weighted sampler (Stage-2 up-weighting).
4. [`scripts/test_asr.py`](scripts/test_asr.py) - evaluates a trained adapter (WER/CER, overall
   and per-source/per-dialect) on held-out manifests.
