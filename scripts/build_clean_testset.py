"""Curate a high-quality Gulf-Arabic test set from existing bake-off data.

Inputs (already in repo):
  - eval/bakeoff_30min/manifest.jsonl
  - eval/bakeoff_30min/audio/*.wav
  - eval/bakeoff_30min/bakeoff/predictions/qwen3-asr-1.7b/*.json
  - eval/bakeoff_30min/bakeoff/predictions/qwen3-asr-uae/*.json

Quality signals (combined to assign a tier):
  Q1. SOURCE_CER     — WorldSpeech ships a CER between its ref and an
                       internal ASR. <=0.10 indicates a faithful gold.
                       (Not available for SADA22; treated as neutral.)
  Q2. REF_STRUCTURAL — eval_v2._ref_looks_broken: standalone "ال",
                       multiple 1-char Arabic tokens.
  Q3. LEN_OUTLIER    — |pred - ref| / ref > 1.5 (manifest misalignment).
                       Computed against the AGREED prediction.
  Q4. MODEL_CONSENSUS — CER(pred_a, pred_b) under arabic-aware norm. Low
                       consensus CER = the models agree on what they
                       heard. High = audio is genuinely ambiguous.
  Q5. REF_DISAGREEMENT — When models agree but the ref differs a lot:
                         min(CER(ref, pred_a), CER(ref, pred_b)) > 0.25
                         with consensus CER < 0.10 → ref is wrong.

Tier assignment (per clip):
  HIGH  : ref looks clean, models agree, both predictions close to ref.
          (Use these for headline WER numbers.)
  MEDIUM: ref is okay, but models or ref show small disagreements.
          (Use these to track regressions but expect higher WER.)
  HARD  : audio is genuinely tough OR ref disagrees with consensus.
          (Diagnostic only — don't blame the model.)
  REJECT: structural ref defects or length mismatch beyond repair.
          (Excluded from the clean test set entirely.)

Output:
  eval/bakeoff_clean/
    manifest.jsonl       (HIGH + MEDIUM only; HARD/REJECT excluded)
    manifest_full.jsonl  (all 180 with tier labels — for diagnostics)
    audio/  (symlinks back to the original audio files)
    README.md            (counts, durations, per-source breakdown)

Each kept clip also gets:
  - transcript          : original raw reference
  - transcript_canonical: arabic-aware normalised reference
  - consensus           : pred_a if it agrees with pred_b, else None
  - tier                : "high" or "medium"
  - reason              : short human-readable explanation of the tier
"""
from __future__ import annotations

import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_v2 import (  # noqa: E402
    cer,
    norm_text,
    _ref_looks_broken,
    _length_outlier,
    source_of,
)

SRC_DIR = PROJECT_ROOT / "eval" / "bakeoff_30min"
OUT_DIR = PROJECT_ROOT / "eval" / "bakeoff_clean"
PRED_BASE = SRC_DIR / "bakeoff" / "predictions" / "qwen3-asr-1.7b"
PRED_UAE = SRC_DIR / "bakeoff" / "predictions" / "qwen3-asr-uae"

# Thresholds — picked empirically from the histograms.
SOURCE_CER_MAX = 0.10        # WorldSpeech-provided CER bound
CONSENSUS_CER_MAX = 0.10     # CER(pred_a, pred_b)
REF_TO_PRED_CER_HIGH = 0.10  # high tier: both preds close to ref
REF_TO_PRED_CER_MED = 0.25   # medium tier upper bound
LENGTH_RATIO_MAX = 1.5


def load_manifest() -> List[Dict]:
    return [json.loads(l) for l in (SRC_DIR / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def load_pred(model_dir: Path, clip_id: str) -> Optional[str]:
    p = model_dir / f"{clip_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("pred", "")
    except Exception:
        return None


def classify(clip: Dict) -> Tuple[str, str, Optional[str]]:
    """Return (tier, reason, consensus_text)."""
    ref_raw = clip.get("transcript", "") or ""
    ref_n = norm_text(ref_raw, fold_numbers=True)

    pred_a_raw = load_pred(PRED_BASE, clip["id"])
    pred_b_raw = load_pred(PRED_UAE, clip["id"])

    # If we are missing predictions, fall back to ref-only signals.
    if pred_a_raw is None or pred_b_raw is None:
        if _ref_looks_broken(ref_n):
            return "REJECT", "broken_ref + no_pred", None
        return "MEDIUM", "no_predictions", None

    pred_a_n = norm_text(pred_a_raw, fold_numbers=True)
    pred_b_n = norm_text(pred_b_raw, fold_numbers=True)

    # Structural defects first.
    if _ref_looks_broken(ref_n):
        return "REJECT", "ref_split_words_or_lone_alef", None

    if _length_outlier(ref_n, pred_a_n, ratio=LENGTH_RATIO_MAX) and \
       _length_outlier(ref_n, pred_b_n, ratio=LENGTH_RATIO_MAX):
        return "REJECT", "ref_length_misaligned_with_audio", None

    # Source-provided CER (WorldSpeech only).
    src_cer = clip.get("cer")
    if isinstance(src_cer, (int, float)) and src_cer > 0.20:
        return "REJECT", f"source_cer={src_cer:.2f}_too_high", None

    # Inter-model consensus.
    cons_cer = cer(pred_a_n, pred_b_n)
    cer_a = cer(ref_n, pred_a_n)
    cer_b = cer(ref_n, pred_b_n)
    cer_min = min(cer_a, cer_b)

    consensus_text = pred_a_raw if cons_cer < CONSENSUS_CER_MAX else None

    # Both models agree AND both disagree heavily with ref → ref is wrong.
    if cons_cer < CONSENSUS_CER_MAX and cer_min > REF_TO_PRED_CER_MED:
        return "HARD", f"models_agree_but_ref_differs (cons={cons_cer:.2f}, cer_min={cer_min:.2f})", consensus_text

    # Models disagree a lot → genuinely hard audio.
    if cons_cer > 0.30:
        return "HARD", f"models_disagree (cons={cons_cer:.2f})", None

    # Clean case: both predictions close to ref.
    if cer_a <= REF_TO_PRED_CER_HIGH and cer_b <= REF_TO_PRED_CER_HIGH:
        # And if source provided a CER, it must be good.
        if src_cer is None or src_cer <= 0.10:
            return "HIGH", f"both_preds_near_ref (cer_a={cer_a:.2f}, cer_b={cer_b:.2f})", consensus_text
        return "MEDIUM", f"good_preds_but_src_cer={src_cer:.2f}", consensus_text

    # Otherwise medium: some disagreement but ref structurally fine.
    if cer_min <= REF_TO_PRED_CER_MED:
        return "MEDIUM", f"moderate_disagreement (cer_min={cer_min:.2f})", consensus_text

    return "HARD", f"high_disagreement (cer_min={cer_min:.2f}, cons={cons_cer:.2f})", consensus_text


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audio").mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()

    full_rows: List[Dict] = []
    keep_rows: List[Dict] = []
    tier_counts: Dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "HARD": 0, "REJECT": 0}
    duration_by_tier: Dict[str, float] = {"HIGH": 0.0, "MEDIUM": 0.0, "HARD": 0.0, "REJECT": 0.0}
    source_by_tier: Dict[str, Dict[str, int]] = {t: {} for t in tier_counts}

    for clip in manifest:
        tier, reason, consensus = classify(clip)
        ref_canonical = norm_text(clip.get("transcript", "") or "", fold_numbers=True)

        row = {
            "id": clip["id"],
            "category": clip.get("category"),
            "language": clip.get("language", "ar"),
            "audio_path": clip["audio_path"],
            "duration_s": clip.get("duration_s"),
            "source": clip.get("source"),
            "tags": clip.get("tags", []),
            "transcript": clip.get("transcript"),
            "transcript_canonical": ref_canonical,
            "tier": tier,
            "tier_reason": reason,
            "consensus": consensus,
            "medical_terms": clip.get("medical_terms", []),
        }
        # Preserve source CER if present
        if "cer" in clip:
            row["source_cer"] = clip["cer"]
        # Preserve dialect / gender if present
        for key in ("dialect", "gender"):
            if key in clip:
                row[key] = clip[key]

        full_rows.append(row)
        tier_counts[tier] += 1
        duration_by_tier[tier] += float(clip.get("duration_s", 0.0))
        src = source_of(clip["id"])
        source_by_tier[tier][src] = source_by_tier[tier].get(src, 0) + 1

        if tier in ("HIGH", "MEDIUM"):
            keep_rows.append(row)

    # Write full manifest (for diagnostics)
    (OUT_DIR / "manifest_full.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in full_rows) + "\n",
        encoding="utf-8",
    )

    # Write clean manifest (HIGH + MEDIUM)
    (OUT_DIR / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in keep_rows) + "\n",
        encoding="utf-8",
    )

    # Symlink (or copy fallback) audio for kept clips
    n_audio = 0
    for row in keep_rows:
        src = SRC_DIR / row["audio_path"]
        dst = OUT_DIR / row["audio_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
        n_audio += 1

    # README
    total_dur = sum(duration_by_tier.values())
    kept_dur = duration_by_tier["HIGH"] + duration_by_tier["MEDIUM"]
    lines: List[str] = []
    lines.append("# Bake-off clean test set")
    lines.append("")
    lines.append(f"Total clips: {len(manifest)}  |  Total duration: {total_dur/60:.1f} min")
    lines.append(f"**Kept (HIGH + MEDIUM):** {tier_counts['HIGH'] + tier_counts['MEDIUM']} clips, {kept_dur/60:.1f} min")
    lines.append("")
    lines.append("## Tier breakdown")
    lines.append("")
    lines.append("| Tier | Clips | Duration | Description |")
    lines.append("|---|---:|---:|---|")
    desc = {
        "HIGH":   "Both models predict near-identical text, both close to reference.",
        "MEDIUM": "Some disagreement but structurally fine reference.",
        "HARD":   "Audio genuinely ambiguous OR reference disagrees with model consensus.",
        "REJECT": "Reference defects (split words, length misalignment, too noisy).",
    }
    for tier in ["HIGH", "MEDIUM", "HARD", "REJECT"]:
        lines.append(
            f"| {tier} | {tier_counts[tier]} | "
            f"{duration_by_tier[tier]/60:.1f} min | {desc[tier]} |"
        )
    lines.append("")
    lines.append("## Per-source breakdown")
    lines.append("")
    all_sources = sorted({s for d in source_by_tier.values() for s in d})
    header = "| source | " + " | ".join(["HIGH", "MEDIUM", "HARD", "REJECT", "kept"]) + " |"
    lines.append(header)
    lines.append("|" + "---|" * 6)
    for src in all_sources:
        h = source_by_tier["HIGH"].get(src, 0)
        m = source_by_tier["MEDIUM"].get(src, 0)
        hd = source_by_tier["HARD"].get(src, 0)
        r = source_by_tier["REJECT"].get(src, 0)
        lines.append(f"| {src} | {h} | {m} | {hd} | {r} | {h + m} |")
    lines.append("")
    lines.append("## How to evaluate against this set")
    lines.append("")
    lines.append("```bash")
    lines.append("# Score the cached predictions against the cleaned references:")
    lines.append("python3 scripts/eval_v2.py --testset eval/bakeoff_clean")
    lines.append("")
    lines.append("# Run a new model on the clean set (DGX):")
    lines.append("python -m scripts.bakeoff \\")
    lines.append("    --eval-dir eval/bakeoff_clean \\")
    lines.append("    --models qwen3 qwen3_uae")
    lines.append("```")
    lines.append("")
    lines.append("## Scoring policy")
    lines.append("")
    lines.append("- Report **CER** (mean over kept clips) as the headline metric. CER")
    lines.append("  is robust to dialect spelling variants (هذه / هزه) and minor")
    lines.append("  formatting differences (digits vs. spelled-out numbers).")
    lines.append("- Report WER as a secondary metric; treat WER differences smaller")
    lines.append("  than 3 percentage points as noise.")
    lines.append("- Apply `norm_text(s, fold_numbers=True)` to both ref and hyp before")
    lines.append("  scoring (see `scripts/eval_v2.py`).")
    lines.append("")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {OUT_DIR / 'manifest.jsonl'}  ({len(keep_rows)} clips kept)")
    print(f"wrote {OUT_DIR / 'manifest_full.jsonl'}  ({len(full_rows)} total)")
    print(f"linked {n_audio} audio files into {OUT_DIR / 'audio'}")
    print()
    print("Tier breakdown:")
    for tier in ["HIGH", "MEDIUM", "HARD", "REJECT"]:
        print(f"  {tier:7s} {tier_counts[tier]:3d} clips  {duration_by_tier[tier]/60:5.1f} min")


if __name__ == "__main__":
    main()
