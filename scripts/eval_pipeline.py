"""End-to-end evaluation of the correction pipeline against the
Medical-Audio-Transcription ground-truth dataset.

Usage
-----
  python -m scripts.eval_pipeline              # use defaults
  python -m scripts.eval_pipeline --audio-dir /path/to/wavs --meta /path/to/metadata.csv

For each WAV in --audio-dir that has a matching row in metadata.csv the
script:
  1. Sends the audio to /api/transcribe (POST multipart/form-data).
  2. Receives the raw Whisper transcription + the corrected text.
  3. Computes Word Error Rate (WER) for both raw and corrected against the
     expected ground truth.
  4. Flags problems:
       BAD_CORRECTION  corrected WER is worse than raw WER
       HIGH_WER        corrected WER > 0.25
       MISMATCH        raw transcript deviates substantially from expected
  5. Prints a per-file report and a summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_AUDIO_DIR = Path(
    "/Users/abderrahmenbejaoui/Medical-Audio-Transcription/data/audio_cache_preprocessed_10"
)
DEFAULT_META = Path(
    "/Users/abderrahmenbejaoui/Medical-Audio-Transcription/data/metadata.csv"
)
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
TRANSCRIBE_URL = f"{API_BASE}/api/transcribe"
WER_WARN_THRESHOLD = 0.25
TIMEOUT_S = 300


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def _normalize(text: str) -> List[str]:
    """Lowercase, remove punctuation, return word list."""
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def wer(reference: str, hypothesis: str) -> float:
    r = _normalize(reference)
    h = _normalize(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


# ---------------------------------------------------------------------------
# Pipeline call
# ---------------------------------------------------------------------------

def transcribe(audio_path: Path) -> Dict:
    with audio_path.open("rb") as f:
        resp = requests.post(
            TRANSCRIBE_URL,
            files={"audio": (audio_path.name, f, "audio/wav")},
            timeout=TIMEOUT_S,
        )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PROBLEMS = {
    "BAD_CORRECTION": "corrected WER > raw WER (pipeline made it worse)",
    "HIGH_WER":       "corrected WER > 25%",
    "FAIL":           "API call raised an exception",
}


def main(audio_dir: Path, meta_path: Path) -> int:
    # Load ground truth keyed by filename.
    gt: Dict[str, str] = {}
    # utf-8-sig strips the UTF-8 BOM (\ufeff) that this file has on the first column.
    with meta_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fname = row.get("filename", "").strip()
            tx = row.get("transcription", "").strip()
            if fname and tx:
                gt[fname] = tx

    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        print(f"[eval] No WAV files found in {audio_dir}", file=sys.stderr)
        return 1

    # Pre-warm: reload the lexicon and wait for the voice model to be ready.
    # The CTC voice model loads lazily on first use; sending a tiny request
    # first avoids a cold-start timeout on the first real file.
    print("[eval] Pre-warming server (loading voice model)...")
    try:
        # Use the smallest file to trigger model load.
        smallest = min(wavs, key=lambda p: p.stat().st_size)
        warmup_resp = transcribe(smallest)
        print(f"[eval] Warm-up OK ({smallest.name}): {(warmup_resp.get('raw_text') or '')[:60]}...")
    except Exception as exc:
        print(f"[eval] Warm-up timed out or failed: {exc}. Continuing anyway.")

    results = []

    for wav in wavs:
        expected = gt.get(wav.name, "")
        # The preprocessed folder uses suffix _Xb20.wav; metadata uses _Xo00.wav.
        # Try the canonical o00 name if the direct lookup misses.
        if not expected:
            canonical = wav.name.replace("b20.wav", "o00.wav")
            expected = gt.get(canonical, "")
        if not expected:
            print(f"  [SKIP] {wav.name} – no ground-truth row in metadata.csv")
            continue

        print(f"\n{'='*72}")
        print(f"FILE: {wav.name}  ({wav.stat().st_size // 1024} KB)")
        print(f"EXPECTED:\n  {expected[:200]}{'...' if len(expected) > 200 else ''}")

        t0 = time.time()
        try:
            resp = transcribe(wav)
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  !! FAIL ({elapsed:.1f}s): {exc}")
            results.append({
                "file": wav.name,
                "flag": "FAIL",
                "raw_wer": None,
                "cor_wer": None,
                "raw": "",
                "corrected": "",
                "expected": expected,
                "error": str(exc),
            })
            continue

        elapsed = time.time() - t0
        raw = resp.get("raw_text") or resp.get("text") or ""
        corrected = resp.get("corrected_text") or raw

        raw_wer = wer(expected, raw)
        cor_wer = wer(expected, corrected)

        flags = []
        if cor_wer > raw_wer + 0.02:
            flags.append("BAD_CORRECTION")
        if cor_wer > WER_WARN_THRESHOLD:
            flags.append("HIGH_WER")

        flag_str = " | ".join(flags) if flags else "OK"

        print(f"RAW       ({elapsed:.1f}s): {raw[:200]}{'...' if len(raw) > 200 else ''}")
        print(f"CORRECTED : {corrected[:200]}{'...' if len(corrected) > 200 else ''}")
        print(f"WER  raw={raw_wer:.3f}  corrected={cor_wer:.3f}  [{flag_str}]")

        if "BAD_CORRECTION" in flags:
            # Show exactly what the pipeline changed.
            raw_words = _normalize(raw)
            cor_words = _normalize(corrected)
            diffs = [(r, c) for r, c in zip(raw_words, cor_words) if r != c]
            if diffs:
                print(f"  CHANGES: { {r: c for r, c in diffs[:10]} }")

        results.append({
            "file": wav.name,
            "flag": flag_str,
            "raw_wer": raw_wer,
            "cor_wer": cor_wer,
            "raw": raw,
            "corrected": corrected,
            "expected": expected,
        })

    # Summary table.
    print(f"\n\n{'='*72}")
    print(f"{'FILE':<30} {'RAW WER':>9} {'COR WER':>9}  FLAG")
    print("-" * 72)
    ok = bad = high = fail = 0
    for r in results:
        flag = r["flag"]
        raw_s = f"{r['raw_wer']:.3f}" if r["raw_wer"] is not None else "  N/A"
        cor_s = f"{r['cor_wer']:.3f}" if r["cor_wer"] is not None else "  N/A"
        print(f"{r['file']:<30} {raw_s:>9} {cor_s:>9}  {flag}")
        if flag == "OK":
            ok += 1
        elif flag == "FAIL":
            fail += 1
        else:
            if "BAD_CORRECTION" in flag:
                bad += 1
            if "HIGH_WER" in flag:
                high += 1

    print(f"\nSUMMARY: {len(results)} files — OK={ok}  BAD_CORRECTION={bad}  HIGH_WER={high}  FAIL={fail}")

    # Write full JSON report next to this script.
    report_path = Path(__file__).parent / "eval_report.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Full report -> {report_path}")

    return 0 if (bad + fail) == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pipeline against ground truth.")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    args = parser.parse_args()
    sys.exit(main(args.audio_dir, args.meta))
