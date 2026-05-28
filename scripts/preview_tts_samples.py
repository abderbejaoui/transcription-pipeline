"""Synthesize a handful of preview clips so we can listen and verify
that the TTS voice cloning is producing the Emirati Gulf accent we want
BEFORE letting the full multi-day synthesis run.

Workflow
--------
1. scripts/extract_uae_references.py    (one-time, ~30s)
2. scripts/preview_tts_samples.py       (this script, ~2 min)
3. scp the wavs to your laptop, listen.
4. If they sound Emirati → resume the full synthesis.
   If not → adjust references or fall back to voice-design mode.

Usage
-----
    python3 scripts/preview_tts_samples.py \\
        --tts-url http://localhost:7900 \\
        --references data/tts_references/references.jsonl \\
        --out data/tts_preview \\
        --n 6
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import requests


# A handful of representative Gulf-clinic sentences spanning the kinds of
# content the full synthesis will produce. Each one contains an English
# medical term embedded in Khaleeji Arabic.
PREVIEW_SENTENCES = [
    "اعطيتها paracetamol مرتين في اليوم للحرارة",
    "خذي augmentin 625 mg كل ٨ ساعات لمدة اسبوع",
    "الدكتور قال عندي asthma لازم استخدم ventolin",
    "وصف لي اوزمبيك ozempic مره في الاسبوع",
    "حسيت بدوخه من lipitor الصبح",
    "ابي panadol للصداع شكرا",
    "عندي type 2 diabetes منذ خمس سنوات",
    "وصف لي omeprazole قبل الفطور",
    "الطفل عنده fever و sore throat",
    "اخذ amlodipine 5 mg صباحا للضغط",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tts-url", default="http://localhost:7900")
    ap.add_argument("--references",
                    default="data/tts_references/references.jsonl",
                    help="Reference manifest from extract_uae_references.py")
    ap.add_argument("--out", default="data/tts_preview",
                    help="Output directory for preview WAVs.")
    ap.add_argument("--n", type=int, default=6,
                    help="How many preview clips to synthesize.")
    ap.add_argument("--no-cloning", action="store_true",
                    help="Skip voice cloning, use voice-design fallback for "
                         "A/B comparison.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    refs = []
    if not args.no_cloning:
        ref_path = Path(args.references)
        if not ref_path.exists():
            raise SystemExit(
                f"[preview] no references found at {ref_path}. "
                f"Run scripts/extract_uae_references.py first, or "
                f"pass --no-cloning to use voice-design mode."
            )
        with ref_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if Path(row.get("audio_path", "")).exists():
                    refs.append(row)
        if not refs:
            raise SystemExit("[preview] no usable refs in manifest.")
        print(f"[preview] using {len(refs)} UAE references for voice cloning")

    # Check TTS service.
    r = requests.get(f"{args.tts_url}/health", timeout=5)
    r.raise_for_status()
    print(f"[preview] TTS: {r.json()}")

    sentences = PREVIEW_SENTENCES[:args.n]
    rng = random.Random(42)
    rng.shuffle(refs)

    for i, text in enumerate(sentences):
        fname = f"preview_{i + 1:02d}.wav"
        path = out / fname
        payload = {"text": text}
        ref_tag = "voice-design"
        if refs:
            ref = refs[i % len(refs)]
            payload["reference_wav_path"] = ref["audio_path"]
            if ref.get("transcript"):
                payload["reference_text"] = ref["transcript"]
            ref_tag = ref["ref_id"]
        else:
            payload["voice_description"] = (
                "Gulf Arabic Emirati male doctor, calm professional tone"
            )

        print(f"[preview] {i + 1}/{len(sentences)}  ref={ref_tag}  "
              f"text: {text[:50]}...")
        try:
            r = requests.post(f"{args.tts_url}/tts", json=payload, timeout=180)
            r.raise_for_status()
            path.write_bytes(r.content)
            print(f"           -> {path}  ({len(r.content) // 1024} KB)")
        except Exception as e:
            print(f"           ERROR: {e}")

    print(f"\n[preview] DONE. {args.n} previews in {out}")
    print(f"[preview] copy them to your laptop:")
    print(f"  scp '<dgx>:{out.resolve()}/*.wav' .")
    print(f"[preview] LISTEN: do they sound Emirati Gulf?")
    print(f"  YES -> resume the full synthesis run.")
    print(f"  NO  -> tweak references or try --no-cloning.")


if __name__ == "__main__":
    main()
