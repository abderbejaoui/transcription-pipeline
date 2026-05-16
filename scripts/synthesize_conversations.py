"""
Synthesize the 100 UAE medical conversations into WAV files via the TTS server.

Usage (after tts_server.py is running on DGX):
    python scripts/synthesize_conversations.py \
        --json  uae_medical_conversations_100.json \
        --out   eval/gulf_medical_v1/wavs \
        --url   http://<DGX_IP>:7900

Each conversation turn → one WAV file.
Naming: conv_{id:03d}_turn_{t:02d}_{speaker}.wav

Also builds a manifest.jsonl compatible with the bake-off harness.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests


VOICE_DOCTOR = "Gulf Arabic male doctor, calm professional tone"
VOICE_PATIENT = "Gulf Arabic male patient, casual conversational tone"


def synthesize_one(url: str, text: str, speaker: str = "patient") -> bytes:
    """Call the VoxCPM2 TTS server and return WAV bytes."""
    voice = VOICE_DOCTOR if speaker == "doctor" else VOICE_PATIENT
    resp = requests.post(
        f"{url}/tts",
        json={"text": text, "voice_description": voice},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to uae_medical_conversations_100.json")
    parser.add_argument("--out", required=True, help="Output directory for WAV files")
    parser.add_argument("--url", default="http://localhost:7900", help="TTS server URL")
    parser.add_argument("--manifest", default=None, help="Output manifest.jsonl path (default: <out>/manifest.jsonl)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip WAVs that already exist")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.jsonl"

    with open(args.json) as f:
        conversations = json.load(f)

    print(f"[synth] {len(conversations)} conversations → {out_dir}")

    # Check server health
    try:
        r = requests.get(f"{args.url}/health", timeout=5)
        print(f"[synth] Server: {r.json()}")
    except Exception as e:
        print(f"[synth] ERROR: Cannot reach TTS server at {args.url}: {e}")
        sys.exit(1)

    manifest_entries = []
    total_turns = sum(len(c["conversation"]) for c in conversations)
    done = 0
    t0 = time.time()

    for conv in conversations:
        conv_id = conv["id"]
        for turn_idx, turn in enumerate(conv["conversation"]):
            speaker = turn["speaker"]
            text = turn["text"]
            fname = f"conv_{conv_id:03d}_turn_{turn_idx:02d}_{speaker}.wav"
            wav_path = out_dir / fname

            if args.skip_existing and wav_path.exists():
                done += 1
                continue

            try:
                wav_bytes = synthesize_one(args.url, text, speaker)
                wav_path.write_bytes(wav_bytes)
                done += 1
                elapsed = time.time() - t0
                eta = (elapsed / done) * (total_turns - done) if done else 0
                print(f"  [{done}/{total_turns}] {fname}  ({len(wav_bytes)//1024}KB)  ETA {eta:.0f}s")
            except Exception as e:
                print(f"  [ERR] {fname}: {e}")
                continue

            manifest_entries.append({
                "audio": str(wav_path),
                "text": text,
                "speaker": speaker,
                "conv_id": conv_id,
                "turn_idx": turn_idx,
                "tag": f"gulf_medical_{speaker}",
            })

    # Write manifest
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(f"\n[synth] Done: {done}/{total_turns} turns in {elapsed:.0f}s")
    print(f"[synth] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
