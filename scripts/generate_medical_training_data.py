"""
Generate ~10,000 Gulf Arabic medical training samples for ASR fine-tuning.

Pipeline:
  1. Read medical lexicon (246 terms)
  2. For each term, ask Ollama LLM to generate ~40 Gulf Arabic sentences
     containing the term (mix of Arabic + English code-switching)
  3. Synthesize each sentence via VoxCPM2 TTS server
  4. Output: WAV files + manifest.jsonl (audio_path, text pairs)

Usage:
    python scripts/generate_medical_training_data.py \
        --tts-url http://localhost:7900 \
        --ollama-url http://100.68.87.28:11434 \
        --out data/training/medical_gulf \
        --samples-per-term 40

Requires: requests
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

# Gulf Arabic sentence templates the LLM must follow.
# Emphasis on REAL Gulf clinical patterns:
#   - Drug names usually written in Latin INSIDE the Arabic sentence
#     (this is what we want the ASR to learn — preserve the English)
#   - Doctor uses MSA/educated Gulf, patient uses pure Khaleeji
#   - Common prescription frames: خذ X, وصف لي X, اعطيتك X, لازم تاخذ X
#   - Dosage: مرتين في اليوم, قبل الاكل, بعد الفطور, لمدة اسبوع
SYSTEM_PROMPT = """You are a Gulf Arabic (Emirati/Saudi/Kuwaiti) medical dialogue generator.
Generate realistic sentences that a PATIENT or DOCTOR would say in a Gulf Arabic clinic.

CRITICAL RULES:
- Use Gulf Arabic dialect (Khaleeji), NOT Modern Standard Arabic.
- Drug names MUST stay in their original LATIN spelling (e.g. write "paracetamol"
  not "باراسيتامول", "voltaren" not "فولتارين"). This is exactly how real bilingual
  doctors/pharmacists write prescriptions in the Gulf.
- However, if `aliases` for the term shows it commonly appears in Arabic script
  (e.g. "بنادول" for panadol), produce SOME sentences using each form so the model
  learns both spellings.
- Mix Arabic + English naturally (code-switching) at PHRASE level, not whole-language switches.
- Use real Gulf clinical patterns:
    * Prescriptions: خذ <DRUG> <DOSE> ملليجرام مرتين في اليوم
    * Indication: اخذ <DRUG> للحرارة / للالم / للضغط / للسكر
    * Brand context: الصيدلي اعطاني <DRUG>
    * Multi-drug: <DRUG1> صباحا و <DRUG2> مساء
    * Time markers: قبل الاكل / بعد الفطور / لمدة اسبوع / كل ٨ ساعات
    * Patient concerns: حسيت بدوخه من <DRUG>, <DRUG> ما عطاني نتيجه
- Gulf dialect markers: وايد، شوي، يالله، خلاص، شو، ليش، هيه، حق، عشان
- 5-20 words per sentence.
- Mix speakers: doctor, pharmacist, patient, parent (about child).

Return ONLY a JSON array of strings. No explanation. No markdown."""

USER_TEMPLATE = """Generate {n} different Gulf Arabic medical sentences containing the term "{term}".
The term type is: {term_type}.
Aliases (Arabic-script forms commonly heard in clinics): {aliases}

Generate a MIX:
- ~70% sentences keep "{term}" in its Latin spelling inside Arabic text.
- ~30% sentences use one of the Arabic-script aliases (if provided).
Make every sentence sound like a real Gulf clinic moment — prescriptions, dosing,
side-effects, patient questions, pharmacist instructions.

Return ONLY a JSON array of strings."""


def generate_sentences(
    term: str,
    term_type: str,
    n: int,
    ollama_url: str,
    model: str,
    aliases: list[str] | None = None,
) -> list[str]:
    """Ask Ollama to generate n Gulf Arabic sentences containing `term`."""
    aliases_str = ", ".join(aliases) if aliases else "(none)"
    resp = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": USER_TEMPLATE.format(
                n=n, term=term, term_type=term_type, aliases=aliases_str,
            ),
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": 0.9, "num_predict": 4096},
        },
        timeout=300,
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "")

    # Parse JSON array from LLM output
    # Try to find [...] in the response
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        print(f"  [WARN] No JSON array found for '{term}', got: {raw[:200]}")
        return []
    try:
        sentences = json.loads(raw[start:end + 1])
        if isinstance(sentences, list):
            return [s for s in sentences if isinstance(s, str) and len(s.strip()) > 5]
    except json.JSONDecodeError:
        print(f"  [WARN] JSON parse failed for '{term}': {raw[start:start+200]}")
    return []


def synthesize_one(tts_url: str, text: str, voice: str) -> bytes:
    """Call VoxCPM2 TTS and return WAV bytes."""
    resp = requests.post(
        f"{tts_url}/tts",
        json={"text": text, "voice_description": voice},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.content


VOICES = [
    "Gulf Arabic male doctor, calm professional tone",
    "Gulf Arabic male patient, casual conversational tone",
    "Gulf Arabic female doctor, professional tone",
    "Gulf Arabic female patient, casual worried tone",
    "Gulf Arabic young male, nervous speaking to doctor",
    "Gulf Arabic elderly male, slow calm speech",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lexicon", nargs="+",
        default=["data/medical_lexicon.jsonl", "data/gulf_drug_brands.jsonl"],
        help="One or more JSONL lexicon files. Lines are merged and "
             "deduplicated by `term`. Default loads the international "
             "lexicon plus the Gulf-specific brand-name lexicon.",
    )
    parser.add_argument("--tts-url", default="http://100.68.87.28:7900")
    parser.add_argument("--ollama-url", default="http://100.68.87.28:11434")
    parser.add_argument("--ollama-model", default="calme-3.2-instruct-78b-GGUF:IQ4_XS")
    parser.add_argument("--out", default="data/training/medical_gulf")
    parser.add_argument("--samples-per-term", type=int, default=40)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--sentences-only", action="store_true",
                        help="Generate sentences JSON only, skip TTS")
    args = parser.parse_args()

    out_dir = Path(args.out)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    sentences_path = out_dir / "sentences.jsonl"
    manifest_path = out_dir / "manifest.jsonl"

    # Load lexicon(s) — multiple files merged, deduped by `term`.
    lexicon_by_term: dict[str, dict] = {}
    for lex_path in args.lexicon:
        if not Path(lex_path).exists():
            print(f"[gen] skipping missing lexicon: {lex_path}")
            continue
        with open(lex_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                term = entry.get("term", "").strip().lower()
                if not term:
                    continue
                # Later files override earlier ones; aliases are merged.
                if term in lexicon_by_term:
                    existing = lexicon_by_term[term]
                    merged_aliases = list({
                        *(existing.get("aliases") or []),
                        *(entry.get("aliases") or []),
                    })
                    existing.update(entry)
                    existing["aliases"] = merged_aliases
                else:
                    lexicon_by_term[term] = entry
    lexicon = list(lexicon_by_term.values())
    print(f"[gen] {len(lexicon)} unique terms across "
          f"{len(args.lexicon)} lexicon file(s), "
          f"target {args.samples_per_term} sentences each")

    # Check services
    try:
        r = requests.get(f"{args.tts_url}/health", timeout=5)
        print(f"[gen] TTS: {r.json()}")
    except Exception as e:
        if not args.sentences_only:
            print(f"[gen] TTS not reachable: {e}")
            sys.exit(1)

    # Phase 1: Generate sentences via LLM
    all_sentences = []
    if sentences_path.exists():
        with open(sentences_path) as f:
            for line in f:
                if line.strip():
                    all_sentences.append(json.loads(line))
        print(f"[gen] Loaded {len(all_sentences)} existing sentences")

    existing_terms = {s["term"] for s in all_sentences}
    remaining = [e for e in lexicon if e["term"] not in existing_terms]

    if remaining:
        print(f"[gen] Generating sentences for {len(remaining)} remaining terms...")
        with open(sentences_path, "a") as fh:
            for i, entry in enumerate(remaining):
                term = entry["term"]
                term_type = entry.get("type", "medical")
                print(f"  [{i+1}/{len(remaining)}] {term} ({term_type})...", end="", flush=True)
                try:
                    sents = generate_sentences(
                        term, term_type, args.samples_per_term,
                        args.ollama_url, args.ollama_model,
                        aliases=entry.get("aliases") or [],
                    )
                    for s in sents:
                        rec = {"term": term, "type": term_type, "text": s}
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        all_sentences.append(rec)
                    print(f" {len(sents)} sentences")
                except Exception as e:
                    print(f" ERROR: {e}")
                    continue

    print(f"[gen] Total sentences: {len(all_sentences)}")

    if args.sentences_only:
        print("[gen] --sentences-only mode, skipping TTS")
        return

    # Phase 2: Synthesize via TTS
    manifest_entries = []
    done = 0
    t0 = time.time()

    for i, rec in enumerate(all_sentences):
        fname = f"med_{i:05d}.wav"
        wav_path = wav_dir / fname

        if args.skip_existing and wav_path.exists():
            manifest_entries.append({
                "audio": str(wav_path),
                "text": rec["text"],
                "term": rec["term"],
                "type": rec.get("type", "medical"),
            })
            done += 1
            continue

        voice = random.choice(VOICES)
        try:
            wav_bytes = synthesize_one(args.tts_url, rec["text"], voice)
            wav_path.write_bytes(wav_bytes)
            done += 1
            elapsed = time.time() - t0
            total = len(all_sentences)
            eta = (elapsed / done) * (total - done) if done else 0
            if done % 50 == 0 or done < 5:
                print(f"  [{done}/{total}] {fname} ({len(wav_bytes)//1024}KB) ETA {eta/60:.0f}m")
        except Exception as e:
            print(f"  [ERR] {fname}: {e}")
            continue

        manifest_entries.append({
            "audio": str(wav_path),
            "text": rec["text"],
            "term": rec["term"],
            "type": rec.get("type", "medical"),
        })

    # Write manifest
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(f"\n[gen] Done: {done}/{len(all_sentences)} samples in {elapsed/60:.0f}m")
    print(f"[gen] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
