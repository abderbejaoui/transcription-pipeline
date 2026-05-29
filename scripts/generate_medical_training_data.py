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
#
# IMPORTANT: drug and disease terms ALWAYS stay in English (Latin script).
# The ASR's target output is English drug names inside Arabic sentences,
# because that's how Gulf doctors write prescriptions in real life. Any
# Arabic-script transliteration of an English drug name in the training set
# would teach the model the wrong thing and we'd have to undo it in
# post-processing.
SYSTEM_PROMPT = """You are a Gulf Arabic (Emirati/Saudi/Kuwaiti) medical dialogue generator.
Generate realistic sentences a PATIENT, DOCTOR, PHARMACIST, or PARENT would say
in a Gulf Arabic clinic or pharmacy.

CRITICAL RULES — read carefully:
1. The medical TERM (drug or disease name) MUST appear in the sentence exactly
   as given, spelled in ENGLISH (Latin letters). Never transliterate it to
   Arabic script. Examples of CORRECT output:
       "اعطاني الدكتور paracetamol مرتين في اليوم"
       "عندي asthma من زمان"
       "الصيدلي قال لي خذ augmentin بعد الاكل"
   Examples of WRONG output (DO NOT produce these):
       "اعطاني الدكتور باراسيتامول"          ← WRONG (transliterated)
       "عندي ربو"                              ← WRONG (translated)
       "خذ اوقمنتين"                           ← WRONG (transliterated)
2. The rest of the sentence is Gulf Arabic dialect (Khaleeji), NOT MSA.
3. Mix Arabic + English at PHRASE level. Numbers and dosing units may be in
   either language: "two times" / "مرتين", "500 mg" / "500 ملي".
4. Use real Gulf clinical patterns:
     * Prescriptions: خذ <TERM> 500 mg مرتين في اليوم
     * Indication:    اخذ <TERM> للحرارة / للالم / للضغط / للسكر
     * Pharmacist:    الصيدلي اعطاني <TERM>
     * Multi-drug:    <TERM1> صباحا و paracetamol مساء
     * Diagnosis:     الدكتور قال عندي <TERM>
     * Symptom:       حسيت بدوخه من <TERM>
     * Compliance:    نسيت اخذ <TERM> الصبح
     * Question:      <TERM> له اعراض جانبيه؟
     * Time markers:  قبل الاكل / بعد الفطور / لمدة اسبوع / كل ٨ ساعات
5. Gulf dialect markers: وايد، شوي، يالله، خلاص، شو، ليش، هيه، حق، عشان، أبغى.
6. 5-20 words per sentence. Vary length naturally.
7. Mix speakers: doctor, pharmacist, patient (male/female), parent about child.

Return ONLY a JSON array of strings. No explanation. No markdown."""


USER_TEMPLATE = """Generate {n} different Gulf Arabic medical sentences.

The medical term to include is: "{term}"
The term type is: {term_type}

Every sentence MUST contain the term "{term}" spelled exactly that way in
English (Latin letters). Do NOT transliterate it to Arabic. Do NOT translate
it. The surrounding sentence is Gulf Arabic.

Make every sentence sound like a real Gulf clinic moment — prescriptions,
dosing instructions, side-effects, patient questions, pharmacist guidance,
parent worried about a child, etc.

Return ONLY a JSON array of strings."""


def generate_sentences(
    term: str,
    term_type: str,
    n: int,
    ollama_url: str,
    model: str,
) -> list[str]:
    """Ask Ollama to generate n Gulf Arabic sentences containing `term`."""
    resp = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": USER_TEMPLATE.format(n=n, term=term, term_type=term_type),
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": 0.9, "num_predict": 4096},
        },
        timeout=300,
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "")

    # Parse JSON array from LLM output.
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        print(f"  [WARN] No JSON array found for '{term}', got: {raw[:200]}")
        return []
    try:
        sentences = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        print(f"  [WARN] JSON parse failed for '{term}': {raw[start:start + 200]}")
        return []
    if not isinstance(sentences, list):
        return []

    # Keep only strings that actually contain the English term verbatim.
    # If the LLM transliterated it, drop the sentence — we don't want bad
    # training pairs in the dataset.
    term_lc = term.lower()
    kept = []
    for s in sentences:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if len(s) < 5:
            continue
        if term_lc not in s.lower():
            # The model transliterated the term. Skip this sentence.
            continue
        kept.append(s)
    return kept


def synthesize_one(
    tts_url: str,
    text: str,
    voice: str | None = None,
    reference_wav_path: str | None = None,
    reference_text: str | None = None,
) -> bytes:
    """Call VoxCPM2 TTS and return WAV bytes.

    Two modes are supported:
      (a) Voice cloning  — pass `reference_wav_path` (and optionally
          `reference_text` for Ultimate Cloning). This is the
          documented way to get a specific dialect / accent out of
          VoxCPM2 and is the path used when references are available.
      (b) Voice design   — pass `voice` (natural-language description).
          Used as a fallback when no references are available. Note
          the VoxCPM2 model card explicitly lists steering for
          gender/age/tone/emotion/pace, not for Arabic dialect — so
          (a) gives much better Gulf-accent results.
    """
    payload: dict = {"text": text}
    if reference_wav_path:
        payload["reference_wav_path"] = reference_wav_path
        if reference_text:
            payload["reference_text"] = reference_text
    elif voice:
        payload["voice_description"] = voice
    resp = requests.post(f"{tts_url}/tts", json=payload, timeout=180)
    resp.raise_for_status()
    return resp.content


# Fallback voice-design strings used ONLY when no UAE reference WAVs
# are configured. With references configured these are ignored.
VOICES = [
    "Gulf Arabic male doctor, calm professional tone",
    "Gulf Arabic male patient, casual conversational tone",
    "Gulf Arabic female doctor, professional tone",
    "Gulf Arabic female patient, casual worried tone",
    "Gulf Arabic young male, nervous speaking to doctor",
    "Gulf Arabic elderly male, slow calm speech",
]


def load_voice_references(refs_path: Path) -> list[dict]:
    """Load the UAE reference manifest produced by
    scripts/extract_uae_references.py. Returns a list of dicts each
    with at least `audio_path` and `transcript`.
    """
    if not refs_path.exists():
        return []
    refs = []
    with refs_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("audio_path") or not Path(row["audio_path"]).exists():
                continue
            refs.append(row)
    return refs


def _samples_for_tier(tier: int, defaults: dict[int, int]) -> int:
    return defaults.get(tier, defaults.get(3, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lexicon", nargs="+",
        default=["data/full_lexicon.jsonl"],
        help="One or more JSONL lexicon files. Lines are merged and "
             "deduplicated by `term`. Default loads the tiered full lexicon "
             "built by scripts/build_full_lexicon.py.",
    )
    parser.add_argument("--tts-url", default="http://localhost:7900")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="hf.co/bartowski/calme-3.2-instruct-78b-GGUF:IQ4_XS")
    parser.add_argument("--out", default="data/training/medical_gulf_v2")
    parser.add_argument("--samples-per-term", type=int, default=0,
                        help="Flat samples per term. If 0 (default), tier-weighted "
                             "sampling is used instead (recommended).")
    parser.add_argument("--tier1-samples", type=int, default=60,
                        help="Sentences per tier-1 term (common drugs/diseases).")
    parser.add_argument("--tier2-samples", type=int, default=12,
                        help="Sentences per tier-2 term.")
    parser.add_argument("--tier3-samples", type=int, default=2,
                        help="Sentences per tier-3 term (long tail).")
    parser.add_argument("--target-hours", type=float, default=70.0,
                        help="Stop generating audio after reaching this many hours.")
    parser.add_argument("--max-tier", type=int, default=3,
                        help="Only generate sentences for terms at this tier or "
                             "lower. Default 3 (all tiers). Set to 1 to only "
                             "process tier-1 terms (~543 terms). Tier-1 alone "
                             "produces ~32k sentences which is already enough "
                             "for the 70h audio budget — use this when speed "
                             "matters more than vocabulary breadth.")
    parser.add_argument("--voice-references",
                        default="data/tts_references/references.jsonl",
                        help="Path to references.jsonl produced by "
                             "scripts/extract_uae_references.py. If present, "
                             "the script uses VoxCPM2 voice cloning with these "
                             "real UAE Emirati reference clips — the documented "
                             "way to get dialect-accurate output. If absent, "
                             "falls back to natural-language voice prompts.")
    parser.add_argument("--no-voice-cloning", action="store_true",
                        help="Disable reference-based voice cloning even if "
                             "references.jsonl exists. Use only for debugging.")
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
                # Later files override earlier ones. We pick the lower tier
                # (more important) when the same term appears in multiple files.
                if term in lexicon_by_term:
                    existing = lexicon_by_term[term]
                    new_tier = entry.get("tier", existing.get("tier", 3))
                    old_tier = existing.get("tier", 3)
                    existing.update(entry)
                    existing["tier"] = min(new_tier, old_tier)
                else:
                    # Default to tier 3 if the file doesn't carry a tier.
                    entry.setdefault("tier", 3)
                    lexicon_by_term[term] = entry
    lexicon = list(lexicon_by_term.values())

    # Tier counts for the log line.
    tier_counts: dict[int, int] = {}
    for e in lexicon:
        t = int(e.get("tier", 3))
        tier_counts[t] = tier_counts.get(t, 0) + 1

    tier_default = {
        1: args.tier1_samples,
        2: args.tier2_samples,
        3: args.tier3_samples,
    }
    if args.samples_per_term > 0:
        # Override: flat sampling.
        tier_default = {t: args.samples_per_term for t in (1, 2, 3)}

    print(f"[gen] {len(lexicon)} unique terms across "
          f"{len(args.lexicon)} lexicon file(s)")
    for t in sorted(tier_counts.keys()):
        print(f"[gen]   tier{t}: {tier_counts[t]} terms × "
              f"{tier_default.get(t, 0)} sentences "
              f"= {tier_counts[t] * tier_default.get(t, 0)} target sentences")
    total_target = sum(tier_counts.get(t, 0) * tier_default.get(t, 0)
                       for t in (1, 2, 3))
    print(f"[gen]   total target: {total_target} sentences "
          f"(~{total_target * 8 / 3600:.1f}h of audio at 8s/clip)")

    # Check services
    try:
        r = requests.get(f"{args.tts_url}/health", timeout=5)
        print(f"[gen] TTS: {r.json()}")
    except Exception as e:
        if not args.sentences_only:
            print(f"[gen] TTS not reachable: {e}")
            sys.exit(1)

    # Phase 1: Generate sentences via LLM (tier-weighted).
    all_sentences: list[dict] = []
    if sentences_path.exists():
        with open(sentences_path) as f:
            for line in f:
                if line.strip():
                    all_sentences.append(json.loads(line))
        print(f"[gen] loaded {len(all_sentences)} existing sentences")

    # How many sentences each term already has, so we can resume cleanly.
    have_per_term: dict[str, int] = {}
    for s in all_sentences:
        have_per_term[s["term"]] = have_per_term.get(s["term"], 0) + 1

    # Process tier 1 first so the most important drugs are covered even if
    # we get killed early.
    lexicon_by_tier = sorted(lexicon, key=lambda e: int(e.get("tier", 3)))

    # Apply --max-tier filter: skip terms above the specified tier.
    filtered_lexicon = [
        e for e in lexicon_by_tier
        if int(e.get("tier", 3)) <= args.max_tier
    ]
    if args.max_tier < 3:
        skipped_tiers = len(lexicon_by_tier) - len(filtered_lexicon)
        print(f"[gen] --max-tier {args.max_tier}: "
              f"processing {len(filtered_lexicon)} terms "
              f"(skipping {skipped_tiers} tier>{args.max_tier} terms)")

    print(f"[gen] generating sentences for {len(filtered_lexicon)} terms ...")
    with open(sentences_path, "a") as fh:
        for i, entry in enumerate(filtered_lexicon):
            term = entry["term"]
            term_type = entry.get("type", "medical")
            tier = int(entry.get("tier", 3))
            want = tier_default.get(tier, 1)
            already = have_per_term.get(term, 0)
            need = max(0, want - already)
            if need == 0:
                continue

            print(f"  [{i + 1}/{len(lexicon_by_tier)}] T{tier} {term} ({term_type}) "
                  f"need={need}...", end="", flush=True)
            try:
                sents = generate_sentences(
                    term, term_type, need,
                    args.ollama_url, args.ollama_model,
                )
                for s in sents:
                    rec = {"term": term, "type": term_type, "tier": tier, "text": s}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    all_sentences.append(rec)
                print(f" got {len(sents)}")
            except Exception as e:
                print(f" ERROR: {e}")
                continue

    print(f"[gen] total sentences: {len(all_sentences)}")

    if args.sentences_only:
        print("[gen] --sentences-only mode, skipping TTS")
        return

    # Phase 2: Synthesize via TTS, stopping when we hit the hour cap.
    #
    # CRITICAL: shuffle sentences with a tier-weighted order so that
    # when we hit the 70h cap, every tier is proportionally represented.
    # Without this shuffle, tier-1 sentences (which lead the list)
    # would consume the entire 70h budget before tier-2 or tier-3
    # ever get synthesized — defeating the whole point of the 10k
    # lexicon.
    #
    # Strategy: interleave tiers by their hour budget ratios. With
    # defaults (60/12/2 samples-per-term), the natural ratio of
    # sentences-per-tier is roughly 32k / 18k / 16k. We shuffle within
    # each tier then round-robin pull from all three at proportional
    # rates so the first 1000 TTS calls span all tiers.
    by_tier: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    for rec in all_sentences:
        by_tier.setdefault(int(rec.get("tier", 3)), []).append(rec)
    rng = random.Random(42)
    for t in by_tier:
        rng.shuffle(by_tier[t])
    # Compute proportional pull rates from the actual counts.
    counts = {t: len(by_tier.get(t, [])) for t in (1, 2, 3)}
    total = max(1, sum(counts.values()))
    print(f"[gen] TTS queue composition: "
          f"T1={counts.get(1, 0)} ({counts.get(1, 0) * 100 / total:.1f}%) "
          f"T2={counts.get(2, 0)} ({counts.get(2, 0) * 100 / total:.1f}%) "
          f"T3={counts.get(3, 0)} ({counts.get(3, 0) * 100 / total:.1f}%)")
    # Round-robin interleave proportionally — like a weighted merge.
    cursors = {t: 0 for t in (1, 2, 3)}
    interleaved: list[dict] = []
    while any(cursors[t] < counts.get(t, 0) for t in (1, 2, 3)):
        for t in (1, 2, 3):
            if cursors[t] >= counts.get(t, 0):
                continue
            # Pull a chunk proportional to this tier's share.
            chunk = max(1, counts.get(t, 0) // 100)
            take = min(chunk, counts[t] - cursors[t])
            interleaved.extend(by_tier[t][cursors[t]:cursors[t] + take])
            cursors[t] += take
    all_sentences = interleaved

    # Load UAE Emirati voice-cloning references (documented dialect-steering
    # method per VoxCPM2 model card). Falls back to natural-language voice
    # design if no references are configured.
    voice_refs: list[dict] = []
    if not args.no_voice_cloning:
        voice_refs = load_voice_references(Path(args.voice_references))
    if voice_refs:
        print(f"[gen] voice cloning ENABLED — {len(voice_refs)} UAE references")
        for r in voice_refs[:5]:
            print(f"        {r['ref_id']}  "
                  f"({r.get('duration_s', 0):.1f}s) "
                  f"{r.get('source', '?')}")
        if len(voice_refs) > 5:
            print(f"        ... and {len(voice_refs) - 5} more")
    else:
        print(f"[gen] voice cloning DISABLED — no references found at "
              f"{args.voice_references}. Using natural-language voice "
              f"prompts. WARNING: these will NOT reliably produce Gulf "
              f"dialect — run scripts/extract_uae_references.py first.")

    manifest_entries: list[dict] = []
    done = 0
    skipped_existing = 0
    audio_seconds = 0.0
    target_seconds = args.target_hours * 3600
    t0 = time.time()

    # If there's already a manifest from a previous run, load it so we
    # know how many seconds we've accumulated. /tts/file returns duration
    # but we only call /tts here, so probe via soundfile lazily.
    import wave  # std-lib, no extra dep

    def _wav_duration_s(path: Path) -> float:
        try:
            with wave.open(str(path), "rb") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            return 0.0

    for i, rec in enumerate(all_sentences):
        if audio_seconds >= target_seconds:
            print(f"[gen] hit target {args.target_hours}h "
                  f"({audio_seconds / 3600:.2f}h) — stopping TTS phase")
            break

        fname = f"med_{i:05d}.wav"
        wav_path = wav_dir / fname

        if args.skip_existing and wav_path.exists():
            dur = _wav_duration_s(wav_path)
            manifest_entries.append({
                "audio": str(wav_path),
                "text": rec["text"],
                "term": rec["term"],
                "type": rec.get("type", "medical"),
                "tier": rec.get("tier", 3),
                "duration_s": dur,
            })
            audio_seconds += dur
            skipped_existing += 1
            continue

        try:
            if voice_refs:
                # Voice cloning mode — rotate through UAE references so the
                # dataset spans multiple Emirati speakers.
                ref = random.choice(voice_refs)
                wav_bytes = synthesize_one(
                    args.tts_url, rec["text"],
                    reference_wav_path=ref["audio_path"],
                    reference_text=ref.get("transcript"),
                )
                ref_tag = ref["ref_id"]
            else:
                # Fallback voice-design mode.
                voice = random.choice(VOICES)
                wav_bytes = synthesize_one(args.tts_url, rec["text"], voice=voice)
                ref_tag = None
            wav_path.write_bytes(wav_bytes)
            dur = _wav_duration_s(wav_path)
            audio_seconds += dur
            done += 1
            elapsed = time.time() - t0
            if done % 50 == 0 or done < 5:
                pct = (audio_seconds / target_seconds) * 100
                eta_s = (
                    (elapsed / done) * max(0, target_seconds - audio_seconds) / dur
                    if dur > 0 and done > 0 else 0
                )
                print(f"  [{done}/{len(all_sentences)}] {fname} "
                      f"({len(wav_bytes) // 1024}KB) "
                      f"audio={audio_seconds / 3600:.2f}h "
                      f"({pct:.1f}%) ETA {eta_s / 60:.0f}m"
                      + (f"  ref={ref_tag}" if ref_tag else ""))
        except Exception as e:
            print(f"  [ERR] {fname}: {e}")
            continue

        manifest_entries.append({
            "audio": str(wav_path),
            "text": rec["text"],
            "term": rec["term"],
            "type": rec.get("type", "medical"),
            "tier": rec.get("tier", 3),
            "duration_s": dur,
            "voice_ref": ref_tag,
        })

    # Write manifest.
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(f"\n[gen] DONE.")
    print(f"[gen]   new clips synthesized : {done}")
    print(f"[gen]   reused existing clips : {skipped_existing}")
    print(f"[gen]   total audio in manifest: {audio_seconds / 3600:.2f}h")
    print(f"[gen]   wall time             : {elapsed / 60:.0f}m")
    print(f"[gen]   manifest              : {manifest_path}")


if __name__ == "__main__":
    main()
