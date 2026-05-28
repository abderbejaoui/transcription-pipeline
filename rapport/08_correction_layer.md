# 08 — Correction Layer (Phonetic Flagger)

The correction layer is the deterministic safety net that sits on top of
the ASR. Its job is to catch medical-vocabulary errors that the LoRA
fine-tune does not yet eliminate, without depending on a slow online
LLM call.

This is the most heavily-iterated component in the repository. The
current version passes a 50/50 hand-curated hard test suite. The code
lives at [app/services/flag.py](../app/services/flag.py).

## 8.1 The medical lexicon used by the flagger

Source: `medical_terms.txt` (~196 terms, English-only, hand-curated).

Each term carries an implicit class flag (drug vs disease) derived from
its presence in `data/medical_lexicon.jsonl`. The 196-term list is the
shortlist of terms we *actively check for* in transcripts — wider than
the formulary alone because it also includes common diagnoses,
procedures, and a handful of high-confusion non-medical words used as
"distractors" to suppress false positives.

## 8.2 The algorithm in plain English

For each consecutive 1, 2, or 3-word window in the transcript:

1. Normalize the window to a phonetic skeleton (consonants only, with
   light Arabic clitic stripping).
2. Compute string similarity against the phonetic skeleton of every
   lexicon term.
3. If any term scores above a calibrated threshold, surface it as a
   candidate.
4. Apply precision filters (LCS length, drug-vs-disease tiebreaker,
   name-lexicon suppression) to drop false positives.
5. If similarity ≥ 0.85 and the LLM judge (if running) agrees, apply
   the correction automatically. Otherwise surface a flag.

## 8.3 Key helpers

### `_translit` and `_strip_arabic_clitics`

Converts the window into a Latin phonetic representation. Handles:

- Common Arabic clitics: ال (the), و (and), ب (with), ك (like), ل (for).
- Arabic consonant inventory mapped to Latin equivalents (ق→q, ك→k,
  ج→j, etc.).
- Arabic vowels stripped (Arabic script doesn't write short vowels).

### `_consonant_skeleton_ar` and `_consonant_skeleton_latin`

Reduces a string to its consonant skeleton:
- Removes vowels (a, e, i, o, u) and y/w semi-vowels.
- Folds equivalent consonants:
  - `q → k` (for Khaleeji where ق often sounds like k)
  - `x → ks` (for words like "axilla" vs "aksila")
- Removes geminations.

Two same-rooted words become the same skeleton:
- "panadol" → "pndl"
- "paneadol" → "pndl"
- "بنادول" → "pndl" (via _translit then skeleton)

### `_longest_common_substring`

Bigram LCS used by the precision filter. We use a true LCS, not LCS of
character n-grams, because phonetic confusions tend to preserve runs
of consonants.

### `_phonetic_candidates`

For one window, returns a sorted list of `(score, term)` candidates.
The score combines:
- Skeleton edit-distance similarity (Jaro-Winkler-like measure)
- LCS bonus: if LCS length ≥ 3 AND LCS covers ≥ 60% of the needle and
  ≥ 50% of the term skeleton, add +0.10
- Drug-vs-disease tiebreaker at 0.03 granularity (drugs win ties)
- Score bucketing at 0.03 so micro-differences in the similarity
  algorithm don't flip the rank between drugs and diseases

### `phonetic_pass`

The main entry point. Walks the transcript with overlapping windows
in this order: bigrams → trigrams → singles. Once a window is flagged
with a high-confidence candidate, the words it covers are masked from
subsequent windows so we don't double-flag.

The bigram-first ordering is important: medical terms are often
multi-word ("vitamin d3", "type 2 diabetes"). If we ran singles first
we would lock in a bad single-word match before the multi-word
candidate has a chance.

### `apply_high_confidence_corrections`

Applies the correction at the text level only if **both** conditions
hold:
1. Phonetic top-1 similarity ≥ 0.85.
2. The candidate term appears in the lexicon.

Otherwise the correction is surfaced as a flag in the UI.

### `_PHONETIC_ALIAS`

A small hard-coded alias table for the few cases the consonant
skeleton can't resolve cleanly. Example:

```
"اف اول قن" → "efferalgan"
```

The Arabic "اف اول قن" reads as three short words ("ef awwal qan"),
and the consonant skeleton is "f-w-l-q-n" — not enough overlap with
"efferalgan"'s skeleton "frlgn". The alias entry forces the right
candidate.

### `_ARABIC_FILLER`

A set of ~70 Arabic first/last names plus common interjections
("والله", "يعني", "بس"). When a window's first or last word is in this
set we apply a similarity penalty, because names and fillers should
not normally cross into medical territory.

This single fix eliminated the most embarrassing class of false
positives ("فؤاد علي النزار" → fluconazole).

### `_is_likely_drug`

Heuristic on the candidate term's character distribution:
- Starts with a consonant cluster typical of medical English
- Contains a characteristic suffix (-azole, -olol, -pril, -sartan,
  -statin, -mab, -dipine, etc.)
- 5–18 characters

Used by the drug-vs-disease tiebreaker — at equal scores the drug
wins because drug-name failures are the dominant error mode.

## 8.4 Calibration: how the thresholds were chosen

Two test suites:
- `scripts/test_flag_hard.py` — 50 hand-picked failure cases pulled
  from the FastAPI session logs. Each case is a transcript + the
  correction it should produce (or "no flag" if it should not flag).
- `eval_corrector.py` — corpus-style evaluation on a larger set.

The 50/50 suite reached 100% pass after the most recent tightening.
Key calibration choices:

| Knob | Value | Justification |
|---|---|---|
| Auto-correct threshold | 0.85 | False positives at 0.80 — flagger was over-correcting plain words |
| Surface-as-flag threshold | 0.55 | Below this, candidates are too noisy to surface |
| LCS precision boost zone | 0.40–0.65 | This is the band where LCS evidence flips ambiguous cases |
| Min LCS length for boost | 3 | Length 2 boosted noise |
| Skeleton score bucket | 0.03 | Smaller buckets caused random drug-vs-disease flips |

## 8.5 Why no LLM in the hot path

We explicitly chose **not** to put an LLM in the correction hot path,
because:
- An LLM call on Calme 78B takes 500–2000 ms per word.
- The flagger runs in < 5 ms per transcript.
- LLMs hallucinate plausible-but-wrong drug names that pass our
  validators.
- The deterministic flagger is auditable: every flag has a numerical
  score the user can read.

The LLM is optionally available as a *judge* in Phase D of the
pipeline (chapter 7), but the flagger itself is pure string algorithms.

## 8.6 Files

| File | Purpose |
|---|---|
| `app/services/flag.py` | The flagger |
| `medical_terms.txt` | The active lexicon |
| `data/medical_lexicon.jsonl` | Larger lexicon with class info |
| `scripts/test_flag_hard.py` | 50-case test suite |
| `eval_corrector.py` | Corpus-style flagger eval |

## 8.7 References

- Jaro & Winkler, **String comparator metrics**, 1989 (the JW
  similarity inspiration for our skeleton score).
- **Soundex / Metaphone** classical phonetic indexing — used as the
  conceptual basis for the consonant-skeleton approach.
- Our own `flag.py` is original code; no external phonetic library
  was a good fit because they all assume Latin-script English input
  and we needed Arabic-aware behaviour.
