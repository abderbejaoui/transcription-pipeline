# 01 — Problem Statement

## 1.1 The clinical scenario

A doctor in a Gulf country (UAE, Saudi Arabia, Kuwait, Qatar, Bahrain, Oman)
is dictating a consultation note in real time. The speech is:

- **Code-switched**: most sentences are Khaleeji Arabic; drug names and most
  technical terms are spoken in English ("اعطيتها paracetamol مرتين في اليوم").
- **Domain-specific**: heavy in pharmaceutical brand names (panadol,
  doliprane, ventolin, lipitor, ozempic …), clinical diagnoses (asthma,
  hypertension, GERD …), and dosage frames.
- **Spontaneous**: not read speech. Disfluencies, restarts, dialectal
  contractions (مو, ما, هيه).

The output must be usable in a clinical record. Two requirements:

1. **Drug names must come out in English, spelled correctly.** A drug
   transliterated to Arabic ("بنادول" instead of "panadol") is unusable
   because it cannot be cross-referenced with formulary databases.
2. **The diagnostic vocabulary must not be hallucinated.** When the doctor
   says "asthma" the ASR must not return "ozma" or "Asthma" mixed with a
   random nearby word. Mistakes that look fluent are more dangerous than
   obviously broken output.

## 1.2 Why off-the-shelf ASR fails

We measured three publicly-available production ASR systems on the
`UBC-NLP/Casablanca` UAE conversational subset (a real Emirati test set,
not synthetic):

| Model                       | WER on UAE | CER on UAE |
|---|---:|---:|
| Qwen3-ASR-1.7B (base)       | 67.67%     | 22.29%     |
| vadimbelsky/qwen3-asr-arabic-uae | 70.85% | 25.58%   |
| Voxtral-Mini-3B             | 70.85%     | 28.43%     |

(See `raw_test_results.md` for the full methodology.)

The leaderboard numbers from Wang et al. 2024 confirm Qwen3-ASR is the best
publicly-available baseline on the Open Universal Arabic ASR Leaderboard,
but **even the best system is at ~67% WER** on real Khaleeji speech. That
is not a usable starting point for a medical product.

The failure modes split into two categories:

### A. Acoustic / dialect failures

- Khaleeji vowel reductions and consonant assimilations not present in MSA
  training data.
- Code-switching at phrase boundaries: the encoder gets confused at the
  language switch and produces garbage for several words on either side.

### B. Vocabulary failures (the medical-specific part)

- Drug brand names get transliterated to Arabic when they should stay in
  English ("voltaren" → "فولتارين" or worse: "فواد علي النزار").
- Disease terms get translated to MSA Arabic equivalents that no doctor
  uses in writing ("asthma" → "ربو شعبي").
- Two drugs with similar phonetic profiles get confused (paracetamol vs
  prostatitis on noisy audio).

## 1.3 Accuracy targets

| Setting | Target WER |
|---|---:|
| Gulf general speech (SADA-style)           | ≤ 10% |
| UAE Emirati conversational                  | ≤ 10% |
| Gulf medical speech (overall)               | ≤ 8%  |
| **Medical-term-only error rate** (drugs + diagnoses) | **≤ 5%** |
| MSA / FLEURS ar (regression check)          | ≤ 10% |

These are stretch targets, not guarantees. Published state of the art on
real Khaleeji conversational speech in 2025 is ~12–18% WER (see SADA paper,
ICASSP 2024). The medical-term metric we defined ourselves — there is no
external benchmark for it.

## 1.4 Why we have to build this ourselves

There is no free Arabic medical conversational corpus. We checked LDC,
ELRA, OpenSLR, HuggingFace, Kaggle, and ~1,200 papers on Semantic Scholar.
What exists is:

- Free general-purpose Gulf Arabic speech (~1,160 hours verified).
- Free pan-Arabic speech for regularization (~2,880 hours).
- Zero free Arabic medical speech.

So the project plan is:
1. Adapt the best public base model (Qwen3-ASR) to Gulf dialect using free
   general data.
2. Patch the medical-vocabulary gap with synthetic data (TTS of LLM-generated
   Gulf clinical sentences) + a hand-curated 10,000-term lexicon.
3. Build a correction / flagging layer on top to catch the residual errors
   in real time.

The chapters that follow document how each of those three steps was
implemented.
