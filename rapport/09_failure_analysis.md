# 09 — Failure Analysis (Why v1 Was Not Enough)

After the v1 LoRA fine-tune (chapter 5) we ran the full system on real
clinic-style audio and characterized the residual failures. This chapter
documents the failure modes, the dead-end fixes we tried first, and the
conclusion that pushed us to v2.

## 9.1 Headline number

Post-v1 WER on `eval/casablanca_emirati_full/` (UAE conversational, 813
clips):

```
67.67% baseline → ~45% after v1 LoRA on 900h Gulf corpus.
```

A 22-point absolute reduction is real progress, but ~45% is
unacceptable for a clinical product. The product target is ≤ 10% WER on
Gulf general speech and ≤ 5% medical-term error rate.

The error categories from chapter 6, ranked:

| Category | Approx % of errors |
|---|---:|
| Medical drug names (transliteration or total mangle) | 35% |
| Code-switching boundary garbage | 20% |
| Dialect contractions not merged | 15% |
| Hamza / teh-marbuta noise (mostly survives normalizer) | 15% |
| Acoustic noise drop-outs | 10% |
| Numeric reading, proper nouns, etc. | 5% |

70% of all errors live in the first three categories. Those are the
ones v2 is designed to attack.

## 9.2 The case that crystallized the v2 decision

A real recording from a UAE-based clinical pilot:

> Patient: "اعطيتها voltaren"
>
> ASR (v1 LoRA): "اعطيتها فؤاد علي النزار"
>
> Correct: "اعطيتها voltaren"

The acoustic decode produced "فؤاد علي النزار" — three Arabic
words that happen to share the consonant skeleton of "voltaren"
(*v* → ف, *l* → ل, *t* → ت, *r* → ر, *n* → ن). The phonetic
flagger then tried to fix it but ran into a precision-vs-recall wall:

- If we trust phonetic similarity 0.55, the flagger correctly proposes
  "voltaren" — but it also proposes "fluconazole" for the name
  "فؤاد" in unrelated transcripts (false positive).
- If we tighten to 0.85, the flagger no longer catches voltaren
  (similarity here is only ~0.62 because the acoustic decode produced
  *three* tokens spanning the drug, breaking the skeleton match).

We invested several rounds in tightening the flagger:
- LCS-3 precision filter for the 0.55–0.65 ambiguous band.
- 70+ Arabic name lexicon to suppress name-as-drug matches.
- Drug-vs-disease tiebreaker.
- LCS boost for short-overlap-but-strong-evidence cases.

These fixes pushed the flagger to 100% on a 50/50 hand-curated suite.
But "voltaren → فؤاد علي النزار" still fails the flagger because the
problem is in the **acoustic decode**, not in the post-processing.
A 3-token Arabic phrase that fluently spans the drug name is a textual
artifact the deterministic flagger cannot reverse.

**Conclusion**: the residual medical errors are an acoustic /
vocabulary problem at the ASR level. Post-processing fixes have
reached their ceiling. The ASR needs to see medical vocabulary at
training time.

## 9.3 What we tried before deciding to retrain

| Attempt | Outcome |
|---|---|
| Add medical_terms.txt as context biasing prompt to Qwen3-ASR | Modest: ~5% relative WER reduction on medical-only clips. Does not handle multi-token mangles. |
| Replace audio encoder with MMS-1B | Encoder swap broke decoder alignment; abandoned. |
| Increase LoRA rank to 128 | Marginal improvement on general WER, no improvement on medical. We are not capacity-bound on the existing 900h, we are vocabulary-bound. |
| Whisper second-pass with VAD | Whisper produced English-only output for code-switched audio. Cannot be used as a primary backend. |
| LLM judge between Qwen and Whisper transcripts | Helps when Qwen mis-recognizes a drug but Whisper got it. Doesn't help when both fail. |

The pattern across all these attempts: **post-processing has a ceiling
of around the model's acoustic confusion margin**. Once the acoustic
decode produces a fluent-sounding Arabic phrase covering an English
drug, no post-processor can recover the drug name without knowing
what the patient actually said.

## 9.4 The v2 decision

Retrain the LoRA on a corpus that includes the medical vocabulary.
Specifically:

1. Generate ~60h of **synthetic Gulf medical audio** by:
   - LLM-generating realistic Gulf clinical sentences (Calme 78B).
   - Forcing drug + disease names to stay in English in the text.
   - Synthesizing audio with VoxCPM2 TTS using Gulf-accent voice
     prompts.

2. Mix back ~45h of the existing 900h general Gulf corpus as a
   **rehearsal anchor** to prevent catastrophic forgetting.

3. Add ~30h of **real Arabic-English code-switched** audio (MASC /
   MGB-3 / Mixat) to teach the boundary cases.

4. Add ~15h of **real English medical** audio (PriMock57 + Common
   Voice filtered by medical vocab) so English drug-name acoustic
   features are seen with native English pronunciation too.

5. Use a **tiered 10,000-term lexicon** (chapter 10) so common drugs
   are repeated 60× in training and the long tail still gets 2×
   exposure.

6. Apply a content filter at sentence-generation time so any LLM
   output that *transliterated* the drug to Arabic is silently
   dropped — bad training pairs never enter the dataset.

The detailed v2 recipe and the actual scripts that implement it are
in [10_finetuning_v2_plan.md](10_finetuning_v2_plan.md).

## 9.5 What this means for the flagger

The phonetic flagger is **not removed in v2**. It remains as the
deterministic safety net, but its job is downgraded: it goes from
"primary medical correction layer" to "last-resort catcher for the
remaining ~5% of medical errors after the new LoRA". The 100%
pass-rate on the hard suite is still expected, but the suite itself
is no longer a stress test for the whole product — it is a regression
guard.

This is the right asymmetry: the ASR does the bulk of the work, the
flagger catches edge cases, the LLM judge is the third line of defence
when both fail.
