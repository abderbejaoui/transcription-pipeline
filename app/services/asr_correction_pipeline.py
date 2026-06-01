"""End-to-end ASR correction pipeline for mixed Arabic/English medical text.

Phases:
1) MedicalDictionaryManager with G2P hook.
2) Dual-metric candidate search (accent-mapped edit distance + phoneme distance).
3) Runtime ASR monitor for low-confidence English clusters.
4) LLM prompt builder for semantic verification.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


_ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06ff]")
_NON_LATIN_RE = re.compile(r"[^a-z]")
_LATIN_VOWELS = set("aeiouy")


@dataclass(frozen=True)
class MedicalEntry:
    name: str
    clean_text: str
    phonemes: str
    meta: Optional[Dict[str, Any]] = None


def sanitize_name(text: str) -> str:
    """Lowercase and strip non-letter characters for matching."""
    return _NON_LATIN_RE.sub("", text.lower())


def _consonant_skeleton(text: str) -> str:
    """Remove vowels to emphasize consonant structure."""
    return "".join(ch for ch in text if ch not in _LATIN_VOWELS)


def is_latin_token(token: str) -> bool:
    """True if token has Latin letters and no Arabic letters."""
    if _ARABIC_LETTER_RE.search(token):
        return False
    return any("a" <= ch.lower() <= "z" for ch in token)


def simple_g2p(word: str) -> str:
    """Lightweight G2P stub that can be replaced by g2p_en/epitran.

    Example:
        "lisinopril" -> "laɪsɪnəprɪl" (approximate)
    """
    w = sanitize_name(word)
    if not w:
        return ""

    digraphs = {
        "ph": "f",
        "sh": "ʃ",
        "ch": "tʃ",
        "th": "θ",
        "ng": "ŋ",
        "qu": "kw",
        "ck": "k",
        "gh": "ɣ",
        "kh": "x",
        "oo": "u",
        "ee": "i",
        "ea": "i",
        "ai": "eɪ",
        "ay": "eɪ",
        "oi": "ɔɪ",
        "ou": "aʊ",
        "ow": "aʊ",
        "er": "ɝ",
        "ar": "ɑr",
        "or": "ɔr",
        "ur": "ɝ",
    }
    single = {
        "a": "æ",
        "e": "ɛ",
        "i": "ɪ",
        "o": "ɒ",
        "u": "ʌ",
        "y": "ɪ",
        "b": "b",
        "c": "k",
        "d": "d",
        "f": "f",
        "g": "g",
        "h": "h",
        "j": "dʒ",
        "k": "k",
        "l": "l",
        "m": "m",
        "n": "n",
        "p": "p",
        "q": "k",
        "r": "r",
        "s": "s",
        "t": "t",
        "v": "v",
        "w": "w",
        "x": "ks",
        "z": "z",
    }

    out: List[str] = []
    i = 0
    while i < len(w):
        if i + 1 < len(w):
            pair = w[i : i + 2]
            if pair in digraphs:
                out.append(digraphs[pair])
                i += 2
                continue
        out.append(single.get(w[i], w[i]))
        i += 1
    return "".join(out)


class MedicalDictionaryManager:
    """Manages medical terms and their phoneme representations."""

    def __init__(
        self,
        names: Sequence[str],
        *,
        g2p: Optional[Callable[[str], str]] = None,
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._g2p = g2p or simple_g2p
        self._entries: List[MedicalEntry] = []
        self._by_clean: Dict[str, MedicalEntry] = {}
        self._build(names, metadata or {})

    def _build(self, names: Sequence[str], metadata: Dict[str, Dict[str, Any]]) -> None:
        for name in names:
            clean = sanitize_name(name)
            if not clean or clean in self._by_clean:
                continue
            phonemes = self._g2p(clean)
            entry = MedicalEntry(
                name=name,
                clean_text=clean,
                phonemes=phonemes,
                meta=metadata.get(clean),
            )
            self._entries.append(entry)
            self._by_clean[clean] = entry

    def entries(self) -> Sequence[MedicalEntry]:
        return tuple(self._entries)

    def phonemes_for(self, text: str) -> str:
        return self._g2p(text)


_ACCENT_SUB_COST: Dict[Tuple[str, str], float] = {
    ("p", "b"): 0.1,
    ("b", "p"): 0.1,
    ("v", "f"): 0.1,
    ("f", "v"): 0.1,
    ("o", "u"): 0.1,
    ("u", "o"): 0.1,
    ("s", "c"): 0.1,
    ("c", "s"): 0.1,
}


def accent_mapped_levenshtein(a: str, b: str) -> float:
    """Weighted Levenshtein with low-cost UAE accent swaps."""
    if not a and not b:
        return 0.0
    if not a:
        return float(len(b))
    if not b:
        return float(len(a))

    n, m = len(a), len(b)
    prev = list(range(m + 1))
    cur = [0.0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = float(i)
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            if ai == bj:
                sub_cost = 0.0
            else:
                sub_cost = _ACCENT_SUB_COST.get((ai, bj), 1.0)
            # DP step: min(delete, insert, substitute) for prefixes.
            cur[j] = min(prev[j] + 1.0, cur[j - 1] + 1.0, prev[j - 1] + sub_cost)
        prev, cur = cur, prev
    return float(prev[m])


def levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein distance on generic strings."""
    if not a and not b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    n, m = len(a), len(b)
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return int(prev[m])


def _normalized_similarity(dist: float, len_a: int, len_b: int) -> float:
    denom = max(len_a, len_b)
    if denom <= 0:
        return 0.0
    return max(0.0, 1.0 - dist / float(denom))


def find_top_k_candidates(
    mangled_word: str,
    dictionary_manager: MedicalDictionaryManager,
    *,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Rank medical terms using consonant-weighted scoring."""
    clean = sanitize_name(mangled_word)
    if not clean:
        return []
    mangled_phonemes = dictionary_manager.phonemes_for(clean)
    mangled_cons = _consonant_skeleton(clean)

    scored: List[Dict[str, Any]] = []
    for entry in dictionary_manager.entries():
        char_dist = accent_mapped_levenshtein(clean, entry.clean_text)
        char_sim = _normalized_similarity(char_dist, len(clean), len(entry.clean_text))

        phon_dist = levenshtein(mangled_phonemes, entry.phonemes)
        phon_sim = _normalized_similarity(phon_dist, len(mangled_phonemes), len(entry.phonemes))

        entry_cons = _consonant_skeleton(entry.clean_text)
        cons_dist = accent_mapped_levenshtein(mangled_cons, entry_cons)
        cons_sim = _normalized_similarity(cons_dist, len(mangled_cons), len(entry_cons))

        score = 0.60 * cons_sim + 0.15 * char_sim + 0.25 * phon_sim
        scored.append(
            {
                "term": entry.name,
                "score": score,
                "char_distance": char_dist,
                "consonant_distance": cons_dist,
                "phoneme_distance": phon_dist,
                "phonemes": entry.phonemes,
                "meta": entry.meta,
            }
        )

    scored.sort(key=lambda c: (-c["score"], c["consonant_distance"], c["char_distance"], c["phoneme_distance"]))
    return scored[: max(1, int(top_k))]


def _iter_english_clusters(tokens: Sequence[str]) -> Iterable[Tuple[int, int]]:
    i = 0
    while i < len(tokens):
        if not is_latin_token(tokens[i]):
            i += 1
            continue
        start = i
        i += 1
        while i < len(tokens) and is_latin_token(tokens[i]):
            i += 1
        yield (start, i)


def process_asr_output(
    text_tokens: Sequence[str],
    logprobs: Sequence[float],
    dictionary_manager: MedicalDictionaryManager,
    *,
    confidence_threshold: float = 0.80,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Scan ASR output and flag low-confidence English clusters."""
    if len(text_tokens) != len(logprobs):
        raise ValueError("text_tokens and logprobs must have equal length")

    flagged: List[Dict[str, Any]] = []
    for start, end in _iter_english_clusters(text_tokens):
        cluster_tokens = text_tokens[start:end]
        cluster_logprobs = logprobs[start:end]
        probs = [math.exp(lp) for lp in cluster_logprobs]
        mean_conf = sum(probs) / max(1, len(probs))
        if mean_conf >= confidence_threshold:
            continue

        raw_text = " ".join(cluster_tokens)
        mangled = "".join(cluster_tokens)
        candidates = find_top_k_candidates(mangled, dictionary_manager, top_k=top_k)
        flagged.append(
            {
                "index_start": start,
                "index_end": end,
                "raw_text": raw_text,
                "confidence": mean_conf,
                "candidates": candidates,
            }
        )
    return flagged


def build_llm_correction_prompt(
    full_asr_transcript: str,
    mangled_word: str,
    top_five_candidates: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    """Create system + user prompt instructing the LLM to choose 1 candidate."""
    mask = "<MASK>"
    if mangled_word in full_asr_transcript:
        masked = full_asr_transcript.replace(mangled_word, mask, 1)
    else:
        masked = full_asr_transcript

    lines: List[str] = []
    for c in top_five_candidates[:5]:
        term = str(c.get("term") or "")
        uses = c.get("meta", {}).get("uses") if isinstance(c.get("meta"), dict) else None
        score = c.get("score")
        if score is None:
            score = c.get("phonetic_similarity")
        score_str = f" (score={float(score):.3f})" if isinstance(score, (int, float)) else ""
        if uses:
            lines.append(f"- {term}{score_str}: {uses}")
        else:
            lines.append(f"- {term}{score_str}")

    system_prompt = (
        "You are a clinical ASR correction assistant for Gulf Arabic + English code-switching. "
        "Pick the single most medically accurate replacement for the masked term. "
        "Use the Arabic context and the candidate list. Output ONLY the chosen word."
    )
    user_prompt = (
        "ASR transcript (masked):\n"
        f"{masked}\n\n"
        "Masked word: "
        f"{mangled_word}\n\n"
        "Top candidate replacements:\n"
        + "\n".join(lines)
    )
    return {"system": system_prompt, "user": user_prompt}


def load_medicine_details_csv(path: Path) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Load medicine names and uses from the CSV file.

    Returns:
        names: list of medicine names
        metadata: dict keyed by sanitized name -> {"uses": <text>}
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    names: List[str] = []
    metadata: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row.get("Medicine Name") or "").strip()
            uses = (row.get("Uses") or "").strip()
            if not name:
                continue
            clean = sanitize_name(name)
            if not clean:
                continue
            names.append(name)
            if uses:
                metadata.setdefault(clean, {})["uses"] = uses
    return names, metadata


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parents[2] / "data" / "Medicine_Details.csv"
    if csv_path.exists():
        names, meta = load_medicine_details_csv(csv_path)
        manager = MedicalDictionaryManager(names, metadata=meta)
    else:
        names = ["Paracetamol", "Lisinopril", "Metformin", "Amoxicillin", "Ventolin"]
        manager = MedicalDictionaryManager(names)

    tokens = ["المريض", "بياخذ", "bacetamol", "عشرة", "ملغ"]
    logprobs = [-0.05, -0.10, -2.50, -0.05, -0.05]
    transcript = "المريض بياخذ bacetamol عشرة ملغ"

    flagged = process_asr_output(tokens, logprobs, manager, confidence_threshold=0.80)
    for item in flagged:
        print("Low-confidence cluster:", item["raw_text"], "conf=", round(item["confidence"], 3))
        print("Top candidates:")
        for cand in item["candidates"]:
            print("  ", cand["term"], "score=", round(cand["score"], 3))

        prompt = build_llm_correction_prompt(
            transcript, item["raw_text"], item["candidates"]
        )
        print("\nSYSTEM PROMPT:\n", prompt["system"])
        print("\nUSER PROMPT:\n", prompt["user"])
