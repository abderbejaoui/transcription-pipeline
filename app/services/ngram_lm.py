"""Pure Python Kneser-Ney interpolated n-gram language model.

No external dependencies beyond Python stdlib. Trained on medical text,
used to detect words that are contextually anomalous (likely ASR errors).

Usage:
    lm = NGramLM(order=4, discount=0.75)
    lm.train(tokenized_sentences)
    ppl = lm.word_perplexity("pain", ["chest", "severe"])   # -log prob
    lm.save("medical_lm.pkl")
    lm2 = NGramLM.load("medical_lm.pkl")
"""

from __future__ import annotations

import math
import pickle
from collections import defaultdict, Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple


class NGramLM:
    """Kneser-Ney interpolated n-gram language model.

    Uses modified Kneser-Ney smoothing with absolute discounting.
    Supports word-level log-probability and context perplexity scoring.
    """

    def __init__(self, order: int = 4, discount: float = 0.75):
        self.order = order
        self.discount = discount
        # n-gram counts: {(w_{i-n+1}, ..., w_i): count}
        self.counts: Dict[Tuple[str, ...], int] = {}
        # context counts: {(w_{i-n+1}, ..., w_{i-1}): total_count}
        self.context_counts: Dict[Tuple[str, ...], int] = {}
        # continuation counts: {w_i: number of distinct contexts w_i follows}
        self.continuation: Dict[str, int] = {}
        # unigram counts for lowest-order fallback
        self.unigram_counts: Dict[str, int] = {}
        self.vocab: Set[str] = set()
        self.total_tokens: int = 0
        self.begin_token: str = "<s>"
        self.end_token: str = "</s>"
        self.unk_token: str = "<unk>"

    def _tokenize_sentence(self, sentence: List[str]) -> List[str]:
        """Add begin/end markers."""
        return [self.begin_token] * (self.order - 1) + sentence + [self.end_token]

    def train(self, sentences: Iterable[List[str]]) -> None:
        """Train the n-gram model from tokenized sentences.

        Args:
            sentences: Iterable of lists of word-level tokens.
        """
        # --- First pass: collect all counts ---
        unigram_counts: Dict[str, int] = Counter()
        bigram_contexts: Dict[str, Set[str]] = defaultdict(set)  # word -> set of prev words
        ngram_counts: Dict[Tuple[str, ...], int] = Counter()

        for sent in sentences:
            if not sent:
                continue
            tokens = self._tokenize_sentence(sent)
            for token in tokens:
                unigram_counts[token] += 1
                self.vocab.add(token)

            # Count all n-grams of orders 1..self.order
            for n in range(1, self.order + 1):
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i + n])
                    ngram_counts[ngram] += 1

                    # For continuation counts: for n >= 2, the last word follows
                    # the first n-1 words as context
                    if n >= 2:
                        context = ngram[:-1]
                        word = ngram[-1]
                        bigram_contexts[word].add(context)

        self.counts = dict(ngram_counts)
        self.unigram_counts = dict(unigram_counts)
        self.total_tokens = sum(unigram_counts.values())
        self.continuation = {w: len(ctxs) for w, ctxs in bigram_contexts.items()}

        # Compute context counts: for each context, total n-gram count starting with it
        context_counts: Dict[Tuple[str, ...], int] = Counter()
        for ngram, count in self.counts.items():
            if len(ngram) >= 2:
                context = ngram[:-1]
                context_counts[context] += count
        self.context_counts = dict(context_counts)

        # Add <unk> token to vocab
        self.vocab.add(self.unk_token)

    def _get_continuation_count(self, word: str) -> int:
        """Get the number of distinct contexts the word follows."""
        return self.continuation.get(word, 0)

    def _get_unigram_prob(self, word: str) -> float:
        """P(w) = count(w) / total_tokens, with <unk> backoff."""
        count = self.unigram_counts.get(word, 0)
        if count > 0:
            return count / self.total_tokens
        # OOV: assign very small probability
        return 0.5 / self.total_tokens

    def _get_continuation_prob(self, word: str) -> float:
        """P_continuation(w) = number of distinct contexts w follows / total distinct contexts."""
        total_continuation = sum(self.continuation.values())
        if total_continuation == 0:
            return self._get_unigram_prob(word)
        cont = self._get_continuation_count(word)
        return cont / total_continuation

    def _get_ngram_count(self, ngram: Tuple[str, ...]) -> int:
        """Get n-gram count, returning 0 if not found."""
        return self.counts.get(ngram, 0)

    def _get_context_count(self, context: Tuple[str, ...]) -> int:
        """Get context count, returning 0 if not found."""
        return self.context_counts.get(context, 0)

    def _interpolated_prob(self, word: str, *context: str) -> float:
        """Compute P_kn(word | context) using interpolated Kneser-Ney.

        Recursively backs off to lower-order models.
        """
        if len(context) == 0:
            # Unigram with continuation smoothing
            return self._get_continuation_prob(word)

        ngram = tuple(context) + (word,)
        ngram_count = self._get_ngram_count(ngram)
        context_count = self._get_context_count(tuple(context))

        if context_count == 0:
            # Unknown context: back off to lower order
            return self._interpolated_prob(word, *context[1:])

        D = self.discount
        max_term = max(ngram_count - D, 0) / context_count

        # Lambda: probability mass discounted away
        # Count how many unique words follow this context
        unique_follows = 0
        for ng, cnt in self.counts.items():
            if len(ng) == len(context) + 1 and ng[:-1] == tuple(context):
                unique_follows += 1

        lam = (D * unique_follows) / context_count

        # Backoff to lower-order
        backoff = self._interpolated_prob(word, *context[1:])

        return max_term + lam * backoff

    def log_prob(self, word: str, *context: str) -> float:
        """Return log10 probability of word given context.

        Returns:
            Log10 probability (negative float). More negative = less likely.
        """
        prob = self._interpolated_prob(word, *context)
        if prob <= 0:
            # Floor to avoid log(0)
            prob = 1e-10
        return math.log10(prob)

    def word_perplexity(self, word: str, context: List[str]) -> float:
        """Compute word-level perplexity contribution.

        Returns -log10(word | context). Higher = more surprising = more suspicious.
        """
        # Pad context to at least order-1 tokens with begin markers
        full_context = [self.begin_token] * (self.order - 1) + context
        # Use the last (order-1) words as context
        ctx = tuple(full_context[-(self.order - 1):])
        return -self.log_prob(word, *ctx)

    def sentence_log_prob(self, tokens: List[str]) -> float:
        """Compute total log10 probability of a sentence."""
        full = [self.begin_token] * (self.order - 1) + tokens + [self.end_token]
        total = 0.0
        for i in range(self.order - 1, len(full)):
            word = full[i]
            context = full[max(0, i - self.order + 1):i]
            total += self.log_prob(word, *context)
        return total

    def sentence_perplexity(self, tokens: List[str]) -> float:
        """Compute standard perplexity of a sentence (lower = more fluent)."""
        log_prob = self.sentence_log_prob(tokens)
        n = len(tokens)
        if n == 0:
            return float('inf')
        return math.pow(10, -log_prob / n)

    def save(self, path: str) -> None:
        """Save model to pickle file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'discount': self.discount,
                'counts': self.counts,
                'context_counts': self.context_counts,
                'continuation': self.continuation,
                'unigram_counts': self.unigram_counts,
                'vocab': self.vocab,
                'total_tokens': self.total_tokens,
                'begin_token': self.begin_token,
                'end_token': self.end_token,
                'unk_token': self.unk_token,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> 'NGramLM':
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        lm = cls(order=data['order'], discount=data['discount'])
        lm.counts = data['counts']
        lm.context_counts = data['context_counts']
        lm.continuation = data['continuation']
        lm.unigram_counts = data['unigram_counts']
        lm.vocab = data['vocab']
        lm.total_tokens = data['total_tokens']
        lm.begin_token = data['begin_token']
        lm.end_token = data['end_token']
        lm.unk_token = data['unk_token']
        return lm

    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)

    def get_total_ngrams(self) -> int:
        """Return the total number of unique n-grams stored."""
        return len(self.counts)


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

import re

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|[\u0600-\u06FF][\u0600-\u06FF\-]*|\d+(?:\.\d+)?|[^\w\s]")


def tokenize(text: str) -> List[str]:
    """Tokenize a text into words. Drops pure-punctuation tokens.

    Uses the same token regex as the MedicalCorrector for consistency.
    """
    tokens = [m.group() for m in _WORD_RE.finditer(text)]
    # Drop standalone punctuation
    return [t for t in tokens if len(t) >= 1 and not _is_punct(t)]


def _is_punct(t: str) -> bool:
    """Check if a string is pure punctuation."""
    return not any(c.isalnum() or '\u0600' <= c <= '\u06FF' for c in t)
