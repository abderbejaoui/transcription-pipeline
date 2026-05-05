"""Optional audio-grounding helpers for transcript correction.

Use this when you have raw audio plus timestamps for a suspicious span.
It ranks candidate medical terms by comparing:

    real audio segment embedding
        vs
    synthesized candidate pronunciation embedding

This is intentionally optional because loading SpeechT5 + wav2vec2 is heavy.
The text correction pipeline in `medical_corrector.py` works without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import soundfile as sf

from pipeline import SAMPLE_RATE, SoundEmbedder, cosine_similarity


@dataclass(frozen=True)
class AudioCandidateScore:
    candidate: str
    score: float


class AudioGrounder:
    def __init__(self, embedder: SoundEmbedder | None = None) -> None:
        self.embedder = embedder or SoundEmbedder.load()

    def rank_candidates_for_segment(
        self,
        audio_path: str,
        start_seconds: float,
        end_seconds: float,
        candidates: Sequence[str],
    ) -> List[AudioCandidateScore]:
        """Rank candidate terms against a real audio segment."""
        waveform = self._load_segment(audio_path, start_seconds, end_seconds)
        audio_emb = self.embedder.audio_to_embedding(waveform)

        scores: List[AudioCandidateScore] = []
        for candidate in candidates:
            text_emb = self.embedder.embed_term(candidate)
            scores.append(
                AudioCandidateScore(
                    candidate=candidate,
                    score=cosine_similarity(audio_emb, text_emb),
                )
            )
        return sorted(scores, key=lambda item: item.score, reverse=True)

    def _load_segment(self, audio_path: str, start_seconds: float, end_seconds: float) -> np.ndarray:
        if end_seconds <= start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")

        waveform, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        start = max(0, int(start_seconds * sr))
        end = min(len(waveform), int(end_seconds * sr))
        segment = waveform[start:end]

        if sr != SAMPLE_RATE:
            from math import gcd
            from scipy.signal import resample_poly

            g = gcd(sr, SAMPLE_RATE)
            segment = resample_poly(segment, SAMPLE_RATE // g, sr // g).astype(np.float32)

        return segment
