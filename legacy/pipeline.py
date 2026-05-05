"""
Text -> Sound (Pronunciation) Embedding pipeline.

Pipeline
--------
    medical term (str)
        |
        v
    TTS  : microsoft/speecht5_tts  (+ HiFi-GAN vocoder)
        |
        v
    16 kHz waveform (numpy array)
        |
        v
    Encoder : facebook/wav2vec2-base  (self-supervised, phonetically rich)
        |
        v
    Mean-pooled last hidden state -> L2-normalized 768-D embedding


Why this design
---------------
The goal is to compare a text-derived embedding with an embedding of real
human audio. If we used two different encoders (one for text, one for audio)
the embeddings would live in different spaces and cosine similarity would be
meaningless. Here we force both paths through the SAME audio encoder
(wav2vec2), so `text_embed(word)` and `audio_embed(recording_of_word)` are
directly comparable with cosine similarity.

wav2vec2's hidden states are well known to encode phonetic / acoustic
content (WavThruVec, SpeechT5 speech pre-net, countless pronunciation
assessment papers). Mean-pooling over time gives a fixed-size utterance
embedding robust to length differences.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)


TTS_MODEL_ID = "microsoft/speecht5_tts"
VOCODER_MODEL_ID = "microsoft/speecht5_hifigan"
ENCODER_MODEL_ID = "facebook/wav2vec2-base"
SAMPLE_RATE = 16_000

# Cache for one good speaker x-vector so we don't need the `datasets` legacy
# loader. The embedding itself is tiny (512 floats).
_SPK_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".speaker_xvector.npy"
)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_speaker_xvector(
    filename: str = "cmu_us_slt_arctic-wav-arctic_a0508.bin",
) -> np.ndarray:
    """Return a single 512-dim x-vector suitable for SpeechT5 TTS.

    Downloads ONE tiny (~2 KB) x-vector file from the
    `Xenova/cmu-arctic-xvectors-extracted` repo, caches it as
    `.speaker_xvector.npy` next to this file, and returns it. Subsequent
    runs read straight from the cache.

    `cmu_us_slt_arctic` is a clear US-English female voice - a good default
    for pronouncing medical terminology.
    """
    if os.path.exists(_SPK_CACHE_PATH):
        return np.load(_SPK_CACHE_PATH).astype(np.float32)

    print("[load] Fetching speaker x-vector (one-time, ~2 KB)...")
    path = hf_hub_download(
        repo_id="Xenova/cmu-arctic-xvectors-extracted",
        filename=filename,
        repo_type="dataset",
    )
    # Each .bin file is a raw float32 x-vector of length 512.
    xv = np.fromfile(path, dtype=np.float32)
    if xv.size != 512:
        raise RuntimeError(
            f"Unexpected x-vector length {xv.size} from {filename}"
        )
    np.save(_SPK_CACHE_PATH, xv)
    return xv


@dataclass
class SoundEmbedder:
    """Bundle of models needed to go from text -> pronunciation embedding."""

    tts_processor: SpeechT5Processor
    tts_model: SpeechT5ForTextToSpeech
    vocoder: SpeechT5HifiGan
    speaker_embedding: torch.Tensor  # shape (1, 512)
    enc_feat: Wav2Vec2FeatureExtractor
    enc_model: Wav2Vec2Model
    device: torch.device

    @classmethod
    def load(
        cls,
        device: Optional[torch.device] = None,
    ) -> "SoundEmbedder":
        device = device or _pick_device()
        print(f"[load] device = {device}")

        print(f"[load] TTS: {TTS_MODEL_ID}")
        tts_processor = SpeechT5Processor.from_pretrained(TTS_MODEL_ID)
        tts_model = SpeechT5ForTextToSpeech.from_pretrained(TTS_MODEL_ID).to(device).eval()

        print(f"[load] Vocoder: {VOCODER_MODEL_ID}")
        vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_MODEL_ID).to(device).eval()

        xv = _load_speaker_xvector()
        spk = torch.from_numpy(xv).unsqueeze(0).to(device)

        print(f"[load] Audio encoder: {ENCODER_MODEL_ID}")
        enc_feat = Wav2Vec2FeatureExtractor.from_pretrained(ENCODER_MODEL_ID)
        enc_model = Wav2Vec2Model.from_pretrained(ENCODER_MODEL_ID).to(device).eval()

        return cls(
            tts_processor=tts_processor,
            tts_model=tts_model,
            vocoder=vocoder,
            speaker_embedding=spk,
            enc_feat=enc_feat,
            enc_model=enc_model,
            device=device,
        )

    @torch.inference_mode()
    def synthesize(self, text: str) -> np.ndarray:
        """Text -> 16 kHz waveform (float32 numpy array)."""
        inputs = self.tts_processor(text=text, return_tensors="pt").to(self.device)
        speech = self.tts_model.generate_speech(
            inputs["input_ids"], self.speaker_embedding, vocoder=self.vocoder
        )
        return speech.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def audio_to_embedding(self, waveform: np.ndarray) -> np.ndarray:
        """16 kHz waveform -> 768-D L2-normalized embedding.

        Mean-pooling over time gives a fixed-size representation of the whole
        utterance that is comparable across different word lengths.
        """
        inputs = self.enc_feat(
            waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).to(self.device)
        out = self.enc_model(**inputs)
        hidden = out.last_hidden_state  # (1, T, 768)
        pooled = hidden.mean(dim=1).squeeze(0)  # (768,)
        pooled = torch.nn.functional.normalize(pooled, dim=0)
        return pooled.detach().cpu().numpy().astype(np.float32)

    def embed_term(self, text: str) -> np.ndarray:
        """text -> 768-D pronunciation embedding."""
        wav = self.synthesize(text)
        return self.audio_to_embedding(wav)

    def embed_terms(
        self,
        terms: Sequence[str],
        save_audio_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """Batch version. Returns shape (N, 768).

        If `save_audio_dir` is given, saves each synthesized waveform as a
        16 kHz mono WAV file named `<index>_<slug>.wav` in that directory
        (useful for debugging / listening to what the TTS produced).
        """
        if save_audio_dir:
            os.makedirs(save_audio_dir, exist_ok=True)

        vectors: List[np.ndarray] = []
        for i, term in enumerate(terms):
            wav = self.synthesize(term)
            emb = self.audio_to_embedding(wav)
            vectors.append(emb)

            if save_audio_dir:
                slug = "".join(c if c.isalnum() else "_" for c in term)[:40]
                sf.write(
                    os.path.join(save_audio_dir, f"{i:03d}_{slug}.wav"),
                    wav,
                    SAMPLE_RATE,
                )
            if verbose:
                print(f"  [{i+1:>3}/{len(terms)}] {term!r}  -> emb shape {emb.shape}")

        return np.stack(vectors, axis=0)

    def embed_audio_file(self, path: str) -> np.ndarray:
        """Embed a real audio recording (any format soundfile can read).

        The file is resampled to 16 kHz mono if needed. Produces a 768-D
        embedding directly comparable with `embed_term(...)`.
        """
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)  # stereo -> mono
        if sr != SAMPLE_RATE:
            from scipy.signal import resample_poly

            from math import gcd

            g = gcd(sr, SAMPLE_RATE)
            wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
        return self.audio_to_embedding(wav)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D embeddings."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def cosine_similarity_matrix(emb: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for a matrix of shape (N, D)."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    normed = emb / norms
    return normed @ normed.T


def load_terms(path: str) -> List[str]:
    """Read medical terms from a text file (one per line, # for comments)."""
    terms: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            terms.append(line)
    return terms
