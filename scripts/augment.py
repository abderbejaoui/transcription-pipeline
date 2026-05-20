"""Audio augmentations used by the Qwen3-ASR LoRA fine-tune.

Plug into `AsrDataset` by wrapping the array returned by `__getitem__`. These
are deliberately CPU-only and stateless so multiple DataLoader workers can run
them in parallel without GPU contention.

Defaults are tuned for Gulf-Arabic dialect training:
  - SpecAugment freq_mask=15, time_mask=70, p=0.5
  - Volume jitter ±6 dB
  - Speed perturbation in {0.9, 1.0, 1.1}
  - Additive MUSAN noise at SNR 10–20 dB, p=0.3 (requires --musan-dir)
  - Reverb p=0.2 (skipped unless --rir-dir is given)

SpecAugment runs on the log-mel features, so the helper here is invoked
*inside* the collator, not on the raw waveform.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence


# ----------------------------------------------------------------------------
# Waveform-level augs
# ----------------------------------------------------------------------------


def volume_jitter(wave, max_db: float = 6.0):
    import numpy as np
    gain_db = random.uniform(-max_db, max_db)
    return (wave * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def speed_perturb(wave, sr: int, choices: Sequence[float] = (0.9, 1.0, 1.1)):
    speed = random.choice(list(choices))
    if speed == 1.0:
        return wave, sr
    try:
        import soxr
        return soxr.resample(wave, sr, int(sr / speed)).astype("float32"), sr
    except ImportError:
        import librosa
        return librosa.effects.time_stretch(wave, rate=speed).astype("float32"), sr


def add_noise(wave, noise, snr_db_range=(10.0, 20.0)):
    """Add `noise` to `wave` at a random SNR in the given range."""
    import numpy as np
    if len(noise) < len(wave):
        reps = (len(wave) // len(noise)) + 1
        noise = np.tile(noise, reps)
    noise = noise[: len(wave)]
    snr_db = random.uniform(*snr_db_range)
    sig_power = float((wave ** 2).mean() + 1e-12)
    noise_power = float((noise ** 2).mean() + 1e-12)
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = (target_noise_power / noise_power) ** 0.5
    return (wave + scale * noise).astype("float32")


# ----------------------------------------------------------------------------
# Mel-spectrogram-level: SpecAugment
# ----------------------------------------------------------------------------


def specaugment(features, freq_mask: int = 15, time_mask: int = 70, num_masks: int = 2):
    """In-place SpecAugment on a (T, F) mel feature tensor."""
    import torch
    if features.ndim != 2:
        return features
    T, F = features.shape
    for _ in range(num_masks):
        # frequency mask
        f = random.randint(0, freq_mask)
        if f > 0 and F > f:
            f0 = random.randint(0, F - f)
            features[:, f0:f0 + f] = 0
        # time mask
        t = random.randint(0, time_mask)
        if t > 0 and T > t:
            t0 = random.randint(0, T - t)
            features[t0:t0 + t, :] = 0
    return features


# ----------------------------------------------------------------------------
# Pipeline wrapper
# ----------------------------------------------------------------------------


class WaveformAugmenter:
    """Stochastic chain of waveform augmentations.

    Used inside `AsrDataset.__getitem__` like:

        sample = ds[i]
        sample["audio"] = aug(sample["audio"], sample["sampling_rate"])
    """

    def __init__(
        self,
        musan_dir: Optional[Path] = None,
        rir_dir: Optional[Path] = None,
        p_vol: float = 1.0,
        p_speed: float = 0.5,
        p_noise: float = 0.3,
        p_reverb: float = 0.2,
        vol_max_db: float = 6.0,
        speed_choices: Sequence[float] = (0.9, 1.0, 1.1),
        snr_range=(10.0, 20.0),
    ):
        self.musan_files = sorted(Path(musan_dir).rglob("*.wav")) if musan_dir else []
        self.rir_files = sorted(Path(rir_dir).rglob("*.wav")) if rir_dir else []
        self.p_vol = p_vol
        self.p_speed = p_speed
        self.p_noise = p_noise
        self.p_reverb = p_reverb
        self.vol_max_db = vol_max_db
        self.speed_choices = speed_choices
        self.snr_range = snr_range

    def __call__(self, wave, sr: int):
        if random.random() < self.p_vol:
            wave = volume_jitter(wave, self.vol_max_db)
        if random.random() < self.p_speed:
            wave, sr = speed_perturb(wave, sr, self.speed_choices)
        if self.musan_files and random.random() < self.p_noise:
            import soundfile as sf
            noise_path = random.choice(self.musan_files)
            noise, nsr = sf.read(str(noise_path), dtype="float32", always_2d=False)
            if nsr != sr:
                try:
                    import soxr
                    noise = soxr.resample(noise, nsr, sr)
                except ImportError:
                    pass
            wave = add_noise(wave, noise, self.snr_range)
        if self.rir_files and random.random() < self.p_reverb:
            import numpy as np
            import soundfile as sf
            rir_path = random.choice(self.rir_files)
            rir, _ = sf.read(str(rir_path), dtype="float32", always_2d=False)
            wave = np.convolve(wave, rir, mode="full")[: len(wave)].astype("float32")
        return wave, sr
