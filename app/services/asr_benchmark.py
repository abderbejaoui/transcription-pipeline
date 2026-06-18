"""Stub for asr_benchmark — the benchmark endpoint is not used in this deployment."""

MODELS: dict = {}


def run_asr(model_key: str, audio_path: str) -> dict:
    raise NotImplementedError("asr_benchmark is not available in this deployment.")
