"""Offline validation harness — feeds labeled WAV files through the pipeline.

Per architecture §5: NO training. We measure the pre-trained components on a
labeled corpus. Drop WAVs into ./test_audio and describe them in MANIFEST.

Run:  python test_harness.py [--chunk-ms 512] [--audio-dir test_audio]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import config
from contracts import EscalationLevel

try:
    import soundfile as sf
except ImportError:
    sf = None


def load_wav_int16(path: str) -> np.ndarray:
    """Load mono 16 kHz int16; assert the canonical contract (no silent resample)."""
    if sf is None:
        raise SystemExit("soundfile not installed; cannot run harness")
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    assert sr == config.SAMPLE_RATE, f"{path}: sr={sr}, expected {config.SAMPLE_RATE}"
    return data[:, 0]  # mono


def iter_chunks(pcm: np.ndarray, chunk_samples: int):
    for i in range(0, len(pcm) - chunk_samples + 1, chunk_samples):
        yield i / config.SAMPLE_RATE, pcm[i : i + chunk_samples]


def run_file(make_pipeline, path: str, chunk_samples: int) -> dict:
    # Fresh pipeline PER FILE: each WAV is an independent recording, so detector
    # debounce state and Vosk's streaming buffer must not carry across files.
    pipeline = make_pipeline()
    pcm = load_wav_int16(path)
    pipeline.prime_noise_baseline(pcm[: int(config.NOISE_BASELINE_SECONDS * config.SAMPLE_RATE)])
    levels: dict[EscalationLevel, int] = {lvl: 0 for lvl in EscalationLevel}
    cue_hits = 0
    last_ts = 0.0
    for ts, chunk in iter_chunks(pcm, chunk_samples):
        decision = pipeline.process_chunk(chunk, ts)
        levels[decision.level] += 1
        if decision.cue_word is not None:
            cue_hits += 1
        last_ts = ts
    final = pipeline.flush(last_ts + chunk_samples / config.SAMPLE_RATE)  # drain buffer
    levels[final.level] += 1
    if final.cue_word is not None:
        cue_hits += 1
    pipeline.close()
    return {"file": os.path.basename(path), "cue_hits": cue_hits,
            "levels": {k.value: v for k, v in levels.items() if v}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-ms", type=int, default=512)
    ap.add_argument("--audio-dir", default="test_audio")
    args = ap.parse_args()

    from pipeline import Pipeline  # imported late so --help works without deps

    chunk_samples = config.SAMPLE_RATE * args.chunk_ms // 1000
    files = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if not files:
        raise SystemExit(f"no .wav files in {args.audio_dir}/ — see README §test corpus")

    for path in files:
        print(run_file(Pipeline, path, chunk_samples))


if __name__ == "__main__":
    main()
