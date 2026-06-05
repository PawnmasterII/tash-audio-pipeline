"""Layer 3 — generate synthetic labeled WAVs so the harness runs with no key.

Creates 16 kHz mono int16 files in ./test_audio:
  * clean_silence.wav      — near-silence (expect APNEA/AMBIGUOUS, no cue)
  * normal_breathing.wav   — ~15 breaths/min envelope (expect NORMAL)
  * agonal_gasps.wav       — sparse irregular gasps (expect LOW_RATE/AGONAL)
  * road_noise.wav         — broadband cabin-noise proxy (negative control)

These exercise Stages 1 & 3. Real "help" utterances are still needed for
Stage 2 (see README test corpus). Run:  python make_test_audio.py
"""
import os

import numpy as np

try:
    import soundfile as sf
except ImportError:
    raise SystemExit("pip install soundfile to generate test audio")

import config

SR = config.SAMPLE_RATE
OUT = "test_audio"
RNG = np.random.default_rng(0)


def _to_int16(x: np.ndarray) -> np.ndarray:
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.6
    return np.clip(x * 32768, -32768, 32767).astype(np.int16)


def _breath_burst(dur_s: float, fc: float = 600.0) -> np.ndarray:
    """One breath: band-ish noise under a smooth attack/decay envelope."""
    n = int(dur_s * SR)
    noise = RNG.standard_normal(n)
    env = np.sin(np.linspace(0, np.pi, n)) ** 2          # smooth bump
    carrier = np.sin(2 * np.pi * fc * np.arange(n) / SR)
    return noise * env * (0.5 + 0.5 * carrier)


def _series(rate_bpm: float, total_s: float, jitter: float = 0.0) -> np.ndarray:
    out = np.zeros(int(total_s * SR), dtype=float)
    period = 60.0 / rate_bpm
    t = 1.0
    while t < total_s - 1.0:
        burst = _breath_burst(0.4)
        i = int(t * SR)
        out[i : i + burst.size] += burst[: out.size - i]
        t += period * (1.0 + RNG.uniform(-jitter, jitter))
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    files = {
        "clean_silence.wav": RNG.standard_normal(SR * 25) * 1e-3,
        "normal_breathing.wav": _series(15, 25, jitter=0.1),
        "agonal_gasps.wav": _series(4, 25, jitter=0.6),       # slow + irregular
        "road_noise.wav": RNG.standard_normal(SR * 25) * 0.3,
    }
    for name, sig in files.items():
        sf.write(os.path.join(OUT, name), _to_int16(sig), SR, subtype="PCM_16")
        print(f"wrote {OUT}/{name}  ({sig.size / SR:.1f}s)")
    print("\nNow run:  python test_harness.py --chunk-ms 512")


if __name__ == "__main__":
    main()
