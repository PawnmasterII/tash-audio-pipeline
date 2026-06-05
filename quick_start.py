"""TASHaudio — automated setup validator.

A new team member runs:  python quick_start.py
→ validates the full setup in ~30 seconds, printing 🟢 / 🔴 (and 🟡 for optional)
  for each step, and the exact command to fix anything that's broken.

Idempotent and read-only: safe to run as many times as you like. It never
downloads, installs, or modifies anything — it only inspects.

Cross-platform: all paths go through os.path / config (no hardcoded separators).
"""
from __future__ import annotations

import importlib
import os
import sys

# ── console glyphs (degrade gracefully if the terminal can't encode emoji) ───
try:
    "🟢🔴🟡✅❌".encode(sys.stdout.encoding or "utf-8")
    OK, BAD, WARN, DONE, FAIL = "🟢", "🔴", "🟡", "✅", "❌"
except (UnicodeEncodeError, TypeError):
    OK, BAD, WARN, DONE, FAIL = "[OK]", "[X]", "[!]", "[OK]", "[FAIL]"

# Activated-venv hint, correct for the current OS.
if os.name == "nt":
    VENV_HINT = r"py -3.12 -m venv .venv312 ; .\.venv312\Scripts\Activate.ps1"
else:
    VENV_HINT = "python3.12 -m venv .venv312 && source .venv312/bin/activate"

# Required runtime deps: import name → distribution name (for version lookup).
REQUIRED = {
    "numpy": "numpy",
    "scipy": "scipy",
    "noisereduce": "noisereduce",
    "vosk": "vosk",
    "pvrecorder": "pvrecorder",
    "librosa": "librosa",
    "soundfile": "soundfile",
}
# Optional: pipeline runs degraded (energy-VAD fallback) without these.
OPTIONAL = {
    "webrtcvad": "webrtcvad-wheels",  # imports as webrtcvad; no wheel on some platforms
}


def _version(dist_name: str) -> str:
    try:
        from importlib.metadata import version
        return version(dist_name)
    except Exception:
        return "?"


def check_python_version() -> bool:
    """Assert Python 3.12+."""
    v = sys.version_info
    if v < (3, 12):
        print(f"  {BAD} Python 3.12+ required. You have {v.major}.{v.minor}.{v.micro}. "
              f"See SETUP.md")
        return False
    print(f"  {OK} Python {v.major}.{v.minor}.{v.micro} found")
    if v >= (3, 13):
        print(f"  {WARN} 3.13+ has no webrtcvad-wheels wheel — Stage 2 will use the "
              f"coarse energy-VAD fallback. Use 3.12 for the real VAD.")
    return True


def check_venv_active() -> bool:
    """Assert we're running inside a virtual environment."""
    if sys.prefix == sys.base_prefix:
        print(f"  {BAD} Not in a virtual environment. Run:\n      {VENV_HINT}")
        return False
    print(f"  {OK} Virtual environment active ({sys.prefix})")
    return True


def check_imports() -> bool:
    """Try to import each dependency; report version. Optional deps never fail."""
    ok = True
    for mod, dist in REQUIRED.items():
        try:
            importlib.import_module(mod)
            print(f"  {OK} {mod} v{_version(dist)}")
        except Exception as e:  # noqa: BLE001 — surface any import-time failure
            ok = False
            print(f"  {BAD} Missing/broken: {mod} ({e}). "
                  f"Run: pip install -r requirements-lock.txt")
    for mod, dist in OPTIONAL.items():
        try:
            importlib.import_module(mod)
            print(f"  {OK} {mod} v{_version(dist)} (optional)")
        except Exception:
            print(f"  {WARN} {mod} not installed (optional) — Stage 2 uses the "
                  f"energy-VAD fallback. Real VAD: "
                  f"pip install --only-binary :all: webrtcvad-wheels")
    return ok


def check_vosk_model() -> bool:
    """Verify the unpacked Vosk model directory exists."""
    import config
    path = config.VOSK_MODEL_PATH
    if not os.path.isdir(path):
        print(f"  {BAD} Vosk model missing at {path}. "
              f"Run: python download_vosk_model.py")
        return False
    # Sanity: a real model has these sub-dirs.
    missing = [d for d in ("am", "conf", "graph") if not os.path.isdir(os.path.join(path, d))]
    if missing:
        print(f"  {BAD} Vosk model at {path} looks incomplete (missing {missing}). "
              f"Re-run: python download_vosk_model.py")
        return False
    print(f"  {OK} Vosk model found (~40 MB) at {os.path.relpath(path)}")
    return True


def check_contracts() -> bool:
    """Verify the stage data contracts import cleanly."""
    try:
        from contracts import (  # noqa: F401
            BreathingEstimate,
            BreathingState,
            CueWordEvent,
            DenoisedChunk,
            EscalationLevel,
            FusionDecision,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  {BAD} Contracts broken: {e}. Check contracts.py")
        return False
    print(f"  {OK} Contracts valid (DenoisedChunk, CueWordEvent, BreathingEstimate, …)")
    return True


def check_pipeline() -> bool:
    """Run a minimal end-to-end pass over 1 second of synthetic silence."""
    import numpy as np

    import config

    import warnings

    stage = "<import>"
    try:
        from stage1_denoise import Denoiser
        from stage2_cueword import CueWordDetector
        from stage3_breathing import BreathingDetector

        silence = np.zeros(config.SAMPLE_RATE, dtype=np.int16)  # 1 s @ 16 kHz

        # Silence makes noisereduce's spectral gate divide by ~0 (benign — stage 1
        # scrubs the resulting NaNs). Mute it so the validator output stays clean.
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        stage = "stage1_denoise"
        denoiser = Denoiser()
        chunk = denoiser.process(silence, ts=0.0)

        stage = "stage2_cueword"
        cue = CueWordDetector()  # raises CueWordError if model/vosk unavailable
        cue.process(silence, ts=0.0)
        cue.close()

        stage = "stage3_breathing"
        breathing = BreathingDetector()
        # Returns None until its rolling window fills — we only assert "no crash".
        breathing.process(chunk.mild_float32, ts=0.0, speech_present=False)
    except Exception as e:  # noqa: BLE001
        print(f"  {BAD} Pipeline failed at {stage}: {e}")
        return False
    print(f"  {OK} Full pipeline loads and processes a chunk successfully")
    return True


CHECKS = [
    ("Python Version", check_python_version),
    ("Venv Active", check_venv_active),
    ("Imports", check_imports),
    ("Vosk Model", check_vosk_model),
    ("Contracts", check_contracts),
    ("Pipeline", check_pipeline),
]

BAR = ("═" * 63) if OK == "🟢" else ("=" * 63)


def main() -> int:
    print(BAR)
    print("TASHaudio — setup validator")
    print(BAR)

    results: list[tuple[str, bool]] = []
    for name, fn in CHECKS:
        print(f"\n{name}:")
        try:
            results.append((name, bool(fn())))
        except Exception as e:  # a check itself should never hard-crash the script
            print(f"  {BAD} {name} check errored: {e}")
            results.append((name, False))

    failed = [name for name, ok in results if not ok]

    print("\n" + BAR)
    if not failed:
        print(f"{DONE} TASHaudio Pipeline Ready!")
        print(BAR)
        print(
            "\nNext steps:\n"
            "  1. Run: python live_mic_test.py\n"
            "     → say \"help\" into your mic; expect: >>> DETECTED 'help' ...\n"
            "  2. Read: README.md (architecture + data flow)\n"
            "  3. See:  tests/ (four-layer test suite)\n"
            "\nQuestions? Ping the team channel (see README.md).\n"
            + BAR
        )
        return 0

    print(f"{FAIL} Setup incomplete — failed: {', '.join(failed)}")
    print("Fix the items marked above, then re-run:\n  python quick_start.py")
    print(BAR)
    return 1


if __name__ == "__main__":
    sys.exit(main())
