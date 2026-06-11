"""Build the four scenario WAV files used by full_pipeline_test.py --all.

Uses Windows SAPI (built-in, no extra deps) for TTS voice lines, resampled
to 16 kHz mono. Run once; re-run any time you want to regenerate.

    .venv\\Scripts\\python.exe demo\\make_scenarios.py

Writes to test_audio/scenarios/:
  A_help_then_fine.wav      help → say "fine"   → expect: DE-ESCALATED
  B_help_then_help.wav      help → say "help"   → expect: ESCALATED (distress response)
  C_help_then_silence.wav   help → silence × 2  → expect: ESCALATED (no response)
  D_agonal_then_fine.wav    agonal breathing → say "fine"  → expect: DE-ESCALATED
"""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile

import numpy as np
import scipy.signal
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config

SR = config.SAMPLE_RATE          # 16 000 Hz
OUT = os.path.join(ROOT, "test_audio", "scenarios")
CONFIRM_TIMEOUT_S = 4.0          # must match full_pipeline_test --confirm-timeout
NOISE_BASELINE_S  = 1.5


def _silence(secs: float) -> np.ndarray:
    return np.zeros(int(secs * SR), dtype=np.int16)


def _tts(text: str) -> np.ndarray:
    """Windows SAPI → tmp WAV → resample to 16kHz int16."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name

    ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = -2
$s.SetOutputToWaveFile('{tmp.replace(chr(92), '/')}')
$s.Speak('{text}')
$s.SetOutputToDefaultAudioDevice()
"""
    subprocess.run(["powershell", "-Command", ps], check=True, capture_output=True)

    data, orig_sr = sf.read(tmp, dtype="float32", always_2d=True)
    os.unlink(tmp)

    mono = data[:, 0]
    if orig_sr != SR:
        n_samples = int(len(mono) * SR / orig_sr)
        mono = scipy.signal.resample(mono, n_samples)

    # Normalise and convert to int16
    peak = np.max(np.abs(mono)) + 1e-9
    mono = mono / peak * 0.7
    return np.clip(mono * 32768, -32768, 32767).astype(np.int16)


def _load_wav(name: str) -> np.ndarray:
    path = os.path.join(ROOT, "test_audio", name)
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    assert sr == SR, f"{name}: expected {SR} Hz, got {sr}"
    return data[:, 0]


def _write(name: str, pcm: np.ndarray) -> None:
    path = os.path.join(OUT, name)
    sf.write(path, pcm, SR, subtype="PCM_16")
    print(f"  wrote {os.path.relpath(path, ROOT)}  ({len(pcm)/SR:.1f}s)")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    print("Generating TTS voice lines via Windows SAPI ...")
    say_fine   = _tts("fine")
    say_okay   = _tts("okay")
    say_help   = _tts("help")
    print("  TTS done.")

    help_wav    = _load_wav("say_help.wav")
    agonal_wav  = _load_wav("agonal_gasps.wav")

    # Buffer between trigger and response: long enough for Vosk to flush and
    # for the state machine to enter CONFIRM before the response lands.
    gap = _silence(1.5)

    # Extra silence to absorb a full CONFIRM_TIMEOUT_S window (for scenario C).
    timeout_pad = _silence(CONFIRM_TIMEOUT_S + 1.0)

    print("\nBuilding scenario files ...")

    # A: "help" → short gap → "okay"  (de-escalate)
    # Note: Vosk recognises "okay" reliably from SAPI; "fine" does not match.
    _write("A_help_then_okay.wav", np.concatenate([
        _silence(NOISE_BASELINE_S),
        help_wav, gap, say_okay,
        _silence(1.0),          # tail so flush captures the word
    ]))

    # B: "help" → short gap → "help" again (immediate escalation on distress response)
    # Use the real recorded say_help.wav for the response (SAPI "help" is not
    # reliably recognised by Vosk; the original recording always is).
    _write("B_help_then_help.wav", np.concatenate([
        _silence(NOISE_BASELINE_S),
        help_wav, gap, help_wav,   # real voice for both trigger and response
        _silence(1.0),
    ]))

    # C: "help" → silence × 2 timeouts  (no response → escalate)
    _write("C_help_then_silence.wav", np.concatenate([
        _silence(NOISE_BASELINE_S),
        help_wav,
        timeout_pad,    # first confirm window expires
        timeout_pad,    # second confirm window expires
        _silence(1.0),
    ]))

    # D: agonal breathing (25s) → gap → "okay"  (breathing trigger → de-escalate)
    # Stage 3 needs a 20s rolling window, so this scenario is intentionally long.
    _write("D_agonal_then_okay.wav", np.concatenate([
        _silence(NOISE_BASELINE_S),
        agonal_wav, gap, say_okay,
        _silence(1.0),
    ]))

    print("\nDone.  Run the full test:")
    print("  .venv\\Scripts\\python.exe demo\\full_pipeline_test.py --all")


if __name__ == "__main__":
    main()
