# TASHaudio — In-Car Health Monitor Audio Pipeline

A three-stage, real-time audio pipeline for an in-car passenger health assistant.
**Pre-trained libraries only — no model training.**

```
mic ─┬─(raw)──────────────────────► Stage 2 Cue Word ──┐
     └► Stage 1 Denoise ─(mild)───► Stage 3 Breathing ─┴─► Fusion ─► Escalation
```

## Modules
| File | Role | Contract out |
|------|------|--------------|
| `config.py` | One source of truth for the signal contract + all tuning constants | `SIZES` (derived frame sizes) |
| `contracts.py` | Dataclasses exchanged between stages | — |
| `stage1_denoise.py` | `noisereduce` + scipy band-pass; emits **two** strengths | `DenoisedChunk` |
| `stage2_cueword.py` | **Vosk** offline ASR + energy/VAD gate; no key, spots "help" | `CueWordEvent` + `speech_present` |
| `stage3_breathing.py` | librosa/scipy heuristic respiration analysis (**advisory**) | `BreathingEstimate` |
| `fusion.py` | Corroboration + escalation; **no stage auto-dispatches alone** | `FusionDecision` |
| `pipeline.py` | Orchestration; one-frame lag breaks the S1↔S2 cycle | `FusionDecision` |
| `test_harness.py` | Offline labeled-WAV evaluation (fresh pipeline per file) | metrics dict |

## Setup
```powershell
pip install -r requirements.txt
python download_vosk_model.py        # ~40 MB, no key/signup
```
No API key required — Vosk runs fully offline and recognizes the arbitrary word
"help" with no custom model or training.

## Testing — four layers
```powershell
python tests\test_fusion.py          # L1: fusion truth table (stdlib only)
python tests\test_reblocker.py       # L2: 480-sample re-blocker (numpy only)
python make_test_audio.py            # L3: synthetic breathing WAVs
python test_harness.py               #     + run the full pipeline
python live_mic_test.py              # L4: say "help" into your mic
```

## Key design decisions (the non-obvious ones, each found by testing)
1. **Stage 2 runs on RAW audio, not denoised.** Vosk is a noise-robust ASR;
   aggressive spectral-gating gives it no benefit. Denoise still feeds Stage 3.
2. **Full-vocabulary Vosk, not a restricted grammar.** A 2-word grammar
   (`["help","[unk]"]`) force-decodes *any* sound into "help" and makes
   confidence meaningless (always 1.0). Full vocab transcribes noise as other
   words ("huh"), so a cue match is genuine. This alone eliminated the
   synthetic-noise false positives.
3. **Confident FINAL results, gated by `speech_present`, with an end-of-stream
   `flush()`.** Eager partials fire on noise; finals don't. But finals only
   emit on a speech→silence edge, so a stream that ends mid-utterance needs
   `flush()` to drain the last words (live use self-corrects on real silence).
4. **Stage 3 is heuristic, not medical-grade.** Advisory only; needs clinical
   validation on real agonal-breathing recordings before any automatic action.
5. **Fusion never auto-dispatches on one signal.** Cue word *or* breathing →
   `CONFIRM` (prompt passenger). Both, or no response → `ESCALATE`.

## Verified result (synthetic corpus)
Both Python 3.14 (energy-VAD fallback) and 3.12 (real `webrtcvad`) give clean
separation: only `say_help.wav` produces a cue hit; silence, noise, breathing,
and the negative-speech clip all score 0.

## Running on Python 3.12 (recommended for production)
`webrtcvad` has no 3.14 wheel and won't compile without MSVC, so on 3.14 Stage 2
uses a coarse RMS gate (works, but coarser). For the real VAD + `librosa`:
```powershell
py install 3.12
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install numpy scipy noisereduce vosk pvrecorder librosa soundfile
.\.venv312\Scripts\python.exe -m pip install --only-binary :all: webrtcvad-wheels
.\.venv312\Scripts\python.exe test_harness.py
```

## Known limitations (prototype)
- Test corpus is synthetic (TTS "help" + generated breathing/noise). It proves
  the data flow and logic; real cabin-noise and clinical recordings are required
  for production metrics. White Gaussian noise is actually *adversarial* for a
  VAD, so real road noise should behave no worse.
