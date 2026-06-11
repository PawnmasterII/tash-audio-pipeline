# TASHaudio — Presentation & Demo Kit

Everything here runs against the **real pipeline** — the same code path live audio
takes. Nothing is mocked.

## 1. Generate the slides (numbers + charts)

```powershell
.\.venv\Scripts\python.exe demo\eval_report.py
```

Produces in `demo/out/`:

| File | What it shows |
|------|---------------|
| `results_dashboard.png` | 4-panel board — detection matrix, cue accuracy vs targets, latency budget, per-clip escalation mix |
| `escalation_timeline.png` | Escalation level over time — agonal & "help" trigger, normal/noise never false-alarm |
| `stats.txt` | The raw numbers, copy-pasteable into a slide |

### The numbers to say out loud (synthetic corpus, 6 clips)

- **Cue-word detection: 100% true-positive, 0% false-positive, 100% precision.**
  Only the spoken-"help" clip fires; silence, road noise, breathing, and non-cue
  speech all score zero. Targets were ≥95% TP and <5% FP — both beaten.
- **Breathing: 1/1 agonal episodes caught, 0/5 false alarms** on benign clips.
- **End-to-end latency budget: 195 ms, under the 200 ms ceiling** — denoise 10 +
  Vosk-final 150 + breathing 20 + fusion 15.
- **No single signal auto-dispatches** — cue word *or* agonal breathing → CONFIRM
  (prompt the passenger); both, or no response → ESCALATE.

> ⚠️ Honesty slide: corpus is synthetic (TTS "help" + generated breathing/noise).
> It proves the data flow and decision logic. Real cabin noise and clinical
> agonal-breathing recordings are needed before quoting production metrics.

## 2. Live demos you can run on stage

### A. Say "help" into the mic (already built — 30 s, high impact)

```powershell
.\.venv\Scripts\python.exe live_mic_test.py
```

Say "help" → `>>> DETECTED 'help' at t=… (conf=…)`. Say random words → nothing.
This is the most convincing live moment: it's real offline ASR, no internet, no key.

### B. Batch replay of the corpus (deterministic, no mic risk)

```powershell
.\.venv\Scripts\python.exe test_harness.py --chunk-ms 512
```

Prints the per-clip decision dict. Pair it with the dashboard PNG on the slide.

## 3. Live console (projector-ready animated dashboard)

A rich terminal dashboard that streams a WAV file or the live mic through the
real pipeline and animates the escalation state in real time — great for a
projector.

**Prerequisites:** `pip install rich` (already in requirements.txt).

```powershell
# Stream a test clip (plays at natural speed, real-time dashboard)
.\.venv\Scripts\python.exe demo\live_console.py --wav test_audio/agonal_gasps.wav
.\.venv\Scripts\python.exe demo\live_console.py --wav test_audio/say_help.wav

# Live mic mode
.\.venv\Scripts\python.exe demo\live_console.py --mic
```

**What you see (updates every ~32 ms):**

- Big colored **ESCALATION** banner — NONE (grey), MONITOR (blue), CONFIRM (amber), ESCALATE (red)
- Latest cue word + confidence score
- Current breathing state + respiratory rate estimate
- Elapsed wall-clock time
- Scrolling log of the last 8 fusion reasons with timestamps

When the level reaches **CONFIRM**, a simulated passenger prompt
("Are you okay?") appears. If no response arrives within the timeout
(default 5 s, configurable via `--confirm-timeout`), it escalates to **ESCALATE**.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--wav PATH` | — | Stream a WAV file at natural speed |
| `--mic` | — | Capture from default microphone |
| `--chunk-ms` | 512 | Chunk size in ms |
| `--confirm-timeout` | 5.0 | Seconds before auto-escalating from CONFIRM |

Degrades gracefully if `rich` or `pvrecorder` is missing, with a clear install hint.

## 4. Suggested 3-minute presentation flow

1. **Problem** (15 s): in-car passenger medical emergency — can audio catch it?
2. **Architecture** (30 s): the 3-stage diagram from `README.md`. Pre-trained
   libraries only, runs fully offline, no API key, no model training.
3. **Show `results_dashboard.png`** (45 s): walk panels A→D. Land on "100% TP,
   0% FP, 195 ms."
4. **Show `escalation_timeline.png`** (20 s): agonal & "help" trigger, benign
   clips stay flat — no false alarms.
5. **Live: `live_mic_test.py`** (45 s): say "help", it fires; say nonsense, silence.
6. **Honesty + next steps** (25 s): synthetic corpus today; real cabin-noise and
   clinical recordings next; clinical validation before any auto-action.
