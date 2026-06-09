# TASHaudio — Setup Guide

Get the in-car health-monitor audio pipeline running from a clean machine.
Pick your platform below. **Python 3.12 is the target** for the full production
stack (real `webrtcvad` + `librosa`); other versions run in a degraded fallback.

> **Fastest path:** after any platform's steps, run `python quick_start.py` — it
> validates everything in ~30 seconds and tells you exactly what to fix.

| Platform | Est. time | Real VAD? |
|----------|-----------|-----------|
| 🪟 Windows | ⏱️ ~10 min | ✅ `webrtcvad-wheels` |
| 🍎 macOS   | ⏱️ ~10 min | ✅ if a wheel exists for your arch, else RMS fallback |
| 🐧 Linux / WSL (Ubuntu 22.04) | ⏱️ ~12 min | ✅ `webrtcvad-wheels` |

---

## 🪟 A) Windows (primary target — PowerShell)

### Step 1 — Verify Python 3.12 is installed

```powershell
py --list-paths
```

🟢 **Expect:** a line containing `-V:3.12`, e.g.
`-V:3.12[-64]  C:\Users\you\AppData\Local\Python\pythoncore-3.12-64\python.exe`

⚠️ **If `3.12` is missing:** install it from
<https://www.python.org/downloads/release/python-31210/> (the "Windows installer
(64-bit)"). During install tick **"Add python.exe to PATH"**. Then validate:

```powershell
py -3.12 --version
```

🟢 **Expect:** `Python 3.12.x`

### Step 2 — Create the virtual environment

```powershell
py -3.12 -m venv .venv312
.venv312\Scripts\python.exe --version
```

🟢 **Expect:** `Python 3.12.10` (any `3.12.x` is fine).

> ℹ️ The rest of this guide calls the venv's interpreter explicitly
> (`.venv312\Scripts\python.exe`) so it works whether or not you've "activated"
> the venv. To activate it for the session instead:
> `.\.venv312\Scripts\Activate.ps1` (then you can just type `python`).
> ⚠️ If activation is blocked by execution policy, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Step 3 — Install dependencies (with the `webrtcvad` workaround)

⚠️ **WHY the workaround:** the original `webrtcvad` package ships **no prebuilt
wheel** for Python 3.12 on PyPI, so `pip install webrtcvad` tries to compile C
from source and fails without an MSVC toolchain. The drop-in
`webrtcvad-wheels` package provides the **same `import webrtcvad` module** as a
prebuilt binary wheel — install it first, `--only-binary` so pip never tries to
compile:

```powershell
.venv312\Scripts\python.exe -m pip install --upgrade pip
.venv312\Scripts\python.exe -m pip install --only-binary :all: webrtcvad-wheels
.venv312\Scripts\python.exe -m pip install -r requirements-lock.txt
```

🟢 **Expect:** `Successfully installed ... vosk-0.3.45 librosa-0.11.0 ...` with no
red error text.

> 💡 Prefer `requirements-lock.txt` (exact pins, reproducible). Use
> `requirements.txt` only if you intentionally want the latest compatible
> versions.

**Troubleshoot:**
- `webrtcvad-wheels` install fails → you can skip it. Stage 2 automatically falls
  back to a coarse RMS energy gate. Just run the `requirements-lock.txt` line and
  remove the `webrtcvad-wheels==…` line from it first.
- `pip install -r …` fails on a single package → re-run with `-v` to see which
  one, then check the [Troubleshooting](#-d-troubleshooting) section.

### Step 4 — Download the Vosk model (~40 MB)

```powershell
.venv312\Scripts\python.exe download_vosk_model.py
```

🟢 **Expect:**
```
downloading https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip (~40 MB) ...
unpacking ...
done -> ...\models\vosk-model-small-en-us-0.15
```

Validate the model landed:

```powershell
dir models\vosk-model-small-en-us-0.15
```

🟢 **Expect:** sub-folders `am`, `conf`, `graph`, `ivector`. (Re-running the
script is safe — it prints `model already present` and exits.)

### Step 5 — Verify the pipeline works

First the offline self-check (no microphone needed):

```powershell
.venv312\Scripts\python.exe quick_start.py
```

🟢 **Expect:** a column of 🟢 checks ending in `✅ TASHaudio Pipeline Ready!`

Then the live mic test:

```powershell
.venv312\Scripts\python.exe live_mic_test.py
```

🟢 **Expect:** `Listening on '<your mic>'. Say one of ['help']. Ctrl+C to stop.`
Say **"help"** clearly; you should see:
```
  >>> DETECTED 'help' at t=3.2s (conf=0.87)
```
To test reassurance detection, pass extra words when constructing the detector
(`cue_words=["help", "fine", "okay", "ok"]`) — see `config.REASSURANCE_WORDS`.
Press **Ctrl+C** to stop.

---

## 🍎 B) macOS (secondary)

### Step 1 — Verify / install Python 3.12

```bash
python3.12 --version
```

⚠️ **If missing**, install via Homebrew:

```bash
brew install python@3.12
python3.12 --version   # 🟢 Python 3.12.x
```

### Step 2 — Create the virtual environment

```bash
python3.12 -m venv .venv312
.venv312/bin/python --version   # 🟢 Python 3.12.x
```

> To activate for the session: `source .venv312/bin/activate`.

### Step 3 — Install dependencies

⚠️ **macOS arm64 note:** `webrtcvad-wheels` may not publish a wheel for your
Python/arch combination. That's fine — Stage 2 **falls back to the RMS energy
gate** automatically. Install it `--only-binary` so pip never tries to compile;
if it can't find a wheel it will error, in which case just skip it.

```bash
.venv312/bin/python -m pip install --upgrade pip
.venv312/bin/python -m pip install --only-binary :all: webrtcvad-wheels   # OK to skip if no wheel
.venv312/bin/python -m pip install -r requirements-lock.txt
```

🟢 **Expect:** `Successfully installed ... vosk-0.3.45 librosa-0.11.0 ...`

> 💡 If `soundfile`/`librosa` complain about a missing `libsndfile` at runtime:
> `brew install libsndfile`.

### Step 4 — Download the Vosk model (~40 MB)

```bash
.venv312/bin/python download_vosk_model.py
ls models/vosk-model-small-en-us-0.15   # 🟢 am  conf  graph  ivector
```

### Step 5 — Verify the pipeline works

```bash
.venv312/bin/python quick_start.py        # 🟢 ✅ TASHaudio Pipeline Ready!
.venv312/bin/python live_mic_test.py      # say "help" → >>> DETECTED 'help' ...
```

⚠️ macOS will prompt for **microphone permission** the first time — allow it for
your terminal/IDE (System Settings → Privacy & Security → Microphone).

---

## 🐧 C) Linux / WSL (Ubuntu 22.04)

### Step 1 — Install Python 3.12 + dev headers

⚠️ Ubuntu 22.04 ships Python 3.10; add the deadsnakes PPA for 3.12. The
`-dev` headers are needed so any source-built dependency (and `vosk`/`cffi`)
links cleanly.

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential libsndfile1
python3.12 --version   # 🟢 Python 3.12.x
```

### Step 2 — Create the virtual environment

```bash
python3.12 -m venv .venv312
.venv312/bin/python --version   # 🟢 Python 3.12.x
```

### Step 3 — Install dependencies

```bash
.venv312/bin/python -m pip install --upgrade pip
.venv312/bin/python -m pip install --only-binary :all: webrtcvad-wheels
.venv312/bin/python -m pip install -r requirements-lock.txt
```

🟢 **Expect:** `Successfully installed ... vosk-0.3.45 librosa-0.11.0 ...`

### Step 4 — Download the Vosk model (~40 MB)

```bash
.venv312/bin/python download_vosk_model.py
ls models/vosk-model-small-en-us-0.15   # 🟢 am  conf  graph  ivector
```

### Step 5 — Verify the pipeline works

```bash
.venv312/bin/python quick_start.py        # 🟢 ✅ TASHaudio Pipeline Ready!
.venv312/bin/python live_mic_test.py
```

⚠️ **WSL has no microphone by default.** `quick_start.py` will still fully pass
(it uses synthetic audio). `live_mic_test.py` needs a real input device — run it
on native Linux/Windows, or see the troubleshooting entry below.

---

## 🔧 D) Troubleshooting

These apply to **all platforms** unless noted.

### "webrtcvad import fails" / won't install
- **Cause:** no prebuilt wheel for the original `webrtcvad` on Python 3.12+.
- **Fix:** install `webrtcvad-wheels` instead (same `import webrtcvad` module):
  `pip install --only-binary :all: webrtcvad-wheels`.
- **No wheel for your platform (e.g. macOS arm64)?** Skip it entirely. Stage 2
  prints `[Stage2] webrtcvad unavailable -> coarse energy VAD fallback` and runs
  on the RMS gate. Lower-precision, but the pipeline works end to end.

### "Vosk model download hangs" / times out
- **Cause:** slow or proxied network; the host `alphacephei.com` is unreachable.
- **Fix:** download the zip manually from
  <https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip>, then
  unzip it so the folder `models/vosk-model-small-en-us-0.15/` (containing `am`,
  `conf`, `graph`, `ivector`) sits at the repo root. Re-run `quick_start.py` to
  confirm. Behind a corporate proxy, set `HTTPS_PROXY` before re-running the
  download script.

### "live_mic_test.py has no audio device"
- **Symptom:** a `PvRecorder` error, an empty device, or silence.
- **WSL / headless / CI:** there is no mic — this is expected. Use
  `python test_harness.py` (synthetic WAVs) to exercise Stage 2 instead.
- **macOS:** grant microphone permission to your terminal/IDE (see §B Step 5).
- **Windows/Linux desktop:** confirm an input device exists and isn't claimed by
  another app; `PvRecorder(device_index=-1)` picks the system default.

### "Python 3.12 not found"
- **Windows:** `py --list-paths` shows no `3.12` → install from python.org (§A
  Step 1), tick "Add to PATH", reopen the terminal.
- **macOS:** `brew install python@3.12`; if `python3.12` still isn't found,
  `brew link python@3.12`.
- **Linux:** add the deadsnakes PPA (§C Step 1); on WSL use the same steps.
- ⚠️ **Don't** use a 3.13/3.14 venv for production — `webrtcvad-wheels` has no
  wheel there and you'll silently drop to the energy-VAD fallback.

### `quick_start.py` says "Not in venv"
- You're running the system Python. Use the venv interpreter explicitly
  (`.venv312\Scripts\python.exe quick_start.py` on Windows,
  `.venv312/bin/python quick_start.py` elsewhere) or activate the venv first.

---

## ✅ You're done

```bash
python quick_start.py     # all 🟢  → pipeline ready
python live_mic_test.py   # say "help"
```

Next: read [README.md](README.md) for the architecture and the four-layer test
suite under [tests/](tests/).
