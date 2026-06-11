"""Full end-to-end pipeline test — every response path in one run.

THREE MODES:

  --all          Run all 4 pre-built scenario WAV files automatically.
                 Generates them first if they don't exist.
                 Shows a pass/fail results table at the end.

  --mic          Guided interactive test.  The script tells you exactly
                 what to say at each step.  Tests all 4 paths in order.

  --wav PATH     Stream one WAV file and watch the state machine live.

State machine (same in all modes):

    MONITORING  ──(trigger: "help" OR agonal breathing)──►
    CONFIRM_1   "Are you okay?"   [timeout]
                ├─ reassurance word → DE-ESCALATED ✓
                ├─ "help"           → ESCALATED ✗ (immediate)
                └─ silence          → CONFIRM_2
    CONFIRM_2   "Are you still there?"  [timeout]
                ├─ reassurance word → DE-ESCALATED ✓
                ├─ "help"           → ESCALATED ✗ (immediate)
                └─ silence          → ESCALATED ✗ (NEXT STEPS TRIGGERED)

Run:
    .venv\\Scripts\\python.exe demo\\full_pipeline_test.py --all
    .venv\\Scripts\\python.exe demo\\full_pipeline_test.py --mic
    .venv\\Scripts\\python.exe demo\\full_pipeline_test.py --wav test_audio/scenarios/A_help_then_fine.wav
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from enum import Enum, auto
from typing import Iterator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SCENARIOS_DIR = os.path.join(ROOT, "test_audio", "scenarios")

import config
from contracts import EscalationLevel

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
except ImportError:
    raise SystemExit("pip install rich  then re-run.")


# ── State machine ─────────────────────────────────────────────────────────────

class AppState(Enum):
    MONITORING  = auto()
    CONFIRM_1   = auto()
    CONFIRM_2   = auto()
    DEESCALATED = auto()
    ESCALATED   = auto()

_STATE_COLOR = {
    AppState.MONITORING:  "cyan",
    AppState.CONFIRM_1:   "yellow",
    AppState.CONFIRM_2:   "yellow",
    AppState.DEESCALATED: "green",
    AppState.ESCALATED:   "red",
}
_STATE_LABEL = {
    AppState.MONITORING:  "MONITORING",
    AppState.CONFIRM_1:   "AWAITING RESPONSE (1st ask)",
    AppState.CONFIRM_2:   "AWAITING RESPONSE (2nd ask)",
    AppState.DEESCALATED: "ALL CLEAR - DE-ESCALATED",
    AppState.ESCALATED:   "NEXT STEPS TRIGGERED",
}
_TERMINAL = {AppState.DEESCALATED, AppState.ESCALATED}


# ── Dashboard renderer ────────────────────────────────────────────────────────

def _ui(state: AppState, elapsed: float, confirm_t: float | None,
        breath: str, cue: str, log: list[str], stream_ts: float,
        title: str = "TASHaudio — Pipeline Test",
        hint: str = "") -> Panel:

    color = _STATE_COLOR[state]
    label = _STATE_LABEL[state]

    if state == AppState.ESCALATED:
        banner = Text(f"  *** {label} ***  ", style="bold white on red")
    elif state == AppState.DEESCALATED:
        banner = Text(f"  *** {label} ***  ", style="bold white on green")
    else:
        banner = Text(f"  {label}  ", style=f"bold {color}")

    tbl = Table(show_header=False, show_edge=False, box=None, padding=(0, 1))
    tbl.add_column(width=22, style="bold dim")
    tbl.add_column()

    tbl.add_row("Status:", banner)

    if state in (AppState.CONFIRM_1, AppState.CONFIRM_2) and confirm_t is not None:
        remaining = max(0.0, config_timeout - (elapsed - confirm_t))
        prompt = "Are you okay?" if state == AppState.CONFIRM_1 else "Are you still there?"
        timer_style = "bold red" if remaining < 2 else "bold yellow"
        tbl.add_row("Prompt:", Text(f'[?]  "{prompt}"', style="bold"))
        tbl.add_row("Auto-escalate in:", Text(f"{remaining:.1f}s", style=timer_style))

    tbl.add_row("Audio time:", Text(f"{stream_ts:.1f}s"))
    tbl.add_row("Breathing:", Text(breath or "—"))
    tbl.add_row("Last word:", Text(cue or "—"))

    if hint:
        tbl.add_row("", Text(""))
        tbl.add_row(">> Say:", Text(hint, style="bold cyan"))

    if log:
        tbl.add_row("", Text(""))
        log_text = Text()
        for line in log[-6:]:
            log_text.append(line + "\n", style="dim")
        tbl.add_row("Events:", log_text)

    return Panel(tbl, title=title, border_style=color,
                 subtitle=f"elapsed {elapsed:.0f}s")


# ── Core processing loop ──────────────────────────────────────────────────────

config_timeout: float = 7.0   # set by CLI arg before run_loop is called
RESPONSE_COOLDOWN_S = 2.0     # ignore cue words for this long after trigger fires
                               # prevents the triggering audio itself from being
                               # mistaken for a passenger response


def run_loop(
    audio_iter: Iterator,
    console: Console,
    title: str = "TASHaudio — Pipeline Test",
    hint_fn=None,               # callable(AppState) -> str for guided mode
    expected: AppState | None = None,
) -> AppState:
    """
    Process audio chunks through the full state machine.
    Returns the final AppState.
    """
    import numpy as np
    from pipeline import Pipeline

    all_words = config.CUE_WORDS + config.REASSURANCE_WORDS
    pipeline = Pipeline(cue_words=all_words)

    state          = AppState.MONITORING
    breath_text    = ""
    cue_text       = ""
    log: list[str] = []
    confirm_t: float | None = None
    response_armed = 0.0   # cue words as responses not accepted until this elapsed time

    baseline_buf  = np.empty(0, dtype=np.int16)
    baseline_done = False

    def ev(msg: str, stream_ts: float) -> None:
        log.append(f"[{stream_ts:.1f}s] {msg}")

    t_start = time.perf_counter()
    def elapsed() -> float:
        return time.perf_counter() - t_start

    try:
        with Live(console=console, refresh_per_second=15, screen=True) as live:
            for stream_ts, pcm in audio_iter:

                # Collect noise baseline
                if not baseline_done:
                    baseline_buf = np.concatenate([baseline_buf, pcm])
                    needed = int(config.NOISE_BASELINE_SECONDS * config.SAMPLE_RATE)
                    if len(baseline_buf) >= needed:
                        pipeline.prime_noise_baseline(baseline_buf[:needed])
                        baseline_done = True
                    live.update(_ui(state, elapsed(), confirm_t, breath_text,
                                    cue_text, log, stream_ts, title,
                                    hint=(hint_fn(state) if hint_fn else "")))
                    continue

                el = elapsed()

                # Check confirm timeouts
                if state == AppState.CONFIRM_1 and confirm_t and el - confirm_t >= config_timeout:
                    state = AppState.CONFIRM_2
                    confirm_t = el
                    ev("No response — asking again", stream_ts)

                elif state == AppState.CONFIRM_2 and confirm_t and el - confirm_t >= config_timeout:
                    state = AppState.ESCALATED
                    ev("No response to 2nd ask — NEXT STEPS TRIGGERED", stream_ts)

                # Run the pipeline
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    decision = pipeline.process_chunk(pcm, stream_ts)

                if decision.breathing is not None:
                    b = decision.breathing
                    rate = f"{b.resp_rate_bpm:.1f} bpm" if b.resp_rate_bpm else "?"
                    breath_text = f"{b.state.value}  {rate}  conf={b.confidence:.2f}"

                cue = decision.cue_word

                # State transitions
                if state == AppState.MONITORING:
                    if cue and cue.category == "distress":
                        cue_text = f"'{cue.keyword}' conf={cue.confidence_proxy:.2f}"
                        state, confirm_t = AppState.CONFIRM_1, el
                        # Cooldown: don't treat the tail of the trigger audio as a response
                        response_armed = el + RESPONSE_COOLDOWN_S
                        ev(f"TRIGGER: distress word '{cue.keyword}'", stream_ts)

                    elif (decision.level in (EscalationLevel.CONFIRM, EscalationLevel.ESCALATE)
                          and decision.breathing
                          and decision.breathing.state.value in ("agonal_suspect", "apnea", "low_rate")):
                        state, confirm_t = AppState.CONFIRM_1, el
                        response_armed = el + RESPONSE_COOLDOWN_S
                        ev(f"TRIGGER: breathing={decision.breathing.state.value}", stream_ts)

                elif state in (AppState.CONFIRM_1, AppState.CONFIRM_2):
                    if cue and el >= response_armed:   # only accept after cooldown
                        cue_text = f"'{cue.keyword}' conf={cue.confidence_proxy:.2f}"
                        if cue.category == "reassurance":
                            state = AppState.DEESCALATED
                            ev(f"Passenger said '{cue.keyword}' - DE-ESCALATED", stream_ts)
                        elif cue.category == "distress":
                            state = AppState.ESCALATED
                            ev(f"Passenger said '{cue.keyword}' - ESCALATED immediately", stream_ts)

                live.update(_ui(state, elapsed(), confirm_t, breath_text,
                                cue_text, log, stream_ts, title,
                                hint=(hint_fn(state) if hint_fn else "")))

                if state in _TERMINAL:
                    time.sleep(3)
                    break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.flush(0)
        pipeline.close()

    return state


# ── Audio iterators ───────────────────────────────────────────────────────────

def _wav_iter(path: str, chunk_samples: int):
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    assert sr == config.SAMPLE_RATE, f"{path}: expected {config.SAMPLE_RATE} Hz"
    pcm = data[:, 0]
    chunk_dur = chunk_samples / config.SAMPLE_RATE
    t_start = time.perf_counter()
    for i in range(0, len(pcm) - chunk_samples + 1, chunk_samples):
        ts = i / config.SAMPLE_RATE
        yield ts, pcm[i : i + chunk_samples]
        wall_target = t_start + ts + chunk_dur
        sleep = wall_target - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)


def _mic_iter(chunk_samples: int):
    import numpy as np
    try:
        from pvrecorder import PvRecorder
    except ImportError:
        raise SystemExit("pip install pvrecorder")
    rec = PvRecorder(frame_length=chunk_samples, device_index=-1)
    rec.start()
    frames = 0
    try:
        while True:
            pcm_list = rec.read()
            ts = frames * chunk_samples / config.SAMPLE_RATE
            frames += 1
            yield ts, np.array(pcm_list, dtype=np.int16)
    finally:
        rec.stop()
        rec.delete()


# ── Modes ─────────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id":       "A",
        "wav":      "A_help_then_okay.wav",
        "title":    "Scenario A: help then okay",
        "expected": AppState.DEESCALATED,
        "desc":     "Trigger 'help', passenger says 'okay' -> should DE-ESCALATE",
    },
    {
        "id":       "B",
        "wav":      "B_help_then_help.wav",
        "title":    "Scenario B: help then help",
        "expected": AppState.ESCALATED,
        "desc":     "Trigger 'help', passenger says 'help' again -> should ESCALATE immediately",
    },
    {
        "id":       "C",
        "wav":      "C_help_then_silence.wav",
        "title":    "Scenario C: help then silence",
        "expected": AppState.ESCALATED,
        "desc":     "Trigger 'help', passenger silent twice -> should ESCALATE",
    },
    {
        "id":       "D",
        "wav":      "D_agonal_then_okay.wav",
        "title":    "Scenario D: agonal breathing then okay",
        "expected": AppState.DEESCALATED,
        "desc":     "Agonal breathing triggers, passenger says 'okay' -> should DE-ESCALATE (30s)",
    },
]

_MIC_HINTS = {
    AppState.MONITORING:  "say 'help' to trigger",
    AppState.CONFIRM_1:   "say 'fine'/'okay'/'ok'  OR  'help'  OR  stay silent",
    AppState.CONFIRM_2:   "say 'fine'/'okay'/'ok'  OR  'help'  OR  stay silent again",
    AppState.DEESCALATED: "(done - Ctrl+C for next path)",
    AppState.ESCALATED:   "(done - Ctrl+C for next path)",
}

_MIC_STEPS = [
    ("Path 1 - trigger then say fine",   AppState.DEESCALATED,
     "say 'help' to trigger, then say 'fine'/'okay'/'ok'"),
    ("Path 2 - trigger then say help",   AppState.ESCALATED,
     "say 'help' to trigger, then say 'help' again"),
    ("Path 3 - trigger then no response", AppState.ESCALATED,
     "say 'help' to trigger, then stay completely silent"),
]


def mode_all(chunk_samples: int, console: Console) -> None:
    """Auto-run all 4 scenario WAV files; show results table at the end."""
    # Build scenarios if missing
    missing = [s for s in SCENARIOS if not os.path.exists(
        os.path.join(SCENARIOS_DIR, s["wav"]))]
    if missing:
        console.print("[bold yellow]Some scenario files are missing — generating now ...[/]")
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "demo", "make_scenarios.py")],
            cwd=ROOT
        )
        if result.returncode != 0:
            raise SystemExit("make_scenarios.py failed — check output above.")

    results: list[dict] = []

    for sc in SCENARIOS:
        path = os.path.join(SCENARIOS_DIR, sc["wav"])
        console.print(f"\n[bold]{sc['title']}[/]")
        console.print(f"  {sc['desc']}")
        console.print()

        final = run_loop(
            _wav_iter(path, chunk_samples),
            console,
            title=sc["title"],
            expected=sc["expected"],
        )
        passed = final == sc["expected"]
        results.append({**sc, "final": final, "passed": passed})
        console.print()

    # Summary table
    console.print(Rule("Results"))
    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("#",        width=3)
    tbl.add_column("Scenario", width=35)
    tbl.add_column("Expected",  width=20)
    tbl.add_column("Got",       width=20)
    tbl.add_column("Result",    width=8)

    for r in results:
        exp_label = _STATE_LABEL[r["expected"]]
        got_label = _STATE_LABEL[r["final"]]
        ok_str = "PASS" if r["passed"] else "FAIL"
        ok_style = "bold green" if r["passed"] else "bold red"
        exp_col = _STATE_COLOR[r["expected"]]
        got_col = _STATE_COLOR[r["final"]]
        tbl.add_row(
            r["id"],
            r["title"].split("—", 1)[-1].strip(),
            Text(exp_label, style=exp_col),
            Text(got_label, style=got_col),
            Text(ok_str, style=ok_style),
        )

    console.print(tbl)
    passed = sum(1 for r in results if r["passed"])
    console.print(f"\n[bold]{passed}/{len(results)} scenarios passed.[/]")


def mode_mic(chunk_samples: int, console: Console) -> None:
    """Guided interactive mic test — walks through all paths."""
    console.print(Rule("[bold cyan]TASHaudio — Guided Mic Test[/]"))
    console.print("\nYou will be prompted what to say for each path.")
    console.print("Press [bold]Ctrl+C[/] after each scenario to move to the next.\n")
    console.print("[dim]Recognized words:  trigger=[bold]help[/dim][dim]"
                  "   reassure=[bold]fine / okay / ok[/dim]\n")

    results = []
    for step_title, expected, instructions in _MIC_STEPS:
        console.print(Rule(f"[bold]{step_title}[/]"))
        console.print(f"  Instructions: [cyan]{instructions}[/]\n")
        time.sleep(1.5)

        final = run_loop(
            _mic_iter(chunk_samples),
            console,
            title=step_title,
            hint_fn=lambda s: _MIC_HINTS.get(s, ""),
        )
        passed = final == expected
        color = "green" if passed else "red"
        label = _STATE_LABEL[final]
        console.print(f"  Result: [{color}]{label}[/]  "
                      f"({'PASS' if passed else 'FAIL'})\n")
        results.append(passed)
        time.sleep(1)

    console.print(Rule("Summary"))
    for i, (passed, (title, _, _)) in enumerate(zip(results, _MIC_STEPS), 1):
        mark = "[green]PASS[/]" if passed else "[red]FAIL[/]"
        console.print(f"  {mark}  Path {i}: {title}")
    total = sum(results)
    console.print(f"\n[bold]{total}/{len(results)} paths passed.[/]")


def mode_single_wav(path: str, chunk_samples: int, console: Console) -> None:
    """Stream one WAV file and show the live state machine."""
    final = run_loop(_wav_iter(path, chunk_samples), console,
                     title=f"WAV: {os.path.basename(path)}")
    console.print(f"\n[bold]Final state: {_STATE_LABEL[final]}[/]")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global config_timeout

    ap = argparse.ArgumentParser(
        description="TASHaudio full pipeline test — all paths in one run"
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                       help="Auto-run all 4 scenario WAV files (recommended)")
    group.add_argument("--mic", action="store_true",
                       help="Guided interactive mic test")
    group.add_argument("--wav", metavar="PATH",
                       help="Stream a single WAV file")
    ap.add_argument("--chunk-ms", type=int, default=512)
    ap.add_argument("--confirm-timeout", type=float, default=7.0,
                    help="Seconds to wait for response before re-asking (default 7)")
    args = ap.parse_args()

    config_timeout = args.confirm_timeout
    chunk_samples  = config.SAMPLE_RATE * args.chunk_ms // 1000
    console = Console(force_terminal=True, legacy_windows=False)

    if args.all:
        mode_all(chunk_samples, console)
    elif args.mic:
        mode_mic(chunk_samples, console)
    else:
        mode_single_wav(args.wav, chunk_samples, console)


if __name__ == "__main__":
    main()
