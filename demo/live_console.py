"""Real-time terminal operator console for presentations.

Drives the real Pipeline (fresh per run, same code path as test_harness.py)
and renders a live rich dashboard showing escalation state, cue words,
breathing, and a scrolling reasons log.

Modes:
    --wav <path>   stream a WAV file at natural speed
    --mic          capture from default microphone via pvrecorder

Run:
    .venv\\Scripts\\python.exe demo\\live_console.py --wav test_audio/agonal_gasps.wav
    .venv\\Scripts\\python.exe demo\\live_console.py --mic
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from contracts import EscalationLevel, FusionDecision

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print(
        "ERROR: 'rich' is not installed.\n"
        "  Install with:  pip install rich\n"
        "  Then re-run this script."
    )
    raise SystemExit(1)

LEVEL_STYLE = {
    EscalationLevel.NONE: ("grey62", "dim"),
    EscalationLevel.MONITOR: ("blue", "bold"),
    EscalationLevel.CONFIRM: ("yellow", "bold"),
    EscalationLevel.ESCALATE: ("red", "bold"),
}

BANNER_WIDTH = 60
CONFIRM_TIMEOUT_S = 5.0
LOG_LINES = 8


def _build_dashboard(
    level: EscalationLevel,
    cue_text: str,
    breath_text: str,
    elapsed: float,
    log_entries: list[tuple[float, str]],
    confirm_shown: bool,
) -> Panel:
    banner_label = level.value.upper()
    color, weight = LEVEL_STYLE[level]
    banner = Text(banner_label.center(BANNER_WIDTH), style=f"{color} {weight}")

    body = Table(show_header=False, show_edge=False, box=None, padding=0)
    body.add_column(width=18, style="bold cyan")
    body.add_column()

    body.add_row("Escalation:", banner)
    body.add_row("Cue word:", Text(cue_text or "—"))
    body.add_row("Breathing:", Text(breath_text or "—"))
    body.add_row("Elapsed:", Text(f"{elapsed:.1f} s"))

    if confirm_shown:
        body.add_row("Prompt:", Text("🔊 'Are you okay?'  (no response → escalate)", style="bold yellow"))

    if log_entries:
        body.add_row(Text())
        log_text = Text()
        for ts, reason in log_entries[-LOG_LINES:]:
            log_text.append(f"[{ts:7.1f}s] ", style="dim")
            log_text.append(reason)
            log_text.append("\n")
        body.add_row("Log:", log_text)

    return Panel(body, title="TASHaudio — Live Console", border_style=color)


def _load_wav_int16(path: str):
    import numpy as np
    try:
        import soundfile as sf
    except ImportError:
        raise SystemExit("soundfile not installed — needed for --wav mode")
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    assert sr == config.SAMPLE_RATE, f"{path}: sr={sr}, expected {config.SAMPLE_RATE}"
    return data[:, 0]


def run_wav(args: argparse.Namespace) -> None:
    import numpy as np
    from pipeline import Pipeline

    pcm = _load_wav_int16(args.wav)
    chunk_samples = config.SAMPLE_RATE * args.chunk_ms // 1000
    chunk_dur = chunk_samples / config.SAMPLE_RATE

    pipeline = Pipeline()
    pipeline.prime_noise_baseline(pcm[: int(config.NOISE_BASELINE_SECONDS * config.SAMPLE_RATE)])

    console = Console()
    level = EscalationLevel.NONE
    cue_text = ""
    breath_text = ""
    log_entries: list[tuple[float, str]] = []
    confirm_shown = False
    confirm_time: float | None = None
    passenger_responded: bool | None = None

    with Live(console=console, refresh_per_second=12, screen=True) as live:
        t_start = time.perf_counter()
        for i in range(0, len(pcm) - chunk_samples + 1, chunk_samples):
            ts = i / config.SAMPLE_RATE

            if passenger_responded is None and confirm_shown and confirm_time is not None:
                if ts - confirm_time >= args.confirm_timeout:
                    passenger_responded = False

            decision = pipeline.process_chunk(pcm[i : i + chunk_samples], ts, passenger_responded)

            level = decision.level
            if decision.cue_word is not None:
                cue_text = f"'{decision.cue_word.keyword}' conf={decision.cue_word.confidence_proxy:.2f}"
            if decision.breathing is not None:
                b = decision.breathing
                rate_str = f"{b.resp_rate_bpm:.1f} bpm" if b.resp_rate_bpm else "?"
                breath_text = f"{b.state.value}  rate={rate_str}  conf={b.confidence:.2f}"
            for reason in decision.reasons:
                log_entries.append((ts, reason))

            if level == EscalationLevel.CONFIRM and not confirm_shown:
                confirm_shown = True
                confirm_time = ts

            elapsed = time.perf_counter() - t_start
            live.update(_build_dashboard(level, cue_text, breath_text, elapsed, log_entries, confirm_shown))

            wall_next = t_start + (i / config.SAMPLE_RATE + chunk_dur)
            sleep_for = wall_next - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

        last_ts = (len(pcm) - len(pcm) % chunk_samples) / config.SAMPLE_RATE
        final = pipeline.flush(last_ts + chunk_dur)
        level = final.level
        for reason in final.reasons:
            log_entries.append((last_ts, reason))
        if final.cue_word is not None:
            cue_text = f"'{final.cue_word.keyword}' conf={final.cue_word.confidence_proxy:.2f}"
        elapsed = time.perf_counter() - t_start
        live.update(_build_dashboard(level, cue_text, breath_text, elapsed, log_entries, confirm_shown))

        pipeline.close()
        time.sleep(2)

    console.print(f"\n[bold]Done.[/]  Final level: [{LEVEL_STYLE[level][0]}]{level.value.upper()}[/]")


def run_mic(args: argparse.Namespace) -> None:
    try:
        from pvrecorder import PvRecorder
    except ImportError:
        raise SystemExit(
            "ERROR: 'pvrecorder' is not installed.\n"
            "  Install with:  pip install pvrecorder\n"
            "  Then re-run with --mic."
        )

    import numpy as np
    from pipeline import Pipeline

    chunk_samples = 512
    rec = PvRecorder(frame_length=chunk_samples, device_index=-1)

    pipeline = Pipeline()
    console = Console()
    level = EscalationLevel.NONE
    cue_text = ""
    breath_text = ""
    log_entries: list[tuple[float, str]] = []
    confirm_shown = False
    confirm_time: float | None = None
    passenger_responded: bool | None = None
    frames = 0

    console.print(f"[bold]Listening on '{rec.selected_device}'[/]  —  Ctrl+C to stop.\n")

    rec.start()
    try:
        with Live(console=console, refresh_per_second=12, screen=True) as live:
            t_start = time.perf_counter()
            while True:
                pcm_list = rec.read()
                pcm = np.array(pcm_list, dtype=np.int16)
                ts = frames * chunk_samples / config.SAMPLE_RATE
                frames += 1

                if passenger_responded is None and confirm_shown and confirm_time is not None:
                    if ts - confirm_time >= args.confirm_timeout:
                        passenger_responded = False

                decision = pipeline.process_chunk(pcm, ts, passenger_responded)

                level = decision.level
                if decision.cue_word is not None:
                    cue_text = f"'{decision.cue_word.keyword}' conf={decision.cue_word.confidence_proxy:.2f}"
                if decision.breathing is not None:
                    b = decision.breathing
                    rate_str = f"{b.resp_rate_bpm:.1f} bpm" if b.resp_rate_bpm else "?"
                    breath_text = f"{b.state.value}  rate={rate_str}  conf={b.confidence:.2f}"
                for reason in decision.reasons:
                    log_entries.append((ts, reason))

                if level == EscalationLevel.CONFIRM and not confirm_shown:
                    confirm_shown = True
                    confirm_time = ts

                elapsed = time.perf_counter() - t_start
                live.update(_build_dashboard(level, cue_text, breath_text, elapsed, log_entries, confirm_shown))

    except KeyboardInterrupt:
        pass
    finally:
        rec.stop()
        final_ts = frames * chunk_samples / config.SAMPLE_RATE
        final = pipeline.flush(final_ts)
        if final.cue_word is not None:
            cue_text = f"'{final.cue_word.keyword}' conf={final.cue_word.confidence_proxy:.2f}"
        pipeline.close()
        rec.delete()

    console.print(f"\n[bold]Stopped.[/]  Final level: [{LEVEL_STYLE[level][0]}]{level.value.upper()}[/]")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="TASHaudio live operator console for presentations"
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", metavar="PATH", help="Stream a WAV file at natural speed")
    group.add_argument("--mic", action="store_true", help="Capture from default microphone")
    ap.add_argument("--chunk-ms", type=int, default=512, help="Chunk size in ms (default: 512)")
    ap.add_argument("--confirm-timeout", type=float, default=CONFIRM_TIMEOUT_S,
                    help="Seconds to wait for passenger response before escalating (default: 5)")
    args = ap.parse_args()

    if args.wav:
        run_wav(args)
    else:
        run_mic(args)


if __name__ == "__main__":
    main()
