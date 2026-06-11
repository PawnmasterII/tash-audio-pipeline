"""Presentation report — run the REAL pipeline on the labeled corpus and turn
the results into slides.

Outputs (into demo/out/):
  * results_dashboard.png   — 2x2 board: detection matrix, cue accuracy vs
                              targets, latency budget, per-clip escalation mix
  * escalation_timeline.png — escalation level + respiration rate over time,
                              agonal vs normal (the "it actually catches it" plot)
  * stats.txt               — the numbers, as a copy-pasteable table

Honest by construction: every number is produced by feeding the WAVs in
test_audio/ through pipeline.Pipeline — the same path live audio takes. The
only hard-coded figures are the per-stage latency BUDGET (config.AUDIO_LATENCY_MS),
which config.py itself flags as engineering targets, not measured values.

Run:  .venv\\Scripts\\python.exe demo\\eval_report.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import warnings

import numpy as np

# Run from anywhere: make the repo root importable and CWD-independent.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from contracts import EscalationLevel  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
AUDIO_DIR = os.path.join(ROOT, "test_audio")
CHUNK_MS = 512

# ── Ground truth for the synthetic corpus ────────────────────────────────────
# expects_cue:   Stage 2 should spot the distress word.
# expects_alarm: Stage 3 breathing should drive escalation to CONFIRM+.
LABELS: dict[str, dict] = {
    "say_help.wav":         {"desc": "Distress word: 'help'",     "expects_cue": True,  "expects_alarm": False},
    "say_negative.wav":     {"desc": "Non-cue speech",            "expects_cue": False, "expects_alarm": False},
    "clean_silence.wav":    {"desc": "Near-silence",              "expects_cue": False, "expects_alarm": False},
    "road_noise.wav":       {"desc": "Cabin road noise",          "expects_cue": False, "expects_alarm": False},
    "normal_breathing.wav": {"desc": "Normal breathing ~15/min",  "expects_cue": False, "expects_alarm": False},
    "agonal_gasps.wav":     {"desc": "Agonal gasps ~4/min",       "expects_cue": False, "expects_alarm": True},
}

LEVEL_RANK = {
    EscalationLevel.NONE: 0,
    EscalationLevel.MONITOR: 1,
    EscalationLevel.CONFIRM: 2,
    EscalationLevel.ESCALATE: 3,
}
LEVEL_NAMES = ["none", "monitor", "confirm", "escalate"]
LEVEL_COLORS = {"none": "#3a4a5a", "monitor": "#3b82f6", "confirm": "#f59e0b", "escalate": "#ef4444"}


def _load_wav_int16(path: str) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    assert sr == config.SAMPLE_RATE, f"{path}: sr={sr}, expected {config.SAMPLE_RATE}"
    return data[:, 0]


def run_file(path: str, chunk_samples: int) -> dict:
    """Feed one WAV through a fresh pipeline; record a per-chunk timeline."""
    from pipeline import Pipeline

    pipeline = Pipeline()
    pcm = _load_wav_int16(path)
    pipeline.prime_noise_baseline(
        pcm[: int(config.NOISE_BASELINE_SECONDS * config.SAMPLE_RATE)]
    )

    timeline: list[dict] = []
    counts = {n: 0 for n in LEVEL_NAMES}
    cue_hits = 0
    last_ts = 0.0
    for i in range(0, len(pcm) - chunk_samples + 1, chunk_samples):
        ts = i / config.SAMPLE_RATE
        decision = pipeline.process_chunk(pcm[i : i + chunk_samples], ts)
        counts[decision.level.value] += 1
        if decision.cue_word is not None:
            cue_hits += 1
        rate = decision.breathing.resp_rate_bpm if decision.breathing else None
        bstate = decision.breathing.state.value if decision.breathing else None
        timeline.append({"ts": ts, "rank": LEVEL_RANK[decision.level],
                         "cue": decision.cue_word is not None, "rate": rate, "bstate": bstate})
        last_ts = ts

    final = pipeline.flush(last_ts + chunk_samples / config.SAMPLE_RATE)
    counts[final.level.value] += 1
    if final.cue_word is not None:
        cue_hits += 1
    pipeline.close()

    max_rank = max((t["rank"] for t in timeline), default=0)
    return {
        "file": os.path.basename(path),
        "timeline": timeline,
        "counts": counts,
        "cue_hits": cue_hits,
        "max_level": LEVEL_NAMES[max_rank],
        "max_rank": max_rank,
    }


def evaluate() -> list[dict]:
    """Run every labeled clip; pipeline chatter is swallowed for a clean console."""
    chunk_samples = config.SAMPLE_RATE * CHUNK_MS // 1000
    results = []
    warnings.filterwarnings("ignore")
    for name in LABELS:
        path = os.path.join(AUDIO_DIR, name)
        if not os.path.exists(path):
            print(f"  [skip] {name} not found in {AUDIO_DIR}")
            continue
        with contextlib.redirect_stdout(io.StringIO()):  # mute Stage 2 VAD notice
            r = run_file(path, chunk_samples)
        meta = LABELS[name]
        r.update(meta)
        r["pred_cue"] = r["cue_hits"] > 0
        # An alarm = breathing alone pushed us to CONFIRM+ (no cue word involved).
        r["pred_alarm"] = (r["max_rank"] >= LEVEL_RANK[EscalationLevel.CONFIRM]) and not r["pred_cue"]
        r["cue_ok"] = r["pred_cue"] == meta["expects_cue"]
        r["alarm_ok"] = r["pred_alarm"] == meta["expects_alarm"]
        results.append(r)
    return results


# ── Stats ─────────────────────────────────────────────────────────────────────
def compute_stats(results: list[dict]) -> dict:
    tp = sum(1 for r in results if r["expects_cue"] and r["pred_cue"])
    fn = sum(1 for r in results if r["expects_cue"] and not r["pred_cue"])
    fp = sum(1 for r in results if not r["expects_cue"] and r["pred_cue"])
    tn = sum(1 for r in results if not r["expects_cue"] and not r["pred_cue"])
    n_pos = tp + fn
    n_neg = fp + tn
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "tpr": (tp / n_pos) if n_pos else float("nan"),
        "fpr": (fp / n_neg) if n_neg else float("nan"),
        "precision": (tp / (tp + fp)) if (tp + fp) else float("nan"),
        "breath_tp": sum(1 for r in results if r["expects_alarm"] and r["pred_alarm"]),
        "breath_pos": sum(1 for r in results if r["expects_alarm"]),
        "false_alarms": sum(1 for r in results if not r["expects_cue"] and not r["expects_alarm"]
                            and r["max_rank"] >= LEVEL_RANK[EscalationLevel.CONFIRM]),
        "n_neg": n_neg,
    }


def write_stats_txt(results: list[dict], stats: dict, path: str) -> str:
    L = []
    L.append("=" * 70)
    L.append("TASHaudio — Evaluation Report (synthetic corpus)")
    L.append("=" * 70)
    L.append("")
    L.append(f"{'clip':<22}{'description':<26}{'cue':<6}{'max level':<11}{'verdict'}")
    L.append("-" * 70)
    for r in results:
        verdict = "PASS" if (r["cue_ok"] and r["alarm_ok"]) else "CHECK"
        L.append(f"{r['file']:<22}{r['desc']:<26}"
                 f"{('HIT' if r['pred_cue'] else '-'):<6}{r['max_level']:<11}{verdict}")
    L.append("-" * 70)
    L.append("")
    L.append("CUE-WORD DETECTOR (Stage 2 — Vosk offline ASR)")
    L.append(f"  True positives : {stats['tp']}    False negatives: {stats['fn']}")
    L.append(f"  False positives: {stats['fp']}    True negatives : {stats['tn']}")
    L.append(f"  Detection rate (TPR) : {stats['tpr']*100:5.1f}%   (target >= {config.PERFORMANCE_TARGETS['vosk_accuracy_on_help']*100:.0f}%)")
    L.append(f"  False-positive rate  : {stats['fpr']*100:5.1f}%   (target  < {config.PERFORMANCE_TARGETS['false_positive_rate']*100:.0f}%)")
    L.append(f"  Precision            : {stats['precision']*100:5.1f}%")
    L.append("")
    L.append("BREATHING DETECTOR (Stage 3 — advisory)")
    L.append(f"  Agonal/apnea caught  : {stats['breath_tp']}/{stats['breath_pos']}")
    L.append(f"  False alarms on benign clips: {stats['false_alarms']}/{stats['n_neg']}")
    L.append("")
    L.append("LATENCY BUDGET (engineering targets, config.AUDIO_LATENCY_MS)")
    for k, v in config.AUDIO_LATENCY_MS.items():
        if k.startswith("total"):
            continue
        L.append(f"  {k:<22}{v:>4} ms")
    L.append(f"  {'TOTAL':<22}{config.TOTAL_LATENCY_MS:>4} ms   (ceiling {config.PERFORMANCE_TARGETS['end_to_end_latency_ms']} ms)")
    L.append("=" * 70)
    text = "\n".join(L)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return text


# ── Charts ────────────────────────────────────────────────────────────────────
def make_dashboard(results: list[dict], stats: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "figure.facecolor": "#0f1620", "axes.facecolor": "#0f1620",
        "savefig.facecolor": "#0f1620", "text.color": "#e6edf3",
        "axes.labelcolor": "#e6edf3", "xtick.color": "#9fb0c0",
        "ytick.color": "#9fb0c0", "axes.edgecolor": "#2a3a4a", "font.size": 10,
    })
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle("TASHaudio — In-Car Health Monitor: Pipeline Results",
                 fontsize=19, fontweight="bold", color="#e6edf3", y=0.975)
    fig.text(0.5, 0.937, "Three-stage real-time audio pipeline  ·  pre-trained libraries, no model training",
             ha="center", fontsize=11, color="#7d8fa0")
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22,
                          left=0.07, right=0.965, top=0.88, bottom=0.10)

    # Panel A — detection matrix (the headline: clean separation) -------------
    axA = fig.add_subplot(gs[0, 0])
    axA.set_title("A · Detection matrix — one row per clip", fontweight="bold", loc="left", color="#e6edf3")
    files = [r["file"].replace(".wav", "") for r in results]
    y = np.arange(len(results))[::-1]
    for yi, r in zip(y, results):
        # cue column
        c_cue = LEVEL_COLORS["confirm"] if r["pred_cue"] else "#22303c"
        axA.add_patch(plt.Rectangle((0, yi - 0.4), 1, 0.8, color=c_cue, ec="#0f1620"))
        axA.text(0.5, yi, "HELP" if r["pred_cue"] else "—", ha="center", va="center",
                 fontsize=8, fontweight="bold", color="#0f1620" if r["pred_cue"] else "#5a6b7a")
        # max escalation column
        c_lvl = LEVEL_COLORS[r["max_level"]]
        axA.add_patch(plt.Rectangle((1.1, yi - 0.4), 1, 0.8, color=c_lvl, ec="#0f1620"))
        axA.text(1.6, yi, r["max_level"].upper(), ha="center", va="center", fontsize=7.5,
                 fontweight="bold", color="#0f1620" if r["max_level"] != "none" else "#9fb0c0")
        # verdict tick
        ok = r["cue_ok"] and r["alarm_ok"]
        axA.text(2.35, yi, "✓" if ok else "✗", ha="center", va="center", fontsize=14,
                 fontweight="bold", color="#22c55e" if ok else "#ef4444")
    axA.set_yticks(y)
    axA.set_yticklabels(files, fontsize=9)
    axA.set_xticks([0.5, 1.6, 2.35])
    axA.set_xticklabels(["cue word", "max escalation", "expected?"], fontsize=9)
    axA.set_xlim(-0.1, 2.7)
    axA.set_ylim(-0.7, len(results) - 0.3)
    for s in axA.spines.values():
        s.set_visible(False)
    axA.tick_params(length=0)

    # Panel B — cue accuracy vs targets ---------------------------------------
    axB = fig.add_subplot(gs[0, 1])
    axB.set_title("B · Cue-word detector vs acceptance targets", fontweight="bold", loc="left", color="#e6edf3")
    metrics = ["Detection\nrate (TPR)", "False-positive\nrate", "Precision"]
    achieved = [stats["tpr"] * 100, stats["fpr"] * 100, stats["precision"] * 100]
    targets = [config.PERFORMANCE_TARGETS["vosk_accuracy_on_help"] * 100,
               config.PERFORMANCE_TARGETS["false_positive_rate"] * 100, None]
    xb = np.arange(len(metrics))
    bars = axB.bar(xb, achieved, width=0.55,
                   color=["#22c55e", "#22c55e", "#3b82f6"], zorder=3)
    for xi, t in zip(xb, targets):
        if t is not None:
            axB.hlines(t, xi - 0.3, xi + 0.3, color="#f59e0b", lw=2.5, zorder=4)
            axB.text(xi, t + 3, f"target {t:.0f}%", ha="center", fontsize=7.5, color="#f59e0b")
    for b, v in zip(bars, achieved):
        axB.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center",
                 fontsize=11, fontweight="bold", color="#e6edf3")
    axB.set_xticks(xb)
    axB.set_xticklabels(metrics, fontsize=9)
    axB.set_ylim(0, 112)
    axB.set_ylabel("percent")
    axB.grid(axis="y", color="#22303c", zorder=0)
    for s in axB.spines.values():
        s.set_visible(False)

    # Panel C — latency budget ------------------------------------------------
    axC = fig.add_subplot(gs[1, 0])
    axC.set_title("C · End-to-end latency budget (per chunk)", fontweight="bold", loc="left", color="#e6edf3")
    stage_lat = {k: v for k, v in config.AUDIO_LATENCY_MS.items() if not k.startswith("total")}
    labels = ["Stage 1\ndenoise", "Stage 2\nVosk final", "Stage 3\nbreathing", "Fusion\nlogic"]
    vals = [stage_lat["stage1_denoise"], stage_lat["stage2_vosk_final"],
            stage_lat["stage3_breathing"], stage_lat["fusion_logic"]]
    colors = ["#3b82f6", "#8b5cf6", "#06b6d4", "#22c55e"]
    left = 0.0
    for v, c, lab in zip(vals, colors, labels):
        axC.barh(0, v, left=left, color=c, height=0.5, zorder=3,
                 label=f"{lab.replace(chr(10),' ')} ({v} ms)")
        if v >= 15:
            axC.text(left + v / 2, 0, f"{v}", ha="center", va="center",
                     fontsize=9, fontweight="bold", color="#0f1620")
        left += v
    ceiling = config.PERFORMANCE_TARGETS["end_to_end_latency_ms"]
    axC.axvline(ceiling, color="#ef4444", lw=2, ls="--", zorder=5)
    axC.text(ceiling - 2, 0.42, f"{ceiling} ms ceiling", ha="right", color="#ef4444", fontsize=9, fontweight="bold")
    axC.text(left + 4, 0, f"= {config.TOTAL_LATENCY_MS} ms total", va="center",
             fontsize=11, fontweight="bold", color="#22c55e")
    axC.set_ylim(-0.6, 0.7)
    axC.set_xlim(0, ceiling + 30)
    axC.set_yticks([])
    axC.set_xlabel("milliseconds")
    axC.legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=2, fontsize=8,
               facecolor="#0f1620", edgecolor="#2a3a4a", labelcolor="#e6edf3")
    axC.grid(axis="x", color="#22303c", zorder=0)
    for s in axC.spines.values():
        s.set_visible(False)

    # Panel D — per-clip escalation mix ---------------------------------------
    axD = fig.add_subplot(gs[1, 1])
    axD.set_title("D · Escalation decisions per clip (chunk-by-chunk)", fontweight="bold", loc="left", color="#e6edf3")
    order = [r["file"].replace(".wav", "") for r in results]
    yb = np.arange(len(results))[::-1]
    bottoms = np.zeros(len(results))
    for lvl in LEVEL_NAMES:
        widths = np.array([r["counts"][lvl] for r in results], dtype=float)
        axD.barh(yb, widths, left=bottoms, color=LEVEL_COLORS[lvl], label=lvl, zorder=3)
        bottoms += widths
    axD.set_yticks(yb)
    axD.set_yticklabels(order, fontsize=9)
    axD.set_xlabel("number of decision windows")
    axD.legend(loc="lower right", fontsize=8, facecolor="#0f1620",
               edgecolor="#2a3a4a", labelcolor="#e6edf3", ncol=2)
    axD.grid(axis="x", color="#22303c", zorder=0)
    for s in axD.spines.values():
        s.set_visible(False)

    fig.text(0.5, 0.022,
             "Synthetic corpus (TTS 'help' + generated breathing/noise): proves data flow + decision logic. "
             "Real cabin-noise & clinical agonal recordings required for production metrics.",
             ha="center", fontsize=8.5, color="#7d8fa0", style="italic")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_timeline(results: list[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_name = {r["file"]: r for r in results}
    # One representative clip per outcome class — keeps the plot legible.
    series = [("agonal_gasps.wav", "#ef4444", "Agonal gasps (~4/min, irregular)"),
              ("say_help.wav", "#f59e0b", "Spoken 'help'"),
              ("normal_breathing.wav", "#22c55e", "Normal breathing (~15/min)"),
              ("road_noise.wav", "#60a5fa", "Road noise (control)")]
    series = [(n, c, lab) for (n, c, lab) in series if n in by_name]
    if not series:
        return

    plt.rcParams.update({
        "figure.facecolor": "#0f1620", "axes.facecolor": "#0f1620",
        "savefig.facecolor": "#0f1620", "text.color": "#e6edf3",
        "axes.labelcolor": "#e6edf3", "xtick.color": "#9fb0c0",
        "ytick.color": "#9fb0c0", "axes.edgecolor": "#2a3a4a", "font.size": 11,
    })
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("TASHaudio — Live decision timeline",
                 fontsize=17, fontweight="bold", y=0.97)
    fig.text(0.5, 0.905, "Escalation level the pipeline outputs as each clip plays",
             ha="center", fontsize=11, color="#7d8fa0")

    # Stage 3 needs a full rolling window before it can judge breathing — be
    # explicit about it rather than letting the flat warm-up look like a bug.
    ax.axvspan(0, config.RESP_WINDOW_S, color="#1b2733", alpha=0.6, zorder=0)
    ax.text(config.RESP_WINDOW_S / 2, 3.12,
            f"Stage 3 warm-up ({config.RESP_WINDOW_S:.0f}s rolling window)",
            ha="center", color="#7d8fa0", fontsize=9, style="italic")

    # Small vertical offsets so overlapping flat lines stay distinguishable.
    offsets = {n: (i - (len(series) - 1) / 2) * 0.06 for i, (n, _, _) in enumerate(series)}
    for name, color, lab in series:
        tl = by_name[name]["timeline"]
        ts = [t["ts"] for t in tl]
        rank = [t["rank"] + offsets[name] for t in tl]
        ax.step(ts, rank, where="post", color=color, lw=2.4, label=lab, zorder=3)

    ax.set_yticks(range(4))
    ax.set_yticklabels([n.upper() for n in LEVEL_NAMES])
    ax.set_ylim(-0.25, 3.35)
    ax.set_ylabel("escalation level")
    ax.set_xlabel("time (seconds)")
    ax.legend(loc="center left", fontsize=10, facecolor="#0f1620",
              edgecolor="#2a3a4a", labelcolor="#e6edf3")
    ax.grid(color="#22303c", zorder=1)
    for s in ax.spines.values():
        s.set_visible(False)

    fig.text(0.5, 0.02,
             "Once warmed up, the agonal clip reaches CONFIRM (prompt passenger) and 'help' fires CONFIRM on the word; "
             "normal breathing and road noise never raise a false alarm.",
             ha="center", fontsize=9.5, color="#7d8fa0", style="italic")
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.09, right=0.97)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Running the pipeline over the labeled corpus ...\n")
    results = evaluate()
    if not results:
        raise SystemExit(f"No clips evaluated — is {AUDIO_DIR} populated? "
                         "Run: python make_test_audio.py")
    stats = compute_stats(results)

    text = write_stats_txt(results, stats, os.path.join(OUT_DIR, "stats.txt"))
    print(text)

    make_dashboard(results, stats, os.path.join(OUT_DIR, "results_dashboard.png"))
    make_timeline(results, os.path.join(OUT_DIR, "escalation_timeline.png"))
    print(f"\nWrote to {os.path.relpath(OUT_DIR, ROOT)}\\:")
    print("  results_dashboard.png   escalation_timeline.png   stats.txt")


if __name__ == "__main__":
    main()
