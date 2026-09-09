#!/usr/bin/env python3
"""Render existing report data. Never run a device probe or infer missing samples.

Run from any directory: python scripts/plot_evidence.py
Requires matplotlib (rendered with 3.10.9). Outputs PNG + SVG and source hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GREEN = "#76B900"
BLUE = "#192BC2"
INK = "#202428"
GRAY = "#62666B"
THOR = "reports/thor_trt_benchmark/thor_trt_benchmark.json"
THERMAL = "reports/thor_trt_benchmark/thermal_samples.json"
ORIN = "reports/jetson/yahboom_day_one/day_one_smoke.json"
ORIN_PROV = "reports/jetson/yahboom_day_one/provenance.json"
LUDO = "reports/ludo_stats.csv"
CORRECTED = "reports/ludo_groot17_eval01/session_summary_corrected.json"
LUDO_PROV = "reports/ludo_soak_s10/provenance.json"
SOURCES = (THOR, THERMAL, ORIN, ORIN_PROV, LUDO, CORRECTED, LUDO_PROV)


def read_json(path):
    return json.loads((ROOT / path).read_text())


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#C9CDD1",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.hashsalt": "physical-ai-evidence-v1",
        }
    )


def finish(fig, out, name, title, subtitle, source, note):
    fig.suptitle(title, x=0.075, y=0.965, ha="left", fontsize=23, weight="bold")
    fig.text(0.075, 0.895, subtitle, ha="left", fontsize=11, color=GRAY)
    fig.text(0.075, 0.065, note, ha="left", fontsize=10, color=INK)
    fig.text(0.075, 0.025, "Source: " + source, ha="left", fontsize=8, color=GRAY)
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.20, top=0.80, wspace=0.42, hspace=0.7)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=160, metadata={"Date": None} if ext == "svg" else {})
    plt.close(fig)


def thor_charts(out):
    data = read_json(THOR)
    modes = [data["pytorch_eager"], data["tensorrt_n17_full_pipeline"]]
    date = data["generated_utc"][:10]
    c = data["conditions"]
    env = (
        f"AGX Thor 128GB | 120W | {date} | BF16, batch {c['batch_size']} | "
        f"n={c['num_iterations']} after {c['warmup']} warmups"
    )
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for y, mode in zip((1, 0), modes, strict=True):
        e = mode["e2e"]
        ax.hlines(y, e["min_ms"], e["max_ms"], color=GREEN, linewidth=4)
        ax.plot(
            [e["min_ms"], e["max_ms"]], [y, y], "|", color=GREEN, markersize=20, markeredgewidth=2
        )
        ax.plot(e["median_ms"], y, "D", color=GREEN, markeredgecolor=INK, markersize=11)
        ax.text(
            e["median_ms"], y + 0.18, f"median {e['median_ms']:.1f} ms", ha="center", weight="bold"
        )
        ax.text(e["min_ms"], y - 0.18, str(e["min_ms"]), ha="center", fontsize=11)
        ax.text(e["max_ms"], y - 0.18, str(e["max_ms"]), ha="center", fontsize=11)
        ax.text(
            143, y, f"mean {e['mean_ms']:.1f}\nSD {e['std_ms']:.1f} ms", va="center", fontsize=11
        )
    ax.set(
        yticks=[1, 0],
        yticklabels=["Eager", "TensorRT"],
        ylim=(-0.5, 1.5),
        xlim=(90, 157),
        xlabel="End-to-end latency (ms); line = recorded min to max",
    )
    ax.grid(axis="x", alpha=0.18)
    finish(
        fig,
        out,
        "thor_latency",
        "Thor inference: recorded spread",
        env,
        THOR,
        "Summaries only: the 100 individual iteration timings are not stored. No "
        "histogram or fitted distribution.",
    )

    fig, ax = plt.subplots(figsize=(12, 5.8))
    keys = ["data_processing_ms_median", "backbone_ms_median", "action_head_ms_median"]
    x = np.arange(3)
    for i, mode in enumerate(modes):
        vals = [mode[k] for k in keys]
        bars = ax.bar(
            x + (i - 0.5) * 0.34,
            vals,
            0.32,
            label=("Eager" if i == 0 else "TensorRT"),
            color=("white" if i == 0 else GREEN),
            edgecolor=GREEN,
            linewidth=2,
            hatch=("//" if i == 0 else None),
        )
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=5, fontsize=12)
    ax.set(
        xticks=x,
        xticklabels=["Preprocessing", "Backbone", "Action head"],
        ylabel="Component median (ms)",
        ylim=(0, 90),
    )
    ax.legend(frameon=False, ncol=2, loc="upper left")
    finish(
        fig,
        out,
        "thor_components",
        "Thor components, measured separately",
        env,
        THOR,
        "Components are NOT summable into the end-to-end figure. Bars are not stacked.",
    )

    samples = read_json(THERMAL)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    t0 = samples[0]["t"]
    axes[0].scatter(
        [s["t"] - t0 for s in samples], [s["gpu-thermal"] for s in samples], color=GREEN, s=34
    )
    axes[0].set(
        title=f"GPU temperature | n={len(samples)}",
        xlabel="Seconds from first recorded sample",
        ylabel="Temperature (C)",
    )
    axes[0].grid(alpha=0.18)
    peaks = data["thermal"]["power_peak_mw"]
    bars = axes[1].bar(list(peaks), list(peaks.values()), color=GREEN, width=0.5)
    axes[1].bar_label(bars, padding=5, fontsize=12)
    axes[1].set(
        title="Rail peaks only",
        ylabel="Recorded peak power (mW)",
        ylim=(0, max(peaks.values()) * 1.25),
    )
    finish(
        fig,
        out,
        "thor_thermal_power",
        "Thor thermals and recorded rail peaks",
        env,
        THERMAL + " + thor_trt_benchmark.json",
        "SHORT BENCHMARK, NOT A SOAK. Temperature points are raw. Rail time series "
        "are absent; no traces inferred.",
    )


def orin_chart(out):
    d = read_json(ORIN)
    p = read_json(ORIN_PROV)
    fig, axs = plt.subplots(2, 2, figsize=(12, 9.5))
    m = d["probes"]["matmul"]["warm"]
    ax = axs[0, 0]
    vals = [m[k] for k in ("p50_ms", "p95_ms", "p99_ms")]
    ax.scatter([0, 1, 2], vals, color=GREEN, s=70)
    for x, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 10), textcoords="offset points", ha="center")
    ax.set(
        xticks=[0, 1, 2],
        xticklabels=["p50", "p95", "p99"],
        ylabel="Warm latency (ms)",
        title=f"FP16 matmul | n={m['n']}",
        xlim=(-0.5, 2.5),
        ylim=(0, 3.5),
    )
    cam = d["probes"]["camera"]["frame_interval"]
    ax = axs[0, 1]
    vals = [cam[k] for k in ("min_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")]
    ax.scatter(range(5), vals, color=GREEN, s=70)
    for x, v in enumerate(vals):
        ax.annotate(
            f"{v:.2f}", (x, v), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10
        )
    ax.set(
        xticks=range(5),
        xticklabels=["min", "p50", "p95", "p99", "max"],
        ylabel="Frame interval (ms)",
        title=f"Camera intervals | n={cam['n']}",
        ylim=(0, 76),
        xlim=(-0.5, 4.5),
    )
    ax = axs[1, 0]
    vals = [d["thermal"]["temp_c_sustained_p50"], d["thermal"]["temp_c_peak"]]
    bars = ax.bar(["Sample p50", "Peak"], vals, color=GREEN, width=0.5)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=5)
    ax.set(
        title=f"Temperature summaries | n={d['thermal']['samples']}",
        ylabel="Temperature (C)",
        ylim=(0, 78),
    )
    ax = axs[1, 1]
    rails = [r for r in d["power"]["rails"] if isinstance(d["power"].get(r), dict)]
    x = np.arange(len(rails))
    for i, key in enumerate(("mW_p50", "mW_peak")):
        vals = [d["power"][r][key] for r in rails]
        bars = ax.bar(
            x + (i - 0.5) * 0.35,
            vals,
            0.32,
            color=("white" if i == 0 else GREEN),
            edgecolor=GREEN,
            linewidth=2,
            label=("p50" if i == 0 else "peak"),
        )
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in vals], padding=4, fontsize=9, rotation=0)
    ax.set(
        title=f"Rail summaries | n={d['power']['samples']}",
        ylabel="Power (mW)",
        xticks=x,
        xticklabels=rails,
        ylim=(0, 15500),
    )
    ax.legend(frameon=False, ncol=2, fontsize=10, loc="upper left")
    finish(
        fig,
        out,
        "orin_day_one",
        "Orin NX: day-one probe summaries",
        f"Orin NX | MAXN_SUPER | {p['timestamp_utc'][:10]} | "
        f"{d['wall_clock_s']} s run | camera 640 x 480",
        ORIN + " + provenance.json",
        "SHORT PROBE, NOT A SOAK. Raw interval, thermal and rail traces are absent. "
        "No distribution reconstructed.",
    )


def percentile(values, p):
    ordered = sorted(values)
    return ordered[round((p / 100) * (len(ordered) - 1))]


def ludo_chart(out):
    with (ROOT / LUDO).open(newline="") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.3))
    groups = [("Pre-fix: seeds 4-9", set(range(4, 10))), ("Post-fix: seeds 10-12", {10, 11, 12})]
    records = []
    for ax, (name, seeds) in zip(axes, groups, strict=True):
        selected = [
            r
            for r in rows
            if int(r["seed"]) in seeds and r["ok"] == "True" and int(r["attempt"]) == 1
        ]
        values = sorted(float(r["err_mm"]) for r in selected if 0 <= float(r["err_mm"]) < 35)
        if not values:
            raise ValueError(f"No eligible first-try placements for {name}")
        # Never use closest_approach_mm: it is a separate sentinel-bearing field.
        ax.scatter(values, np.arange(1, len(values) + 1), color=BLUE, s=25, alpha=0.85)
        for p, ls in ((50, "--"), (95, ":")):
            val = percentile(values, p)
            ax.axvline(val, color=BLUE, linestyle=ls, alpha=0.65)
        p50, p95 = percentile(values, 50), percentile(values, 95)
        ax.set(
            title=f"{name}\nn={len(values)} | p50 {p50:.1f}, p95 {p95:.1f} mm",
            xlabel="Successful placement error (mm)",
            ylabel="Sorted successful first-try sample",
            xlim=(0, 35),
        )
        records.append({"group": name, "n": len(values), "p50_mm": p50, "p95_mm": p95})
    finish(
        fig,
        out,
        "ludo_placement",
        "Ludo placement error: separate configurations",
        "ISAAC SIM | RTX 5090 | frozen campaign summary 2026-08-20 | scripted "
        "components + kinematic attach",
        LUDO,
        "Only successful first tries. Separate axes; no pooled curve. No clearance "
        "values, smoothing or interpolation.",
    )
    return records


def correction_chart(out):
    d = read_json(CORRECTED)
    if len(d["false_positive_episodes"]) != 1:
        raise ValueError("This correction figure requires the recorded single false positive")
    e = d["false_positive_episodes"][0]
    if any(e[k] for k in ("grasped", "lifted", "released")):
        raise ValueError("Correction event flags no longer support the figure caption")
    n = d["funnel"]["episodes"]
    corrected = d["first_try_ok_corrected"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 6.3))
    for ax in axs:
        ax.axis("off")
    axs[0].text(0.5, 0.90, "OLD GEOMETRIC RULE", ha="center", weight="bold", fontsize=13)
    axs[0].text(
        0.5,
        0.60,
        f"{corrected + len(d['false_positive_episodes'])}/{n}",
        ha="center",
        fontsize=54,
        color=GRAY,
        weight="bold",
    )
    axs[0].text(0.5, 0.38, "apparent success", ha="center", fontsize=16)
    axs[0].text(
        0.5,
        0.17,
        f"Target error {e['err_mm']} mm\nFinal tilt {e['final_tilt_deg']:.1f} degrees",
        ha="center",
        fontsize=13,
    )
    axs[1].text(0.5, 0.90, "SIMULATOR EVENT RECORD", ha="center", weight="bold", fontsize=13)
    axs[1].text(
        0.5,
        0.64,
        "Never grasped\nNever lifted\nNever released",
        ha="center",
        va="center",
        fontsize=20,
        color=BLUE,
        linespacing=1.6,
        weight="bold",
    )
    axs[1].text(
        0.5,
        0.17,
        f"Moved from pick: {e['moved_from_pick_mm']} mm\nAdjacent target triggered old rule",
        ha="center",
        fontsize=12,
    )
    axs[2].text(0.5, 0.90, "CORRECTED RESULT", ha="center", weight="bold", fontsize=13)
    axs[2].text(0.5, 0.60, f"{corrected}/{n}", ha="center", fontsize=54, color=BLUE, weight="bold")
    axs[2].text(0.5, 0.38, "successful episodes", ha="center", fontsize=16)
    axs[2].text(
        0.5, 0.17, "A grasp is now required.\nRaw record retained.", ha="center", fontsize=13
    )
    finish(
        fig,
        out,
        "gr00t_correction",
        "The cup never moved through a grasp",
        f"GR00T Ludo eval01 | ISAAC SIM | {n} episodes | record corrected by grasp requirement",
        CORRECTED,
        "Different rules are shown in separate panels, not on a shared metric axis. "
        "This is not physical ground truth.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/evidence/figures")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    style()
    thor_charts(out)
    orin_chart(out)
    groups = ludo_chart(out)
    correction_chart(out)
    manifest = {
        "generator": "scripts/plot_evidence.py",
        "matplotlib": matplotlib.__version__,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
        "ludo_groups": groups,
        "gaps": [
            "Thor per-iteration latency samples absent; range/median/mean/SD only.",
            "Thor thermal_samples.json has no rail samples; power peaks are in benchmark JSON.",
            "Orin raw timing, temperature and rail traces absent; percentiles/summaries only.",
            "Ludo historical pooled p50 19.3 / p95 30.1 spans configurations; shown "
            "separately instead.",
            "Correction uses simulator event ground truth, not physical object-success "
            "ground truth.",
        ],
    }
    (out / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
