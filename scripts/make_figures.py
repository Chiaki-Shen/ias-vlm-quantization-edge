#!/usr/bin/env python3
"""Make the three README / report figures from the committed result tables.

    python scripts/make_figures.py            # from the repo root, writes figures/*.png

Inputs (read-only, all already in the repo):
  results/table_b_deltas.csv                    Figure 1 (Table B paired deltas + CIs)
  results/crossruntime_table.csv                Figure 2 (matched-record accuracy)
  results_latency/overnight/                    Figure 3 (Oct 5 vision sweep, Configs 2/5/9/10/8/11)
  results_latency/config3_nov/                  Figure 3 (Nov 14 same-session Configs 2/3)
  results_latency_trt/trio/                     Figure 3 (TensorRT T8/T4/T16)

Each latency folder gives the TextVQA median wall-clock from latency_summary.csv and the
peak system RAM (tegrastats, timed phase, strict) from summary.txt, the same numbers as the
README tables. BLINK is left out of the figures: with 124/143 questions its intervals are
too wide to resolve the effects shown here.

Needs matplotlib (tested with 3.x). No numpy needed.
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ---- palette (light mode; categorical slots validated all-pairs for 3 series) -------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

DATASETS = [("cv_bench", "CV-Bench", BLUE), ("scienceqa", "ScienceQA", ORANGE),
            ("textvqa", "TextVQA", AQUA)]

# Table B vision encoders: config -> (label, mmproj size MB). Sizes as in the README.
VISION = {"2": ("F16", 782), "5": ("Q8_0", 421), "9": ("Q6_K", 329),
          "10": ("Q5_K_M", 295), "8": ("Q4_0", 229), "11": ("Q3_K_M", 202)}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
    })


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def title(ax, main, sub):
    ax.set_title(main, loc="left", color=INK, fontweight="bold", pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


# ---- Figure 1 ------------------------------------------------------------------------
def fig_vision_sweep(repo, out):
    rows = [r for r in read_csv(repo / "results/table_b_deltas.csv") if r["baseline"] == "2"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    dodge = {"cv_bench": 1 / 1.018, "scienceqa": 1.0, "textvqa": 1.018}  # so CIs don't overlap
    for key, name, color in DATASETS:
        pts = [(VISION["2"][1], 0.0, 0.0, 0.0)]
        for r in rows:
            if r["dataset"] == key and r["config"] in VISION:
                pts.append((VISION[r["config"]][1], float(r["delta_pts"]),
                            float(r["ci_lo"]), float(r["ci_hi"])))
        pts.sort(key=lambda p: -p[0])
        xs = [p[0] * (dodge[key] if p[0] != VISION["2"][1] else 1) for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, lw=2, zorder=2)
        for x, (size, d, lo, hi) in zip(xs, pts):
            if size == VISION["2"][1]:
                ax.plot(x, 0, "o", ms=8, color=color, mec=SURFACE, mew=1.5, zorder=4)
                continue
            ax.plot([x, x], [lo, hi], color=color, lw=1.4, alpha=0.75, zorder=1,
                    solid_capstyle="round")
            sig = lo > 0 or hi < 0
            ax.plot(x, d, "o", ms=8, zorder=4, color=color if sig else SURFACE,
                    mec=color, mew=1.8)
        # direct label at the low-precision end
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(-9, 0), textcoords="offset points",
                    va="center", ha="right", color=INK, fontsize=9.5, fontweight="bold")

    ax.axhline(0, color=AXIS, lw=1, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(150, 850)  # ascending size; F16 at the right
    ax.minorticks_off()
    ax.set_xticks([v[1] for v in VISION.values()])
    ax.set_xticklabels([f"{v[0]}\n{v[1]}" for v in VISION.values()], fontsize=8.5)
    ax.set_xlabel("Vision encoder: mmproj file size in MB (log scale)")
    ax.set_ylabel("Change from F16 vision (accuracy points)")
    ax.set_ylim(-3.8, 1.4)
    ax.grid(axis="x", visible=False)
    ax.axvspan(150, 262, color=GRID, alpha=0.35, zorder=0, lw=0)
    ax.text(258, 1.15, "4 bits and below", color=INK_2, fontsize=9, ha="right")
    title(ax, "Vision quantization is nearly free down to 5 bits",
          "Cosmos-Reason2-2B on Jetson Orin Nano, llama.cpp, language decoder fixed at Q8_0. "
          "Paired 95% bootstrap CI.")
    legend = [Line2D([], [], ls="none", marker="o", ms=8, color=INK_2, label="CI excludes 0"),
              Line2D([], [], ls="none", marker="o", ms=8, color=SURFACE, mec=INK_2, mew=1.8,
                     label="CI includes 0")]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig1_vision_sweep.png", dpi=200)
    plt.close(fig)


# ---- Figure 2 ------------------------------------------------------------------------
CROSS = [("C2", "llama.cpp\nQ8_0", BLUE), ("T8", "TRT\nINT8 SQ", ORANGE),
         ("C3", "llama.cpp\nQ4_K_M", BLUE), ("T4", "TRT\nINT4 AWQ", ORANGE),
         ("T16", "TRT\nFP16", MUTED)]


def fig_crossruntime(repo, out):
    acc = {(r["config"], r["dataset"]): (float(r["accuracy_pct"]), int(r["n_matched"]))
           for r in read_csv(repo / "results/crossruntime_table.csv")}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0), sharey=True)
    xpos = [0, 1, 2.5, 3.5, 5]  # gap between the 8-bit pair, 4-bit pair and FP16 reference
    for ax, (key, name, _) in zip(axes, DATASETS):
        n = acc[("C2", key)][1]
        for x, (cfg, _lab, color) in zip(xpos, CROSS):
            v = acc[(cfg, key)][0]
            ax.bar(x, v, width=0.86, color=color, linewidth=0)
            ax.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5,
                    color=INK_2)
        ax.set_xticks(xpos)
        ax.set_xticklabels([c[0] for c in CROSS], color=INK_2)
        for x, txt in [(0.5, "8-bit"), (3.0, "4-bit"), (5, "ref")]:
            ax.text(x, -0.16, txt, transform=ax.get_xaxis_transform(), ha="center",
                    color=MUTED, fontsize=8.5)
        ax.set_title(f"{name}  (n={n:,})", loc="left", color=INK, fontsize=10.5)
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Accuracy (%), matched records")
    fig.suptitle("Matched bit width, two runtimes: TensorRT's quantized engines lose more "
                 "on ScienceQA", x=0.01, ha="left", color=INK, fontweight="bold", fontsize=12)
    fig.text(0.01, 0.885, "Vision F16/FP16 in every bar. TRT = TensorRT Edge-LLM; "
             "SQ = SmoothQuant W8A8; AWQ = W4A16. Pairs differ in runtime and method.",
             color=INK_2, fontsize=9, ha="left")
    handles = [Patch(color=BLUE, label="llama.cpp (weight-only GGUF)"),
               Patch(color=ORANGE, label="TensorRT Edge-LLM, quantized"),
               Patch(color=MUTED, label="TensorRT Edge-LLM, FP16 reference")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    fig.savefig(out / "fig2_crossruntime_accuracy.png", dpi=200)
    plt.close(fig)


# ---- Figure 3 ------------------------------------------------------------------------
def latency_points(folder):
    """{config: (textvqa median ms, timed peak RAM MB strict)} from one latency run."""
    med = {r["config"]: float(r["wall_med"]) for r in read_csv(folder / "latency_summary.csv")
           if r["dataset"] == "textvqa"}
    ram = {}
    pat = re.compile(r"^\s*(\S+)\s+timed\s+(\d+)\s+(\d+)\s+")
    in_tegra = False
    for line in (folder / "summary.txt").read_text().splitlines():
        if line.startswith("== tegrastats"):
            in_tegra = True
        m = pat.match(line) if in_tegra else None
        if m:
            ram[m.group(1)] = int(m.group(3))  # 'max strict'
    return {c: (med[c], ram[c]) for c in med}


def fig_latency_memory(repo, out):
    oct5 = latency_points(repo / "results_latency/overnight")
    nov = latency_points(repo / "results_latency/config3_nov")
    trt = latency_points(repo / "results_latency_trt/trio")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    # The 8 llama.cpp points sit within ~700 MB and ~150 ms of each other, so they are
    # labeled in a zoomed inset. Label offsets (points) are set by hand to avoid collisions.
    lab_oct = {"2": ("C2 F16 (Oct 5)", 0, 9, "center"), "5": ("Q8_0", 0, -11, "center"),
               "9": ("Q6_K", 0, 9, "center"), "10": ("Q5_K_M", 7, 0, "left"),
               "8": ("Q4_0", 0, -11, "center"), "11": ("Q3_K_M", -7, 0, "right")}
    lab_nov = {"2": ("C2 F16 (Nov 14)", 0, -11, "center"),
               "3": ("C3 F16 + Q4_K_M lang", 8, 0, "left")}
    lab_trt = {"T8": ("T8 INT8 SQ", 8, -4, "left"), "T4": ("T4 INT4 AWQ", 8, -4, "left"),
               "T16": ("T16 FP16", 0, 11, "center")}
    styles = {"oct": ("o", BLUE, True), "nov": ("s", BLUE, False), "trt": ("^", ORANGE, True)}

    def draw(axis, points, kind, labels=None, ms=8.5):
        marker, color, filled = styles[kind]
        for c, (lat, mb) in points.items():
            axis.plot(mb, lat, marker, ms=ms, zorder=3,
                      color=color if filled else SURFACE,
                      mec=SURFACE if filled else color, mew=1.2 if filled else 1.8)
            if labels:
                text, dx, dy, ha = labels[c]
                axis.annotate(text, (mb, lat), xytext=(dx, dy), textcoords="offset points",
                              fontsize=8, color=INK_2, va="center", ha=ha)

    draw(ax, oct5, "oct")
    draw(ax, nov, "nov")
    draw(ax, trt, "trt", lab_trt)
    ax.set_xlim(4000, 7200)
    ax.set_ylim(0, 3200)
    fmt = matplotlib.ticker.StrMethodFormatter("{x:,.0f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel("Peak system RAM during timed requests (MB, tegrastats; board total 7,620)")
    ax.set_ylabel("TextVQA median latency per request (ms)")

    ins = ax.inset_axes([0.38, 0.47, 0.6, 0.45])
    draw(ins, oct5, "oct", lab_oct, ms=7.5)
    draw(ins, nov, "nov", lab_nov, ms=7.5)
    ins.set_xlim(4150, 5050)
    ins.set_ylim(2580, 2840)
    ins.xaxis.set_major_formatter(fmt)
    ins.yaxis.set_major_formatter(fmt)
    ins.tick_params(labelsize=7.5)
    for side in ("top", "right"):
        ins.spines[side].set_visible(True)
    ins.set_title("llama.cpp configs (zoomed)", fontsize=8.5, color=INK_2, loc="left")
    ax.indicate_inset_zoom(ins, edgecolor=MUTED, alpha=0.8)

    title(ax, "TensorRT is 2.4 to 2.8x faster on TextVQA; llama.cpp uses less memory",
          "200 frozen records x 3 rounds, 15W, one request at a time. Lower-left is better.")
    handles = [
        Line2D([], [], ls="none", marker="o", ms=8, color=BLUE,
               label="llama.cpp, vision sweep, language Q8_0 (Oct 5 run)"),
        Line2D([], [], ls="none", marker="s", ms=8, color=SURFACE, mec=BLUE, mew=1.8,
               label="llama.cpp, F16 vision, Q8_0 / Q4_K_M language (Nov 14 run)"),
        Line2D([], [], ls="none", marker="^", ms=8, color=ORANGE,
               label="TensorRT Edge-LLM, FP16 vision"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "fig3_latency_vs_memory.png", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default=".", help="repo root (default: current directory)")
    p.add_argument("--out-dir", default="figures")
    a = p.parse_args()
    repo = Path(a.repo)
    out = repo / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    style()
    fig_vision_sweep(repo, out)
    fig_crossruntime(repo, out)
    fig_latency_memory(repo, out)
    print(f"wrote 3 figures to {out}/")


if __name__ == "__main__":
    main()
