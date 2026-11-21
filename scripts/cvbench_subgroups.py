#!/usr/bin/env python3
"""CV-Bench per-task subgroup analysis (Count, Relation, Depth, Distance).

For each comparison and task: accuracy of both sides on the records valid in both, the paired change
in points with a 95% paired percentile bootstrap CI, and the exact McNemar p-value with the discordant
counts (worse = baseline right and other wrong). It uses the loaders and statistics from
make_table_b.py, so the per-task numbers add up to Table B's CV-Bench column.

Comparisons:
  vision sweep (language Q8_0)   Configs 5, 9, 10, 8, 11 vs Config 2 (F16 vision)
  language decoder               Config 3 (Q4_K_M) vs Config 2 (Q8_0)
  TensorRT quantization          T8, T4 vs T16
  cross-runtime                  T16 vs Config 2

Also reports position bias: the share of predictions that are "A" per task, against the share of
ground-truth answers that are "A", to see whether low-bit models drift toward the first option.

Bootstrap: 10,000 resamples, default_rng([seed, comparison number, task number]), seed 249.
Intervals are not adjusted for multiple comparisons (4 tasks x 9 comparisons).

Usage: python3 scripts/cvbench_subgroups.py [--out-dir results/analysis]
Writes cvbench_subgroups.md and cvbench_subgroups.csv.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_table_b import load_results, paired_delta  # noqa: E402

TASKS = ["Count", "Relation", "Depth", "Distance"]
FILES = {
    "C2": "results/config2_F16v_Q8l__cv_bench.jsonl",
    "C5": "results/config5_Q8v_Q8l__cv_bench.jsonl",
    "C9": "results/config9_Q6v_Q8l__cv_bench.jsonl",
    "C10": "results/config10_Q5v_Q8l__cv_bench.jsonl",
    "C8": "results/config8_Q4v_Q8l__cv_bench.jsonl",
    "C11": "results/config11_Q3v_Q8l__cv_bench.jsonl",
    "C3": "results/config3_F16v_Q4l__cv_bench.jsonl",
    "T16": "results_trt/configT16_TRT_F16v_F16l__cv_bench.jsonl",
    "T8": "results_trt/configT8_TRT_F16v_INT8l__cv_bench.jsonl",
    "T4": "results_trt/configT4_TRT_F16v_INT4l__cv_bench.jsonl",
}
LABELS = {
    "C2": "C2 F16v/Q8_0", "C5": "C5 Q8_0 vision", "C9": "C9 Q6_K vision", "C10": "C10 Q5_K_M vision",
    "C8": "C8 Q4_0 vision", "C11": "C11 Q3_K_M vision", "C3": "C3 Q4_K_M language",
    "T16": "T16 TRT FP16", "T8": "T8 TRT INT8 SQ", "T4": "T4 TRT INT4 AWQ",
}
COMPARISONS = [  # (group, baseline, other)
    ("vision", "C2", "C5"), ("vision", "C2", "C9"), ("vision", "C2", "C10"),
    ("vision", "C2", "C8"), ("vision", "C2", "C11"),
    ("language", "C2", "C3"),
    ("tensorrt", "T16", "T8"), ("tensorrt", "T16", "T4"),
    ("runtime", "C2", "T16"),
]


def task_of(path):
    """{idx: (task, predicted letter, gt letter)} from the raw file (all rows, errors included)."""
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            out[r["idx"]] = (r["task"], r.get("predicted"), r.get("gt_answer"))
    return out


def subset(d, ids):
    return {"valid": {i: v for i, v in d["valid"].items() if i in ids}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=249)
    args = ap.parse_args()

    data = {k: load_results(p) for k, p in FILES.items()}
    info = {k: task_of(p) for k, p in FILES.items()}
    ref = info["C2"]
    for k, inf in info.items():  # same idx must be the same question/task in every file
        bad = [i for i in ref if inf.get(i, (None, None, None))[0] != ref[i][0] or inf[i][2] != ref[i][2]]
        if bad:
            sys.exit(f"ERROR: {k}: {len(bad)} records differ in task/gt from C2 (e.g. idx {bad[:5]})")
    # Universe: the records valid in C2 (2,520; Table B and the cross-runtime table use the same set).
    # Config 3 and the TensorRT runs also answered the 118 records that fail in C2; they are left out.
    universe = set(data["C2"]["valid"])
    ids_by_task = {t: {i for i, v in ref.items() if v[0] == t and i in universe} for t in TASKS}

    rows = []
    for ci, (group, base, other) in enumerate(COMPARISONS):
        for ti, task in enumerate(TASKS + ["all"]):
            ids = universe if task == "all" else ids_by_task[task]
            rng = np.random.default_rng([args.seed, ci, ti])
            r = paired_delta(subset(data[base], ids), subset(data[other], ids), args.n_boot, rng)
            rows.append({"group": group, "baseline": base, "other": other, "task": task,
                         "n_paired": r["n_paired"], "acc_baseline": r["base_acc"], "acc_other": r["acc"],
                         "delta": r["delta"], "ci_lo": r["ci"][0], "ci_hi": r["ci"][1],
                         "worse": r["worse"], "better": r["better"], "p_mcnemar": r["p_mcnemar"]})

    # Position bias: share of "A" answers among valid records, per task and config.
    bias = {}
    for k in FILES:
        for task in TASKS:
            v = [info[k][i][1] for i in ids_by_task[task] if i in data[k]["valid"]]
            bias[(k, task)] = 100 * sum(p == "A" for p in v) / len(v)
    gt_a = {t: 100 * sum(ref[i][2] == "A" for i in ids_by_task[t]) / len(ids_by_task[t]) for t in TASKS}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cvbench_subgroups.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    def cell(r):
        s = f"{r['delta']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
        return f"**{s}**" if r["ci_lo"] > 0 or r["ci_hi"] < 0 else s

    n_task = {t: next(r["n_paired"] for r in rows if r["task"] == t and r["other"] == "C5") for t in TASKS}
    L = ["# CV-Bench per-task subgroup analysis", "",
         "Generated by `scripts/cvbench_subgroups.py`. All rows use the 2,520 records valid in C2 (as Table B and the cross-runtime table): "
         + ", ".join(f"{t} {n_task[t]}" for t in TASKS) + ". "
         "Change in points vs the baseline, paired 95% bootstrap CI (10,000 resamples, seed 249). "
         "**Bold** = interval excludes 0. Not adjusted for multiple comparisons.", "",
         "## Accuracy per task", "",
         "| config | " + " | ".join(TASKS) + " | all |", "|---|" + "---:|" * (len(TASKS) + 1)]
    seen = set()
    for _, base, other in COMPARISONS:
        for k in (base, other):
            if k in seen:
                continue
            seen.add(k)
            accs = []
            for task in TASKS + ["all"]:
                ids = universe if task == "all" else ids_by_task[task]
                v = [s for i, s in data[k]["valid"].items() if i in ids]
                accs.append(f"{100 * sum(v) / len(v):.2f}")
            L.append(f"| {LABELS[k]} | " + " | ".join(accs) + " |")
    L += ["", "## Paired change per task", "",
          "| comparison | " + " | ".join(TASKS) + " | all |", "|---|" + "---|" * (len(TASKS) + 1)]
    for group, base, other in COMPARISONS:
        cells = [cell(next(r for r in rows if r["other"] == other and r["baseline"] == base and r["task"] == t))
                 for t in TASKS + ["all"]]
        L.append(f"| {LABELS[other]} vs {base} | " + " | ".join(cells) + " |")
    L += ["", "## Discordant pairs (worse / better) and exact McNemar p", "",
          "| comparison | " + " | ".join(TASKS) + " |", "|---|" + "---|" * len(TASKS)]
    for group, base, other in COMPARISONS:
        cells = []
        for t in TASKS:
            r = next(r for r in rows if r["other"] == other and r["baseline"] == base and r["task"] == t)
            cells.append(f"{r['worse']} / {r['better']}, p={r['p_mcnemar']:.3g}")
        L.append(f"| {LABELS[other]} vs {base} | " + " | ".join(cells) + " |")
    L += ["", "## Position bias: share of predictions that are \"A\" (%)", "",
          "| config | " + " | ".join(TASKS) + " |", "|---|" + "---:|" * len(TASKS),
          "| ground truth | " + " | ".join(f"{gt_a[t]:.1f}" for t in TASKS) + " |"]
    for k in FILES:
        L.append(f"| {LABELS[k]} | " + " | ".join(f"{bias[(k, t)]:.1f}" for t in TASKS) + " |")
    text = "\n".join(L) + "\n"
    (out / "cvbench_subgroups.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
