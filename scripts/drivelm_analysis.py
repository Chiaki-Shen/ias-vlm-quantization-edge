#!/usr/bin/env python3
"""DriveLM driving-scene check: vision sweep accuracy with paired CIs.

Reads the six per-record result files (read-only; never modifies them):
    results/config{N}_{short}__drivelm.jsonl   for N in 2, 5, 9, 10, 8, 11
and writes
    results/analysis/drivelm.md      tables for the report / README
    results/analysis/drivelm.csv     one row per (config, subset), with deltas

Subsets: traffic_light (150, 3-way MCQ), sign_box (150, yes/no) and all (300).

Statistics follow make_table_b.py exactly (same functions are imported):
paired percentile bootstrap on the per-record score difference vs Config 2
(F16 vision), 10,000 resamples, seed 249, independent stream per
(config, subset): default_rng([seed, config, subset_index]); exact two-sided
McNemar p. 95% intervals, NOT adjusted for multiple comparisons.

Also reports the answer distribution (which colour / Yes-No each config
predicts) and accuracy per ground-truth class, to check whether accuracy
changes come from a shift in answer bias rather than better perception.

Usage (repo root):  python scripts/drivelm_analysis.py
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_table_b import CONFIG_ORDER, CONFIG_INFO, paired_delta  # noqa: E402

SUBSETS = ["traffic_light", "sign_box", "all"]
LETTERS = "ABCDEFGH"


def load(path):
    """-> list of dicts (one per record) with idx, task, score, pred/gt text."""
    rows, seen = [], set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["idx"] in seen:
                sys.exit(f"ERROR: duplicate idx {r['idx']} in {path}")
            seen.add(r["idx"])
            if r.get("error_type") == "ServerUnavailable":
                sys.exit(f"ERROR: {path} has never-attempted rows; run incomplete")
            ch = r.get("choices") or []

            def text(letter):
                if letter and letter in LETTERS[:len(ch)]:
                    return ch[LETTERS.index(letter)]
                return "(unparsed)"

            rows.append({
                "idx": r["idx"], "task": r.get("task"),
                "error": bool(r.get("error")),
                "score": float(r["score"]) if r.get("score") is not None
                else (1.0 if r.get("correct") else 0.0),
                "pred": text(r.get("predicted")),
                "gt": text(r.get("gt_answer")),
            })
    return rows


def as_tb(rows, subset):
    """Shape expected by make_table_b.paired_delta."""
    sel = [r for r in rows if subset == "all" or r["task"] == subset]
    return {"valid": {r["idx"]: r["score"] for r in sel if not r["error"]},
            "err": {r["idx"] for r in sel if r["error"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=249)
    args = ap.parse_args()

    res = Path(args.results_dir)
    data = {}
    for cfg in CONFIG_ORDER:
        hits = sorted(res.glob(f"config{cfg}_*__drivelm.jsonl"))
        if len(hits) != 1:
            sys.exit(f"ERROR: expected one drivelm file for config {cfg}, found {hits}")
        data[cfg] = load(hits[0])
        n, e = len(data[cfg]), sum(r["error"] for r in data[cfg])
        print(f"config {cfg:>2}: {hits[0].name}  n={n}  errors={e}")
        if n != 300:
            sys.exit(f"ERROR: config {cfg} has {n} records, expected 300")

    # alignment: same idx -> same task and same ground truth in every config
    ref = {r["idx"]: (r["task"], r["gt"]) for r in data[CONFIG_ORDER[0]]}
    for cfg in CONFIG_ORDER[1:]:
        cur = {r["idx"]: (r["task"], r["gt"]) for r in data[cfg]}
        if cur != ref:
            sys.exit(f"ERROR: config {cfg} records do not align with config 2")

    base = CONFIG_ORDER[0]
    base_tb = {s: as_tb(data[base], s) for s in SUBSETS}
    csv_rows, results = [], {}
    for cfg in CONFIG_ORDER:
        for si, s in enumerate(SUBSETS):
            tb = as_tb(data[cfg], s)
            acc = 100 * np.mean(list(tb["valid"].values()))
            row = {"config": cfg, "vision": CONFIG_INFO[cfg][0], "subset": s,
                   "n": len(tb["valid"]), "n_err": len(tb["err"]),
                   "acc": round(acc, 2)}
            if cfg != base:
                rng = np.random.default_rng([args.seed, cfg, si])
                d = paired_delta(base_tb[s], tb, args.n_boot, rng)
                row.update({"delta": round(d["delta"], 2),
                            "ci_lo": round(d["ci"][0], 2), "ci_hi": round(d["ci"][1], 2),
                            "worse": d["worse"], "better": d["better"],
                            "p_mcnemar": round(d["p_mcnemar"], 4)})
            results[(cfg, s)] = row
            csv_rows.append(row)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["config", "vision", "subset", "n", "n_err", "acc", "delta",
              "ci_lo", "ci_hi", "worse", "better", "p_mcnemar"]
    with open(out / "drivelm.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)

    md = ["# DriveLM driving-scene check (llama.cpp vision sweep)", "",
          "300 DriveLM-nuScenes CAM_FRONT records: traffic_light (150, 3-way MCQ "
          "Red/Green/Yellow, chance 33.3%) and sign_box (150, \"is the boxed object "
          "a traffic sign or road barrier?\" yes/no, chance 50%). Language decoder "
          "Q8_0 fixed. Generated by `scripts/drivelm_analysis.py`.", "",
          "## Accuracy (%)", "",
          "| Config | Vision | traffic_light | sign_box | All 300 | Errors |",
          "|---|---|---|---|---|---|"]
    for cfg in CONFIG_ORDER:
        r = {s: results[(cfg, s)] for s in SUBSETS}
        md.append(f"| {cfg} | {CONFIG_INFO[cfg][0]} | {r['traffic_light']['acc']:.1f} | "
                  f"{r['sign_box']['acc']:.1f} | {r['all']['acc']:.1f} | {r['all']['n_err']} |")

    md += ["", f"## Change from Config {base} (F16 vision), points, paired 95% bootstrap CI", "",
           "Cell: delta [CI low, CI high], McNemar p, (records F16 got right and this "
           "config got wrong / the reverse).", "",
           "| Config | Vision | traffic_light | sign_box | All 300 |", "|---|---|---|---|---|"]
    for cfg in CONFIG_ORDER[1:]:
        cells = []
        for s in SUBSETS:
            r = results[(cfg, s)]
            star = " *" if r["ci_lo"] > 0 or r["ci_hi"] < 0 else ""
            cells.append(f"{r['delta']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}], "
                         f"p={r['p_mcnemar']:.3f} ({r['worse']}/{r['better']}){star}")
        md.append(f"| {cfg} | {CONFIG_INFO[cfg][0]} | " + " | ".join(cells) + " |")
    md += ["", "`*` = CI excludes 0. Not adjusted for multiple comparisons "
           "(5 configs x 3 subsets)."]

    # answer distribution + per-class accuracy
    for task in ["traffic_light", "sign_box"]:
        gts = sorted({r["gt"] for r in data[base] if r["task"] == task})
        preds = sorted({r["pred"] for cfg in CONFIG_ORDER for r in data[cfg]
                        if r["task"] == task})
        gt_count = collections.Counter(r["gt"] for r in data[base] if r["task"] == task)
        md += ["", f"## {task}: predicted answers (count of 150)", "",
               "Ground truth: " + ", ".join(f"{g} {gt_count[g]}" for g in gts), "",
               "| Config | Vision | " + " | ".join(preds) + " |",
               "|---|---|" + "---|" * len(preds)]
        for cfg in CONFIG_ORDER:
            c = collections.Counter(r["pred"] for r in data[cfg] if r["task"] == task)
            md.append(f"| {cfg} | {CONFIG_INFO[cfg][0]} | "
                      + " | ".join(str(c[p]) for p in preds) + " |")
        md += ["", f"## {task}: accuracy by ground-truth class (%)", "",
               "| Config | Vision | " + " | ".join(f"{g} (n={gt_count[g]})" for g in gts) + " |",
               "|---|---|" + "---|" * len(gts)]
        for cfg in CONFIG_ORDER:
            cells = []
            for g in gts:
                sel = [r for r in data[cfg] if r["task"] == task and r["gt"] == g]
                cells.append(f"{100 * np.mean([r['score'] for r in sel]):.1f}")
            md.append(f"| {cfg} | {CONFIG_INFO[cfg][0]} | " + " | ".join(cells) + " |")

    (out / "drivelm.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'drivelm.md'} and {out / 'drivelm.csv'}")


if __name__ == "__main__":
    main()
