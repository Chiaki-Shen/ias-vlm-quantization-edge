#!/usr/bin/env python3
"""Cross-runtime table: llama.cpp (Configs 2, 3) vs TensorRT Edge-LLM (T16, T8, T4).

Reads per-record result JSONL files (read-only; never modifies them) and writes
  crossruntime_table.md, crossruntime_table.csv, crossruntime_deltas.csv,
  crossruntime_latency.csv, crossruntime_provenance.json

Usage (from the repo root; these are the defaults, so plain
`python scripts/make_crossruntime_table.py` does the same):
    python scripts/make_crossruntime_table.py \
        --llamacpp-dir results --trt-dir results_trt \
        --latency-dir results_latency/overnight --latency-dir results_latency_trt/trio \
        --out-dir results
When the Config 3 latency run lands, add its folder with another --latency-dir.

Matched records (the main difference from make_table_b.py):
  * Per dataset, a record is EXCLUDED for every config if any loaded config has
    a non-empty `error` on it, or if it is listed in a llama.cpp
    `*__{dataset}_poison.jsonl` file. The legacy Config 3 CV-Bench file has no
    `error` field (its 118 poison rows look like plain wrong answers), so the
    poison files are what remove them there.
  * All configs are then scored on the same record set, so every accuracy in a
    column uses the same denominator and every delta is paired on it.
  * A missing file (Config 3 ScienceQA/TextVQA until its run lands) is shown as
    "pending" and does not enter the exclusion union. When it lands, rerun:
    if it has errors of its own, the matched set shrinks and the other cells in
    that column can move by a few hundredths of a point.
  * Every loaded file must cover the same idx set and carry the same question
    and ground truth per idx; otherwise the script aborts (this catches a
    half-finished run copied off the Jetson).

Per-record score = `score` if present and not null, else float(correct)
(TextVQA VQA accuracy is fractional). Pairing is on `idx`.

Statistics: paired percentile bootstrap (default 10,000 resamples, seed 249) on
the per-record score difference, independent stream per (pair, dataset):
default_rng([seed, pair_index, dataset_index]). Exact two-sided McNemar p for
binary (MCQ) datasets. 95% intervals, NOT adjusted for multiple comparisons.

Latency comes from each --latency-dir's latency_summary.csv (median and p95
wall ms) and summary.txt (tegrastats peak system RAM, timed phase, strict).
Wall-clock definitions differ: llama.cpp is an HTTP round trip to llama-server,
TensorRT is one in-process LLM.generate() call.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

# key, runtime, vision, language, results-dir kind, file stem
CONFIGS = [
    ("C2", "llama.cpp", "F16", "Q8_0", "llamacpp", "config2_F16v_Q8l"),
    ("T8", "TensorRT Edge-LLM", "FP16", "INT8 SmoothQuant (W8A8)", "trt",
     "configT8_TRT_F16v_INT8l"),
    ("C3", "llama.cpp", "F16", "Q4_K_M", "llamacpp", "config3_F16v_Q4l"),
    ("T4", "TensorRT Edge-LLM", "FP16", "INT4 AWQ (W4A16)", "trt",
     "configT4_TRT_F16v_INT4l"),
    ("T16", "TensorRT Edge-LLM", "FP16", "FP16", "trt", "configT16_TRT_F16v_F16l"),
]
CFG = {c[0]: c for c in CONFIGS}
# Latency summary labels -> config keys
LAT_LABEL = {"2": "C2", "3": "C3", "T16": "T16", "T8": "T8", "T4": "T4"}

# (base, other, description). Order fixes the RNG stream index; append only.
PAIRS = [
    ("C2", "T8", "8-bit language, cross-runtime (Q8_0 vs INT8 SQ)"),
    ("C3", "T4", "4-bit language, cross-runtime (Q4_K_M vs INT4 AWQ)"),
    ("T16", "T8", "INT8 SQ, within TensorRT"),
    ("T16", "T4", "INT4 AWQ, within TensorRT"),
    ("C2", "C3", "Q8_0 to Q4_K_M, within llama.cpp"),
    ("C2", "T16", "llama.cpp Q8_0 vs TensorRT FP16 (not a pure runtime comparison)"),
]

DATASET_COLS = [
    ("blink_depth", "BLINK depth"),
    ("blink_spatial", "BLINK spatial"),
    ("cv_bench", "CV-Bench"),
    ("scienceqa", "ScienceQA"),
    ("textvqa", "TextVQA"),
]
SMALL = {"blink_depth", "blink_spatial"}


# ----------------------------------------------------------------- loading --
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_score(r):
    s = r.get("score")
    if s is not None:
        return float(s)
    c = r.get("correct")
    if c is None:
        raise ValueError(f"record idx={r.get('idx')} has neither score nor correct")
    return 1.0 if c else 0.0


def record_key(r):
    """What must be identical across configs for the same idx."""
    gt = r.get("gt_answer")
    if gt is None and r.get("answers") is not None:
        gt = tuple(r["answers"])
    return (r.get("question"), gt)


def load_results(path):
    scores, keys, errs = {}, {}, set()
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            idx = r["idx"]
            if idx in keys:
                sys.exit(f"ERROR: duplicate idx {idx} in {path} (line {ln})")
            if r.get("error_type") == "ServerUnavailable":
                sys.exit(f"ERROR: {path} contains never-attempted placeholder "
                         f"rows (idx {idx}); the run is not complete")
            keys[idx] = record_key(r)
            if r.get("error"):
                errs.add(idx)
            else:
                scores[idx] = record_score(r)
    return {"scores": scores, "keys": keys, "err": errs}


def load_poison(path):
    out = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                out.add(json.loads(line)["idx"])
    return out


# -------------------------------------------------------------- statistics --
def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_bootstrap(diff, n_boot, rng, chunk=500):
    n = len(diff)
    means = np.empty(n_boot)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        ix = rng.integers(0, n, size=(m, n))
        means[done:done + m] = diff[ix].mean(axis=1)
        done += m
    return np.percentile(means, [2.5, 97.5])


def paired_delta(a, b, n_boot, rng):
    diff = b - a
    out = {"n": len(a), "base_acc": a.mean() * 100, "acc": b.mean() * 100,
           "delta": diff.mean() * 100}
    if np.allclose(diff, 0):
        out["ci"] = (0.0, 0.0)
    else:
        lo, hi = paired_bootstrap(diff, n_boot, rng)
        out["ci"] = (lo * 100, hi * 100)
    if set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}:
        out["worse"] = int(((a == 1) & (b == 0)).sum())
        out["better"] = int(((a == 0) & (b == 1)).sum())
        out["p_mcnemar"] = mcnemar_exact(out["worse"], out["better"])
    return out


# ----------------------------------------------------------------- latency --
TEG_RE = re.compile(r"^\s*(\S+)\s+timed\s+(\d+)\s+(\d+)\s+(\d+)")


def load_latency(dirs):
    """-> {cfg_key: {"dir", "ram_peak", dataset: {med, p95, valid}}}"""
    lat = {}
    for d in dirs:
        d = Path(d)
        csv_p = d / "latency_summary.csv"
        if not csv_p.exists():
            sys.exit(f"ERROR: {csv_p} not found")
        ram = {}
        txt = d / "summary.txt"
        if txt.exists():
            in_teg = False
            for line in txt.read_text().splitlines():
                if line.startswith("== tegrastats"):
                    in_teg = True
                elif line.startswith("=="):
                    in_teg = False
                elif in_teg:
                    m = TEG_RE.match(line)
                    if m:
                        ram[m.group(1)] = int(m.group(3))  # max strict
        with open(csv_p) as f:
            for row in csv.DictReader(f):
                key = LAT_LABEL.get(row["config"])
                if key is None:
                    continue
                entry = lat.setdefault(key, {"dir": str(d), "ram_peak": ram.get(row["config"])})
                if entry["dir"] != str(d):
                    sys.exit(f"ERROR: latency for {key} found in both {entry['dir']} and {d}; "
                             "pass only one --latency-dir per config")
                entry[row["dataset"]] = {"med": float(row["wall_med"]),
                                         "p95": float(row["wall_p95"]),
                                         "valid": int(row["valid"])}
    return lat


# -------------------------------------------------------------------- main --
def fmt_delta(d, small):
    lo, hi = d["ci"]
    star = "*" if (lo > 0 or hi < 0) else ""
    s = f"{d['delta'] + 0.0:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}"
    return s + (" †" if small else "")


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llamacpp-dir", default=str(root / "results"))
    ap.add_argument("--trt-dir", default=str(root / "results_trt"))
    ap.add_argument("--latency-dir", action="append",
                    help="folder with latency_summary.csv + summary.txt (repeatable); "
                         "default: results_latency/overnight and results_latency_trt/trio")
    ap.add_argument("--out-dir", default=str(root / "results"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=249)
    args = ap.parse_args()
    if not args.latency_dir:
        args.latency_dir = [str(root / "results_latency/overnight"),
                            str(root / "results_latency_trt/trio")]
    dirs = {"llamacpp": Path(args.llamacpp_dir), "trt": Path(args.trt_dir)}

    data, files, excl, notes = {}, [], {}, {}
    for ds, _ in DATASET_COLS:
        loaded, poison = {}, set()
        for key, _, _, _, kind, stem in CONFIGS:
            p = dirs[kind] / f"{stem}__{ds}.jsonl"
            if not p.exists():
                continue
            loaded[key] = load_results(p)
            files.append({"config": key, "dataset": ds, "path": str(p.relative_to(root))
                          if p.is_relative_to(root) else str(p), "sha256": sha256_file(p),
                          "records": len(loaded[key]["keys"]),
                          "errors": len(loaded[key]["err"])})
        for pp in sorted(dirs["llamacpp"].glob(f"config*__{ds}_poison.jsonl")):
            poison |= load_poison(pp)
        if not loaded:
            continue
        # Same idx set and same question/ground truth across configs.
        ref_key = next(iter(loaded))
        ref = loaded[ref_key]["keys"]
        for key, d in loaded.items():
            if set(d["keys"]) != set(ref):
                sys.exit(f"ERROR: {ds}: {key} has {len(d['keys'])} records, {ref_key} has "
                         f"{len(ref)} (or different idx). Is one run incomplete?")
            bad = sum(1 for i in ref if ref[i] != d["keys"][i])
            if bad:
                sys.exit(f"ERROR: {ds}: {bad} records differ in question/ground truth "
                         f"between {ref_key} and {key}; pairing on idx is not valid.")
        err_union = set().union(*(d["err"] for d in loaded.values()))
        extra_poison = poison - err_union
        drop = err_union | (poison & set(ref))
        keep = sorted(set(ref) - drop)
        excl[ds] = {"n_total": len(ref), "n_excluded": len(drop), "n_matched": len(keep),
                    "from_error_field": len(err_union),
                    "from_poison_files_only": len(extra_poison & set(ref)),
                    "errors_by_config": {k: len(d["err"]) for k, d in loaded.items()}}
        for key, d in loaded.items():
            missing = [i for i in keep if i not in d["scores"]]
            if missing:
                sys.exit(f"ERROR: {ds}/{key}: {len(missing)} matched records lack a score")
            data[(key, ds)] = np.array([d["scores"][i] for i in keep])
        notes[ds] = keep

    # Accuracy rows
    acc_rows = []
    for key, rt, vis, lang, _, _ in CONFIGS:
        row = {"key": key, "runtime": rt, "vision": vis, "language": lang}
        for ds, _ in DATASET_COLS:
            a = data.get((key, ds))
            row[ds] = None if a is None else a.mean() * 100
        acc_rows.append(row)

    # Paired deltas
    deltas = []
    for pi, (b, o, desc) in enumerate(PAIRS):
        for di, (ds, _) in enumerate(DATASET_COLS):
            if (b, ds) not in data or (o, ds) not in data:
                deltas.append({"base": b, "other": o, "desc": desc, "dataset": ds,
                               "pending": True})
                continue
            rng = np.random.default_rng([args.seed, pi, di])
            d = paired_delta(data[(b, ds)], data[(o, ds)], args.n_boot, rng)
            d.update({"base": b, "other": o, "desc": desc, "dataset": ds, "pending": False})
            deltas.append(d)

    lat = load_latency(args.latency_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- md ---
    lbl = dict(DATASET_COLS)
    md = ["# Cross-runtime comparison: llama.cpp vs TensorRT Edge-LLM", "",
          "Generated by `scripts/make_crossruntime_table.py`. Do not edit by hand.", "",
          "Matched records only: a record is dropped for every config if any config "
          "errored on it (or it is a known CV-Bench/TextVQA poison record), so each column "
          "uses one denominator. Accuracy in %. Do not average columns.", ""]
    hdr = ["Config", "Runtime", "Vision", "Language"] + [
        f"{lbl[ds]} (n={excl[ds]['n_matched']})" if ds in excl else lbl[ds]
        for ds, _ in DATASET_COLS]
    md += ["| " + " | ".join(hdr) + " |", "|---|---|---|---|" + "---:|" * len(DATASET_COLS)]
    for r in acc_rows:
        cells = [r["key"], r["runtime"], r["vision"], r["language"]]
        cells += ["pending" if r[ds] is None else f"{r[ds]:.2f}" for ds, _ in DATASET_COLS]
        md.append("| " + " | ".join(cells) + " |")
    md += ["", "Excluded records per dataset: " + ", ".join(
        f"{lbl[ds]} {excl[ds]['n_excluded']} of {excl[ds]['n_total']}" for ds in excl) + ".", ""]

    md += ["## Paired deltas (other minus base, points, 95% bootstrap CI)", "",
           f"{args.n_boot:,} resamples, seed {args.seed}. `*` = CI excludes 0 (unadjusted). "
           "`†` = BLINK, n = 124/143, report for completeness only.", ""]
    hdr = ["Base to other", "What it isolates"] + [lbl[ds] for ds, _ in DATASET_COLS]
    md += ["| " + " | ".join(hdr) + " |", "|---|---|" + "---|" * len(DATASET_COLS)]
    for b, o, desc in PAIRS:
        cells = [f"{b} to {o}", desc]
        for ds, _ in DATASET_COLS:
            d = next(x for x in deltas if x["base"] == b and x["other"] == o
                     and x["dataset"] == ds)
            cells.append("pending" if d["pending"] else fmt_delta(d, ds in SMALL))
        md.append("| " + " | ".join(cells) + " |")
    md += ["", "C2 to T16 changes both runtime and language precision (Q8_0 vs FP16). "
           "The clean statements are within a runtime (T16 to T8/T4 here, Table A/B for "
           "llama.cpp) and the matched bit-width pairs (C2 to T8, C3 to T4), which still "
           "differ in quantization method (weight-only Q8_0/Q4_K_M vs SmoothQuant W8A8 / "
           "AWQ W4A16).", ""]

    md += ["## Discordant pairs and McNemar p (MCQ datasets)", "",
           "| Base to other | Dataset | n | base right, other wrong | base wrong, other right | p |",
           "|---|---|---:|---:|---:|---:|"]
    for d in deltas:
        if not d["pending"] and "p_mcnemar" in d:
            md.append(f"| {d['base']} to {d['other']} | {lbl[d['dataset']]} | {d['n']} | "
                      f"{d['worse']} | {d['better']} | {d['p_mcnemar']:.3g} |")

    md += ["", "## Latency and memory (200 frozen records, 3 rounds, 15W)", "",
           "| Config | Runtime | Language | ScienceQA median ms | ScienceQA p95 | "
           "TextVQA median ms | TextVQA p95 | Peak RAM MB (timed) | Source |",
           "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for key, rt, _, lang, _, _ in CONFIGS:
        e = lat.get(key)
        if not e:
            md.append(f"| {key} | {rt} | {lang} | pending | pending | pending | pending | "
                      "pending | - |")
            continue
        s, t = e.get("scienceqa"), e.get("textvqa")
        f = lambda x, k: "-" if not x else f"{x[k]:,.0f}"
        ram = "-" if e["ram_peak"] is None else f"{e['ram_peak']:,}"
        src = Path(e["dir"]).relative_to(root) if Path(e["dir"]).is_relative_to(root) else e["dir"]
        md.append(f"| {key} | {rt} | {lang} | {f(s,'med')} | {f(s,'p95')} | {f(t,'med')} | "
                  f"{f(t,'p95')} | {ram} | `{src}` |")
    sp = []
    for k in ("T8", "T4", "T16"):
        if k in lat and "C2" in lat:
            for ds in ("scienceqa", "textvqa"):
                if ds in lat[k] and ds in lat["C2"]:
                    sp.append(f"{k} {lbl[ds]} {lat['C2'][ds]['med'] / lat[k][ds]['med']:.2f}x")
    if sp:
        md += ["", "Median speedup vs C2: " + ", ".join(sp) + "."]
    md += ["", "Wall-clock is not defined identically: llama.cpp is an HTTP round trip to "
           "llama-server, TensorRT is one in-process `LLM.generate()` call (image load, "
           "preprocess, encode, prefill, decode). The HTTP overhead is small next to the gap, "
           "but it is a caveat. Peak RAM is whole-system tegrastats (strict), not per process. "
           "Answers are 2 to 3 tokens, so latency is mostly vision encode plus prefill.", ""]
    (out / "crossruntime_table.md").write_text("\n".join(md))

    # --------------------------------------------------------------- csvs ---
    with open(out / "crossruntime_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "runtime", "vision", "language", "dataset", "accuracy_pct",
                    "n_matched", "n_excluded"])
        for r in acc_rows:
            for ds, _ in DATASET_COLS:
                if r[ds] is not None:
                    w.writerow([r["key"], r["runtime"], r["vision"], r["language"], ds,
                                f"{r[ds]:.4f}", excl[ds]["n_matched"], excl[ds]["n_excluded"]])
    with open(out / "crossruntime_deltas.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["base", "other", "dataset", "n", "base_acc_pct", "acc_pct", "delta_pts",
                    "ci_lo", "ci_hi", "base_right_other_wrong", "base_wrong_other_right",
                    "p_mcnemar", "status"])
        for d in deltas:
            if d["pending"]:
                w.writerow([d["base"], d["other"], d["dataset"]] + [""] * 9 + ["pending"])
            else:
                w.writerow([d["base"], d["other"], d["dataset"], d["n"], f"{d['base_acc']:.4f}",
                            f"{d['acc']:.4f}", f"{d['delta']:.4f}", f"{d['ci'][0]:.4f}",
                            f"{d['ci'][1]:.4f}", d.get("worse", ""), d.get("better", ""),
                            d.get("p_mcnemar", ""), "ok"])
    with open(out / "crossruntime_latency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "dataset", "valid", "wall_med_ms", "wall_p95_ms",
                    "peak_ram_mb_timed", "source_dir"])
        for key, *_ in CONFIGS:
            e = lat.get(key)
            if not e:
                continue
            for ds in ("scienceqa", "textvqa"):
                if ds in e:
                    w.writerow([key, ds, e[ds]["valid"], f"{e[ds]['med']:.3f}",
                                f"{e[ds]['p95']:.3f}", e["ram_peak"], e["dir"]])
    (out / "crossruntime_provenance.json").write_text(json.dumps({
        "script": Path(__file__).name, "script_sha256": sha256_file(__file__),
        "n_boot": args.n_boot, "seed": args.seed,
        "exclusion_rule": "union over loaded configs of error rows, plus llama.cpp poison files",
        "exclusions": excl, "input_files": files,
        "latency_dirs": args.latency_dir}, indent=2))

    print("\n".join(md))
    print(f"\nWrote outputs to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
