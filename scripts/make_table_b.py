#!/usr/bin/env python3
"""Table B generator: vision-encoder sensitivity matrix with paired CIs.

Reads per-record result JSONL files (read-only; never modifies them) and writes
  table_b.md, table_b.csv, table_b_deltas.csv, table_b_provenance.json

Inputs, found by filename in one or more --results-dir folders (recursive):
    config{N}_{short}__{dataset}.jsonl
Datasets: blink_depth, blink_spatial, cv_bench, textvqa (or textvqa_1500),
scienceqa. Missing datasets/configs are shown as "-", so this script can be run
now and re-run unchanged when the new results arrive.

Record handling (both runner formats):
  * Old Config 2/5/8 files have only idx + correct (no rec_no, no raw text).
    New p2.1 files also have rec_no, score, prediction_raw, ...
  * Per-record score = `score` if present and not null, else float(correct).
    TextVQA scores are fractional (0, 0.3, 0.6, 0.9, 1.0), so it is not binary.
  * Pairing is on `idx` only.
  * Rows with a non-empty `error` (e.g. the 118 HTTP 400 CV-Bench records) are
    EXCLUDED from accuracy, not counted as wrong. They are reported as n_err.
    A config's accuracy uses its own valid rows; every delta vs the baseline
    uses only rows valid in BOTH configs.
  * For cv_bench the error set must be identical across configs (the exclusion
    rule assumes it is not encoder-dependent); the script aborts if it is not,
    unless --allow-error-set-mismatch is given.

Statistics: paired percentile bootstrap (default 10,000 resamples, seed 249) on
the per-record score difference. For binary datasets an exact two-sided McNemar
p-value is also given. Intervals are 95% and NOT adjusted for multiple
comparisons.
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

# Precision order from the plan (Table B).
CONFIG_ORDER = [2, 5, 9, 10, 8, 11]
CONFIG_INFO = {
    2: ("F16", "Q8_0"), 5: ("Q8_0", "Q8_0"), 9: ("Q6_K", "Q8_0"),
    10: ("Q5_K_M", "Q8_0"), 8: ("Q4_0", "Q8_0"), 11: ("Q3_K_M", "Q8_0 (optional)"),
}
DATASET_COLS = [
    ("blink_depth", "BLINK depth"),
    ("blink_spatial", "BLINK spatial"),
    ("cv_bench", "CV-Bench"),
    ("textvqa", "TextVQA VQA acc."),
    ("scienceqa", "ScienceQA IMG acc."),
]
FILE_RE = re.compile(r"^config(\d+)_(.+?)__(.+)\.jsonl$")


# ----------------------------------------------------------------- loading --
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(dirs, exclude_dirs):
    """Return {(config_num, dataset): Path}. Aborts on duplicates."""
    found = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*.jsonl")):
            if any(part in exclude_dirs for part in p.parts):
                continue
            m = FILE_RE.match(p.name)
            if not m or p.name.endswith("_poison.jsonl"):
                continue
            key = (int(m.group(1)), m.group(3))
            if key in found:
                sys.exit(f"ERROR: two files for config {key[0]} / {key[1]}:\n"
                         f"  {found[key]}\n  {p}")
            found[key] = p
    return found


def record_score(r):
    s = r.get("score")
    if s is not None:
        return float(s)
    c = r.get("correct")
    if c is None:
        raise ValueError(f"record idx={r.get('idx')} has neither score nor correct")
    return 1.0 if c else 0.0


def load_results(path):
    """-> dict with valid {idx: score}, err_idx set, extras. Read-only."""
    valid, errs, meta = {}, set(), {}
    tok_s, mem = [], []
    seen = set()
    n_lines = 0
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            n_lines += 1
            idx = r["idx"]
            if idx in seen:
                sys.exit(f"ERROR: duplicate idx {idx} in {path} (line {ln})")
            seen.add(idx)
            if r.get("error_type") == "ServerUnavailable":
                sys.exit(f"ERROR: {path} contains never-attempted placeholder "
                         f"rows (idx {idx}); the run is not complete")
            if r.get("error"):
                errs.add(idx)
                continue
            valid[idx] = record_score(r)
            meta[idx] = (r.get("gt_answer"), r.get("question"))
            if r.get("tok_s"):
                tok_s.append(float(r["tok_s"]))
            if r.get("mem_avail_mb") is not None:
                mem.append(float(r["mem_avail_mb"]))
    return {"valid": valid, "err": errs, "meta": meta, "n_lines": n_lines,
            "tok_s": tok_s, "mem": mem}


# -------------------------------------------------------------- statistics --
def mcnemar_exact(b, c):
    """Two-sided exact McNemar p from discordant counts b, c."""
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


def paired_delta(base, other, n_boot, rng):
    common = sorted(set(base["valid"]) & set(other["valid"]))
    if not common:
        return None
    a = np.array([base["valid"][i] for i in common])
    b = np.array([other["valid"][i] for i in common])
    diff = b - a
    out = {"n_paired": len(common), "delta": float(diff.mean()) * 100,
           "base_acc": float(a.mean()) * 100, "acc": float(b.mean()) * 100}
    if np.allclose(diff, 0):
        out["ci"] = (0.0, 0.0)
    else:
        lo, hi = paired_bootstrap(diff, n_boot, rng)
        out["ci"] = (float(lo) * 100, float(hi) * 100)
    binary = set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}
    if binary:
        worse = int(((a == 1) & (b == 0)).sum())   # base right, other wrong
        better = int(((a == 0) & (b == 1)).sum())
        out["worse"], out["better"] = worse, better
        out["p_mcnemar"] = mcnemar_exact(worse, better)
    return out


# ---------------------------------------------------------------- sanity ---
def check_alignment(ds, loaded):
    """Same idx must carry the same gt_answer/question across configs."""
    ref_cfg, ref = None, None
    for cfg, d in loaded.items():
        if ref is None:
            ref_cfg, ref = cfg, d
            continue
        bad = 0
        for i in set(ref["meta"]) & set(d["meta"]):
            if ref["meta"][i] != d["meta"][i]:
                bad += 1
        if bad:
            sys.exit(f"ERROR: {ds}: {bad} records have different question/"
                     f"gt_answer between config {ref_cfg} and {cfg} for the "
                     f"same idx. Pairing on idx is not valid.")


def check_error_sets(ds, loaded, allow):
    sets = {cfg: d["err"] for cfg, d in loaded.items()}
    ref_cfg = next(iter(sets))
    for cfg, s in sets.items():
        if s != sets[ref_cfg]:
            msg = (f"{ds}: error record set differs between config {ref_cfg} "
                   f"({len(sets[ref_cfg])}) and config {cfg} ({len(s)}).")
            if ds == "cv_bench" and not allow:
                sys.exit("ERROR: " + msg + " Exclusion rule assumes encoder-"
                         "independent failures; inspect before continuing.")
            print("WARNING: " + msg, file=sys.stderr)


def check_summary(cfg, ds, d, summary_dir):
    """Cross-check against the runner's own summary.json if available."""
    if not summary_dir:
        return None
    hits = sorted(Path(summary_dir).glob(f"config{cfg}_*summary.json"))
    if not hits:
        return None
    s = json.loads(hits[0].read_text())
    sd = s.get("datasets", {}).get(ds)
    if not sd:
        return None
    msgs = []
    if sd.get("errors") is not None and sd["errors"] != len(d["err"]):
        msgs.append(f"errors {len(d['err'])} vs summary {sd['errors']}")
    if sd.get("score_sum") is not None:
        mine = sum(d["valid"].values())
        if abs(mine - sd["score_sum"]) > 1e-6:
            msgs.append(f"score_sum {mine} vs summary {sd['score_sum']}")
    return msgs or ["ok"]


# ------------------------------------------------------------------ main ---
def fmt_cell(acc, n, n_err):
    if acc is None:
        return "-"
    return f"{acc:.2f}" + (f" (n={n})" if n_err else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", action="append", required=True,
                    help="folder with result JSONL (repeatable, recursive)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--baseline", type=int, default=2)
    ap.add_argument("--textvqa-set", choices=["auto", "textvqa", "textvqa_1500"],
                    default="auto", help="auto prefers the full set when present")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=249)
    ap.add_argument("--summary-dir", help="folder with *summary.json for cross-check")
    ap.add_argument("--sha-file", help="sha256sum output to verify input files against")
    ap.add_argument("--exclude-dir", action="append", default=["smoke"],
                    help="directory names to skip (default: smoke)")
    ap.add_argument("--allow-error-set-mismatch", action="store_true")
    args = ap.parse_args()

    found = discover(args.results_dir, set(args.exclude_dir))
    rng = np.random.default_rng(args.seed)

    expected = {}
    if args.sha_file:
        for line in Path(args.sha_file).read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                expected[Path(parts[1]).name] = parts[0]

    # Resolve which TextVQA folder name to use.
    tv_name = "textvqa"
    present_tv = {ds for (_, ds) in found if ds in ("textvqa", "textvqa_1500")}
    if args.textvqa_set != "auto":
        tv_name = args.textvqa_set
    elif "textvqa" not in present_tv and "textvqa_1500" in present_tv:
        tv_name = "textvqa_1500"
    col_ds = [(tv_name if k == "textvqa" else k, label) for k, label in DATASET_COLS]

    # Load everything that exists.
    data, prov_files, summary_checks = {}, [], {}
    for cfg in CONFIG_ORDER:
        for ds, _ in col_ds:
            p = found.get((cfg, ds))
            if not p:
                continue
            digest = sha256_file(p)
            ok = None
            if expected:
                ok = expected.get(p.name) == digest if p.name in expected else None
                if ok is False:
                    sys.exit(f"ERROR: sha256 mismatch for {p.name}")
            data[(cfg, ds)] = load_results(p)
            prov_files.append({"config": cfg, "dataset": ds, "file": p.name,
                               "path": str(p), "sha256": digest,
                               "sha_verified": ok, "lines": data[(cfg, ds)]["n_lines"]})
            sc = check_summary(cfg, ds, data[(cfg, ds)], args.summary_dir)
            if sc:
                summary_checks[f"config{cfg}/{ds}"] = sc
                if sc != ["ok"]:
                    print(f"WARNING: summary cross-check config{cfg}/{ds}: {sc}",
                          file=sys.stderr)

    # Per-dataset sanity checks across the configs that exist.
    for ds, _ in col_ds:
        loaded = {c: data[(c, ds)] for c in CONFIG_ORDER if (c, ds) in data}
        if len(loaded) >= 2:
            check_alignment(ds, loaded)
            check_error_sets(ds, loaded, args.allow_error_set_mismatch)

    # Accuracy table + deltas.
    base_cfg = args.baseline
    rows, delta_rows = [], []
    for cfg in CONFIG_ORDER:
        v, l = CONFIG_INFO[cfg]
        row = {"config": cfg, "vision": v, "language": l}
        for ds, label in col_ds:
            d = data.get((cfg, ds))
            if d is None or not d["valid"]:
                row[label] = None
                continue
            acc = float(np.mean(list(d["valid"].values()))) * 100
            row[label] = {"acc": acc, "n": len(d["valid"]), "n_err": len(d["err"])}
            if cfg != base_cfg and (base_cfg, ds) in data:
                pd = paired_delta(data[(base_cfg, ds)], d, args.n_boot, rng)
                if pd:
                    pd.update({"config": cfg, "dataset": ds, "label": label})
                    delta_rows.append(pd)
        allm = [x for (c, _), d in data.items() if c == cfg for x in d["mem"]]
        allt = [x for (c, _), d in data.items() if c == cfg for x in d["tok_s"]]
        row["min_mem_avail_mb"] = min(allm) if allm else None
        row["mean_tok_s"] = float(np.mean(allt)) if allt else None
        rows.append(row)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dmap = {(r["config"], r["dataset"]): r for r in delta_rows}

    # ---- markdown
    md = ["# Table B: vision-encoder sensitivity (language decoder fixed Q8_0)", ""]
    md += ["Accuracy in %. Rows with request errors are excluded from accuracy "
           "(n shown when any were excluded). Do not average columns.", ""]
    hdr = ["Config", "Vision", "Language"] + [lb for _, lb in col_ds]
    md += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        cells = [str(r["config"]), r["vision"], r["language"]]
        for _, label in col_ds:
            c = r[label]
            cells.append("-" if c is None else fmt_cell(c["acc"], c["n"], c["n_err"]))
        md.append("| " + " | ".join(cells) + " |")
    md += ["", f"## Change from Config {base_cfg} (paired, 95% bootstrap CI, points)", ""]
    hdr2 = ["Config", "Vision"] + [lb for _, lb in col_ds]
    md += ["| " + " | ".join(hdr2) + " |", "|" + "---|" * len(hdr2)]
    for r in rows:
        if r["config"] == base_cfg:
            continue
        cells = [str(r["config"]), r["vision"]]
        for ds, label in col_ds:
            d = dmap.get((r["config"], ds))
            if not d:
                cells.append("-")
                continue
            lo, hi = d["ci"]
            star = "*" if (lo > 0 or hi < 0) else ""
            cells.append(f"{d["delta"]+0.0:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}")
        md.append("| " + " | ".join(cells) + " |")
    md += ["", "`*` = 95% CI excludes 0 (unadjusted for multiple comparisons). "
           "Deltas use only records valid in both configs.", ""]
    md += ["## Discordant pairs and McNemar p (binary datasets)", "",
           "| Config | Dataset | n paired | base right / other wrong | base wrong / other right | p |",
           "|---|---|---:|---:|---:|---:|"]
    for d in delta_rows:
        if "p_mcnemar" in d:
            md.append(f"| {d['config']} | {d['label']} | {d['n_paired']} | "
                      f"{d['worse']} | {d['better']} | {d['p_mcnemar']:.3g} |")
    md += ["", "## Edge measurements from JSONL (proxies)", "",
           "| Config | Mean tok/s | Lowest mem_avail_mb |", "|---|---:|---:|"]
    for r in rows:
        t = "-" if r["mean_tok_s"] is None else f"{r['mean_tok_s']:.1f}"
        m = "-" if r["min_mem_avail_mb"] is None else f"{r['min_mem_avail_mb']:.0f}"
        md.append(f"| {r['config']} | {t} | {m} |")
    md += ["", "Lowest available memory is a proxy for peak usage, not a direct "
           "peak-memory measurement. Tok/s is decode speed, not end-to-end latency.", ""]
    (out / "table_b.md").write_text("\n".join(md))

    # ---- csvs
    with open(out / "table_b.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "vision", "language", "dataset", "accuracy_pct",
                    "n_valid", "n_error"])
        for r in rows:
            for ds, label in col_ds:
                c = r[label]
                if c:
                    w.writerow([r["config"], r["vision"], r["language"], ds,
                                f"{c['acc']:.4f}", c["n"], c["n_err"]])
    with open(out / "table_b_deltas.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "baseline", "dataset", "n_paired", "base_acc_pct",
                    "acc_pct", "delta_pts", "ci_lo", "ci_hi", "base_right_other_wrong",
                    "base_wrong_other_right", "p_mcnemar"])
        for d in delta_rows:
            w.writerow([d["config"], base_cfg, d["dataset"], d["n_paired"],
                        f"{d['base_acc']:.4f}", f"{d['acc']:.4f}", f"{d['delta']:.4f}",
                        f"{d['ci'][0]:.4f}", f"{d['ci'][1]:.4f}", d.get("worse", ""),
                        d.get("better", ""), d.get("p_mcnemar", "")])

    (out / "table_b_provenance.json").write_text(json.dumps({
        "script": Path(__file__).name,
        "script_sha256": sha256_file(__file__),
        "baseline_config": base_cfg, "n_boot": args.n_boot, "seed": args.seed,
        "textvqa_folder_used": tv_name,
        "error_rule": "rows with non-empty error excluded from accuracy",
        "summary_crosschecks": summary_checks,
        "input_files": prov_files}, indent=2))

    print("\n".join(md))
    print(f"\nWrote outputs to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
