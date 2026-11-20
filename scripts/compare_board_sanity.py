#!/usr/bin/env python3
"""Board sanity check (Shared Setup Agreement): compare llama.cpp Config 2 run on Max's board
with Aeon's Config 2 results, record by record.

Usage: python3 scripts/compare_board_sanity.py [--new results_board_sanity] \
           [--ref results --ref results_rerun_oct4] [--config F16v_Q8l] [--out results_board_sanity/compare.md]

For each dataset and reference: records in both, accuracy on each side (errors count as wrong,
over all records, and over records answered on both sides), identical-prediction rate, the
records whose prediction differs, error-set agreement (poison images), and median tok/s.
Reference-vs-reference rows (Aeon's own rerun) are included as the run-to-run noise floor.
"""
import argparse
import json
import statistics
from pathlib import Path

DATASETS = ["blink_depth", "blink_spatial", "cv_bench", "scienceqa", "textvqa"]


def load(dirpath, cfg, ds):
    """Return {idx: row} from the main file plus {idx: row} for errors (main or _poison file)."""
    rows, errs = {}, {}
    main = Path(dirpath) / f"config2_{cfg}__{ds}.jsonl"
    if not main.exists():
        return None, None
    for line in main.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["idx"]] = r
            if r.get("error"):
                errs[r["idx"]] = r
    poison = Path(dirpath) / f"config2_{cfg}__{ds}_poison.jsonl"
    if poison.exists():
        for line in poison.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                errs[r["idx"]] = r
    return rows, errs


def answered(r):
    return r is not None and not r.get("error") and r.get("predicted") is not None


def compare(a_name, a, a_err, b_name, b, b_err):
    common = sorted(set(a) & set(b))
    both_ans = [i for i in common if answered(a[i]) and answered(b[i])]
    # MCQ rows carry a boolean `correct`; TextVQA rows carry a fractional VQA `score`.
    val = lambda r: float(r["score"]) if r.get("score") is not None else float(bool(r.get("correct")))
    acc = lambda rows, ids: sum(val(rows[i]) for i in ids) / len(ids) if ids else float("nan")
    # Only compare error sets over records present on both sides (a run may still be in progress).
    a_err = {i: v for i, v in a_err.items() if i in b}
    b_err = {i: v for i, v in b_err.items() if i in a}
    same_pred = [i for i in both_ans if str(a[i]["predicted"]).strip() == str(b[i]["predicted"]).strip()]
    diff = [i for i in both_ans if i not in set(same_pred)]
    flips = sum(val(a[i]) != val(b[i]) for i in both_ans)
    med = lambda rows, ids: statistics.median([rows[i]["tok_s"] for i in ids if rows[i].get("tok_s")]) if ids else float("nan")
    return {
        "a": a_name, "b": b_name, "n_a": len(a), "n_b": len(b), "common": len(common),
        "both_answered": len(both_ans),
        "acc_a_all": acc(a, common), "acc_b_all": acc(b, common),
        "acc_a_both": acc(a, both_ans), "acc_b_both": acc(b, both_ans),
        "same_pred": len(same_pred), "same_pct": 100 * len(same_pred) / len(both_ans) if both_ans else float("nan"),
        "correct_flips": flips,
        "err_a": len(a_err), "err_b": len(b_err), "err_same_set": set(a_err) == set(b_err),
        "err_only_a": sorted(set(a_err) - set(b_err))[:10], "err_only_b": sorted(set(b_err) - set(a_err))[:10],
        "toks_a": med(a, both_ans), "toks_b": med(b, both_ans),
        "diff_examples": [(i, a[i]["predicted"], b[i]["predicted"], a[i].get("gt_answer")) for i in diff[:8]],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new", default="results_board_sanity")
    p.add_argument("--ref", action="append", default=None)
    p.add_argument("--config", default="F16v_Q8l")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    refs = args.ref or ["results", "results_rerun_oct4"]

    lines = ["# Board sanity check: llama.cpp Config 2 on Max's board vs Aeon's", "",
             "Accuracy counts errors as wrong (MCQ: exact match; TextVQA: mean VQA score). "
             "'both' = records answered on both sides; error sets compared over common records. "
             "tok/s = median over records answered on both sides.", "",
             "| dataset | A | B | common | both answered | acc A (all) | acc B (all) | identical preds | correct flips | errors A / B, same set | tok/s A / B |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    details = []
    for ds in DATASETS:
        new, new_err = load(args.new, args.config, ds)
        if new is None:
            lines.append(f"| {ds} | {args.new} | (not run yet) | | | | | | | | |")
            continue
        loaded = [(r, *load(r, args.config, ds)) for r in refs]
        loaded = [x for x in loaded if x[1] is not None]
        pairs = [(args.new, new, new_err, r, rr, re) for r, rr, re in loaded]
        if len(loaded) >= 2:  # Aeon vs Aeon: run-to-run noise floor on his board
            pairs.append((loaded[0][0], loaded[0][1], loaded[0][2], loaded[1][0], loaded[1][1], loaded[1][2]))
        for pa in pairs:
            c = compare(*pa)
            lines.append(
                f"| {ds} | {c['a']} | {c['b']} | {c['common']} | {c['both_answered']} | "
                f"{100*c['acc_a_all']:.2f}% | {100*c['acc_b_all']:.2f}% | {c['same_pred']} ({c['same_pct']:.2f}%) | "
                f"{c['correct_flips']} | {c['err_a']} / {c['err_b']}, {'yes' if c['err_same_set'] else 'NO'} | "
                f"{c['toks_a']:.2f} / {c['toks_b']:.2f} |")
            if c["diff_examples"] or not c["err_same_set"]:
                details.append(f"- **{ds}, {c['a']} vs {c['b']}**: differing predictions (idx, A, B, gt) "
                               f"{c['diff_examples']}; errors only in A {c['err_only_a']}, only in B {c['err_only_b']}")
    if details:
        lines += ["", "## Differences (first few per pair)", ""] + details
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
