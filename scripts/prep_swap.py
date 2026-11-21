#!/usr/bin/env python3
"""Build the answer-order swap set for the depth follow-up (README Current Status, BLINK depth item).

Writes <out>/blink_depth/data.jsonl (all 124 records) and <out>/cv_bench/data.jsonl (the 600 Depth
records only), each a copy of the frozen benchmark with the two choices in reverse order and the
ground-truth letter flipped. idx, question text and image paths are unchanged, so the results pair
with the original runs by idx. The folder names stay `blink_depth` and `cv_bench`, so
`run_experiment.py --data-dir <out>` uses the same legacy prompt and parser as the original runs.

Question: is Q4_K_M language's depth gain (BLINK +12.10, CV-Bench Depth +2.33 vs Q8_0) real depth
perception, or a different preference for answer position? With the order reversed, a position
preference turns into the opposite error, while real perception gives the same answers.

Usage: python3 scripts/prep_swap.py [--src benchmarks] [--out benchmarks_swap]
"""
import argparse
import json
import re
from pathlib import Path


def swap(rec):
    assert len(rec["choices"]) == 2, rec["idx"]
    m = re.fullmatch(r"\(([AB])\)", rec["answer"].strip())
    assert m, (rec["idx"], rec["answer"])
    out = dict(rec)
    out["choices"] = rec["choices"][::-1]
    out["answer"] = "(B)" if m.group(1) == "A" else "(A)"
    out["swapped"] = True
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="benchmarks")
    ap.add_argument("--out", default="benchmarks_swap")
    args = ap.parse_args()
    for ds, keep in [("blink_depth", lambda r: True), ("cv_bench", lambda r: r["task"] == "Depth")]:
        rows = [json.loads(l) for l in open(Path(args.src) / ds / "data.jsonl") if l.strip()]
        rows = [swap(r) for r in rows if keep(r)]
        d = Path(args.out) / ds
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "data.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_a = sum(r["answer"] == "(A)" for r in rows)
        print(f"{ds}: {len(rows)} records -> {d / 'data.jsonl'} (gt A {n_a}, B {len(rows) - n_a})")


if __name__ == "__main__":
    main()
