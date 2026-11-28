#!/usr/bin/env python3
"""TensorRT audit, part 2b: does the image-token rounding difference change accuracy?

Records where llama.cpp gets more image tokens (results/audit/tokens.md) vs the rest: the paired T16 - C2 difference
should be the same if the rounding does not matter. Usage: python3 tensorrt/audit_rounding_effect.py"""
import json

import numpy as np

tok_l = {json.loads(l)["idx"]: json.loads(l) for l in open("results/audit/tokens_llamacpp_scienceqa.jsonl")}
tok_t = {json.loads(l)["idx"]: json.loads(l) for l in open("results/audit/tokens_trt_scienceqa.jsonl")}
affected = {i for i in tok_t if tok_l[i]["prompt_n"] + (tok_l[i].get("cache_n") or 0) != tok_t[i]["prefill_tokens"]}


def load(p):
    return {json.loads(l)["idx"]: float(json.loads(l)["score"] if json.loads(l).get("score") is not None
                                          else bool(json.loads(l)["correct"])) for l in open(p)}


c2 = load("results/config2_F16v_Q8l__scienceqa.jsonl")
t16 = load("results_trt/configT16_TRT_F16v_F16l__scienceqa.jsonl")
rng = np.random.default_rng(249)
for name, ids in [("affected (llama.cpp gets more image tokens)", sorted(affected)),
                  ("unaffected", sorted(set(c2) - affected))]:
    d = np.array([t16[i] - c2[i] for i in ids])
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{name}: n={len(ids)}  C2 {100 * np.mean([c2[i] for i in ids]):.2f}  T16 {100 * np.mean([t16[i] for i in ids]):.2f}"
          f"  T16 - C2 {100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]")
