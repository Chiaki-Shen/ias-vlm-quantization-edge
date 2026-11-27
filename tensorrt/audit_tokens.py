#!/usr/bin/env python3
"""TensorRT audit, part 2: does each runtime feed the language model the same number of tokens?

For every record, run one request with max_tokens=1 and record how many tokens were prefilled:
  --runtime llamacpp   llama-server (Config 2, same build and flags as the accuracy runs):
                       timings.prompt_n = text + image tokens
  --runtime trt        Edge-LLM (any engine; template, tokenizer and visual engine are shared):
                       MultimodalMetrics.total_image_tokens and LLMPrefillMetrics.computed_tokens deltas
Then `--compare` (anywhere) joins the two files. Equal totals on a record mean the chat template, the
tokenization and the image-token count all match for it; the image-token column says which part differs
if they do not.

Board usage (from the repo root):
  # llama.cpp, inside cmpelkk/jetson-llm:latest with /opt/llama-cpp-new mounted (as the accuracy runs)
  python3 tensorrt/audit_tokens.py --runtime llamacpp --datasets scienceqa,textvqa --textvqa-limit 500
  # Edge-LLM, inside jetson-llm:v1 via tensorrt/jetson/run_in_container.sh
  python3 tensorrt/audit_tokens.py --runtime trt --engine cr2b-int8sq --datasets scienceqa,textvqa --textvqa-limit 500
  python3 tensorrt/audit_tokens.py --compare
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tensorrt"))
import run_experiment as rx  # noqa: E402

OUT = ROOT / "results" / "audit"


def records(ds: str, data_dir: Path, limit: int | None):
    spec = rx.DATASET_SPECS[ds]
    prompt_fn = rx.FORMATS[spec["format"]]["prompts"][spec["prompt"]]
    recs = rx.load_jsonl(data_dir / ds / "data.jsonl")
    if limit:
        recs = recs[:limit]
    for r in recs:
        yield r, rx.resolve_image_paths(r, data_dir / ds), prompt_fn(r)


def run_llamacpp(args, ds: str, out_path: Path) -> None:
    config = rx.get_config(2)
    rx.kill_stale_servers()
    proc = rx.start_server(config)
    if not rx.wait_for_server():
        sys.exit("llama-server did not start")
    n = 0
    with open(out_path, "w") as f:
        for r, imgs, prompt in records(ds, args.data_dir, args.textvqa_limit if ds == "textvqa" else None):
            for attempt in range(2):
                resp = rx.call_server(imgs, prompt, 1)
                t = resp.get("timings") if isinstance(resp, dict) else None
                if t and t.get("prompt_n") is not None:
                    break
                rx.stop_server(proc)            # e.g. the KV-cache retry case: restart once and retry
                proc = rx.start_server(config)
                rx.wait_for_server()
            f.write(json.dumps({"idx": r["idx"], "prompt_n": (t or {}).get("prompt_n"),
                                "cache_n": (t or {}).get("cache_n"),
                                "error": None if t else str(resp.get("error"))}) + "\n")
            f.flush()
            n += 1
            if n % 200 == 0:
                print(f"  {ds} {n}  {rx.mem_str()}", flush=True)
    rx.stop_server(proc)


def run_trt(args, ds: str, out_path: Path) -> None:
    from run_trt_experiment import EdgeLLMBackend
    backend = EdgeLLMBackend(args.workspace / args.engine / "engines")
    backend._llm._rt.set_profiling_enabled(True)   # the runtime's token metrics are only collected with profiling on
    rt = backend._llm._runtime
    n = 0
    with open(out_path, "w") as f:
        for r, imgs, prompt in records(ds, args.data_dir, args.textvqa_limit if ds == "textvqa" else None):
            mm0, pf0 = rt.get_multimodal_metrics().total_image_tokens, rt.get_prefill_metrics().computed_tokens
            err = None
            try:
                backend.generate(imgs, prompt, 1)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            mm1, pf1 = rt.get_multimodal_metrics().total_image_tokens, rt.get_prefill_metrics().computed_tokens
            f.write(json.dumps({"idx": r["idx"], "image_tokens": mm1 - mm0, "prefill_tokens": pf1 - pf0,
                                "error": err}) + "\n")
            f.flush()
            n += 1
            if n == 1 and not err and (mm1 - mm0 <= 0 or pf1 - pf0 <= 0):
                sys.exit(f"runtime token metrics did not move ({mm1 - mm0}, {pf1 - pf0}); refusing to write zeros")
            if n % 200 == 0:
                print(f"  {ds} {n}", flush=True)


PATCH = 32                      # Qwen3-VL: 16-px patches merged 2x2 -> one image token per 32x32 px
MIN_PX, MAX_PX = 4 * PATCH * PATCH, 1024 * PATCH * PATCH


def grid_tokens(w: int, h: int, rounder) -> int | None:
    """Image tokens from the smart_resize rounding step; None when a min/max pixel bound would rescale."""
    hb, wb = rounder(h / PATCH) * PATCH, rounder(w / PATCH) * PATCH
    if not (MIN_PX <= hb * wb <= MAX_PX) or hb == 0 or wb == 0:
        return None
    return (hb // PATCH) * (wb // PATCH)


def half_up(x: float) -> int:
    return math.floor(x + 0.5)


def compare(datasets, data_dir: Path) -> int:
    from PIL import Image
    lines = ["# TensorRT audit, part 2: tokens prefilled per record (llama.cpp vs Edge-LLM)", "",
             "Generated by `tensorrt/audit_tokens.py --compare`. llama.cpp total = `timings.prompt_n + timings.cache_n` "
             "(Config 2; cache_n = prompt-prefix tokens reused from the previous request). Edge-LLM total = prefill "
             "`computed_tokens`, image tokens = `total_image_tokens` (per-request deltas). Text tokens are identical "
             "(`results/audit/template.md`), so a total difference is an image-token difference.", "",
             "Image-token prediction from the image size: smart_resize rounds each side to a multiple of 32 px. "
             "'half-even' = Python `round()` (the HF reference), 'half-up' = `floor(x + 0.5)`. They differ only when "
             "a side is an odd multiple of 16 px.", "",
             "| dataset | records | totals equal | total diff (llama.cpp - Edge-LLM): records | Edge-LLM image tokens = half-even rule | llama.cpp image tokens = half-up rule | records with a .5 patch boundary | errors |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    sizes = collections.Counter()
    for ds in datasets:
        recs = {r["idx"]: r for r in rx.load_jsonl(data_dir / ds / "data.jsonl")}
        a = {r["idx"]: r for r in rx.load_jsonl(OUT / f"tokens_llamacpp_{ds}.jsonl")}
        b = {r["idx"]: r for r in rx.load_jsonl(OUT / f"tokens_trt_{ds}.jsonl")}
        ids = sorted(set(a) & set(b))
        ok = [i for i in ids if not a[i].get("error") and not b[i].get("error") and a[i].get("prompt_n") is not None]
        total = {i: a[i]["prompt_n"] + (a[i].get("cache_n") or 0) for i in ok}
        diffs = collections.Counter(total[i] - b[i]["prefill_tokens"] for i in ok)
        eq = diffs.get(0, 0)
        n_pred = edge_ok = llama_ok = half = 0
        for i in ok:
            w, h = Image.open(rx.resolve_image_paths(recs[i], data_dir / ds)[0]).size
            even, up = grid_tokens(w, h, round), grid_tokens(w, h, half_up)
            if even is None or up is None:
                continue
            n_pred += 1
            edge_ok += b[i]["image_tokens"] == even
            llama_ok += b[i]["image_tokens"] + total[i] - b[i]["prefill_tokens"] == up
            if even != up:
                half += 1
                sizes[(ds, w, h, up - even)] += 1
        lines.append(f"| {ds} | {len(ok)} | {eq} ({100 * eq / len(ok):.1f}%) | {dict(sorted(diffs.items()))} | "
                     f"{edge_ok} of {n_pred} | {llama_ok} of {n_pred} | {half} | {len(ids) - len(ok)} |")
    if sizes:
        lines += ["", "Image sizes with a .5 patch boundary (dataset, width, height, extra llama.cpp image tokens): records",
                  ""] + [f"- {k}: {v}" for k, v in sizes.most_common(15)]
    text = "\n".join(lines) + "\n"
    print(text)
    (OUT / "tokens.md").write_text(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime", choices=["llamacpp", "trt"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--engine", default="cr2b-int8sq", help="Edge-LLM engine dir under --workspace")
    ap.add_argument("--workspace", type=Path, default=Path.home() / "tensorrt-edgellm-workspace")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "benchmarks")
    ap.add_argument("--datasets", default="scienceqa,textvqa")
    ap.add_argument("--textvqa-limit", type=int, default=500)
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    if args.compare:
        return compare(datasets, args.data_dir)
    if not args.runtime:
        ap.error("--runtime or --compare is required")
    for ds in datasets:
        t0 = time.time()
        out_path = OUT / f"tokens_{'llamacpp' if args.runtime == 'llamacpp' else 'trt'}_{ds}.jsonl"
        (run_llamacpp if args.runtime == "llamacpp" else run_trt)(args, ds, out_path)
        print(f"{ds}: wrote {out_path} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
