#!/usr/bin/env python3
"""Latency breakdown (vision encode / LLM prefill / decode / overhead) for llama.cpp.

Reads a bench_latency.py run folder that was recorded with
  --server-logs --server-extra="--parallel 1 --cache-ram 0 -lv 4"
and splits every timed request into:

  encode   = vision encoder time: "encoding mtmd batch" -> first "decoding image batch"
             (the encoder copies its output to the host, so this interval is synchronous)
  prefill  = prompt_ms - encode   (LLM prefill of text + image tokens)
  decode   = predicted_ms         (token generation)
  overhead = wall_ms - prompt_ms - predicted_ms  (HTTP, base64, image preprocessing)

llama-server's own "image decoded (batch i/n) in X ms" number is NOT used: CUDA work is
queued asynchronously, so most of the image-token prefill cost lands after that timer.

prompt_ms (llama-server timings) starts at launch_slot, so it already contains the encode;
the split above never double counts.

Usage (Mac or container, standard library only):
  python scripts/latency_breakdown.py results_latency/breakdown
Writes <dir>/breakdown_requests.csv, <dir>/breakdown_summary.csv and <dir>/breakdown.md
"""
import argparse
import csv
import json
import random
import re
import statistics
import sys
from pathlib import Path

ORDER = [2, 5, 9, 10, 8, 11]
VISION = {2: "F16", 5: "Q8_0", 9: "Q6_K", 10: "Q5_K_M", 8: "Q4_0", 11: "Q3_K_M"}
COMPONENTS = ("encode", "prefill", "decode", "overhead")

TS = re.compile(r"^(\d+)\.(\d\d)\.(\d{3})\.(\d{3}) ")
LAUNCH = re.compile(r"launch_slot_: .*task (\d+) \|")
MTMD = re.compile(r"process_mtmd: .*task (\d+) \| encoding mtmd batch")
DEC_IMG = re.compile(r"decoding image batch (\d+)/(\d+), n_tokens_batch = (\d+)")
SAMPLER = re.compile(r"init_sampler: .*task (\d+) \|.*tokens: text = (\d+), total = (\d+)")
PROMPT = re.compile(r"print_timing: .*task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
EVAL = re.compile(r"print_timing: .*task (\d+) \|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens")


def ts_seconds(line):
    """llama.cpp log stamp is minutes.seconds.milliseconds.microseconds since start."""
    m = TS.match(line)
    if not m:
        return None
    mi, s, ms, us = map(int, m.groups())
    return mi * 60 + s + ms / 1e3 + us / 1e6


def parse_log(path):
    """Completed tasks in log order: dicts with task, encode_ms, image/text tokens, prompt/eval ms."""
    tasks, cur, out = {}, None, []
    for line in open(path, errors="ignore"):
        t = ts_seconds(line)
        if (m := LAUNCH.search(line)):
            cur = int(m.group(1))
            tasks[cur] = {"task": cur, "t_launch": t, "t_mtmd": None, "t_dec": None, "n_img_batches": 0}
        elif (m := MTMD.search(line)):
            tid = int(m.group(1))
            if tid in tasks:
                tasks[tid]["t_mtmd"] = t
                cur = tid
        elif (m := DEC_IMG.search(line)):
            # no task id on this line; with --parallel 1 it belongs to the current task
            if cur in tasks:
                d = tasks[cur]
                if d["t_dec"] is None:
                    d["t_dec"] = t
                d["n_img_batches"] = max(d["n_img_batches"], int(m.group(2)))
        elif (m := SAMPLER.search(line)):
            tid = int(m.group(1))
            if tid in tasks:
                tasks[tid]["text_tokens"] = int(m.group(2))
                tasks[tid]["image_tokens"] = int(m.group(3)) - int(m.group(2))
        elif (m := PROMPT.search(line)):
            tid = int(m.group(1))
            if tid in tasks:
                tasks[tid]["prompt_ms"] = float(m.group(2))
                tasks[tid]["prompt_n"] = int(m.group(3))
        elif (m := EVAL.search(line)):
            tid = int(m.group(1))
            if tid in tasks and "prompt_ms" in tasks[tid]:
                d = tasks[tid]
                d["predicted_ms"] = float(m.group(2))
                d["predicted_n"] = int(m.group(3))
                if d["t_mtmd"] is not None and d["t_dec"] is not None:
                    d["encode_ms"] = (d["t_dec"] - d["t_mtmd"]) * 1000.0
                out.append(d)
    return out


def join(run_dir):
    """Match timed requests.jsonl rows to log tasks (same server, same order, same prompt_ms)."""
    reqs = [json.loads(l) for l in open(run_dir / "requests.jsonl") if l.strip()]
    by_server = {}
    for r in reqs:
        by_server.setdefault(r["server_id"], []).append(r)
    rows, problems = [], []
    for sid, rs in sorted(by_server.items()):
        logs = sorted((run_dir / "server_logs").glob(f"server_{sid:03d}_cfg*.log"))
        if not logs:
            problems.append(f"server {sid}: no log file")
            continue
        tasks = parse_log(logs[0])
        rs = sorted([r for r in rs if r.get("valid")], key=lambda r: r["t"])
        j = 0
        for r in rs:
            # advance through log tasks until prompt_ms and prompt_n agree with the response
            while j < len(tasks) and not (abs(tasks[j]["prompt_ms"] - (r["prompt_ms"] or -1)) < 0.05
                                          and tasks[j]["prompt_n"] == r["prompt_n"]):
                j += 1
            if j == len(tasks):
                problems.append(f"server {sid}: request k={r['k']} pos={r['pos']} not found in log")
                break
            d = tasks[j]
            j += 1
            if r["phase"] != "timed" or r.get("retried"):
                continue
            if d.get("encode_ms") is None:
                problems.append(f"server {sid} task {d['task']}: no encode markers")
                continue
            enc = d["encode_ms"]
            rows.append({
                "config": r["config"], "vision": VISION.get(r["config"], "?"), "dataset": r["dataset"],
                "round": r["round"], "pos": r["pos"], "server_id": sid, "task": d["task"],
                "prompt_n": r["prompt_n"], "image_tokens": d.get("image_tokens"),
                "text_tokens": d.get("text_tokens"), "n_img_batches": d["n_img_batches"],
                "wall_ms": r["wall_ms"], "prompt_ms": r["prompt_ms"], "predicted_ms": r["predicted_ms"],
                "predicted_n": r["predicted_n"],
                "encode": round(enc, 2),
                "prefill": round(r["prompt_ms"] - enc, 2),
                "decode": round(r["predicted_ms"], 2),
                "overhead": round(r["wall_ms"] - r["prompt_ms"] - r["predicted_ms"], 2),
            })
    return rows, problems, reqs


def boot_mean_diff(diffs, n=10000, seed=249):
    rng = random.Random(seed)
    k = len(diffs)
    ms = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return statistics.fmean(diffs), ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--baseline", type=int, default=2, help="config used as reference for paired deltas")
    a = ap.parse_args()
    d = Path(a.dir)
    rows, problems, reqs = join(d)
    n_timed = sum(1 for r in reqs if r["phase"] == "timed" and r.get("valid") and not r.get("retried"))
    if not rows:
        sys.exit("no rows parsed; problems:\n  " + "\n  ".join(problems))

    with open(d / "breakdown_requests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cfgs = sorted({r["config"] for r in rows}, key=lambda c: ORDER.index(c) if c in ORDER else 99)
    dss = [ds for ds in ("scienceqa", "textvqa") if any(r["dataset"] == ds for r in rows)]
    summ, md = [], []
    md.append(f"# Latency breakdown (llama.cpp, Q8_0 language)\n")
    md.append(f"Source: `{d}`. Parsed {len(rows)} of {n_timed} valid, non-retried timed requests.\n")
    md.append("encode = vision encoder; prefill = prompt_ms - encode (LLM, text + image tokens); "
              "decode = token generation; overhead = wall - prompt_ms - predicted_ms.\n")
    md.append("Means are used for the stacked components (means add up to the mean wall time; medians do not). "
              "Median wall time shown for comparison with the Oct 5 run.\n")
    for ds in dss:
        md.append(f"\n## {ds}\n")
        md.append("| Vision | n | image tok | encode | prefill | decode | overhead | mean wall | median wall | encode share |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c in cfgs:
            rs = [r for r in rows if r["config"] == c and r["dataset"] == ds]
            if not rs:
                continue
            mean = {k: statistics.fmean(r[k] for r in rs) for k in COMPONENTS}
            wall = statistics.fmean(r["wall_ms"] for r in rs)
            itok = statistics.median(r["image_tokens"] for r in rs if r["image_tokens"] is not None)
            row = {"config": c, "vision": VISION.get(c), "dataset": ds, "n": len(rs), "image_tokens_med": itok,
                   **{f"{k}_mean": round(v, 1) for k, v in mean.items()},
                   **{f"{k}_median": round(statistics.median(r[k] for r in rs), 1) for k in COMPONENTS},
                   "wall_mean": round(wall, 1), "wall_median": round(statistics.median(r["wall_ms"] for r in rs), 1),
                   "encode_share": round(mean["encode"] / wall, 3),
                   "encode_per_round_median": [round(statistics.median(r["encode"] for r in rs if r["round"] == q))
                                               for q in sorted({r["round"] for r in rs})]}
            summ.append(row)
            md.append(f"| {row['vision']} | {row['n']} | {itok:.0f} | {mean['encode']:.0f} | {mean['prefill']:.0f} | "
                      f"{mean['decode']:.0f} | {mean['overhead']:.0f} | {wall:.0f} | {row['wall_median']:.0f} | "
                      f"{row['encode_share']:.0%} |")

        # paired deltas vs baseline (same record, same round)
        base = {(r["pos"], r["round"]): r for r in rows if r["config"] == a.baseline and r["dataset"] == ds}
        if base:
            md.append(f"\nPaired mean difference vs {VISION.get(a.baseline)} (same record and round), ms "
                      f"[95% bootstrap CI, 10,000 resamples, seed 249]:\n")
            md.append("| Vision | encode | prefill | decode | wall |")
            md.append("|---|---|---|---|---|")
            for c in cfgs:
                if c == a.baseline:
                    continue
                pairs = [(r, base[(r["pos"], r["round"])]) for r in rows
                         if r["config"] == c and r["dataset"] == ds and (r["pos"], r["round"]) in base]
                if not pairs:
                    continue
                cells = []
                for k in ("encode", "prefill", "decode", "wall_ms"):
                    m, lo, hi = boot_mean_diff([x[k] - y[k] for x, y in pairs])
                    cells.append(f"{m:+.0f} [{lo:+.0f}, {hi:+.0f}]")
                md.append(f"| {VISION.get(c)} | " + " | ".join(cells) + " |")

    with open(d / "breakdown_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    if problems:
        md.append("\n## Parser problems\n")
        md += [f"- {p}" for p in problems]
    (d / "breakdown.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {d/'breakdown.md'}, breakdown_summary.csv, breakdown_requests.csv")


if __name__ == "__main__":
    main()
