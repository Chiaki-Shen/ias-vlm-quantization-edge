#!/usr/bin/env python3
"""
CMPE 249 -- TensorRT Edge-LLM version of scripts/bench_latency.py (Max).

Same workload and protocol as the llama.cpp latency/memory benchmark
(results_latency/NOTES.md, latency_benchmark_spec.md):
  * the same frozen records (results_latency/records_frozen.json: 100 ScienceQA + 100 TextVQA)
  * the same prompts and max_tokens (imported from scripts/run_experiment.py)
  * a fresh engine process per config slot (= llama.cpp's fresh server per slot,
    --restart-every 0), 3 warm-up requests after every start (1 ScienceQA + 2 TextVQA)
  * rounds with rotated config order, cool-down (>= 60 s and until <= 50 C, max 300 s)
  * one request at a time, greedy decoding, batch size 1
  * wall-clock latency per request is the headline number

requests.jsonl / events.jsonl / run_meta.json use the same fields as bench_latency.py,
so its summarizer works unchanged:
    python3 scripts/bench_latency.py summarize results_latency_trt/<run> --tegrastats <log>

Differences that come from the runtime, not the protocol:
  * Edge-LLM runs in-process (Python API), so wall_ms is the time of one
    LLM.generate() call (image load + preprocessing + vision encode + prefill + decode).
    llama.cpp's wall_ms is an HTTP round trip to llama-server on localhost; its base64
    image encoding happened before the timer, the server-side decode inside it.
  * The Python API does not report prompt tokens or a prefill/decode split, so
    prompt_n / prompt_ms / predicted_ms / cache_n are null. predicted_n is the number of
    generated tokens. Each generate() call is independent (no prompt cache across calls).
  * A "server death" here is the engine worker process exiting during a request
    (e.g. SIGKILL from the OOM killer). It is restarted with warm-up and the record is
    resent once, flagged retried=True, exactly like bench_latency.py.

Run INSIDE the jetson-llm container (tensorrt/jetson/run_in_container.sh); use
tensorrt/jetson/bench_latency_trt.sh on the host to also record tegrastats.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx  # noqa: E402  -- prompts, specs, helpers
import bench_latency as bl  # noqa: E402  -- protocol constants and helpers
from run_trt_experiment import CONFIGS, EDGE_LLM_VERSION, RUNTIME, git_rev, sampling_settings, build_messages  # noqa: E402

DATASETS = bl.DATASETS
WORKSPACE_DEFAULT = Path.home() / "tensorrt-edgellm-workspace"


def jobs_for(frozen, timed_lim):
    """Timed (dataset, pos) list for one slot, in bench_latency.py order: all ScienceQA, then TextVQA."""
    out = []
    for ds in DATASETS:
        t = frozen["datasets"][ds]["timed"]
        out += [(ds, p) for p in (t[:timed_lim] if timed_lim else t)]
    return out


def warmup_jobs(frozen):
    w = []
    for ds in DATASETS:
        lst = frozen["datasets"][ds]["warmup"][:bl.N_WARMUP_PER_START]
        lst = lst[:2] if ds == "textvqa" else lst[:1]
        w += [(ds, p) for p in lst]
    return w


class Log:
    def __init__(self, out: Path):
        self.out = out

    def write(self, fname, obj):
        with open(self.out / fname, "a") as f:
            f.write(json.dumps(obj) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def interval(self, phase, t0, t1, **kw):
        from datetime import datetime
        self.write("events.jsonl", {"event": "interval", "phase": phase, "t0": t0, "t1": t1,
                                    "iso0": datetime.fromtimestamp(t0).astimezone().isoformat(timespec="milliseconds"),
                                    **kw})


# ── worker: one engine lifetime ──────────────────────────────────────────────

def cmd_worker(a):
    """Load engines, warm up, then run timed jobs[start:]. Exits 0 when done.
    If the process is killed mid-request, the orchestrator sees which job was in flight
    (the last 'inflight' marker without a matching request line)."""
    frozen = json.loads(Path(a.records).read_text())
    data = {ds: rx.load_jsonl(Path(a.data_dir) / ds / "data.jsonl") for ds in DATASETS}
    cfg = CONFIGS[a.config]
    engine_root = Path(a.workspace) / cfg["engine_dir"] / "engines"
    log = Log(Path(a.out))
    ctx = {"round": a.round, "slot": a.slot, "config": cfg["num"]}
    jobs = jobs_for(frozen, a.limit_per_dataset)

    t0 = time.time()
    edge_path = os.environ.get("EDGE_LLM_PATH", str(Path.home() / "TensorRT-Edge-LLM"))
    sys.path.insert(0, edge_path)
    from experimental.server import LLM, SamplingParams  # type: ignore[import-not-found]
    llm = LLM(engine_dir=str(engine_root / "llm"), visual_engine_dir=str(engine_root / "visual"))
    t1 = time.time()
    block0 = jobs[a.start][0] + ":0" if a.start < len(jobs) else "done"
    tag = block0 if a.restart_no == 0 else f"{block0}+r{a.restart_no}"
    log.interval("load", t0, t1, **ctx, block=tag, load_s=t1 - t0)

    def one(ds, pos, phase, k, block, retried):
        rec = data[ds][pos]
        ddir = Path(a.data_dir) / ds
        imgs = rx.resolve_image_paths(rec, ddir)
        prompt = rx.mcq_prompt_context(rec) if ds == "scienceqa" else rx.vqa_prompt_short(rec)
        params = SamplingParams(**sampling_settings(rx.DATASET_SPECS[ds]["max_tokens"]))
        msgs = [build_messages(imgs, prompt)]                       # untimed
        mem, swap_free, _ = bl.mem_info_mb()
        temp = bl.max_temp_c()
        if phase == "timed":
            log.write("inflight.jsonl", {**ctx, "k": k, "server_id": a.server_id})
        t_epoch = time.time()
        err, text, n_out, finish = None, "", None, None
        t_start = time.perf_counter()
        try:
            out = llm.generate(msgs, params)[0]                      # timed
            text, n_out, finish = out.text, len(out.token_ids), out.finish_reason
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        wall_ms = (time.perf_counter() - t_start) * 1000.0
        valid = err is None and isinstance(text, str)
        log.write("requests.jsonl", {
            **ctx, "block": block, "server_id": a.server_id, "valid": valid, "retried": retried,
            "restarts_in_block": a.restart_no, "timings_complete": False,
            "dataset": ds, "phase": phase, "k": k, "pos": pos,
            "t": t_epoch, "wall_ms": round(wall_ms, 2), "http": None, "error": err,
            "prompt_n": None, "prompt_ms": None, "cache_n": None, "predicted_n": n_out,
            "predicted_ms": None, "predicted_per_second": None, "completion_tokens": n_out,
            "finish_reason": finish, "mem_avail_mb": mem, "swap_free_mb": swap_free, "temp_c": temp,
            "reply": (text or "")[:40],
        })
        return valid

    w0 = time.time()
    ok = all([one(ds, pos, "warmup", i, tag, False) for i, (ds, pos) in enumerate(warmup_jobs(frozen))])
    log.interval("warmup", w0, time.time(), **ctx, block=tag, ok=ok)
    if not ok:
        return 3

    # timed jobs, one 'timed' interval per dataset block (as bench_latency.py with --restart-every 0)
    k = a.start
    while k < len(jobs):
        ds = jobs[k][0]
        block = f"{ds}:0"
        b0, n = time.time(), 0
        while k < len(jobs) and jobs[k][0] == ds:
            one(ds, jobs[k][1], "timed", k, block, a.retry_first and k == a.start)
            k, n = k + 1, n + 1
        log.interval("timed", b0, time.time(), **ctx, block=block, dataset=ds, n=n)
    return 0


# ── orchestrator ─────────────────────────────────────────────────────────────

class Bench:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.frozen = json.loads(Path(a.records).read_text())
        for ds in DATASETS:
            have = rx.sha256_file(Path(a.data_dir) / ds / "data.jsonl")
            if have != self.frozen["datasets"][ds]["data_sha256"]:
                sys.exit(f"ERROR: {ds}/data.jsonl hash differs from the frozen record file")
        self.log = Log(self.out)
        self.server_id = 0
        self.failures = []

    def timed_lines(self, r, slot):
        p = self.out / "requests.jsonl"
        if not p.exists():
            return []
        return [x for x in rx.load_jsonl(p) if x["phase"] == "timed" and x["round"] == r and x["slot"] == slot]

    def last_inflight(self, r, slot, server_id):
        p = self.out / "inflight.jsonl"
        if not p.exists():
            return None
        m = [x for x in rx.load_jsonl(p) if x["round"] == r and x["slot"] == slot and x["server_id"] == server_id]
        return m[-1]["k"] if m else None

    def spawn(self, cfg_key, r, slot, start, retry_first, restart_no):
        self.server_id += 1
        a = self.a
        cmd = [sys.executable, str(Path(__file__).resolve()), "_worker",
               "--records", a.records, "--out", a.out, "--config", cfg_key, "--workspace", a.workspace,
               "--data-dir", a.data_dir, "--round", str(r), "--slot", str(slot),
               "--server-id", str(self.server_id), "--start", str(start), "--restart-no", str(restart_no)]
        if retry_first:
            cmd.append("--retry-first")
        if a.limit_per_dataset:
            cmd += ["--limit-per-dataset", str(a.limit_per_dataset)]
        logdir = self.out / "server_logs"
        logdir.mkdir(parents=True, exist_ok=True)
        lf = logdir / f"server_{self.server_id:03d}_cfg{CONFIGS[cfg_key]['num']}.log"
        with open(lf, "wb") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        return rc, lf

    def cooldown(self, ctx):
        a = self.a
        t0 = time.time()
        time.sleep(a.min_cooldown)
        temp = bl.max_temp_c()
        timed_out = False
        while temp is not None and temp > a.cooldown_temp:
            if time.time() - t0 > a.cooldown_max:
                timed_out = True
                break
            time.sleep(10)
            temp = bl.max_temp_c()
        self.log.interval("cooldown", t0, time.time(), **ctx, temp_end_c=temp, timed_out=timed_out)

    def run_slot(self, cfg_key, r, slot, ctx):
        """Worker per slot; on a worker death mid-request, restart + resend that record once."""
        a = self.a
        n_jobs = len(jobs_for(self.frozen, a.limit_per_dataset))
        start, retry_first, restarts, start_fails = 0, False, 0, 0
        while True:
            rc, lf = self.spawn(cfg_key, r, slot, start, retry_first, restarts)
            if rc == 0:
                return True
            k = self.last_inflight(r, slot, self.server_id)
            logged = {x["k"] for x in self.timed_lines(r, slot) if x["server_id"] == self.server_id}
            died_in_request = k is not None and k not in logged
            if died_in_request:
                self.log.write("events.jsonl", {"event": "server_died", "t": time.time(), **ctx,
                                                "dataset": jobs_for(self.frozen, a.limit_per_dataset)[k][0], "k": k,
                                                "server_id": self.server_id, "returncode": rc,
                                                "note": "-9 = SIGKILL (check dmesg for OOM); -11 = segfault; -6 = abort"})
                print(f"  !! engine worker (id {self.server_id}) exited with {rc} during timed request {k}; see {lf}")
                if restarts >= a.max_restarts:
                    self.failures.append({"round": r, "config": ctx["config"], "reason": "too many worker deaths"})
                    return False
                restarts += 1
                if retry_first and k == start:      # this record already killed a worker twice: skip it
                    start, retry_first = k + 1, False
                else:
                    start, retry_first = k, True
                if start >= n_jobs:
                    return True
                continue
            # died during load / warm-up (or exited non-zero otherwise): one retry, like start_checked()
            start_fails += 1
            print(f"  start/warm-up attempt {start_fails} failed for config {ctx['config']} (exit {rc}); see {lf}")
            if start_fails >= 2:
                self.failures.append({"round": r, "config": ctx["config"], "reason": "engine load or warm-up failed"})
                self.log.write("events.jsonl", {"event": "slot_failed", "t": time.time(), **ctx})
                return False

    def run(self):
        a = self.a
        keys = [c.strip() for c in a.configs.split(",")]
        for c in keys:
            if c not in CONFIGS:
                sys.exit(f"unknown config {c}; known {sorted(CONFIGS)}")
            if not (Path(a.workspace) / CONFIGS[c]["engine_dir"] / "engines" / "llm").is_dir():
                sys.exit(f"engines for {c} not found under {a.workspace}")
        jobs = jobs_for(self.frozen, a.limit_per_dataset)
        if a.dry_run:
            for ds in DATASETS:
                n = sum(1 for j in jobs if j[0] == ds)
                print(f"{ds}: {n} timed records, warm-up {[p for d, p in warmup_jobs(self.frozen) if d == ds]}")
            print(f"configs {keys}, rounds {a.rounds}, expected timed requests {a.rounds * len(keys) * len(jobs)}")
            return 0
        self.out.mkdir(parents=True, exist_ok=True)
        edge = Path(os.environ.get("EDGE_LLM_PATH", Path.home() / "TensorRT-Edge-LLM"))
        meta = {"started": bl.now_iso(), "args": vars(a), "runtime": RUNTIME,
                "edge_llm_version": EDGE_LLM_VERSION, "edge_llm_rev": git_rev(edge),
                "runner_revision": rx.runner_revision(),
                "bench_script_sha256": rx.sha256_file(__file__)[:16],
                "reference_bench_script_sha256": rx.sha256_file(PROJECT_ROOT / "scripts" / "bench_latency.py")[:16],
                "frozen_records_sha256": rx.sha256_file(a.records),
                "n_timed": {ds: sum(1 for j in jobs if j[0] == ds) for ds in DATASETS},
                "sync_gpu_allocator": os.environ.get("EDGELLM_SYNC_GPU_ALLOCATOR"),
                "engines": {c: str(Path(a.workspace) / CONFIGS[c]["engine_dir"] / "engines") for c in keys}}
        (self.out / "run_meta.json").write_text(json.dumps(meta, indent=1))

        n, status = len(keys), "running"
        try:
            for r in range(a.rounds):
                shift = (r * n) // a.rounds
                order = keys[shift:] + keys[:shift]
                self.log.write("events.jsonl", {"event": "round_order", "round": r, "t": time.time(), "order": order})
                print(f"\n=== round {r + 1}/{a.rounds}  order {order}  [{bl.now_iso()}]", flush=True)
                for slot, ck in enumerate(order):
                    ctx = {"round": r, "slot": slot, "config": CONFIGS[ck]["num"]}
                    print(f"  config {ck}: cool-down ...", flush=True)
                    self.cooldown(ctx)
                    base, swp_free, _ = bl.mem_info_mb()
                    self.log.write("events.jsonl", {"event": "baseline", "t": time.time(), **ctx,
                                                    "mem_avail_mb": base, "swap_free_mb": swp_free})
                    print(f"    idle MemAvailable {base} MB, SwapFree {swp_free} MB", flush=True)
                    if a.min_avail_mb and base is not None and base < a.min_avail_mb:
                        raise SystemExit(f"ABORT: idle MemAvailable {base} MB < --min-avail-mb {a.min_avail_mb}. "
                                         "Stop the desktop and other programs and rerun.")
                    ok = self.run_slot(ck, r, slot, ctx)
                    print(f"  config {ck}: {'done' if ok else 'FAILED'} [{bl.now_iso()}]", flush=True)
            status = "completed"
        except KeyboardInterrupt:
            status = "interrupted"
        except SystemExit as e:
            status = "aborted"
            print(e)
        finally:
            comp = self.completion(n, jobs)
            if status == "completed" and (self.failures or comp["timed_valid"] != comp["timed_expected"]):
                status = "completed_with_problems"
            report = {**meta, "finished": bl.now_iso(), "status": status, "completion": comp,
                      "slot_failures": self.failures}
            (self.out / "run_meta.json").write_text(json.dumps(report, indent=1))
            print(f"\nSTATUS: {status}   timed requests valid {comp['timed_valid']}/{comp['timed_expected']}"
                  f"  invalid-attempts {comp['timed_invalid']}  worker deaths {comp['server_deaths']}"
                  f"  slot failures {len(self.failures)}")
        return 0 if status == "completed" else 1

    def completion(self, n_cfg, jobs):
        expected = self.a.rounds * n_cfg * len(jobs)
        valid = invalid = 0
        p = self.out / "requests.jsonl"
        if p.exists():
            for r in rx.load_jsonl(p):
                if r["phase"] == "timed":
                    valid += bool(r.get("valid"))
                    invalid += not r.get("valid")
        ev = self.out / "events.jsonl"
        deaths = sum(1 for l in open(ev) if '"server_died"' in l) if ev.exists() else 0
        return {"timed_expected": expected, "timed_valid": valid, "timed_invalid": invalid, "server_deaths": deaths}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--records", default=str(PROJECT_ROOT / "results_latency" / "records_frozen.json"))
        s.add_argument("--out", required=True)
        s.add_argument("--workspace", default=str(WORKSPACE_DEFAULT))
        s.add_argument("--data-dir", default=str(PROJECT_ROOT / "benchmarks"))
        s.add_argument("--limit-per-dataset", type=int, default=None)

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--configs", default="T8,T4")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--cooldown-temp", type=float, default=50.0)
    r.add_argument("--cooldown-max", type=float, default=300.0)
    r.add_argument("--min-cooldown", type=float, default=60.0)
    r.add_argument("--max-restarts", type=int, default=5)
    r.add_argument("--min-avail-mb", type=int, default=0)
    r.add_argument("--dry-run", action="store_true")

    w = sub.add_parser("_worker")
    common(w)
    w.add_argument("--config", required=True)
    w.add_argument("--round", type=int, required=True)
    w.add_argument("--slot", type=int, required=True)
    w.add_argument("--server-id", type=int, required=True)
    w.add_argument("--start", type=int, default=0)
    w.add_argument("--restart-no", type=int, default=0)
    w.add_argument("--retry-first", action="store_true")

    a = p.parse_args()
    if a.cmd == "_worker":
        sys.exit(cmd_worker(a))
    if not a.dry_run and Path(a.out).exists() and any(Path(a.out).glob("requests.jsonl")):
        sys.exit(f"ERROR: {a.out} already has requests.jsonl; use a fresh --out folder")
    sys.exit(Bench(a).run())


if __name__ == "__main__":
    main()
