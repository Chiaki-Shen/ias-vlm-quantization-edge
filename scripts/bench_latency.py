#!/usr/bin/env python3
"""Controlled latency + memory benchmark for Table B configs (fixed records).

Subcommands
  select     freeze the record set (ScienceQA + TextVQA) into a JSON file
  run        run the timed benchmark (server per config, rotated order, cool-down)
  summarize  latency table from the request log; optional tegrastats RAM per phase

Design (see chat): same records for every config, fresh server per config, warm-up
after every server start, rotated config order per round, cache_prompt=false,
wall-clock request latency as the headline, llama-server timings kept as extra
fields. Run INSIDE the container (llama-server lives there). Put this file in
scripts/ next to run_experiment.py; it reuses that file's prompts and constants so
the requests match the main experiment.
"""
import argparse
import collections
import csv
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATASETS = ("scienceqa", "textvqa")
KNOWN_TEXTVQA_ERRORS = {691, 1185, 1578, 2497, 4041}   # HTTP 400 in all six configs
N_WARMUP_PER_START = 3


# ── small helpers ────────────────────────────────────────────────────────────

def rx_module():
    import run_experiment as rx
    return rx


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def mem_info_mb():
    """(MemAvailable, SwapFree, SwapTotal) in MB from /proc/meminfo."""
    want = {"MemAvailable:": None, "SwapFree:": None, "SwapTotal:": None}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                p = line.split()
                if p[0] in want:
                    want[p[0]] = int(p[1]) // 1024
    except OSError:
        pass
    return want["MemAvailable:"], want["SwapFree:"], want["SwapTotal:"]


def max_temp_c():
    temps = []
    for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            t = int(p.read_text().strip()) / 1000.0
            if 0 < t < 150:          # ignore bogus/disabled zones
                temps.append(t)
        except Exception:        # some zones raise odd errors inside the container
            pass
    return max(temps) if temps else None


def image_dims(path):
    """(width, height) from the file header (PNG / JPEG); PIL if installed; else None."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        pass
    try:
        import struct
        with open(path, "rb") as f:
            head = f.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return struct.unpack(">II", head[16:24])
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    b = f.read(1)
                    while b and b != b"\xff":
                        b = f.read(1)
                    m = f.read(1)
                    while m == b"\xff":
                        m = f.read(1)
                    if not m:
                        return None
                    seg = f.read(2)
                    ln = struct.unpack(">H", seg)[0]
                    if m[0] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        data = f.read(5)
                        h, w = struct.unpack(">HH", data[1:5])
                        return (w, h)
                    f.seek(ln - 2, 1)
    except Exception:
        pass
    return None


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ── select ───────────────────────────────────────────────────────────────────

def cmd_select(a):
    rx = rx_module()
    data_dir = Path(a.data_dir) if a.data_dir else rx.PROJECT_ROOT / "benchmarks"
    rng = random.Random(a.seed)
    out = {"seed": a.seed, "protocol": "latency-v1", "datasets": {}}

    # ScienceQA: random sample + warm-up records
    sdir = data_dir / "scienceqa"
    srecs = rx.load_jsonl(sdir / "data.jsonl")
    pos = list(range(len(srecs)))
    rng.shuffle(pos)
    out["datasets"]["scienceqa"] = {
        "data_sha256": rx.sha256_file(sdir / "data.jsonl"),
        "timed": pos[:a.n_sqa],
        "warmup": pos[a.n_sqa:a.n_sqa + N_WARMUP_PER_START],
    }

    # TextVQA: n_large images with the most PIXELS (valid ones only) + random rest.
    # Every selected question uses a distinct image file. If dimensions cannot be
    # read, file bytes are used and a warning is printed.
    tdir = data_dir / "textvqa"
    trecs = rx.load_jsonl(tdir / "data.jsonl")
    size_key, kb, px_info, img_of = {}, {}, {}, {}
    n_nodims = 0
    for i, rec in enumerate(trecs):
        if i in KNOWN_TEXTVQA_ERRORS:
            continue
        paths = rx.resolve_image_paths(rec, tdir)
        img_of[i] = tuple(str(p) for p in paths)
        kb[i] = sum(os.path.getsize(p) for p in paths) / 1024
        pix, ok = 0, True
        for p in paths:
            d = image_dims(p)
            if d is None:
                ok = False
            else:
                pix += d[0] * d[1]
        if ok:
            px_info[i] = pix
        else:
            n_nodims += 1
        size_key[i] = pix if ok else None
    use_px = n_nodims == 0
    if not use_px:
        print(f"WARNING: could not read dimensions for {n_nodims} records; ranking by file bytes instead")
    key = (lambda i: size_key[i]) if use_px else (lambda i: kb[i])
    ranked = sorted(size_key, key=key, reverse=True)
    used_imgs, large = set(), []
    for i in ranked:                      # distinct images only
        if img_of[i] not in used_imgs:
            used_imgs.add(img_of[i])
            large.append(i)
        if len(large) == a.n_large:
            break
    rest = [i for i in size_key if img_of[i] not in used_imgs]
    rng.shuffle(rest)
    picked, timed_rest = [], []
    for i in rest:
        if img_of[i] in used_imgs:
            continue
        used_imgs.add(img_of[i])
        picked.append(i)
        if len(picked) == (a.n_tvqa - a.n_large) + N_WARMUP_PER_START:
            break
    timed_rest, warm = picked[:a.n_tvqa - a.n_large], picked[a.n_tvqa - a.n_large:]
    timed = large + timed_rest
    rng.shuffle(timed)
    out["datasets"]["textvqa"] = {
        "data_sha256": rx.sha256_file(tdir / "data.jsonl"),
        "timed": timed, "warmup": warm, "large_included": large,
        "ranking": "pixels" if use_px else "file_bytes",
        "image_kb": {str(i): round(kb[i], 1) for i in timed},
        "image_megapixels": {str(i): round(px_info[i] / 1e6, 2) for i in timed if i in px_info},
        "excluded_known_errors": sorted(KNOWN_TEXTVQA_ERRORS),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    kbs = [kb[i] for i in timed]
    print(f"wrote {a.out}")
    print(f"  scienceqa timed={len(out['datasets']['scienceqa']['timed'])}")
    print(f"  textvqa   timed={len(timed)} (distinct images)  file KB: median {statistics.median(kbs):.0f}, max {max(kbs):.0f}")
    if use_px:
        mp = [px_info[i] / 1e6 for i in timed]
        print(f"            megapixels: median {statistics.median(mp):.2f}, max {max(mp):.2f}  ({a.n_large} largest by pixels included)")


# ── run ──────────────────────────────────────────────────────────────────────

class Bench:
    def __init__(self, a):
        self.a = a
        self.rx = rx_module()
        self.data_dir = Path(a.data_dir) if a.data_dir else self.rx.PROJECT_ROOT / "benchmarks"
        self.model_dir = Path(a.model_dir) if a.model_dir else self.rx.MODEL_DIR
        self.server_bin = Path(a.server_bin) if a.server_bin else self.rx.LLAMA_SRV
        self.out = Path(a.out)
        self.port = a.port
        self.base = f"http://localhost:{self.port}"
        self.frozen = json.loads(Path(a.records).read_text())
        self.data = {}
        for ds in DATASETS:
            self.data[ds] = self.rx.load_jsonl(self.data_dir / ds / "data.jsonl")
            have = self.rx.sha256_file(self.data_dir / ds / "data.jsonl")
            if have != self.frozen["datasets"][ds]["data_sha256"]:
                sys.exit(f"ERROR: {ds}/data.jsonl hash differs from the frozen record file")
        self.proc = None
        self.server_id = 0
        self.server_dead = False
        self.failures = []          # (round, config, reason)

    # -- request building (NOT timed) --
    def build_body(self, ds, pos):
        rx = self.rx
        rec = self.data[ds][pos]
        ddir = self.data_dir / ds
        content = [{"type": "image_url", "image_url": {"url": rx.image_data_uri(p)}}
                   for p in rx.resolve_image_paths(rec, ddir)]
        prompt = rx.mcq_prompt_context(rec) if ds == "scienceqa" else rx.vqa_prompt_short(rec)
        content.append({"type": "text", "text": prompt})
        return json.dumps({
            "model": "local",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": rx.DATASET_SPECS[ds]["max_tokens"],
            "temperature": rx.TEMPERATURE,
            "cache_prompt": False,
        }).encode("utf-8")

    def post_timed(self, body):
        req = urllib.request.Request(f"{self.base}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        status, err, raw = None, None, None
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.rx.REQUEST_TIMEOUT) as r:
                status = r.status
                raw = r.read()
        except urllib.error.HTTPError as e:
            status = e.code
            try:
                body_txt = e.read(400).decode("utf-8", "replace")
            except Exception:
                body_txt = ""
            err = f"HTTPError {e.code}: {body_txt}"
        except Exception as e:  # timeout, connection reset, ...
            err = f"{type(e).__name__}: {e}"
        wall_ms = (time.perf_counter() - t0) * 1000.0
        resp = None
        if raw is not None:
            try:
                resp = json.loads(raw.decode("utf-8"))
            except ValueError as e:
                err = f"bad json: {e}"
        return status, err, wall_ms, resp

    # -- server control --
    def kill_servers(self):
        subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
        time.sleep(1)

    def start_server(self, cfg):
        self.kill_servers()
        cmd = [str(self.server_bin),
               "--model", str(self.model_dir / cfg["model"]),
               "--mmproj", str(self.model_dir / cfg["mmproj"]),
               "--ctx-size", str(self.rx.SERVER_CTX),
               "--n-gpu-layers", str(self.rx.SERVER_NGL),
               "--port", str(self.port)]
        if not self.a.server_logs:
            cmd.append("--log-disable")
        if self.a.server_extra:
            import shlex
            cmd += shlex.split(self.a.server_extra)
        self.server_id += 1
        logdir = self.out / "server_logs"
        logdir.mkdir(parents=True, exist_ok=True)
        self.server_log = logdir / f"server_{self.server_id:03d}_cfg{cfg['num']}.log"
        self._logf = open(self.server_log, "wb")
        self.proc = subprocess.Popen(cmd, stdout=self._logf, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 180:
            if self.proc.poll() is not None:
                print(f"  server exited early (code {self.proc.returncode}); see {self.server_log}")
                return None
            try:
                with urllib.request.urlopen(f"{self.base}/health", timeout=2) as r:
                    if r.status == 200:
                        return time.time() - t0
            except Exception:
                time.sleep(0.5)
        return None

    def stop_server(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
            self.proc = None
        self.kill_servers()

    # -- logging --
    def log(self, fname, obj):
        with open(self.out / fname, "a") as f:
            f.write(json.dumps(obj) + "\n")
            f.flush()

    def interval(self, phase, t0, t1, **kw):
        self.log("events.jsonl", {"event": "interval", "phase": phase, "t0": t0, "t1": t1,
                                  "iso0": datetime.fromtimestamp(t0).astimezone().isoformat(timespec="milliseconds"),
                                  **kw})

    # -- one block of requests --
    def check_dead(self, ctx, cfg, ds, k):
        """After a connection-level failure: is the server process gone? Record its exit code."""
        time.sleep(0.5)
        rc = self.proc.poll() if self.proc is not None else "no process"
        if rc is None:
            return False
        self.server_dead = True
        self.log("events.jsonl", {"event": "server_died", "t": time.time(), **ctx, "config": cfg["num"],
                                  "dataset": ds, "k": k, "server_id": self.server_id,
                                  "pid": getattr(self.proc, "pid", None), "returncode": rc,
                                  "note": "-9 = SIGKILL (check dmesg for OOM); -11 = segfault; -6 = abort"})
        print(f"  !! llama-server (id {self.server_id}) is gone, exit code {rc}, during {ds} request {k}")
        return True

    def run_requests(self, cfg, ds, positions, phase, ctx):
        """Send positions in order. If the server dies during a TIMED request: restart it
        (with warm-up) and resend that record once, flagged retried=True (latency stats
        exclude retried requests). If the same record kills it again, it is skipped.
        More than --max-restarts restarts in one block -> give up (server_dead stays True)."""
        self.server_dead = False
        self.block_restarts = 0
        k, retry_flag = 0, False
        while k < len(positions):
            pos = positions[k]
            body = self.build_body(ds, pos)                    # untimed
            mem, swap_free, _ = mem_info_mb()
            temp = max_temp_c()
            t_epoch = time.time()
            status, err, wall_ms, resp = self.post_timed(body)  # timed
            tm = (resp or {}).get("timings") or {}
            usage = (resp or {}).get("usage") or {}
            text, has_text = "", False
            try:
                c = resp["choices"][0]["message"]["content"]
                has_text = isinstance(c, str)
                text = c or ""
            except Exception:
                pass
            valid = (status == 200 and err is None and has_text)
            tkeys = ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "cache_n")
            if phase == "warmup":
                self.warm_ok = getattr(self, "warm_ok", True) and valid
            self.log("requests.jsonl", {
                **ctx, "server_id": self.server_id, "valid": valid, "retried": retry_flag,
                "restarts_in_block": self.block_restarts,
                "timings_complete": all(tm.get(x) is not None for x in tkeys),
                "config": cfg["num"], "dataset": ds, "phase": phase, "k": k, "pos": pos,
                "t": t_epoch, "wall_ms": round(wall_ms, 2), "http": status, "error": err,
                "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                "cache_n": tm.get("cache_n"), "predicted_n": tm.get("predicted_n"),
                "predicted_ms": tm.get("predicted_ms"),
                "predicted_per_second": tm.get("predicted_per_second"),
                "completion_tokens": usage.get("completion_tokens"),
                "mem_avail_mb": mem, "swap_free_mb": swap_free, "temp_c": temp,
                "reply": text[:40],
            })
            if not valid and status is None and self.check_dead(ctx, cfg, ds, k):
                if phase != "timed" or self.block_restarts >= self.a.max_restarts:
                    return                                   # caller sees server_dead
                self.block_restarts += 1
                n_restart = self.block_restarts
                if retry_flag:                               # this record already killed a server twice
                    k, retry_flag = k + 1, False
                else:
                    retry_flag = True
                self.stop_server()
                base_ctx = {kk: vv for kk, vv in ctx.items() if kk != "block"}
                ok = self.warm_start(cfg, base_ctx, f"{ctx['block']}+r{n_restart}")
                if not ok:
                    self.server_dead = True
                    return
                self.server_dead = False
                self.block_restarts = n_restart
                continue
            k, retry_flag = k + 1, False

    def blocks(self, positions):
        n = self.a.restart_every
        if n <= 0:
            return [positions]
        return [positions[i:i + n] for i in range(0, len(positions), n)]

    def warm_start(self, cfg, ctx, block_tag):
        t0 = time.time()
        load_s = self.start_server(cfg)
        t1 = time.time()
        self.interval("load", t0, t1, **ctx, block=block_tag, load_s=load_s)
        if load_s is None:
            return False
        w0 = time.time()
        self.warm_ok = True
        for ds in DATASETS:
            w = self.frozen["datasets"][ds]["warmup"][:N_WARMUP_PER_START]
            if ds == "textvqa":
                w = w[:2]
            elif ds == "scienceqa":
                w = w[:1]
            self.run_requests(cfg, ds, w, "warmup", {**ctx, "block": block_tag})
        self.interval("warmup", w0, time.time(), **ctx, block=block_tag, ok=self.warm_ok)
        return self.warm_ok and not self.server_dead

    def start_checked(self, cfg, ctx, tag):
        """Start + warm up; one retry. Returns False if it still fails."""
        for attempt in (1, 2):
            if self.warm_start(cfg, ctx, tag):
                return True
            print(f"  start/warm-up attempt {attempt} failed for config {cfg['num']} ({tag})")
            self.stop_server()
        return False

    def cooldown(self, ctx):
        t0 = time.time()
        time.sleep(self.a.min_cooldown)
        temp = max_temp_c()
        timed_out = False
        while temp is not None and temp > self.a.cooldown_temp:
            if time.time() - t0 > self.a.cooldown_max:
                timed_out = True
                break
            time.sleep(10)
            temp = max_temp_c()
        self.interval("cooldown", t0, time.time(), **ctx, temp_end_c=temp, timed_out=timed_out)

    def run(self):
        a, rx = self.a, self.rx
        cfgs = [rx.get_config(int(c)) for c in a.configs.split(",")]
        if any(c is None for c in cfgs):
            sys.exit("unknown config number")
        lim = a.limit_per_dataset
        timed = {ds: self.frozen["datasets"][ds]["timed"][:lim] if lim else self.frozen["datasets"][ds]["timed"]
                 for ds in DATASETS}
        if a.dry_run:
            for ds in DATASETS:
                b = self.build_body(ds, timed[ds][0])
                print(f"{ds}: first body {len(b)/1024:.0f} KB ok")
            print("dry run ok; server binary:", self.server_bin, "exists:", self.server_bin.exists())
            return 0
        self.out.mkdir(parents=True, exist_ok=True)
        meta = {"started": now_iso(), "args": vars(a), "runner_revision": rx.runner_revision(),
                "bench_script_sha256": rx.sha256_file(__file__)[:16],
                "frozen_records_sha256": rx.sha256_file(a.records),
                "n_timed": {ds: len(timed[ds]) for ds in DATASETS}, "model_hashes": {}}
        cache = self.out.parent / ".model_hash_cache.json"
        for c in cfgs:
            try:
                meta["model_hashes"][c["num"]] = rx.compute_model_hashes(c, str(cache))
            except Exception as e:
                meta["model_hashes"][c["num"]] = f"unavailable: {e}"
        try:
            r = subprocess.run([str(self.server_bin), "--version"], capture_output=True, text=True, timeout=20)
            meta["server_version"] = (r.stdout + r.stderr).strip()[-300:]
        except Exception as e:
            meta["server_version"] = f"unavailable: {e}"
        (self.out / "run_meta.json").write_text(json.dumps(meta, indent=1))

        n = len(cfgs)
        status = "running"
        try:
            for r in range(a.rounds):
                shift = (r * n) // a.rounds
                order = cfgs[shift:] + cfgs[:shift]
                self.log("events.jsonl", {"event": "round_order", "round": r, "t": time.time(),
                                          "order": [c["num"] for c in order]})
                print(f"\n=== round {r + 1}/{a.rounds}  order {[c['num'] for c in order]}  [{now_iso()}]")
                for slot, cfg in enumerate(order):
                    ctx = {"round": r, "slot": slot, "config": cfg["num"]}
                    print(f"  config {cfg['num']}: cool-down ...", flush=True)
                    self.cooldown(ctx)
                    base, swp_free, _ = mem_info_mb()
                    self.log("events.jsonl", {"event": "baseline", "t": time.time(), **ctx,
                                              "mem_avail_mb": base, "swap_free_mb": swp_free})
                    print(f"    idle MemAvailable {base} MB, SwapFree {swp_free} MB", flush=True)
                    if a.min_avail_mb and base is not None and base < a.min_avail_mb:
                        for _ in range(6):
                            time.sleep(10)
                            base, _sf, _t = mem_info_mb()
                            if base >= a.min_avail_mb:
                                break
                        if base < a.min_avail_mb:
                            raise SystemExit(f"ABORT: idle MemAvailable {base} MB < --min-avail-mb {a.min_avail_mb}. "
                                             "Close other programs (desktop, browser) and rerun.")
                    slot_ok, first = True, True
                    for ds in DATASETS:
                        if not slot_ok:
                            break
                        for bi, blk in enumerate(self.blocks(timed[ds])):
                            tag = f"{ds}:{bi}"
                            if first or a.restart_every > 0:
                                if not first:
                                    self.stop_server()
                                if not self.start_checked(cfg, ctx, tag):
                                    slot_ok = False
                                    self.failures.append({"round": r, "config": cfg["num"], "block": tag,
                                                          "reason": "server start or warm-up failed"})
                                    self.log("events.jsonl", {"event": "slot_failed", "t": time.time(), **ctx, "block": tag})
                                    print(f"  !! config {cfg['num']} round {r}: slot FAILED at {tag}; continuing with next slot")
                                    break
                                first = False
                            t0 = time.time()
                            self.run_requests(cfg, ds, blk, "timed", {**ctx, "block": tag})
                            self.interval("timed", t0, time.time(), **ctx, block=tag, dataset=ds, n=len(blk))
                            if self.server_dead:
                                slot_ok = False
                                self.failures.append({"round": r, "config": cfg["num"], "block": tag,
                                                      "reason": "server died during timed block"})
                                print(f"  !! config {cfg['num']} round {r}: server died in {tag}; rest of slot skipped")
                                break
                    self.stop_server()
                    print(f"  config {cfg['num']}: {'done' if slot_ok else 'FAILED'} [{now_iso()}]", flush=True)
            status = "completed"
        except KeyboardInterrupt:
            status = "interrupted"
        except SystemExit as e:
            status = "aborted"
            print(e)
        except Exception:
            status = "crashed"
            import traceback
            traceback.print_exc()
        finally:
            self.stop_server()
            comp = self.completion(len(cfgs), timed)
            if status == "completed" and (self.failures or comp["timed_valid"] != comp["timed_expected"]):
                status = "completed_with_problems"
            report = {**meta, "finished": now_iso(), "status": status, "completion": comp,
                      "slot_failures": self.failures}
            (self.out / "run_meta.json").write_text(json.dumps(report, indent=1))
            print(f"\nSTATUS: {status}   timed requests valid {comp['timed_valid']}/{comp['timed_expected']}"
                  f"  invalid-attempts {comp['timed_invalid']}  server deaths {comp['server_deaths']}  slot failures {len(self.failures)}")
        return 0 if status == "completed" else 1

    def completion(self, n_cfg, timed):
        expected = self.a.rounds * n_cfg * sum(len(v) for v in timed.values())
        valid = invalid = 0
        path = self.out / "requests.jsonl"
        if path.exists():
            for l in open(path):
                if not l.strip():
                    continue
                r = json.loads(l)
                if r["phase"] == "timed":
                    if r.get("valid"):
                        valid += 1
                    else:
                        invalid += 1
        deaths = 0
        ev = self.out / "events.jsonl"
        if ev.exists():
            deaths = sum(1 for l in open(ev) if '"server_died"' in l)
        return {"timed_expected": expected, "timed_valid": valid, "timed_invalid": invalid, "server_deaths": deaths}


# ── summarize ────────────────────────────────────────────────────────────────

def load_lines(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def parse_tegrastats(path, tz):
    """[(epoch_float, ram_used_mb, swap_used_mb)]. tegrastats stamps whole seconds only,
    so samples sharing a second are spread evenly across it (approximation, +-1 sample)."""
    from zoneinfo import ZoneInfo
    zi = ZoneInfo(tz)
    ts_re = re.compile(r"^(\d\d)-(\d\d)-(\d{4}) (\d\d):(\d\d):(\d\d)")
    ram_re = re.compile(r"RAM (\d+)/(\d+)MB")
    swp_re = re.compile(r"SWAP (\d+)/(\d+)MB")
    groups = []
    for line in open(path, errors="ignore"):
        m, r, s_ = ts_re.match(line), ram_re.search(line), swp_re.search(line)
        if not (m and r):
            continue
        mo, d, y, H, M, S = map(int, m.groups())
        t = datetime(y, mo, d, H, M, S, tzinfo=zi).timestamp()
        rec = (int(r.group(1)), int(s_.group(1)) if s_ else 0)
        if groups and groups[-1][0] == t:
            groups[-1][1].append(rec)
        else:
            groups.append((t, [rec]))
    out = []
    for t, recs in groups:
        for i, (ram, swp) in enumerate(recs):
            out.append((t + i / len(recs), ram, swp))
    return out


def cmd_summarize(a):
    d = Path(a.dir)
    reqs = load_lines(d / "requests.jsonl")
    events = load_lines(d / "events.jsonl") if (d / "events.jsonl").exists() else []
    if (d / "run_meta.json").exists():
        meta = json.loads((d / "run_meta.json").read_text())
        comp = meta.get("completion", {})
        print(f"run status: {meta.get('status', 'unknown (no final status)')}  "
              f"timed valid {comp.get('timed_valid')}/{comp.get('timed_expected')}  "
              f"slot failures: {len(meta.get('slot_failures', []))}")
        if meta.get("status") != "completed":
            print("  !! run did not complete cleanly; do not treat these numbers as the full benchmark")
    timed = [r for r in reqs if r["phase"] == "timed"]
    order = [2, 5, 9, 10, 8, 11]
    cfgs = sorted({r["config"] for r in timed}, key=lambda c: order.index(c) if c in order else 99)

    print("\n== Latency (wall-clock per request, ms; valid, non-retried requests) ==")
    print(f"{'cfg':>3} {'dataset':<9} {'valid':>5} {'bad':>3} {'med':>7} {'p25':>7} {'p75':>7} {'p95':>7} "
          f"{'prompt_ms':>9} {'gen_ms':>7} {'gen_tok':>7}  per-round medians")
    rows = []
    for c in cfgs:
        for ds in DATASETS:
            rs = [r for r in timed if r["config"] == c and r["dataset"] == ds]
            ok = [r for r in rs if r.get("valid") and not r.get("retried")]
            if not rs:
                continue
            w = [r["wall_ms"] for r in ok]
            def med(key):
                v = [r[key] for r in ok if r.get(key) is not None]
                return statistics.median(v) if v else float("nan")
            rounds = sorted({r["round"] for r in ok})
            per_round = [round(statistics.median([r["wall_ms"] for r in ok if r["round"] == q])) for q in rounds]
            row = dict(config=c, dataset=ds, valid=len(ok), invalid=len(rs) - len(ok),
                       wall_med=statistics.median(w) if w else float("nan"),
                       wall_p25=pct(w, .25), wall_p75=pct(w, .75), wall_p95=pct(w, .95),
                       prompt_ms_med=med("prompt_ms"), gen_ms_med=med("predicted_ms"),
                       gen_tokens_med=med("predicted_n"))
            rows.append(row)
            f0 = lambda x: f"{x:>7.0f}" if x is not None else f"{'n/a':>7}"
            print(f"{c:>3} {ds:<9} {row['valid']:>5} {row['invalid']:>3} {f0(row['wall_med'])} {f0(row['wall_p25'])} "
                  f"{f0(row['wall_p75'])} {f0(row['wall_p95'])} {row['prompt_ms_med']:>9.0f} "
                  f"{row['gen_ms_med']:>7.0f} {row['gen_tokens_med']:>7.0f}  {per_round}")
    with open(d / "latency_summary.csv", "w", newline="") as f:
        if rows:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

    deaths = collections.Counter(e["config"] for e in events if e.get("event") == "server_died")
    retried_ok = collections.Counter(r["config"] for r in timed if r.get("retried") and r.get("valid"))
    skipped = collections.Counter(r["config"] for r in timed if r.get("retried") and not r.get("valid"))
    if deaths:
        print("\n== Server deaths (llama-server exited during a request) ==")
        print("   latency table above EXCLUDES requests that were retried after a restart")
        for c in cfgs:
            if deaths.get(c):
                print(f"  config {c}: deaths {deaths[c]}   retried-and-valid {retried_ok.get(c, 0)}   skipped after 2nd death {skipped.get(c, 0)}")

    print("\n== Cache check (cache_prompt was sent as false) ==")
    ok_t = [r for r in timed if r.get("valid")]
    miss = sum(1 for r in ok_t if r.get("cache_n") is None)
    hits = sum(1 for r in ok_t if (r.get("cache_n") or 0) > 0)
    print(f"  valid timed requests: {len(ok_t)}   cache_n missing: {miss}   cache_n > 0: {hits}")
    if miss:
        print("  !! cache_n is missing from some responses, so 'no cache hits' is NOT proven by that field")
    incomplete = sum(1 for r in ok_t if not r.get("timings_complete"))
    if incomplete:
        print(f"  !! {incomplete} valid requests lack some timing fields")
    # independent check: same record + same config -> same prompt_n in every round
    by_rec = {}
    for r in ok_t:
        if r.get("prompt_n") is not None:
            by_rec.setdefault((r["config"], r["dataset"], r["pos"]), set()).add(r["prompt_n"])
    varying = sum(1 for v in by_rec.values() if len(v) > 1)
    print(f"  records whose prompt_n differs between rounds (same config): {varying} of {len(by_rec)}  (expect 0)")
    by_pos = {}
    for (c, ds, pos), v in by_rec.items():
        by_pos.setdefault((ds, pos), set()).update(v)
    xcfg = sum(1 for v in by_pos.values() if len(v) > 1)
    print(f"  records whose prompt_n differs across configs: {xcfg} of {len(by_pos)}  (expect 0 if image tokens do not depend on the encoder)")

    print("\n== Memory drift over each SERVER LIFETIME (MemAvailable, first-10 vs last-10 timed requests, MB) ==")
    print("   negative = available memory shrinking while the server stays up")
    servers = {}
    for r in timed:
        servers.setdefault((r["config"], r["round"], r["server_id"]), []).append(r)
    drift = {}
    for (c, rnd, sid), rs in servers.items():
        rs.sort(key=lambda x: x["t"])
        m = [x["mem_avail_mb"] for x in rs if x["mem_avail_mb"] is not None]
        if len(m) >= 20:
            drift.setdefault(c, []).append((len(m), round(statistics.median(m[-10:]) - statistics.median(m[:10]))))
    for c in cfgs:
        if c in drift:
            print(f"  config {c}: (requests, drift) per server {drift[c]}")
    swaps = [r["swap_free_mb"] for r in reqs if r["swap_free_mb"] is not None]
    if swaps:
        print(f"  lowest SwapFree seen: {min(swaps)} MB")
    temps = [r["temp_c"] for r in reqs if r["temp_c"] is not None]
    if temps:
        print(f"  max temperature seen at request start: {max(temps):.1f} C")
    to = [e for e in events if e.get("phase") == "cooldown" and e.get("timed_out")]
    if to:
        print(f"  WARNING: {len(to)} cool-downs hit the timeout (start temperature was above target)")

    if a.tegrastats:
        samples = parse_tegrastats(a.tegrastats, a.tegra_tz)
        if not samples:
            print("\n!! no usable tegrastats lines (need 'MM-DD-YYYY HH:MM:SS RAM x/yMB' format)")
            return
        print("\n== tegrastats: max sampled SYSTEM RAM used (MB) on this fixed workload ==")
        print("   whole-system reading, not per-process. 'strict' ignores samples within 1 s of a phase edge")
        print("   (tegrastats stamps whole seconds, so edge samples are ambiguous); 'incl' keeps them.")
        idle = [e for e in events if e.get("phase") == "cooldown"]
        by = {}
        for e in events:
            if e.get("event") != "interval" or e["phase"] not in ("load", "warmup", "timed"):
                continue
            inc = [x for x in samples if e["t0"] <= x[0] <= e["t1"]]
            strict = [x for x in samples if e["t0"] + 1 <= x[0] <= e["t1"] - 1]
            if inc:
                k = (e["config"], e["phase"])
                v = by.setdefault(k, {"incl": [], "strict": [], "swap": []})
                v["incl"].append(max(x[1] for x in inc))
                if strict:
                    v["strict"].append(max(x[1] for x in strict))
                v["swap"].append(max(x[2] for x in inc))
        base = []
        for e in idle:
            w = [x[1] for x in samples if e["t0"] <= x[0] <= e["t1"]]
            if w:
                base.append(min(w))
        if base:
            print(f"   idle (server stopped, cool-down windows): median of window minima = {statistics.median(base):.0f} MB")
        print(f"{'cfg':>3} {'phase':<7} {'max incl':>8} {'max strict':>10} {'max swap':>8}   per-interval strict maxima")
        for c in cfgs:
            for ph in ("load", "warmup", "timed"):
                v = by.get((c, ph))
                if v:
                    st = max(v["strict"]) if v["strict"] else "n/a"
                    print(f"{c:>3} {ph:<7} {max(v['incl']):>8} {st:>10} {max(v['swap']):>8}   {v['strict']}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select")
    s.add_argument("--out", required=True)
    s.add_argument("--data-dir", default=None)
    s.add_argument("--n-sqa", type=int, default=100)
    s.add_argument("--n-tvqa", type=int, default=100)
    s.add_argument("--n-large", type=int, default=10, help="largest valid TextVQA images to include")
    s.add_argument("--seed", type=int, default=249)

    r = sub.add_parser("run")
    r.add_argument("--records", required=True)
    r.add_argument("--out", required=True, help="fresh results folder (appended to; use a new one each run)")
    r.add_argument("--configs", default="2,5,9,10,8,11")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--restart-every", type=int, default=50, help="0 = no restart within a slot")
    r.add_argument("--limit-per-dataset", type=int, default=None)
    r.add_argument("--cooldown-temp", type=float, default=50.0)
    r.add_argument("--cooldown-max", type=float, default=300.0, help="timeout seconds")
    r.add_argument("--min-cooldown", type=float, default=60.0)
    r.add_argument("--port", type=int, default=8080)
    r.add_argument("--data-dir", default=None)
    r.add_argument("--model-dir", default=None)
    r.add_argument("--server-bin", default=None)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--server-extra", default="", help='extra llama-server args, e.g. --server-extra="--cache-ram 0"')
    r.add_argument("--max-restarts", type=int, default=5, help="restarts allowed per block after server deaths")
    r.add_argument("--server-logs", action="store_true", help="diagnostics: do not pass --log-disable to llama-server")
    r.add_argument("--min-avail-mb", type=int, default=0,
                   help="abort the run if idle MemAvailable (server stopped) stays below this before a slot; 0 = off")

    m = sub.add_parser("summarize")
    m.add_argument("dir")
    m.add_argument("--tegrastats", default=None)
    m.add_argument("--tegra-tz", default="America/Los_Angeles", help="timezone of the tegrastats timestamps (host)")

    a = p.parse_args()
    if a.cmd == "select":
        cmd_select(a)
    elif a.cmd == "run":
        if not a.dry_run and Path(a.out).exists() and any(Path(a.out).glob("requests.jsonl")):
            sys.exit(f"ERROR: {a.out} already has requests.jsonl; use a fresh --out folder")
        sys.exit(Bench(a).run())
    else:
        cmd_summarize(a)


if __name__ == "__main__":
    main()
