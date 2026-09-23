"""
CMPE 249 — Experiment Runner
Interactive menu to select a quantization config, launch llama-server,
run inference across all benchmarks, then save a summary report.

Usage (on Jetson, inside sjsujetsontool shell):
    export LD_LIBRARY_PATH=/opt/llama-cpp-new/build/bin:$LD_LIBRARY_PATH
    cd /Developer/ias-vlm-quantization-edge
    python3 scripts/run_experiment.py

Optional flags:
    --data-dir   Path to benchmark data folder (default: <project-root>/benchmarks)
    --limit      Max records per dataset, for smoke tests (default: all)
    --port       llama-server port (default: 8080)

Expected project layout on Jetson:
    /Developer/ias-vlm-quantization-edge/
    ├── scripts/
    │   ├── run_experiment.py   ← this file
    │   ├── run_inference.py
    │   └── prep_datasets.py
    ├── benchmarks/
    │   ├── blink_depth/
    │   ├── blink_spatial/
    │   └── cv_bench/
    └── results/                ← created automatically

Changes from previous version:
    - RESTART_EVERY lowered from 75 to 50
    - Retry-on-error: failed records get one retry after a server restart
    - RAM cleanup after experiment (kill stale processes, drop caches)
    - Summary now reports retries and true errors separately
    - Accuracy calculated on answered records (errors excluded) + raw accuracy

v3 changes (Sep 23):
    - Timestamps use Pacific time (US/Pacific)
    - Error logging: captures error message and image file sizes for diagnosis
    - Fixed cache drop (was failing on read-only /proc in container)
    - Poison-image tracking: logs which record indices consistently fail
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Pacific time: UTC-7 (PDT) — adjust to -8 after Nov daylight saving
_PACIFIC = timezone(timedelta(hours=-7))

def now_pacific():
    return datetime.now(_PACIFIC)

# ── Paths ────────────────────────────────────────────────────────────────────

MODEL_DIR   = Path("/Developer/models/cosmos-reason2-2b")
LLAMA_SRV   = Path("/opt/llama-cpp-new/build/bin/llama-server")
SCRIPT_DIR  = Path(__file__).parent.parent  # project root: /Developer/ias-vlm-quantization-edge
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config table ─────────────────────────────────────────────────────────────

CONFIGS = [
    {
        "num":      1,
        "label":    "F16 vision  +  F16 language  [BASELINE — highest quality, slowest]",
        "short":    "F16v_F16l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-F16.gguf",
        "model":    "Cosmos-Reason2-2B-F16.gguf",
        "official": True,
    },
    {
        "num":      2,
        "label":    "F16 vision  +  Q8_0 language",
        "short":    "F16v_Q8l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-F16.gguf",
        "model":    "Cosmos-Reason2-2B-Q8_0.gguf",
        "official": True,
    },
    {
        "num":      3,
        "label":    "F16 vision  +  Q4_K_M language",
        "short":    "F16v_Q4l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-F16.gguf",
        "model":    "Cosmos-Reason2-2B-Q4_K_M.gguf",
        "official": True,
    },
    {
        "num":      4,
        "label":    "Q8_0 vision  +  F16 language",
        "short":    "Q8v_F16l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-Q8_0.gguf",
        "model":    "Cosmos-Reason2-2B-F16.gguf",
        "official": True,
    },
    {
        "num":      5,
        "label":    "Q8_0 vision  +  Q8_0 language",
        "short":    "Q8v_Q8l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-Q8_0.gguf",
        "model":    "Cosmos-Reason2-2B-Q8_0.gguf",
        "official": True,
    },
    {
        "num":      6,
        "label":    "Q8_0 vision  +  Q4_K_M language  [fastest confirmed config]",
        "short":    "Q8v_Q4l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-Q8_0.gguf",
        "model":    "Cosmos-Reason2-2B-Q4_K_M.gguf",
        "official": True,
    },
    {
        "num":      7,
        "label":    "Q4_0 vision  +  Q4_K_M language  [STRETCH — unofficial Q4 mmproj]",
        "short":    "Q4v_Q4l",
        "mmproj":   "mmproj-Cosmos-Reason2-2B-Q4_0.gguf",
        "model":    "Cosmos-Reason2-2B-Q4_K_M.gguf",
        "official": False,
    },
]

FILE_SIZES = {
    "mmproj-Cosmos-Reason2-2B-F16.gguf":  "782 MB",
    "mmproj-Cosmos-Reason2-2B-Q8_0.gguf": "421 MB",
    "mmproj-Cosmos-Reason2-2B-Q4_0.gguf": "229 MB",
    "Cosmos-Reason2-2B-F16.gguf":          "3.8 GB",
    "Cosmos-Reason2-2B-Q8_0.gguf":         "2.1 GB",
    "Cosmos-Reason2-2B-Q4_K_M.gguf":       "1.2 GB",
}

DATASETS = ["blink_depth", "blink_spatial", "cv_bench"]

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=str(SCRIPT_DIR / "benchmarks"))
parser.add_argument("--limit",    default=None, type=int)
parser.add_argument("--port",     default=8080, type=int)
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
PORT     = args.port
BASE_URL = f"http://localhost:{PORT}"

# ── Menu ──────────────────────────────────────────────────────────────────────

def print_menu():
    print("\n" + "="*68)
    print("  CMPE 249 — Quantization Sensitivity Experiment Runner")
    print("  Model : Cosmos-Reason2-2B  |  Hardware : Jetson Orin Nano 8GB")
    print("="*68 + "\n")

    for c in CONFIGS:
        mmproj_ok = "✓" if (MODEL_DIR / c["mmproj"]).exists() else "✗ MISSING"
        model_ok  = "✓" if (MODEL_DIR / c["model"]).exists()  else "✗ MISSING"
        tag = "  ← unofficial" if not c["official"] else ""

        print(f"  [{c['num']}] {c['label']}{tag}")
        print(f"       Vision encoder  : {c['mmproj']:<48} {FILE_SIZES[c['mmproj']]:>6}  {mmproj_ok}")
        print(f"       Language decoder: {c['model']:<48} {FILE_SIZES[c['model']]:>6}  {model_ok}")
        print()

    print("  [0] Exit\n")

# ── Server ────────────────────────────────────────────────────────────────────

def kill_stale_servers():
    """Kill any leftover llama-server processes."""
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(1)


def start_server(config):
    kill_stale_servers()
    cmd = [
        str(LLAMA_SRV),
        "--model",        str(MODEL_DIR / config["model"]),
        "--mmproj",       str(MODEL_DIR / config["mmproj"]),
        "--ctx-size",     "2048",
        "--n-gpu-layers", "999",
        "--port",         str(PORT),
        "--log-disable",
    ]
    print(f"\n  Starting llama-server ...")
    print(f"  Vision encoder  : {config['mmproj']}  ({FILE_SIZES[config['mmproj']]})")
    print(f"  Language decoder: {config['model']}  ({FILE_SIZES[config['model']]})\n")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_server(timeout=120):
    print("  Waiting for server", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as r:
                if r.status == 200:
                    print(f"  ready in {time.time()-start:.1f}s  [{mem_str()}]")
                    return True
        except Exception:
            print(".", end="", flush=True)
            time.sleep(1)
    print("\n  ERROR: server not ready in time.")
    return False


def stop_server(proc):
    print("\n  Stopping llama-server ...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    # Kill any stragglers
    kill_stale_servers()
    print("  Server stopped.")


def cleanup_ram():
    """Release GPU and system memory after experiment ends."""
    print("\n  Cleaning up memory ...")
    kill_stale_servers()
    # Try to drop kernel page caches — may fail in containers with read-only /proc
    try:
        subprocess.run(["sync"], timeout=5)
        result = subprocess.run(
            ["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            timeout=5, capture_output=True
        )
        if result.returncode == 0:
            print("  Page caches dropped.")
        else:
            print("  Cache drop skipped (read-only /proc — normal in containers).")
    except Exception as e:
        print(f"  Cache drop skipped: {e}")
    print("  Cleanup complete.")

# ── RAM monitoring ────────────────────────────────────────────────────────────

def get_mem_mb():
    """Read available memory from /proc/meminfo. Returns (avail_MB, total_MB) or (None, None)."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if parts[0] in ("MemTotal:", "MemAvailable:"):
                    info[parts[0]] = int(parts[1])  # kB
            total = info.get("MemTotal:", 0) // 1024
            avail = info.get("MemAvailable:", 0) // 1024
            return avail, total
    except Exception:
        return None, None


def mem_str():
    """Short string like '1842/7764 MB' for log output."""
    avail, total = get_mem_mb()
    if avail is None:
        return "mem=N/A"
    return f"mem={avail}/{total}MB"

# ── Inference helpers ─────────────────────────────────────────────────────────

def image_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(question, choices):
    letters = [chr(ord('A') + i) for i in range(26)]
    choice_str = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{choice_str}\n\n"
        "Answer with only the letter of the correct choice. "
        "Do not explain."
    )


def get_image_sizes(image_paths):
    """Return list of (path, file_size_KB) for error diagnostics."""
    sizes = []
    for p in image_paths:
        try:
            sizes.append((p, round(os.path.getsize(p) / 1024, 1)))
        except OSError:
            sizes.append((p, -1))
    return sizes


def call_server(image_paths, prompt):
    content = []
    for p in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_to_b64(p)}"}
        })
    content.append({"type": "text", "text": prompt})

    payload = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 16,
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        return {"error": str(e), "error_type": type(e).__name__}


def extract_answer(response):
    try:
        text = response["choices"][0]["message"]["content"].strip()
        return text[0].upper() if text else ""
    except (KeyError, IndexError):
        return ""


def extract_tok_s(response):
    try:
        return response["timings"]["predicted_per_second"]
    except (KeyError, TypeError):
        return 0.0

# ── Inference loop ────────────────────────────────────────────────────────────

def run_inference(config, proc):
    summary = {
        "config_num":   config["num"],
        "config_label": config["label"],
        "config_short": config["short"],
        "mmproj":       config["mmproj"],
        "model":        config["model"],
        "timestamp":    now_pacific().strftime("%Y-%m-%d %H:%M:%S"),
        "limit":        args.limit,
        "datasets":     {},
    }

    RESTART_EVERY = 50  # restart server every N records to avoid memory leak

    for ds_name in DATASETS:
        jsonl_path = DATA_DIR / ds_name / "data.jsonl"
        if not jsonl_path.exists():
            print(f"\n  [SKIP] {ds_name} — not found at {jsonl_path}")
            continue

        records = [json.loads(l) for l in open(jsonl_path)]
        if args.limit:
            records = records[:args.limit]

        out_path = RESULTS_DIR / f"config{config['num']}_{config['short']}__{ds_name}.jsonl"
        print(f"\n  [{ds_name}]  {len(records)} records  →  {out_path.name}")

        correct, total, errors, retries = 0, 0, 0, 0
        poison_records = []  # record indices that fail even after retry
        since_restart = 0  # count records since last restart
        tok_s_list = []
        ds_start   = time.time()

        with open(out_path, "w") as out_f:
            for i, record in enumerate(records):
                prompt   = build_prompt(record["question"], record["choices"])
                response = call_server(record["image_paths"], prompt)

                # ── Retry logic: if error, restart server and try once more ──
                retried = False
                error_msg = ""
                if "error" in response:
                    retries += 1
                    retried = True
                    error_msg = response.get("error", "unknown")
                    img_sizes = get_image_sizes(record["image_paths"])
                    size_str = ", ".join(f"{s[1]}KB" for s in img_sizes)
                    print(f"    [!] Record {i+1} failed ({size_str}) {mem_str()}, restarting for retry ...")
                    stop_server(proc)
                    proc = start_server(config)
                    if not wait_for_server():
                        # Server won't come back — mark rest as errors and bail
                        proc.kill()
                        errors += len(records) - i
                        total  += len(records) - i
                        break
                    since_restart = 0
                    response = call_server(record["image_paths"], prompt)

                # ── Process response ──
                if "error" in response:
                    errors += 1
                    predicted = ""
                    tok_s = 0.0
                    error_msg = response.get("error", "unknown")
                    avail_mb, total_mb = get_mem_mb()
                    poison_records.append({"idx": record["idx"], "record_num": i+1,
                                           "error": error_msg,
                                           "image_sizes": get_image_sizes(record["image_paths"]),
                                           "mem_avail_mb": avail_mb, "mem_total_mb": total_mb})
                else:
                    predicted = extract_answer(response)
                    tok_s     = extract_tok_s(response)
                    if tok_s > 0:
                        tok_s_list.append(tok_s)
                    error_msg = ""  # clear: retry succeeded (or no error at all)

                gt = str(record["answer"]).strip().upper().strip("()")
                if gt.isdigit():
                    gt = chr(ord('A') + int(gt))

                is_correct = (predicted == gt)
                if is_correct:
                    correct += 1
                total += 1
                since_restart += 1

                avail_now, _ = get_mem_mb()
                out_f.write(json.dumps({
                    "idx":          record["idx"],
                    "dataset":      ds_name,
                    "config":       config["short"],
                    "task":         record.get("task", ""),
                    "question":     record["question"],
                    "choices":      record["choices"],
                    "gt_answer":    gt,
                    "predicted":    predicted,
                    "correct":      is_correct,
                    "tok_s":        tok_s,
                    "retried":      retried,
                    "error":        error_msg if error_msg else None,
                    "mem_avail_mb": avail_now,
                    "image_paths":  record["image_paths"],
                }) + "\n")

                # ── Scheduled restart ──
                if RESTART_EVERY and since_restart >= RESTART_EVERY and (i + 1) < len(records):
                    stop_server(proc)
                    proc = start_server(config)
                    if not wait_for_server():
                        proc.kill()
                        errors += len(records) - (i + 1)
                        total  += len(records) - (i + 1)
                        break
                    since_restart = 0

                # ── Progress ──
                if (i + 1) % 25 == 0 or (i + 1) == len(records):
                    acc     = correct / total * 100 if total else 0
                    answered = total - errors
                    acc_ans = correct / answered * 100 if answered else 0
                    avg_tok = sum(tok_s_list) / len(tok_s_list) if tok_s_list else 0
                    elapsed = time.time() - ds_start
                    print(f"    [{i+1:4d}/{len(records)}]  acc={acc:.1f}% ({acc_ans:.1f}% of answered)  "
                          f"avg_tok/s={avg_tok:.1f}  elapsed={elapsed:.0f}s  "
                          f"errors={errors} retries={retries}  {mem_str()}")

        ds_elapsed = time.time() - ds_start
        avg_tok    = sum(tok_s_list) / len(tok_s_list) if tok_s_list else 0
        answered   = total - errors

        summary["datasets"][ds_name] = {
            "total":          total,
            "answered":       answered,
            "correct":        correct,
            "accuracy_raw":   round(correct / total * 100, 2) if total else 0,
            "accuracy_ans":   round(correct / answered * 100, 2) if answered else 0,
            "avg_tok_s":      round(avg_tok, 1),
            "errors":         errors,
            "retries":        retries,
            "poison_records": len(poison_records),
            "elapsed_s":      round(ds_elapsed, 1),
            "output_file":    str(out_path),
        }

        # Save poison record log for cross-config analysis
        if poison_records:
            poison_path = RESULTS_DIR / f"config{config['num']}_{config['short']}__{ds_name}_poison.jsonl"
            with open(poison_path, "w") as pf:
                for pr in poison_records:
                    pf.write(json.dumps(pr) + "\n")
            print(f"    Poison records ({len(poison_records)}) saved → {poison_path.name}")

    return summary

# ── Summary ───────────────────────────────────────────────────────────────────

def save_summary(summary):
    ts    = now_pacific().strftime("%Y-%m-%d_%H-%M")
    fname = f"config{summary['config_num']}_{summary['config_short']}_{ts}.md"
    path  = RESULTS_DIR / fname

    total_correct  = sum(d["correct"]  for d in summary["datasets"].values())
    total_records  = sum(d["total"]    for d in summary["datasets"].values())
    total_answered = sum(d["answered"] for d in summary["datasets"].values())
    total_errors   = sum(d["errors"]   for d in summary["datasets"].values())
    total_retries  = sum(d["retries"]  for d in summary["datasets"].values())
    overall_raw    = total_correct / total_records * 100 if total_records else 0
    overall_ans    = total_correct / total_answered * 100 if total_answered else 0

    lines = [
        "# CMPE 249 — Experiment Summary",
        "",
        "## Config",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Config number | {summary['config_num']} |",
        f"| Config label | {summary['config_label']} |",
        f"| Vision encoder | {summary['mmproj']} ({FILE_SIZES[summary['mmproj']]}) |",
        f"| Language decoder | {summary['model']} ({FILE_SIZES[summary['model']]}) |",
        f"| Timestamp | {summary['timestamp']} |",
        f"| Record limit | {summary['limit'] or 'all (full run)'} |",
        f"| Server restart interval | every {50} records |",
        "",
        "## Results by Dataset",
        "",
        "| Dataset | Total | Answered | Correct | Acc (raw) | Acc (answered) | Avg tok/s | Errors | Retries | Time |",
        "|---------|-------|----------|---------|-----------|----------------|-----------|--------|---------|------|",
    ]

    for ds_name, ds in summary["datasets"].items():
        lines.append(
            f"| {ds_name} | {ds['total']} | {ds['answered']} | {ds['correct']} | "
            f"{ds['accuracy_raw']:.2f}% | {ds['accuracy_ans']:.2f}% | {ds['avg_tok_s']} | "
            f"{ds['errors']} | {ds['retries']} | {ds['elapsed_s']}s |"
        )

    lines += [
        f"| **TOTAL** | **{total_records}** | **{total_answered}** | **{total_correct}** | "
        f"**{overall_raw:.2f}%** | **{overall_ans:.2f}%** | — | "
        f"**{total_errors}** | **{total_retries}** | — |",
        "",
        "**Acc (raw)** = correct / total (errors count as wrong)  ",
        "**Acc (answered)** = correct / answered (errors excluded) — use this for cross-config comparison  ",
        "",
        "## Output Files",
        "",
    ]
    for ds_name, ds in summary["datasets"].items():
        lines.append(f"- `{Path(ds['output_file']).name}`")

    path.write_text("\n".join(lines))
    print(f"\n  Summary saved → {path}")
    return path

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print_menu()

    if not DATA_DIR.exists():
        print(f"  WARNING: benchmark data not found at {DATA_DIR}")
        print(f"  Transfer the benchmarks/ folder to the Jetson first.\n")

    # Selection
    while True:
        try:
            choice = input("  Select config number (0 to exit): ").strip()
        except KeyboardInterrupt:
            print("\n  Exiting.")
            sys.exit(0)

        if choice == "0":
            print("  Exiting.")
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(CONFIGS):
            config = CONFIGS[int(choice) - 1]
            break
        print(f"  Enter a number 1–{len(CONFIGS)}, or 0 to exit.")

    # File check
    missing = [
        str(MODEL_DIR / f)
        for f in [config["mmproj"], config["model"]]
        if not (MODEL_DIR / f).exists()
    ]
    if missing:
        print("\n  ERROR: Missing model files:")
        for m in missing:
            print(f"    {m}")
        sys.exit(1)

    if not config["official"]:
        print(f"\n  NOTE: Config {config['num']} uses an unofficial Q4_0 mmproj.")
        print(f"  Output quality may be lower than standard configs.")

    print(f"\n  Selected: [{config['num']}] {config['label']}")
    if args.limit:
        print(f"  Smoke-test mode: {args.limit} records per dataset")

    try:
        confirm = input("\n  Start experiment? (y/n): ").strip().lower()
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        sys.exit(0)

    if confirm != "y":
        print("  Cancelled.")
        sys.exit(0)

    # Run
    proc = start_server(config)
    if not wait_for_server():
        proc.kill()
        sys.exit(1)

    exp_start = time.time()
    try:
        summary = run_inference(config, proc)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        summary = {"config_num": config["num"], "config_label": config["label"],
                   "config_short": config["short"], "mmproj": config["mmproj"],
                   "model": config["model"], "timestamp": now_pacific().strftime("%Y-%m-%d %H:%M:%S"),
                   "limit": args.limit, "datasets": {}}
    finally:
        stop_server(proc)

    total_elapsed = time.time() - exp_start

    # Print summary
    print(f"\n{'='*68}")
    print(f"  EXPERIMENT COMPLETE — Config [{config['num']}] {config['short']}")
    print(f"  Total time: {total_elapsed/60:.1f} min")
    print(f"{'='*68}")
    for ds_name, ds in summary.get("datasets", {}).items():
        ans = ds.get("answered", ds["total"])
        print(f"  {ds_name:<22} acc={ds['accuracy_raw']:.2f}% (answered: {ds['accuracy_ans']:.2f}%)  "
              f"avg_tok/s={ds['avg_tok_s']}  errors={ds['errors']}")
    print()

    summary["total_elapsed_s"] = round(total_elapsed, 1)
    save_summary(summary)

    # Clean up RAM so the Jetson is usable after the run
    cleanup_ram()


if __name__ == "__main__":
    main()