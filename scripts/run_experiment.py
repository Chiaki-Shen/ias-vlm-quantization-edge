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
from datetime import datetime
from pathlib import Path

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
parser.add_argument("--data-dir", default=str(SCRIPT_DIR / "benchmarks"))  # /Developer/ias-vlm-quantization-edge/benchmarks
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

def start_server(config):
    cmd = [
        str(LLAMA_SRV),
        "--model",        str(MODEL_DIR / config["model"]),
        "--mmproj",       str(MODEL_DIR / config["mmproj"]),
        "--ctx-size",     "4096",
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
                    print(f"  ready in {time.time()-start:.1f}s")
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
    print("  Server stopped.")

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
        return {"error": str(e)}


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

def run_inference(config):
    summary = {
        "config_num":   config["num"],
        "config_label": config["label"],
        "config_short": config["short"],
        "mmproj":       config["mmproj"],
        "model":        config["model"],
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "limit":        args.limit,
        "datasets":     {},
    }

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

        correct, total, errors = 0, 0, 0
        tok_s_list = []
        ds_start   = time.time()

        with open(out_path, "w") as out_f:
            for i, record in enumerate(records):
                prompt   = build_prompt(record["question"], record["choices"])
                response = call_server(record["image_paths"], prompt)

                if "error" in response:
                    errors += 1
                    predicted = ""
                    tok_s = 0.0
                else:
                    predicted = extract_answer(response)
                    tok_s     = extract_tok_s(response)
                    if tok_s > 0:
                        tok_s_list.append(tok_s)

                gt = str(record["answer"]).strip().upper().strip("()")
                if gt.isdigit():
                    gt = chr(ord('A') + int(gt))

                is_correct = (predicted == gt)
                if is_correct:
                    correct += 1
                total += 1

                out_f.write(json.dumps({
                    "idx":         record["idx"],
                    "dataset":     ds_name,
                    "config":      config["short"],
                    "task":        record.get("task", ""),
                    "question":    record["question"],
                    "choices":     record["choices"],
                    "gt_answer":   gt,
                    "predicted":   predicted,
                    "correct":     is_correct,
                    "tok_s":       tok_s,
                    "image_paths": record["image_paths"],
                }) + "\n")

                if (i + 1) % 25 == 0 or (i + 1) == len(records):
                    acc     = correct / total * 100
                    avg_tok = sum(tok_s_list) / len(tok_s_list) if tok_s_list else 0
                    elapsed = time.time() - ds_start
                    print(f"    [{i+1:4d}/{len(records)}]  acc={acc:.1f}%  "
                          f"avg_tok/s={avg_tok:.1f}  elapsed={elapsed:.0f}s  errors={errors}")

        ds_elapsed = time.time() - ds_start
        avg_tok    = sum(tok_s_list) / len(tok_s_list) if tok_s_list else 0

        summary["datasets"][ds_name] = {
            "total":      total,
            "correct":    correct,
            "accuracy":   round(correct / total * 100, 2) if total else 0,
            "avg_tok_s":  round(avg_tok, 1),
            "errors":     errors,
            "elapsed_s":  round(ds_elapsed, 1),
            "output_file": str(out_path),
        }

    return summary

# ── Summary ───────────────────────────────────────────────────────────────────

def save_summary(summary):
    ts    = datetime.now().strftime("%Y-%m-%d_%H-%M")
    fname = f"config{summary['config_num']}_{summary['config_short']}_{ts}.md"
    path  = RESULTS_DIR / fname

    total_correct = sum(d["correct"] for d in summary["datasets"].values())
    total_records = sum(d["total"]   for d in summary["datasets"].values())
    overall_acc   = total_correct / total_records * 100 if total_records else 0

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
        "",
        "## Results by Dataset",
        "",
        "| Dataset | Total | Correct | Accuracy | Avg tok/s | Errors | Time |",
        "|---------|-------|---------|----------|-----------|--------|------|",
    ]

    for ds_name, ds in summary["datasets"].items():
        lines.append(
            f"| {ds_name} | {ds['total']} | {ds['correct']} | "
            f"{ds['accuracy']:.2f}% | {ds['avg_tok_s']} | "
            f"{ds['errors']} | {ds['elapsed_s']}s |"
        )

    lines += [
        f"| **TOTAL** | **{total_records}** | **{total_correct}** | "
        f"**{overall_acc:.2f}%** | — | — | — |",
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
        summary = run_inference(config)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        summary = {"config_num": config["num"], "config_label": config["label"],
                   "config_short": config["short"], "mmproj": config["mmproj"],
                   "model": config["model"], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        print(f"  {ds_name:<22} acc={ds['accuracy']:.2f}%  avg_tok/s={ds['avg_tok_s']}")
    print()

    summary["total_elapsed_s"] = round(total_elapsed, 1)
    save_summary(summary)


if __name__ == "__main__":
    main()
