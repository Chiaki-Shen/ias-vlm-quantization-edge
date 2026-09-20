"""
CMPE 249 — Benchmark Inference Script
Runs all benchmark records against a running llama-server instance,
saves raw outputs to JSONL for offline scoring.

Usage (on Jetson, inside sjsujetsontool shell):
    python3 run_inference.py --config "Q8_vision_Q4_language" --limit 10
    python3 run_inference.py --config "Q8_vision_Q4_language"

Arguments:
    --config    Label for this run (used in output filename)
    --host      llama-server host (default: localhost)
    --port      llama-server port (default: 8080)
    --limit     Max records per dataset (default: all) — use for smoke tests
    --datasets  Comma-separated list: blink_depth,blink_spatial,cv_bench (default: all)
    --data-dir  Path to benchmark data folder (default: ./benchmarks)
"""

import argparse
import base64
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime


# ── Argument parsing ────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--config",   required=True, help="Label for this quant config, e.g. Q8v_Q4l")
parser.add_argument("--host",     default="localhost")
parser.add_argument("--port",     default=8080, type=int)
parser.add_argument("--limit",    default=None, type=int, help="Max records per dataset (for smoke tests)")
parser.add_argument("--datasets", default="blink_depth,blink_spatial,cv_bench")
parser.add_argument("--data-dir", default="./benchmarks", help="Path to folder containing blink_depth/, blink_spatial/, cv_bench/")
args = parser.parse_args()

BASE_URL = f"http://{args.host}:{args.port}/v1/chat/completions"
DATA_DIR = Path(args.data_dir)
OUT_DIR  = Path("./results")
OUT_DIR.mkdir(exist_ok=True)

DATASETS = [d.strip() for d in args.datasets.split(",")]


# ── Helpers ─────────────────────────────────────────────────────────────────

def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(question: str, choices: list) -> str:
    """Format MCQ question with lettered choices."""
    letters = ["A", "B", "C", "D", "E"]
    choice_str = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{choice_str}\n\n"
        "Answer with only the letter of the correct choice (A, B, C, or D). "
        "Do not explain."
    )


def call_server(image_paths: list, prompt: str) -> dict:
    """Send one MCQ record to llama-server. Returns raw response dict."""
    # Build content list — images first, then text
    content = []
    for img_path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_to_b64(img_path)}"
            }
        })
    content.append({"type": "text", "text": prompt})

    payload = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 16,      # MCQ answer is just one letter
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        BASE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return {"error": str(e)}


def extract_answer(response: dict) -> str:
    """Pull the model's letter answer out of the response."""
    try:
        text = response["choices"][0]["message"]["content"].strip()
        # Take just the first character in case model adds punctuation
        return text[0].upper() if text else ""
    except (KeyError, IndexError):
        return ""


def extract_tok_s(response: dict) -> float:
    """Extract generation tok/s from timings field."""
    try:
        return response["timings"]["predicted_per_second"]
    except (KeyError, TypeError):
        return 0.0


# ── Main loop ───────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  CMPE 249 Inference Run")
print(f"  Config : {args.config}")
print(f"  Server : {BASE_URL}")
print(f"  Limit  : {args.limit or 'all'}")
print(f"  Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")

for dataset_name in DATASETS:
    jsonl_path = DATA_DIR / dataset_name / "data.jsonl"
    if not jsonl_path.exists():
        print(f"[SKIP] {dataset_name} — {jsonl_path} not found")
        continue

    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))

    if args.limit:
        records = records[:args.limit]

    out_path = OUT_DIR / f"{args.config}__{dataset_name}.jsonl"
    print(f"[{dataset_name}] {len(records)} records → {out_path.name}")

    correct = 0
    total   = 0
    tok_s_list = []
    start_time = time.time()

    with open(out_path, "w") as out_f:
        for i, record in enumerate(records):
            prompt   = build_prompt(record["question"], record["choices"])
            response = call_server(record["image_paths"], prompt)

            predicted = extract_answer(response)
            tok_s     = extract_tok_s(response)

            # Ground truth: BLINK uses letter directly; CV-Bench same
            gt_answer = str(record["answer"]).strip().upper()
            # Handle cases where answer is an index (0,1,2,3) vs letter (A,B,C,D)
            if gt_answer.isdigit():
                gt_answer = ["A","B","C","D"][int(gt_answer)]

            is_correct = (predicted == gt_answer)
            if is_correct:
                correct += 1
            total += 1

            if tok_s > 0:
                tok_s_list.append(tok_s)

            result = {
                "idx":        record["idx"],
                "dataset":    dataset_name,
                "config":     args.config,
                "task":       record.get("task", ""),
                "question":   record["question"],
                "choices":    record["choices"],
                "gt_answer":  gt_answer,
                "predicted":  predicted,
                "correct":    is_correct,
                "tok_s":      tok_s,
                "image_paths": record["image_paths"],
            }
            out_f.write(json.dumps(result) + "\n")

            # Progress every 25 records
            if (i + 1) % 25 == 0 or (i + 1) == len(records):
                elapsed = time.time() - start_time
                acc = correct / total * 100
                avg_tok = sum(tok_s_list) / len(tok_s_list) if tok_s_list else 0
                print(f"  [{i+1:4d}/{len(records)}] acc={acc:.1f}%  "
                      f"avg_tok/s={avg_tok:.1f}  elapsed={elapsed:.0f}s")

    elapsed = time.time() - start_time
    final_acc = correct / total * 100 if total else 0
    avg_tok   = sum(tok_s_list) / len(tok_s_list) if tok_s_list else 0

    print(f"\n  ✓ {dataset_name} DONE")
    print(f"    Accuracy : {correct}/{total} = {final_acc:.2f}%")
    print(f"    Avg tok/s: {avg_tok:.1f}")
    print(f"    Time     : {elapsed:.0f}s\n")

print("All datasets complete.")
print(f"Results saved to: {OUT_DIR.resolve()}")
