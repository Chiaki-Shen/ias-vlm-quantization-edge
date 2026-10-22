"""
CMPE 249 -- TensorRT Edge-LLM experiment runner (Max).

Runs Cosmos-Reason2-2B TensorRT Edge-LLM engines over the same benchmark records as the
llama.cpp runner and writes result lines in the same format, so both runtimes merge into one
table (see the Shared Setup Agreement and tensorrt/NOTES.md).

Everything that decides a score -- dataset specs, prompts, parsers, scorers, the JSONL line
layout and the metrics -- is imported from scripts/run_experiment.py, not copied. Only the
inference backend differs: engines are loaded in-process through Edge-LLM's Python API
(experimental.server.LLM) instead of calling llama-server over HTTP.

Usage (on the Jetson, after building engines into <workspace>/<engine dir>/engines/{llm,visual}):
    export EDGE_LLM_PATH=$HOME/TensorRT-Edge-LLM
    python3 tensorrt/run_trt_experiment.py --config T8 --datasets blink_depth --limit 20
    python3 tensorrt/run_trt_experiment.py --config T8 --datasets scienceqa,textvqa --resume
    python3 tensorrt/run_trt_experiment.py --config T4 --dry-run     # checks only, no engines

Outputs go to results_trt/ (results_trt/smoke/ with --limit), never to results/.
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
import run_experiment as rx  # noqa: E402  -- shared prompts, parsers, scorers, line format

RUNTIME = "tensorrt-edgellm"
EDGE_LLM_VERSION = "v0.9.1"

# Matched pairs from the Shared Setup Agreement. "num" fills the same slot as llama.cpp's
# config numbers in file names; the T prefix keeps them from colliding with Configs 1-11.
CONFIGS = {
    "T8": {"num": "T8", "short": "TRT_F16v_INT8l", "engine_dir": "cr2b-int8sq",
           "label": "FP16 vision + INT8 SmoothQuant language (pairs with Config 2)"},
    "T4": {"num": "T4", "short": "TRT_F16v_INT4l", "engine_dir": "cr2b-int4awq",
           "label": "FP16 vision + INT4 AWQ language (pairs with Config 3)"},
    "T16": {"num": "T16", "short": "TRT_F16v_F16l", "engine_dir": "cr2b-fp16",
            "label": "FP16 vision + FP16 language (bonus, if it fits)"},
}


def sampling_settings(max_tokens: int) -> dict:
    """Greedy decoding, set explicitly. llama.cpp runs at temperature 0.0; Edge-LLM's own
    defaults (temperature 0.7, top_p 0.9) would make outputs non-deterministic."""
    return {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "max_tokens": max_tokens, "enable_thinking": False}


def build_messages(image_paths, prompt: str) -> list[dict]:
    # Same order as llama.cpp's call_server(): every image first, then the prompt text.
    content = [{"type": "image", "image": str(Path(p).resolve())} for p in image_paths]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def as_chat_response(text: str, finish_reason: str | None) -> dict:
    """Shape an Edge-LLM output like the OpenAI chat response rx.build_line() reads.

    tok_s stays 0.0: llama.cpp's predicted_per_second is decode-only throughput, which the
    Edge-LLM Python API does not report. Speed is compared with latency_ms instead."""
    return {"choices": [{"message": {"content": text}, "finish_reason": finish_reason}]}


class EdgeLLMBackend:
    """In-process Edge-LLM engines. Imported lazily so --dry-run works without TensorRT."""

    def __init__(self, engine_root: Path) -> None:
        edge_path = os.environ.get("EDGE_LLM_PATH", str(Path.home() / "TensorRT-Edge-LLM"))
        sys.path.insert(0, edge_path)
        from experimental.server import LLM, SamplingParams  # type: ignore[import-not-found]

        self._sampling_cls = SamplingParams
        self._llm = LLM(engine_dir=str(engine_root / "llm"), visual_engine_dir=str(engine_root / "visual"))

    def generate(self, image_paths, prompt: str, max_tokens: int) -> tuple[dict, float, int]:
        params = self._sampling_cls(**sampling_settings(max_tokens))
        t0 = time.perf_counter()
        out = self._llm.generate([build_messages(image_paths, prompt)], params)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return as_chat_response(out.text, out.finish_reason), latency_ms, len(out.token_ids)


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def output_paths(config: dict, ds_name: str, out_dir: Path) -> tuple[Path, Path]:
    stem = f"config{config['num']}_{config['short']}__{ds_name}"
    return out_dir / f"{stem}.jsonl", out_dir / f"{stem}.meta.json"


def load_records(ds_name: str, data_dir: Path, limit: int | None) -> tuple[list[dict], Path, list[str]]:
    """Load and validate one dataset with the llama.cpp runner's own checks."""
    spec = rx.DATASET_SPECS[ds_name]
    ds_dir = data_dir / ds_name
    jsonl = ds_dir / "data.jsonl"
    if not jsonl.exists():
        return [], ds_dir, [f"{ds_name}: {jsonl} not found"]
    records = rx.load_jsonl(jsonl)
    if limit:
        records = records[:limit]
    problems = [f"{ds_name} record {i}: {msg}" for i, r in enumerate(records)
                if (msg := rx.validate_record(spec, r, bool(spec.get("strict"))))]
    missing = [str(p) for r in records for p in rx.resolve_image_paths(r, ds_dir) if not rx.file_size_bytes(p)]
    if missing:
        problems.append(f"{ds_name}: {len(missing)} image file(s) missing/empty, e.g. {missing[:3]}")
    return records, ds_dir, problems


def done_indices(out_path: Path) -> set:
    """idx values already written, for --resume. Errors count as attempted, like the llama.cpp
    runner's default (it keeps them unless --retry-errors)."""
    if not out_path.exists():
        return set()
    return {line["idx"] for line in rx.load_jsonl(out_path)}


def run_dataset(backend, config: dict, ds_name: str, records: list[dict], ds_dir: Path,
                out_path: Path, resume: bool) -> list[dict]:
    spec = rx.DATASET_SPECS[ds_name]
    prompt_fn = rx.FORMATS[spec["format"]]["prompts"][spec["prompt"]]
    skip = done_indices(out_path) if resume else set()
    todo = [(i, r) for i, r in enumerate(records) if r["idx"] not in skip]
    print(f"  {ds_name}: {len(records)} records, {len(skip)} already done, {len(todo)} to run")

    with open(out_path, "a" if resume else "w", encoding="utf-8") as out_f:
        for k, (i, record) in enumerate(todo):
            img_paths = rx.resolve_image_paths(record, ds_dir)
            prompt = prompt_fn(record)
            error_msg, error_type, latency_ms, n_out = "", None, None, None
            try:
                response, latency_ms, n_out = backend.generate(img_paths, prompt, spec["max_tokens"])
            except Exception as exc:  # noqa: BLE001 -- one bad record must not stop the run
                response, error_msg, error_type = {}, str(exc), type(exc).__name__
            line = rx.build_line(config, ds_name, spec, record, i, response, False, rx.get_mem_mb()[0],
                                 error_msg=error_msg, error_type=error_type)
            line.update({"runtime": RUNTIME, "latency_ms": latency_ms, "n_output_tokens": n_out})
            out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
            out_f.flush()
            if (k + 1) % 50 == 0 or k + 1 == len(todo):
                print(f"    {k + 1}/{len(todo)}  {rx.mem_str()}")

    lines = rx.load_jsonl(out_path)
    return lines


def write_meta(meta_path: Path, config: dict, ds_name: str, ds_dir: Path, engine_root: Path, n_records: int) -> None:
    spec = rx.DATASET_SPECS[ds_name]
    meta = {
        "runtime": RUNTIME,
        "edge_llm_version": EDGE_LLM_VERSION,
        "edge_llm_rev": git_rev(Path(os.environ.get("EDGE_LLM_PATH", Path.home() / "TensorRT-Edge-LLM"))),
        "runner_rev": git_rev(PROJECT_ROOT),
        "config": config,
        "engine_root": str(engine_root),
        "dataset": ds_name,
        "data_sha256": rx.sha256_file(ds_dir / "data.jsonl"),
        "n_records": n_records,
        "protocol_version": rx.PROTOCOL_VERSION,
        "spec": spec,
        "sampling": sampling_settings(spec["max_tokens"]),
        "written_at": rx.now_pacific().isoformat(),
    }
    rx.write_json(meta_path, meta)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, choices=sorted(CONFIGS))
    p.add_argument("--datasets", default="blink_depth,blink_spatial,cv_bench,scienceqa,textvqa")
    p.add_argument("--workspace", type=Path, default=Path.home() / "tensorrt-edgellm-workspace",
                   help="contains <engine dir>/engines/{llm,visual}")
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "benchmarks")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results_trt")
    p.add_argument("--limit", type=int, help="smoke test: first N records, written to <out-dir>/smoke/")
    p.add_argument("--resume", action="store_true", help="skip records already in the output file")
    p.add_argument("--overwrite", action="store_true", help="replace existing output files")
    p.add_argument("--dry-run", action="store_true", help="validate data and print the plan; no engines")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = CONFIGS[args.config]
    engine_root = args.workspace / config["engine_dir"] / "engines"
    out_dir = args.out_dir / "smoke" if args.limit else args.out_dir
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in rx.DATASET_SPECS]
    if unknown:
        print(f"Unknown dataset(s): {unknown}. Known: {sorted(rx.DATASET_SPECS)}")
        return 2

    print(f"Config {config['num']}: {config['label']}\n  engines: {engine_root}\n  output:  {out_dir}")
    plans, problems = [], []
    for ds in datasets:
        records, ds_dir, ds_problems = load_records(ds, args.data_dir, args.limit)
        problems += ds_problems
        out_path, meta_path = output_paths(config, ds, out_dir)
        if out_path.exists() and not (args.resume or args.overwrite):
            problems.append(f"{out_path} exists; pass --resume or --overwrite")
        plans.append((ds, records, ds_dir, out_path, meta_path))
    if not args.dry_run and not (engine_root / "llm").is_dir():
        problems.append(f"LLM engine dir not found: {engine_root / 'llm'}")
    if problems:
        print("Pre-flight problems:\n  " + "\n  ".join(problems))
        return 1
    if args.dry_run:
        for ds, records, *_ in plans:
            print(f"  {ds}: {len(records)} records OK")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    backend = EdgeLLMBackend(engine_root)
    for ds, records, ds_dir, out_path, meta_path in plans:
        write_meta(meta_path, config, ds, ds_dir, engine_root, len(records))
        lines = run_dataset(backend, config, ds, records, ds_dir, out_path, args.resume)
        m = rx.compute_metrics(lines, rx.DATASET_SPECS[ds]["format"], target=len(records))
        score = m["score_raw"]
        latencies = sorted(x["latency_ms"] for x in lines if x.get("latency_ms") is not None)
        median = latencies[len(latencies) // 2] if latencies else None
        print(f"  {ds}: {m['metric']} {score:.4f} over {m['total']} attempted, {m['errors']} errors, "
              f"median latency {median:.0f} ms" if score is not None and median is not None
              else f"  {ds}: {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
