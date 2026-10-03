"""
CMPE 249 - ScienceQA prep (Phase 3)

Builds benchmarks/scienceqa/ in the format run_experiment.py (protocol p2.1) expects:

    benchmarks/scienceqa/
        data.jsonl       one record per line, RELATIVE image_paths
        images/00000.jpg ...
        manifest.json    provenance, hashed into the resume fingerprint

Selection: derek-thomas/ScienceQA, split "test", image-bearing questions only
(expected 2,017 records).

Leakage rule: only the `hint` is exposed (as `context`). `lecture` and
`solution` are never written to data.jsonl.

Usage (on the Mac):
    pip install datasets pillow
    python3 prep_scienceqa.py                       # writes ./benchmarks/scienceqa
    python3 prep_scienceqa.py --out-dir ~/cmpe249/benchmarks
    python3 prep_scienceqa.py --limit 20 --out-dir /tmp/sqa_test   # quick trial
    python3 prep_scienceqa.py --overwrite           # rebuild an existing folder
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DATASET_ID = "derek-thomas/ScienceQA"
SOURCE_URL = "https://huggingface.co/datasets/derek-thomas/ScienceQA"
SPLIT = "test"
EXPECTED_COUNT = 2017
JPEG_QUALITY = 95


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_record(idx: int, orig_idx: int, row: dict, rel_image: str) -> dict:
    choices = [str(c) for c in row["choices"]]
    answer = int(row["answer"])
    if len(choices) < 2:
        raise ValueError(f"orig_idx {orig_idx}: fewer than 2 choices")
    if not (0 <= answer < len(choices)):
        raise ValueError(f"orig_idx {orig_idx}: answer {answer} outside {len(choices)} choices")
    return {
        "idx": idx,
        "orig_idx": orig_idx,
        "format": "mcq",
        "image_paths": [rel_image],
        "question": str(row["question"]).strip(),
        "choices": choices,
        "answer": answer,                       # index into choices
        "context": (row.get("hint") or "").strip(),   # hint only, never lecture/solution
        "task": "scienceqa",
        "subject": row.get("subject", ""),
        "grade": row.get("grade", ""),
        "category": row.get("category", ""),
        "skill": row.get("skill", ""),
        "topic": row.get("topic", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="./benchmarks", help="parent folder; scienceqa/ is created inside")
    ap.add_argument("--limit", type=int, default=None, help="keep only the first N image records (trial runs)")
    ap.add_argument("--overwrite", action="store_true", help="rebuild if scienceqa/ already has a data.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset  # imported late so --help works without it

    root = Path(args.out_dir).expanduser().resolve() / "scienceqa"
    img_dir = root / "images"
    data_path = root / "data.jsonl"
    manifest_path = root / "manifest.json"

    if data_path.exists() and not args.overwrite:
        sys.exit(f"[ABORT] {data_path} already exists. Pass --overwrite to rebuild.")
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ScienceQA] loading {DATASET_ID} split={SPLIT} ...")
    ds = load_dataset(DATASET_ID, split=SPLIT)
    print(f"  total test rows: {len(ds)}")

    records = []
    kept = 0
    for orig_idx, row in enumerate(ds):
        img = row.get("image")
        if img is None:
            continue                            # image-only subset
        if args.limit is not None and kept >= args.limit:
            break
        rel = f"images/{kept:05d}.jpg"
        out_img = root / rel
        img.convert("RGB").save(out_img, "JPEG", quality=JPEG_QUALITY)
        records.append(build_record(kept, orig_idx, row, rel))
        kept += 1
        if kept % 500 == 0:
            print(f"  processed {kept} image records ...")

    # ---- validation (mirrors what the runner enforces) ----
    ids = [r["idx"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate idx"
    for r in records:
        p = root / r["image_paths"][0]
        assert p.exists() and p.stat().st_size > 0, f"missing/empty image: {p}"
        assert not Path(r["image_paths"][0]).is_absolute(), "image path must be relative"
    n_hint = sum(1 for r in records if r["context"])
    print(f"  kept {len(records)} image records ({n_hint} with a hint)")
    if args.limit is None and len(records) != EXPECTED_COUNT:
        print(f"  [WARN] expected {EXPECTED_COUNT} records, got {len(records)}. "
              f"Check the dataset revision before running anything official.")

    # ---- write data.jsonl (utf-8, one record per line, trailing newline) ----
    tmp = data_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(data_path)

    # ---- manifest ----
    import datasets as _ds
    manifest = {
        "name": "scienceqa",
        "source": DATASET_ID,
        "source_url": SOURCE_URL,
        "split": SPLIT,
        "filter": "image questions only",
        "record_count": len(records),
        "records_with_hint": n_hint,
        "answer_format": "index into choices",
        "context_policy": "hint only; lecture and solution excluded",
        "image_format": f"JPEG quality {JPEG_QUALITY}, RGB",
        "limit": args.limit,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "prep_script": "prep_scienceqa.py",
        "prep_commit": git_commit(),
        "datasets_version": _ds.__version__,
        "data_jsonl_sha256": sha256_file(data_path),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\nDone: {root}")
    print(f"  data.jsonl sha256: {manifest['data_jsonl_sha256']}")


if __name__ == "__main__":
    main()
