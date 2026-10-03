"""
CMPE 249 - TextVQA prep (Phase 3)

Builds TWO dataset folders in the format run_experiment.py (protocol p2.1) expects:

    <out-dir>/textvqa/        all 5,000 validation questions, seeded-shuffled order
    <out-dir>/textvqa_1500/   the first 1,500 records of that same shuffled order

Each folder contains:
    data.jsonl      one record per line, RELATIVE image_paths
    images/*.jpg
    manifest.json   provenance, hashed into the runner's resume fingerprint

Design notes
  * The 5,000 file is shuffled once with a fixed seed. The 1,500 subset is its
    exact prefix, so a subset is a uniform random sample of the validation set
    and idx values match across the two folders (idx 0..1499 are the same
    questions). Paired analysis can therefore key on idx for both.
  * A partly finished run of the full set is also a fair random sample.
  * Each record has exactly 10 human answers (the runner rejects anything else).
  * ocr_tokens and other reference annotations are NOT written (not model input).
  * Images are re-encoded as RGB JPEG (quality 95). No resizing is applied.

Usage (Mac, conda env cmpe249, use `python`, not `python3`):
    python scripts/prep_textvqa.py --limit 30 --subset-size 10 --out-dir /tmp/tv_trial
    python scripts/prep_textvqa.py --out-dir /Users/Aeon/cmpe249-benchmarks
    python scripts/prep_textvqa.py --overwrite      # rebuild (invalidates resumable results)
"""

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DATASET_ID = "lmms-lab/textvqa"   # parquet. facebook/textvqa is script-based and fails on datasets>=4
SPLIT = "validation"
EXPECTED_COUNT = 5000
SEED = 249
JPEG_QUALITY = 95


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_record(idx: int, orig_idx: int, row: dict, rel_image: str) -> dict:
    answers = row["answers"]
    if not (isinstance(answers, list) and len(answers) == 10 and all(isinstance(a, str) for a in answers)):
        raise ValueError(f"orig_idx {orig_idx}: answers must be exactly 10 strings, got {answers!r}")
    return {
        "idx": idx,
        "orig_idx": orig_idx,                       # row position in the HF validation split
        "question_id": row.get("question_id"),
        "image_id": row.get("image_id"),
        "format": "open_vqa",
        "image_paths": [rel_image],
        "question": str(row["question"]).strip(),
        "answers": answers,
        "task": "textvqa",
    }


def write_jsonl(path: Path, records: list):
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_manifest(folder: Path, records: list, extra: dict, ds_version: str):
    manifest = {
        "source": DATASET_ID,
        "split": SPLIT,
        "record_count": len(records),
        "answers_per_question": 10,
        "image_format": f"JPEG quality {JPEG_QUALITY}, RGB, no resize",
        "shuffle_seed": SEED,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "prep_script": "prep_textvqa.py",
        "prep_commit": git_commit(),
        "datasets_version": ds_version,
        "data_jsonl_sha256": sha256_file(folder / "data.jsonl"),
    }
    manifest.update(extra)
    with open(folder / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def validate_folder(folder: Path, records: list):
    ids = [r["idx"] for r in records]
    assert len(ids) == len(set(ids)), f"{folder.name}: duplicate idx"
    for r in records:
        assert not Path(r["image_paths"][0]).is_absolute(), "image path must be relative"
        p = folder / r["image_paths"][0]
        assert p.exists() and p.stat().st_size > 0, f"missing/empty image: {p}"
        assert len(r["answers"]) == 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="./benchmarks", help="parent folder for textvqa/ and textvqa_1500/")
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--subset-size", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=None, help="trial runs: keep only N records in the full set")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset
    import datasets as _ds

    parent = Path(args.out_dir).expanduser().resolve()
    full_dir = parent / "textvqa"
    sub_name = f"textvqa_{args.subset_size}"
    sub_dir = parent / sub_name

    for d in (full_dir, sub_dir):
        if (d / "data.jsonl").exists() and not args.overwrite:
            sys.exit(f"[ABORT] {d / 'data.jsonl'} already exists. Pass --overwrite to rebuild.")

    print(f"[TextVQA] loading {args.dataset_id} split={SPLIT} ...")
    ds = load_dataset(args.dataset_id, split=SPLIT)
    missing = {"image", "question", "answers"} - set(ds.column_names)
    if missing:
        sys.exit(f"[ABORT] dataset is missing columns {sorted(missing)}; found {ds.column_names}")
    for d in (full_dir, sub_dir):                    # created only after a successful load
        (d / "images").mkdir(parents=True, exist_ok=True)
    n_total = len(ds)
    print(f"  validation rows: {n_total}")
    if n_total != EXPECTED_COUNT:
        print(f"  [WARN] expected {EXPECTED_COUNT} rows, got {n_total}")

    order = list(range(n_total))
    random.Random(SEED).shuffle(order)               # deterministic across machines
    if args.limit is not None:
        order = order[:args.limit]

    records = []
    for idx, orig_idx in enumerate(order):
        row = ds[orig_idx]
        rel = f"images/{idx:05d}.jpg"
        row["image"].convert("RGB").save(full_dir / rel, "JPEG", quality=JPEG_QUALITY)
        records.append(build_record(idx, orig_idx, row, rel))
        if (idx + 1) % 500 == 0:
            print(f"  processed {idx + 1} / {len(order)}")

    validate_folder(full_dir, records)
    write_jsonl(full_dir / "data.jsonl", records)

    # ---- subset: exact prefix of the shuffled full set (same idx, same image names) ----
    sub_records = records[:args.subset_size]
    for r in sub_records:
        shutil.copy2(full_dir / r["image_paths"][0], sub_dir / r["image_paths"][0])
    validate_folder(sub_dir, sub_records)
    write_jsonl(sub_dir / "data.jsonl", sub_records)

    m_full = write_manifest(full_dir, records, {
        "name": "textvqa",
        "selection": "all validation questions, seeded shuffle",
        "limit": args.limit,
    }, _ds.__version__)
    m_sub = write_manifest(sub_dir, sub_records, {
        "name": sub_name,
        "selection": f"first {len(sub_records)} records of the seeded shuffle of textvqa (exact prefix)",
        "parent_dataset": "textvqa",
        "parent_data_jsonl_sha256": m_full["data_jsonl_sha256"],
        "limit": args.limit,
    }, _ds.__version__)

    print(f"\nDone.")
    print(f"  {full_dir}  ({len(records)} records)  sha256 {m_full['data_jsonl_sha256']}")
    print(f"  {sub_dir}  ({len(sub_records)} records)  sha256 {m_sub['data_jsonl_sha256']}")


if __name__ == "__main__":
    main()
