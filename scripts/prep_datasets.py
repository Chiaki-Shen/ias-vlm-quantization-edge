"""
CMPE 249 - BLINK + CV-Bench prep

Builds three dataset folders in the format run_experiment.py (protocol p2.1) expects:

    <out-dir>/blink_depth/     BLINK Relative_Depth, val split   (124 records)
    <out-dir>/blink_spatial/   BLINK Spatial_Relation, val split (143 records)
    <out-dir>/cv_bench/        CV-Bench, test split              (2,638 records)

Each folder contains data.jsonl (one record per line) and images/*.jpg.
<out-dir> defaults to benchmarks/ at the repo root, wherever you run the script from.

Image paths
  * Default: RELATIVE paths ("images/00000.jpg"), resolved by the runners against the
    dataset folder. The folder can be copied anywhere.
  * --legacy-abs-root DIR: write ABSOLUTE paths under DIR, the way the original script did.
    The frozen copies used for every published result were built this way with
    DIR = /Developer/ias-vlm-quantization-edge/benchmarks. Use this only to try to
    reproduce those exact data.jsonl hashes (listed in the README).

Record fields, their order, the JSON formatting and the JPEG settings (Pillow default
quality) are unchanged from the original script, so a rebuild differs from the frozen
files only in the image paths, unless the Hugging Face datasets or Pillow changed since.
No manifest.json is written for these three folders, matching the frozen copies (the
runner's fingerprint records manifest_sha256 = null for them).

Usage (from anywhere; use the Python that has `datasets` installed):
    pip install datasets Pillow
    python scripts/prep_datasets.py
    python scripts/prep_datasets.py --out-dir /tmp/bench_trial
    python scripts/prep_datasets.py --legacy-abs-root /Developer/ias-vlm-quantization-edge/benchmarks
    python scripts/prep_datasets.py --overwrite      # rebuild existing folders
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BLINK_ID = "BLINK-Benchmark/BLINK"
BLINK_SPLIT = "val"
BLINK_TASKS = {                       # HF config name -> output folder
    "Relative_Depth":   "blink_depth",
    "Spatial_Relation": "blink_spatial",
}
CVBENCH_ID = "nyu-visionx/CV-Bench"
CVBENCH_SPLIT = "test"

EXPECTED_COUNTS = {"blink_depth": 124, "blink_spatial": 143, "cv_bench": 2638}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def path_for_record(rel: str, out_name: str, abs_root):
    """Relative path by default; absolute under abs_root/<out_name> in legacy mode."""
    if abs_root is None:
        return rel
    return str(Path(abs_root) / out_name / rel)


def save_jpeg(img, path: Path) -> None:
    # Pillow default JPEG quality, exactly as the original script, so images match.
    if not path.exists():
        img.convert("RGB").save(path, "JPEG")


def write_jsonl(path: Path, records: list) -> None:
    # json.dumps defaults (ensure_ascii=True), as the original script; keeps hashes stable.
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def validate_folder(folder: Path, records: list) -> None:
    ids = [r["idx"] for r in records]
    assert len(ids) == len(set(ids)), f"{folder.name}: duplicate idx"
    for r in records:
        assert r["image_paths"], f"{folder.name} idx {r['idx']}: no images"
        for p in r["image_paths"]:
            if Path(p).is_absolute():
                continue            # legacy mode: the path is for the Jetson, not this machine
            f = folder / p
            assert f.exists() and f.stat().st_size > 0, f"missing/empty image: {f}"


def build_blink(load_dataset, task_name: str, out_name: str, root: Path, abs_root) -> list:
    print(f"\n[BLINK] {task_name} -> {out_name}")
    ds = load_dataset(BLINK_ID, task_name, split=BLINK_SPLIT)
    folder = root / out_name
    (folder / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(ds):
        image_paths = []
        for img_key in ["image_1", "image_2", "image_3", "image_4"]:   # most rows use only image_1
            img = row.get(img_key)
            if img is not None:
                rel = f"images/{i:05d}_{img_key}.jpg"
                save_jpeg(img, folder / rel)
                image_paths.append(path_for_record(rel, out_name, abs_root))
        records.append({
            "idx": i,
            "image_paths": image_paths,
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "task": task_name,
            "sub_task": row.get("sub_task", ""),
        })
    return records


def build_cvbench(load_dataset, root: Path, abs_root) -> list:
    print("\n[CV-Bench] test -> cv_bench")
    ds = load_dataset(CVBENCH_ID, split=CVBENCH_SPLIT)
    folder = root / "cv_bench"
    (folder / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(ds):
        rel = f"images/{i:05d}.jpg"
        save_jpeg(row["image"], folder / rel)
        records.append({
            "idx": i,
            "image_paths": [path_for_record(rel, "cv_bench", abs_root)],
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "task": row["task"],
            "type": row["type"],
        })
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "benchmarks"),
                    help="parent folder for blink_depth/, blink_spatial/, cv_bench/ (default: <repo>/benchmarks)")
    ap.add_argument("--legacy-abs-root", default=None, metavar="DIR",
                    help="write absolute image paths under DIR like the frozen copies "
                         "(e.g. /Developer/ias-vlm-quantization-edge/benchmarks)")
    ap.add_argument("--overwrite", action="store_true", help="rebuild folders that already have a data.jsonl")
    args = ap.parse_args(argv)

    root = Path(args.out_dir).expanduser().resolve()
    names = list(BLINK_TASKS.values()) + ["cv_bench"]
    existing = [n for n in names if (root / n / "data.jsonl").exists()]
    if existing and not args.overwrite:
        sys.exit(f"[ABORT] data.jsonl already exists in {root} for {existing}. "
                 f"Pass --overwrite to rebuild.")
    if args.legacy_abs_root and not Path(args.legacy_abs_root).is_absolute():
        sys.exit("[ABORT] --legacy-abs-root must be an absolute path")

    from datasets import load_dataset   # imported late so --help works without it

    built = {}
    for task_name, out_name in BLINK_TASKS.items():
        built[out_name] = build_blink(load_dataset, task_name, out_name, root, args.legacy_abs_root)
    built["cv_bench"] = build_cvbench(load_dataset, root, args.legacy_abs_root)

    print(f"\nDone: {root}")
    for name, records in built.items():
        folder = root / name
        validate_folder(folder, records)
        write_jsonl(folder / "data.jsonl", records)
        warn = ""
        if len(records) != EXPECTED_COUNTS[name]:
            warn = f"  [WARN] expected {EXPECTED_COUNTS[name]} records"
        print(f"  {name:<14} {len(records):>5} records  sha256 {sha256_file(folder / 'data.jsonl')}{warn}")
    if args.legacy_abs_root:
        print(f"\n  Image paths point to {args.legacy_abs_root}/<dataset>/images. Copy the folders there "
              f"before running.")


if __name__ == "__main__":
    main()
