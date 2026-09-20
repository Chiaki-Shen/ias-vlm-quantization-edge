"""
CMPE 249 — Benchmark Dataset Prep
Downloads BLINK (Relative_Depth + Spatial_Relation) and CV-Bench,
saves images to disk, writes inference-ready JSONL files.
"""

import json
from pathlib import Path
from datasets import load_dataset

BASE = Path(__file__).parent

# ── BLINK ──────────────────────────────────────────────────────────────────
# BLINK uses image_1..image_4 columns; most spatial tasks only use image_1

BLINK_TASKS = {
    "Relative_Depth":   "blink_depth",
    "Spatial_Relation": "blink_spatial",
}

for task_name, out_name in BLINK_TASKS.items():
    print(f"\n[BLINK] Downloading {task_name}...")
    ds = load_dataset("BLINK-Benchmark/BLINK", task_name, split="val")

    img_dir = BASE / out_name / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = BASE / out_name / "data.jsonl"

    with open(jsonl_path, "w") as f:
        for i, row in enumerate(ds):
            # Save all non-None images; most rows only have image_1
            image_paths = []
            for img_key in ["image_1", "image_2", "image_3", "image_4"]:
                img = row.get(img_key)
                if img is not None:
                    img_path = img_dir / f"{i:05d}_{img_key}.jpg"
                    if not img_path.exists():
                        img.convert("RGB").save(img_path, "JPEG")
                    image_paths.append(str(img_path))

            record = {
                "idx": i,
                "image_paths": image_paths,
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"],
                "task": task_name,
                "sub_task": row.get("sub_task", ""),
            }
            f.write(json.dumps(record) + "\n")

    print(f"  → {len(ds)} records saved to {jsonl_path}")

# ── CV-Bench ───────────────────────────────────────────────────────────────

print("\n[CV-Bench] Loading (already cached)...")
ds = load_dataset("nyu-visionx/CV-Bench", split="test")

img_dir = BASE / "cv_bench" / "images"
img_dir.mkdir(parents=True, exist_ok=True)
jsonl_path = BASE / "cv_bench" / "data.jsonl"

with open(jsonl_path, "w") as f:
    for i, row in enumerate(ds):
        img_path = img_dir / f"{i:05d}.jpg"
        if not img_path.exists():
            row["image"].convert("RGB").save(img_path, "JPEG")

        record = {
            "idx": i,
            "image_paths": [str(img_path)],
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "task": row["task"],
            "type": row["type"],
        }
        f.write(json.dumps(record) + "\n")

print(f"  → {len(ds)} records saved to {jsonl_path}")
print("\nDone. Folder structure:")
for p in sorted(BASE.rglob("data.jsonl")):
    lines = sum(1 for _ in open(p))
    print(f"  {p.relative_to(BASE)}  ({lines} records)")
