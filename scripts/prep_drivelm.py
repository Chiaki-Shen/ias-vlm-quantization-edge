"""
CMPE 249 - DriveLM driving-scene check prep (handoff_10, Phase 1)

Builds benchmarks/drivelm/ in the format run_experiment.py (protocol p2.1) expects:

    drivelm/
        data.jsonl       one record per line, RELATIVE image_paths
        images/00000.jpg ...
        manifest.json    provenance, hashed into the resume fingerprint

Source: OpenDriveLab/DriveLM on Hugging Face (gated), DriveLM-nuScenes train split
    v1_1_train_nus.json            (QA, 696 scenes / 4,072 key frames)
    drivelm_nus_imgs_train.zip     (nuScenes camera subset, unzipped to nuscenes/samples/)
Only the train split has public answers; we evaluate on it, no training happens.

Selection: front camera (CAM_FRONT) only, one image per record, closed answers only,
scored by exact match like ScienceQA (format "mcq", prompt "context", parser v2).

  task "traffic_light"  (150)  DriveLM question "What is the traffic signal that the
        ego vehicle should pay attention to?" restricted to frames whose answer is
        Red / Green / Yellow light. All yellow frames (36) + an equal red/green split
        of the rest. 3-way MCQ; choice order shuffled per record with the fixed seed.
        "None" answers are dropped (a light can be visible but not marked relevant).
        Image is the original nuScenes JPEG bytes (no re-encode).

  task "sign_box"  (150)  DriveLM question "Is <obj> a traffic sign or a road
        barrier?" for CAM_FRONT objects. The object is drawn as a red box from
        key_object_infos[...]["2d_bbox"] and the question asks about "the object in
        the red box". Balanced 75 Yes / 75 No, at most one object per frame.
        Choices fixed as A. Yes / B. No. Image re-encoded JPEG q95 after drawing.

Fixed seed, deterministic order. Records are sorted by task, then sampled order.

Usage (Mac, conda env cmpe249):
    python scripts/prep_drivelm.py --raw-dir ~/cmpe249-benchmarks/drivelm_raw \
                                   --out-dir ~/cmpe249-benchmarks
    python scripts/prep_drivelm.py ... --max-side 1280   # optional downscale (long side)
    python scripts/prep_drivelm.py ... --overwrite        # rebuild an existing folder
"""

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SOURCE = "OpenDriveLab/DriveLM (DriveLM-nuScenes, train)"
SOURCE_URL = "https://huggingface.co/datasets/OpenDriveLab/DriveLM"
QA_FILE = "v1_1_train_nus.json"
SEED = 249
CAMERA = "CAM_FRONT"
JPEG_QUALITY = 95
BOX_COLOR = (255, 0, 0)
BOX_WIDTH = 4

N_LIGHT = 150
N_SIGN = 150                      # half Yes, half No
LIGHT_LABELS = {"Red light.": "Red light", "Green light.": "Green light", "Yellow light.": "Yellow light"}
LIGHT_Q = "What is the traffic signal that the ego vehicle should pay attention to?"
SIGN_Q_SRC = re.compile(r"^Is (<c\d+,CAM_FRONT,[\d.]+,[\d.]+>) a traffic sign or a road barrier\?$")
SIGN_Q = "Is the object in the red box a traffic sign or a road barrier?"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def resolve_image(raw_dir: Path, rel: str) -> Path:
    # JSON paths look like "../nuscenes/samples/CAM_FRONT/xxx.jpg"
    p = raw_dir / "nuscenes" / rel.split("nuscenes/", 1)[1]
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def collect_candidates(qa: dict):
    lights, signs = [], []
    for scene_id, scene in qa.items():
        for frame_id, frame in scene["key_frames"].items():
            img = frame["image_paths"][CAMERA]
            for cat, qs in frame["QA"].items():
                for q in qs:
                    Q, A = q["Q"].strip(), q["A"].strip()
                    if Q == LIGHT_Q and A in LIGHT_LABELS:
                        lights.append({"scene": scene_id, "frame": frame_id, "src_image": img,
                                       "label": LIGHT_LABELS[A], "src_q": Q, "src_a": A})
                    m = SIGN_Q_SRC.match(Q)
                    if m and A in ("Yes.", "No."):
                        tag = m.group(1)
                        info = frame["key_object_infos"].get(tag)
                        if not info or not info.get("2d_bbox"):
                            continue
                        signs.append({"scene": scene_id, "frame": frame_id, "src_image": img,
                                      "tag": tag, "bbox": [float(v) for v in info["2d_bbox"]],
                                      "obj_category": info.get("Category", ""),
                                      "obj_description": info.get("Visual_description", ""),
                                      "label": A.rstrip("."), "src_q": Q, "src_a": A})
    return lights, signs


def sample_lights(lights, rng):
    by = {k: [c for c in lights if c["label"] == k] for k in LIGHT_LABELS.values()}
    for v in by.values():
        rng.shuffle(v)
    yellow = by["Yellow light"][:N_LIGHT]
    rest = N_LIGHT - len(yellow)
    n_red = rest // 2 + rest % 2
    n_green = rest - n_red
    if len(by["Red light"]) < n_red or len(by["Green light"]) < n_green:
        sys.exit("[ABORT] not enough red/green traffic light questions")
    picked = yellow + by["Red light"][:n_red] + by["Green light"][:n_green]
    rng.shuffle(picked)
    return picked


def sample_signs(signs, rng):
    half = N_SIGN // 2
    out = []
    for label in ("Yes", "No"):
        pool = [c for c in signs if c["label"] == label]
        rng.shuffle(pool)
        seen_frames = set()
        for c in pool:
            if c["frame"] in seen_frames:
                continue
            seen_frames.add(c["frame"])
            out.append(c)
            if sum(1 for o in out if o["label"] == label) == half:
                break
        if sum(1 for o in out if o["label"] == label) < half:
            sys.exit(f"[ABORT] not enough distinct-frame sign_box '{label}' candidates")
    rng.shuffle(out)
    return out


def save_image(src: Path, dst: Path, bbox=None, max_side=None):
    """Copy original bytes when nothing changes; otherwise draw / resize and re-encode."""
    if bbox is None and max_side is None:
        shutil.copyfile(src, dst)
        from PIL import Image
        with Image.open(dst) as im:
            return im.size
    from PIL import Image, ImageDraw
    with Image.open(src) as im:
        im = im.convert("RGB")
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            ImageDraw.Draw(im).rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
        if max_side is not None and max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        im.save(dst, "JPEG", quality=JPEG_QUALITY)
        return im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="~/cmpe249-benchmarks/drivelm_raw",
                    help="folder with v1_1_train_nus.json and nuscenes/samples/")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "benchmarks"),
                    help="parent folder; drivelm/ is created inside")
    ap.add_argument("--max-side", type=int, default=None,
                    help="optional: downscale so the long side is at most this many px")
    ap.add_argument("--overwrite", action="store_true", help="rebuild if drivelm/ already has a data.jsonl")
    args = ap.parse_args()

    raw = Path(args.raw_dir).expanduser().resolve()
    root = Path(args.out_dir).expanduser().resolve() / "drivelm"
    img_dir = root / "images"
    data_path = root / "data.jsonl"
    manifest_path = root / "manifest.json"

    if data_path.exists() and not args.overwrite:
        sys.exit(f"[ABORT] {data_path} already exists. Pass --overwrite to rebuild.")
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    qa_path = raw / QA_FILE
    print(f"[DriveLM] loading {qa_path} ...")
    qa = json.load(open(qa_path, encoding="utf-8"))
    lights, signs = collect_candidates(qa)
    print(f"  candidates: traffic_light {len(lights)} {dict(Counter(c['label'] for c in lights))}, "
          f"sign_box {len(signs)} {dict(Counter(c['label'] for c in signs))}")

    rng = random.Random(SEED)
    picked_lights = sample_lights(lights, rng)
    picked_signs = sample_signs(signs, rng)

    records = []
    sizes = Counter()
    for task, items in (("traffic_light", picked_lights), ("sign_box", picked_signs)):
        for c in items:
            idx = len(records)
            rel = f"images/{idx:05d}.jpg"
            src = resolve_image(raw, c["src_image"])
            size = save_image(src, root / rel, bbox=c.get("bbox"), max_side=args.max_side)
            sizes[size] += 1
            if task == "traffic_light":
                choices = list(LIGHT_LABELS.values())
                rng.shuffle(choices)
                question = LIGHT_Q
            else:
                choices = ["Yes", "No"]
                question = SIGN_Q
            rec = {
                "idx": idx,
                "orig_idx": f"{c['scene']}/{c['frame']}",
                "format": "mcq",
                "image_paths": [rel],
                "question": question,
                "choices": choices,
                "answer": choices.index(c["label"]),     # index into choices
                "context": "",
                "task": task,
                "camera": CAMERA,
                "src_image": Path(c["src_image"]).name,
                "src_question": c["src_q"],
                "src_answer": c["src_a"],
            }
            if task == "sign_box":
                rec.update({"bbox": c["bbox"], "obj_tag": c["tag"],
                            "obj_category": c["obj_category"],
                            "obj_description": c["obj_description"]})
            records.append(rec)

    # ---- validation (mirrors what the runner enforces for strict datasets) ----
    ids = [r["idx"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate idx"
    for r in records:
        p = root / r["image_paths"][0]
        assert p.exists() and p.stat().st_size > 0, f"missing/empty image: {p}"
        assert not Path(r["image_paths"][0]).is_absolute()
        assert 0 <= r["answer"] < len(r["choices"])
    task_counts = Counter(r["task"] for r in records)
    label_counts = Counter((r["task"], r["choices"][r["answer"]]) for r in records)
    letter_counts = Counter((r["task"], "ABC"[r["answer"]]) for r in records)
    print(f"  kept {len(records)} records: {dict(task_counts)}")
    print(f"  labels: {dict(label_counts)}")
    print(f"  answer letters: {dict(letter_counts)}")
    print(f"  image sizes: {dict(sizes)}")

    tmp = data_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(data_path)

    import PIL
    manifest = {
        "name": "drivelm",
        "source": SOURCE,
        "source_url": SOURCE_URL,
        "qa_file": QA_FILE,
        "qa_file_sha256": sha256_file(qa_path),
        "split": "train (only public split with answers; evaluation only)",
        "camera": CAMERA,
        "seed": SEED,
        "filter": {
            "traffic_light": "LIGHT_Q with answer Red/Green/Yellow light; all yellow + equal red/green; "
                             "choices shuffled per record",
            "sign_box": "'Is <obj> a traffic sign or a road barrier?' with CAM_FRONT object; "
                        "75 Yes / 75 No; one object per frame; red box drawn from 2d_bbox",
        },
        "record_count": len(records),
        "task_counts": dict(task_counts),
        "label_counts": {f"{t}:{l}": n for (t, l), n in label_counts.items()},
        "answer_format": "index into choices (mcq, exact match)",
        "image_format": {
            "traffic_light": "original nuScenes JPEG bytes" if args.max_side is None
                             else f"resized long side <= {args.max_side}, JPEG q{JPEG_QUALITY}",
            "sign_box": f"red box width {BOX_WIDTH}px drawn, JPEG q{JPEG_QUALITY}"
                        + ("" if args.max_side is None else f", resized long side <= {args.max_side}"),
        },
        "max_side": args.max_side,
        "image_sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "prep_script": "prep_drivelm.py",
        "prep_commit": git_commit(),
        "pillow_version": PIL.__version__,
        "data_jsonl_sha256": sha256_file(data_path),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\nDone: {root}")
    print(f"  data.jsonl sha256: {manifest['data_jsonl_sha256']}")


if __name__ == "__main__":
    main()
