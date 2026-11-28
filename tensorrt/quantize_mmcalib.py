#!/usr/bin/env python3
"""Re-quantize the Cosmos-Reason2-2B language model with multimodal, in-domain calibration.

Audit follow-up (Prof. Liu: "did the calibration set match the multimodal VQA prompts?"). The original
T8 / T4 engines were calibrated the Edge-LLM way for a language-only quantization: 512 CNN/DailyMail
news articles, text only. Edge-LLM v0.9.1 only switches to image + question calibration when the
visual tower is also quantized (FP8, which Orin cannot run).

This script runs the same Edge-LLM `quantize_and_export` (same ModelOpt INT8 SmoothQuant / INT4 AWQ
configs, same export) with exactly two changes:
  1. calibration data: 128 (image, prompt) pairs from the TRAIN splits of ScienceQA (64, image
     questions only) and TextVQA (64), formatted with this project's own prompt builders, through the
     model's chat template (image first, then text), images capped at 1,024 image tokens like the
     engines. The evaluation uses ScienceQA test and TextVQA validation, so there is no overlap.
  2. the visual tower is NOT quantized: Edge-LLM's multimodal branch is entered by passing
     visual_quantization, and build_quant_config is wrapped to drop it again, so the quantizer config
     is identical to the original language-only run.

Usage (x86 host, Edge-LLM venv):
  python tensorrt/quantize_mmcalib.py --quantization int8_sq --model_dir models/Cosmos-Reason2-2B \
      --output_dir cr2b-int8sq-mm/quantized
Then: tensorrt-edgellm-export cr2b-int8sq-mm/quantized cr2b-int8sq-mm/onnx --skip-visual
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import run_experiment as rx  # noqa: E402  -- the exact prompt builders used at evaluation time

MAX_IMAGE_TOKENS = 1024          # visual engine --maxImageTokensPerImage
MIN_IMAGE_TOKENS = 4             # visual engine --minImageTokens
PATCH_PIXELS = 32 * 32           # Qwen3-VL: one image token per 32x32 pixels after merging


def prepare(calib_dir: Path, n_sqa: int, n_tvqa: int, seed: int) -> None:
    """Download the calibration pairs once: images as JPEG + prompts.jsonl (fixed, re-usable set)."""
    from datasets import load_dataset

    img_dir = calib_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def take(ds_id, n, make_prompt, meta):
        ds = load_dataset(ds_id, split="train", streaming=True).shuffle(seed=seed, buffer_size=2000)
        k = 0
        for row in ds:
            if row.get("image") is None or not row.get("question"):
                continue
            prompt = make_prompt(row)
            name = f"{len(rows):03d}.jpg"
            row["image"].convert("RGB").save(img_dir / name, quality=95)
            rows.append({"image": f"images/{name}", "prompt": prompt, "source": f"{ds_id}:train", **meta(row)})
            k += 1
            if k >= n:
                break

    take("derek-thomas/ScienceQA", n_sqa,
         lambda r: rx.mcq_prompt_context({"question": r["question"], "choices": r["choices"],
                                          "context": (r.get("hint") or "").strip()}),
         lambda r: {"question": r["question"]})
    take("lmms-lab/textvqa", n_tvqa, lambda r: rx.vqa_prompt_short({"question": r["question"]}),
         lambda r: {"question": r["question"], "question_id": r.get("question_id")})
    with open(calib_dir / "prompts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"saved {len(rows)} calibration pairs to {calib_dir}")


def load_pairs(calib_dir: Path):
    from PIL import Image
    rows = [json.loads(l) for l in open(calib_dir / "prompts.jsonl") if l.strip()]

    def gen():
        for r in rows:
            yield Image.open(calib_dir / r["image"]).convert("RGB"), r["prompt"]
    return rows, gen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib_dir", required=True, help="where the calibration images + prompts.jsonl live")
    ap.add_argument("--prepare", action="store_true", help="only download the calibration set to --calib_dir")
    ap.add_argument("--quantization", choices=["int8_sq", "int4_awq"])
    ap.add_argument("--model_dir")
    ap.add_argument("--output_dir")
    ap.add_argument("--n_sqa", type=int, default=64)
    ap.add_argument("--n_tvqa", type=int, default=64)
    ap.add_argument("--seed", type=int, default=249)
    args = ap.parse_args()
    calib_dir = Path(args.calib_dir)
    if args.prepare:
        prepare(calib_dir, args.n_sqa, args.n_tvqa, args.seed)
        return
    if not (args.quantization and args.model_dir and args.output_dir):
        ap.error("--quantization, --model_dir and --output_dir are required unless --prepare")

    from tensorrt_edgellm.quantization import quantize as q

    # (2) keep the visual tower unquantized while still taking the multimodal calibration branch.
    original_build = q.build_quant_config

    def build_without_visual(*a, **kw):
        kw["visual_quantization"] = None
        return original_build(*a, **kw)

    q.build_quant_config = build_without_visual

    # Images at the engines' resolution cap (the multimodal branch builds its own AutoProcessor).
    original_auto = q.AutoProcessor

    class CappedAutoProcessor:
        @staticmethod
        def from_pretrained(path, **kw):
            kw.setdefault("min_pixels", MIN_IMAGE_TOKENS * PATCH_PIXELS)
            kw.setdefault("max_pixels", MAX_IMAGE_TOKENS * PATCH_PIXELS)
            return original_auto.from_pretrained(path, **kw)

    q.AutoProcessor = CappedAutoProcessor

    rows, image_dataset = load_pairs(calib_dir)
    image_dataset.calib_name = f"{calib_dir.name} ({len(rows)} image+prompt pairs)"
    out = q.quantize_and_export(args.model_dir, args.output_dir, quantization=args.quantization,
                                visual_quantization="fp8",   # only selects the multimodal branch; see above
                                image_dataset=image_dataset, num_samples=len(rows))
    digest = hashlib.sha256((calib_dir / "prompts.jsonl").read_bytes()).hexdigest()
    Path(args.output_dir, "calibration_manifest.json").write_text(json.dumps(
        {"quantization": args.quantization, "calib_dir": str(calib_dir), "prompts_sha256": digest,
         "max_image_tokens": MAX_IMAGE_TOKENS, "n_samples": len(rows)}, indent=1))
    print(f"done: {out}  ({len(rows)} calibration samples)")


if __name__ == "__main__":
    main()
