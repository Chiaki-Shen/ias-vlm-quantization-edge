# TensorRT pipeline audit (Nov 2026)

Prof. Liu asked us to rule out implementation or calibration bugs before reading Table 2's gap (TensorRT
INT8 SmoothQuant / INT4 AWQ lose 4 to 6 points on ScienceQA where llama.cpp's Q8_0 / Q4_K_M do not) as a
property of the quantization methods. His three questions: was the calibration text only or real image +
question prompts; does each runtime use the same number of image tokens per image; is the chat template the
same. All three are answered below, plus a recalibration experiment that tests whether the calibration data
explains the gap. Board runs: Nov 26-27, Max's Jetson, 15W, desktop stopped (same conditions as T8 / T4).

## Conclusion

**No pipeline bug.** The template, tokenizer and output handling are identical across runtimes and engines.
The one image preprocessing difference is a rounding rule that affects 8% of ScienceQA images and does not
change accuracy. **The gap is not a calibration bug either, but INT8 SmoothQuant depends strongly on the
calibration data, while INT4 AWQ does not.** With calibration on in-domain image + question prompts, INT8
gains 2.8 points on ScienceQA and loses 1.6 on TextVQA. INT4 barely moves. Neither calibration set brings
INT8 to FP16 level on both datasets (it trails TensorRT FP16 by 3.4 to 6.2 points on ScienceQA, and by 0.7 to
2.3 on TextVQA), and INT4 trails by 3.5 to 5 points on both datasets with either set. That is a property of
SmoothQuant / AWQ on this model as Edge-LLM implements them, not of our pipeline.

| Check | Result | Evidence |
|---|---|---|
| 1. Calibration data | **Text only.** 512 CNN/DailyMail news articles (up to 512 tokens each), no images, no chat template, no question format. Edge-LLM v0.9.1 uses image + question calibration only when the vision encoder is also quantized (FP8, which Orin cannot run), so a language-only quantization always falls back to text. This is Edge-LLM's default, not a bug in our pipeline. | `logs-int8sq-quantize.log` ("Text calibration dataset: cnn_dailymail"); `tensorrt_edgellm/quantization/quantize.py` |
| 1b. Does the calibration explain the gap? | **Partly for INT8, not for INT4.** The recalibrated engines T8m / T4m use the same quantization with 128 in-domain image + prompt calibration pairs. See the table below. | `results/audit/recalibration.md` |
| 2. Image tokens | **One small, fully explained difference.** Both runtimes resize to multiples of 32 px. Edge-LLM rounds half to even (as the HF reference does) and llama.cpp rounds half up, so images with a side that is an odd multiple of 16 px get one more row or column of patches in llama.cpp. This rule predicts the image tokens of all 2,017 ScienceQA and 500 TextVQA records exactly, in both runtimes. It affects 164 ScienceQA images (8.1%, +3 to +25 tokens) and 1 of 500 TextVQA images. It does not change accuracy: T16 - C2 is +1.22 [-2.44, +4.88] on the affected records and +1.03 [-0.38, +2.37] on the rest. It is identical in T16 / T8 / T4, so it cannot explain the quantized drop. | `results/audit/tokens.md`, `results/audit/rounding_effect.txt` |
| 3. Chat template | **Same.** Edge-LLM's prompt is byte-identical to the model's own HF chat template (no system prompt, image before text), is the same in T16 / T8 / T4, and prefills exactly as many text tokens as llama.cpp. The re-saved T8 / T4 tokenizer file tokenizes all 9,922 benchmark prompts to the same ids as T16's. | `results/audit/template.md` |
| 4. Output handling (extra) | **No bug.** Every ScienceQA reply in all configs is a bare 2-token letter, with no truncation. TextVQA answer lengths, empty answers and stop reasons are the same across T16 / T8 / T8m / T4 / T4m. | `results/audit/outputs.md` |

## Recalibration results

T8m / T4m = the T8 / T4 quantization (same ModelOpt config: W8A8 SmoothQuant per-channel / W4A16 AWQ group 128,
same two excluded modules), calibrated on 64 ScienceQA train + 64 TextVQA train image + prompt pairs in this
project's prompt format and chat template (`tensorrt/quantize_mmcalib.py`, set in `tensorrt/calib/`). The
evaluation uses ScienceQA test and TextVQA validation, so there is no overlap. Paired change in points, 95%
bootstrap CI.

| | ScienceQA (2,017) | TextVQA (4,995) |
|---|---:|---:|
| T16 FP16 | 83.99 | 76.24 |
| T8 INT8 SQ, text calibration | 77.79 | 75.52 |
| T8m INT8 SQ, image + prompt calibration | 80.57 | 73.95 |
| T4 INT4 AWQ, text calibration | 79.03 | 71.95 |
| T4m INT4 AWQ, image + prompt calibration | 79.08 | 72.77 |
| C2 llama.cpp Q8_0 / C3 Q4_K_M | 82.94 / 83.29 | 76.41 / 74.82 |
| **T8 -> T8m** (recalibration, INT8) | **+2.78 [+1.14, +4.41]** | **-1.57 [-2.36, -0.78]** |
| **T4 -> T4m** (recalibration, INT4) | +0.05 [-1.29, +1.39] | +0.82 [-0.01, +1.63] |
| T16 -> T8m | **-3.42 [-4.66, -2.18]** | **-2.28 [-3.06, -1.55]** |
| T16 -> T4m | **-4.91 [-6.30, -3.52]** | **-3.47 [-4.31, -2.64]** |
| C2 -> T8m / C3 -> T4m | **-2.38** / **-4.21** | **-2.46** / **-2.05** |

- **INT8 SmoothQuant:** the calibration set moves accuracy by 1.5 to 3 points, in opposite directions on the
  two datasets. SmoothQuant quantizes activations with static per-tensor scales fixed at calibration time, so
  the calibration data decides which inputs are represented well. Neither calibration set gives FP16-level
  accuracy on both datasets: the best per dataset is -3.4 on ScienceQA (image + prompt) and -0.7 on TextVQA (text).
- **INT4 AWQ:** weight-only, and nearly insensitive to calibration (+0.05 and +0.82, both n.s.). Its 3.5 to 5
  point loss comes from the method on this model.
- **Caveat:** ScienceQA's calibration prompts come from its own train split, so +2.78 is a best case for an
  in-domain calibration set. Only 128 pairs were used (Edge-LLM's cap for multimodal calibration) vs 512 texts.

## What changes in the README

- Table 2 / Findings 2-3 are no longer preliminary. Finding 2 (TensorRT's quantized engines lose more at
  matched bit width) holds on both datasets with either calibration. Finding 3 should say the gap comes from
  the quantization method **and its calibration**, not the runtime: TensorRT FP16 matches llama.cpp Q8_0, and
  INT8's ScienceQA gap shrinks from 5.2 to 2.4 points (vs C2) with in-domain calibration but does not close.
- Caveat to add: llama.cpp and Edge-LLM round image sizes differently at .5 patch boundaries (8% of ScienceQA
  images get slightly more image tokens in llama.cpp); no measurable accuracy effect.

## Reproduce

```bash
# x86 host, Edge-LLM venv
python tensorrt/audit_template.py --workspace ~/edgellm-workspace
python tensorrt/quantize_mmcalib.py --calib_dir calib/mm128 --prepare
python tensorrt/quantize_mmcalib.py --calib_dir calib/mm128 --quantization int8_sq \
    --model_dir models/Cosmos-Reason2-2B --output_dir cr2b-int8sq-mm/quantized   # and int4_awq -> cr2b-int4awq-mm
tensorrt-edgellm-export cr2b-int8sq-mm/quantized cr2b-int8sq-mm/onnx --skip-visual
# Jetson: copy onnx/llm + check tensorrt/manifests/*-mm_onnx.sha256, then
bash tensorrt/jetson/run_audit.sh        # builds, T8m/T4m ScienceQA + TextVQA, token audit on both runtimes
# anywhere
python3 tensorrt/audit_outputs.py --out results/audit/outputs.md
python3 tensorrt/audit_recalibration.py
python3 tensorrt/audit_rounding_effect.py
```
