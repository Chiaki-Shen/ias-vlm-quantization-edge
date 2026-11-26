# TensorRT pipeline audit (Nov 2026)

Prof. Liu asked us to rule out implementation or calibration bugs before reading Table 2's gap (TensorRT
INT8 SmoothQuant / INT4 AWQ lose 4 to 6 points on ScienceQA where llama.cpp's Q8_0 / Q4_K_M do not) as a
property of the quantization methods. Three questions: was calibration text-only or real image + question
prompts; does each runtime use the same number of image tokens per image; is the chat template the same.

Status: questions 1 and 3 answered. Question 2 and the recalibration experiment wait for the board.

## Summary

| Check | Result | Evidence |
|---|---|---|
| 1. Calibration data | **Text only.** 512 CNN/DailyMail news articles (up to 512 tokens each), no images, no chat template, no question format. Edge-LLM v0.9.1 uses image + question calibration only when the vision encoder is also quantized (FP8, not supported on Orin), so a language-only quantization always falls back to text. Not a bug in our pipeline, but exactly the mismatch Prof. Liu suspected. | `logs-int8sq-quantize.log` ("Text calibration dataset: cnn_dailymail"); `tensorrt_edgellm/quantization/quantize.py` (`visual_quantization` branch) |
| 1b. Does it explain the gap? | **Experiment ready, waiting for the board.** T8 / T4 re-quantized with the same settings but calibrated on 128 in-domain image + prompt pairs (ScienceQA train, TextVQA train, our exact prompts and template): T8m / T4m. | `tensorrt/quantize_mmcalib.py`; quantizer config identical to T8 / T4 (same algorithm, same 2 excluded modules) |
| 2. Image tokens | **Pending (board).** Same visual engine and image-token cap (1,024) in T16, T8 and T4, so an image-token difference cannot explain a T16 vs T8 / T4 gap. Direct per-record measurement against llama.cpp is scripted. | `tensorrt/audit_tokens.py` |
| 3. Chat template | **Same.** Edge-LLM's prompt is byte-identical to the model's own HF chat template (no system prompt, image before text), identical in T16 / T8 / T4, and prefills exactly as many text tokens as llama.cpp on all 100 records with llama.cpp log counts. The re-saved T8 / T4 tokenizer file tokenizes all 9,922 benchmark prompts to the same ids as T16's. | `results/audit/template.md` (`tensorrt/audit_template.py`) |
| 4. Output handling (extra) | **No bug.** Every ScienceQA reply in all five configs is a bare letter of 2 tokens; no truncation; no parser disagreement beyond 1 to 4 records per config. | `results/audit/outputs.md` (`tensorrt/audit_outputs.py`) |

## Where the gap is (from the existing results)

- It is not concentrated in long text: T16 - T8 is +4.5 to +7.5 points in every prompt-length quartile.
- For INT8 it is largest on ScienceQA questions with **no** text context (+8.7 points, 760 questions), where
  the answer depends most on the image. INT8 SmoothQuant quantizes activations with static scales calibrated
  on news text, which never contained image-token embeddings; this is consistent with the calibration
  explanation. INT4 AWQ (weights only) loses about 5 points everywhere.
- T16 (same runtime, template, tokenizer and visual engine as T8 / T4) matches llama.cpp Q8_0 on ScienceQA and
  TextVQA, so the shared pipeline is sound; the drop appears only with the quantized language engines.

## To finish on the board

1. Token audit: `tensorrt/audit_tokens.py --runtime llamacpp` (llama.cpp container) and `--runtime trt`
   (Edge-LLM container), then `--compare` -> `results/audit/tokens.md`.
2. Copy `cr2b-int8sq-mm/onnx/llm` and `cr2b-int4awq-mm/onnx/llm` to `~/tensorrt-edgellm-workspace/`, check
   `onnx_manifest.sha256`, build with `tensorrt/jetson/build_llm_only.sh <cfg>` (INT8 needs the NVMe swap file).
3. `run_trt_experiment.py --config T8m` and `--config T4m` on ScienceQA and TextVQA (then CV-Bench / BLINK if
   time), paired against T8 / T4 / T16.
