# TensorRT Edge-LLM pipeline (Max)

The TensorRT side of the llama.cpp vs TensorRT-LLM comparison. Matched configs and shared
settings follow the *llama.cpp vs TensorRT-LLM: Shared Setup Agreement* (Oct 8, 2026).

| Pair | llama.cpp (Aeon) | TensorRT Edge-LLM (this folder) |
|---|---|---|
| 8-bit language | Config 2: F16 vision + Q8_0 language | FP16 vision + INT8 SmoothQuant language |
| 4-bit language | Config 3: F16 vision + Q4_K_M language | FP16 vision + INT4 AWQ language |
| bonus | (F16 + F16 does not fit) | FP16 vision + FP16 language, if it fits |

## Versions

| Item | Value | Why |
|---|---|---|
| TensorRT Edge-LLM | **v0.9.1** (`7f061f2`) | Last release that supports JetPack 6.2 on Orin ("JetPack 6.2+ remains compatible for FP16, INT8, and INT4 workflows"). v0.10+ require JetPack 7.2 / CUDA 13 on Orin, which would put the two boards on different stacks. Lists `nvidia/Cosmos-Reason2-2B` as supported. |
| JetPack on the Jetson | 6.2 (L4T R36.4.x), same as Aeon's board (R36.4.3) | Agreement: same software stack on both boards |
| Source checkpoint | `nvidia/Cosmos-Reason2-2B` @ `9ce19a195e423419c349abfc86fd07178b230561` | Original BF16 weights; `model.safetensors` SHA-256 `fa5a6e6ef4fce40216b185cc48a3b24d31637ac3e2ba69c107ed1f389c1e6ede` |
| Export host | x86-64 (WSL2 Ubuntu), RTX 4080 Laptop 12 GB (SM89), Python 3.12, torch 2.12.0+cu130, transformers 5.9.0, nvidia-modelopt 0.44.0 | Quantization and ONNX export run on x86; engines are built on the Jetson |

NVIDIA's pre-quantized Cosmos-Reason2 checkpoints are FP8 / NVFP4 only, which Orin (SM87) cannot run,
so the INT8 and INT4 checkpoints are quantized here from the BF16 source.

## Precision details (agreement: "FP16 vs BF16" row)

- Weights are exported as **FP16** (`--dtype fp16` is the only option; the BF16 source is cast).
  This matches the llama.cpp side, which uses F16 GGUFs.
- The **vision encoder stays FP16**: Edge-LLM only offers FP8 visual quantization, which needs SM89+.
- The **LM head stays FP16** (`--lm_head_quantization` not set). On the llama.cpp side the output
  tensor's type depends on the GGUF quant mix; report this as part of "runtime plus its quantization
  implementation".

## Calibration data

`int8_sq` and `int4_awq` calibrate on the first 512 articles of the CNN/DailyMail 3.0.0 train split
(Edge-LLM default, `--num_samples 512`). Edge-LLM v0.9.1 loads it by the legacy id `cnn_dailymail`,
which current `huggingface_hub` rejects (`Repository id must be 'namespace/name'`). Instead of patching
Edge-LLM, the same 512 articles were saved from `abisee/cnn_dailymail` (config `3.0.0`, revision
`96df5e686bee6baa90b8bee7c28b81fa3fa6223d`, the revision Edge-LLM requests) to a local JSONL and passed
through Edge-LLM's documented override:

```bash
export EDGELLM_QUANT_DATASET_CNN_DAILYMAIL=$WORKSPACE/calib/cnn_dailymail_train_first512.jsonl
# SHA-256 bc025de44299420e27829cf9d4f7802d745b79b953259d1779ea824ca0055d8d
```

## Image resolution (must match llama.cpp)

llama.cpp sends images at native size and its Qwen3-VL preprocessing caps them at about 1 MP.
Aeon's vLLM test matched llama.cpp's image token counts on 100/100 TextVQA records with a
1,048,576-pixel cap (`results_vllm/NOTES.md`). Edge-LLM's Qwen3-VL resize mirrors HF `smart_resize`
with `maxPixels = maxImageTokensPerImage * 32 * 32`, so the visual engine must be built with
`--maxImageTokensPerImage 1024`. **The default is 512**, which would silently downscale most TextVQA
images (about 700 to 1,000 tokens in llama.cpp). `--minImageTokens` stays at its default of 4
(4,096 pixels; vLLM matched with 3,136). Token counts still have to be verified on the board.

## Prompt and decoding (must match llama.cpp)

- **Prompts, parsers, scorers and the result-line format** are imported from
  `scripts/run_experiment.py` by `tensorrt/run_trt_experiment.py`, not copied.
- **Chat template:** the checkpoint's template adds no default system prompt and no thinking tags,
  so a request renders as
  `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n`.
  Images go before the text, as in llama.cpp's `call_server()`. Still to verify on the board: that
  llama-server renders the same string from the GGUF's embedded template.
- **Decoding:** greedy, set explicitly (`temperature 0.0, top_k 1, top_p 1.0`). The model's
  `generation_config.json` defaults to sampling (temperature 0.7, top_k 20, top_p 0.8), and Edge-LLM's
  Python API defaults to temperature 0.7, top_p 0.9. This closes the "other sampling settings" gap
  flagged in the agreement, since llama.cpp at temperature 0 is also greedy.
- **Speed metric:** Edge-LLM's Python API reports no decode-only tokens/s, so `tok_s` is left at 0 and
  each line records wall-clock `latency_ms` (the same definition `bench_latency.py` uses).

## Steps

### 1. Quantize and export (x86 host)

```bash
cd ~/TensorRT-Edge-LLM && git checkout v0.9.1 && git submodule update --init --recursive
python3.12 -m venv ~/venvs/edgellm-0.9.1
~/venvs/edgellm-0.9.1/bin/pip install . ".[tools]"
export PYTHONPATH=$HOME/TensorRT-Edge-LLM
export EDGELLM_QUANT_DATASET_CNN_DAILYMAIL=$WORKSPACE/calib/cnn_dailymail_train_first512.jsonl

tensorrt-edgellm-quantize llm --model_dir models/Cosmos-Reason2-2B \
  --output_dir cr2b-int8sq/quantized --quantization int8_sq
tensorrt-edgellm-quantize llm --model_dir models/Cosmos-Reason2-2B \
  --output_dir cr2b-int4awq/quantized --quantization int4_awq
tensorrt-edgellm-export cr2b-int8sq/quantized cr2b-int8sq/onnx
tensorrt-edgellm-export cr2b-int4awq/quantized cr2b-int4awq/onnx
```

### 2. Jetson checklist

Do these in order; stop and record the problem in the Log below if a step fails.

**a. System matches the agreement**
```bash
cat /etc/nv_tegra_release          # must be R36.4.x (JetPack 6.2); otherwise reflash to 6.2 first
sudo nvpmodel -m 0 && sudo nvpmodel -q   # 15W mode (mode 0 on Orin Nano), as on Aeon's board
bash tensorrt/collect_board_info.sh > tensorrt/board_info_$(hostname).txt
```
Fill in the Max column of the Shared Setup Agreement from that file.

**b. Build Edge-LLM v0.9.1 (C++ runtime + Python bindings)**
```bash
sudo apt update && sudo apt install -y cmake build-essential git python3-pip python3-venv
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git && cd TensorRT-Edge-LLM
git checkout v0.9.1 && git submodule update --init --recursive
nvcc --version                     # 12.6
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=12.6 -DENABLE_CUTE_DSL=ALL
make -j$(nproc)                    # if the build gets OOM-killed, use make -j2
cd .. && pip3 install pybind11==2.13.6
TRT_PACKAGE_DIR=/usr python3 experimental/server/setup_pybind.py build_ext --inplace
```
Loading prebuilt engines through `experimental.server.LLM(engine_dir=...)` does not need PyTorch on
the board: it falls back to loading `build/pybind/_edgellm_runtime*.so` directly. **If the Python
bindings will not build, fall back to `build/examples/llm/llm_inference`** (JSON in, JSON out) and the
runner's backend gets swapped; everything else in the runner stays.

**c. Copy ONNX from the laptop** (`~/edgellm-workspace/cr2b-*/onnx`) to
`~/tensorrt-edgellm-workspace/cr2b-*/onnx` on the board, on NVMe like Aeon's models.

**d. Build engines** (per config; `<cfg>` = `cr2b-int8sq` or `cr2b-int4awq`)
```bash
W=~/tensorrt-edgellm-workspace
./build/examples/llm/llm_build --onnxDir $W/<cfg>/onnx/llm --engineDir $W/<cfg>/engines/llm \
  --maxBatchSize 1 --maxInputLen 2048 --maxKVCacheCapacity 2048
./build/examples/multimodal/visual_build --onnxDir $W/<cfg>/onnx/visual --engineDir $W/<cfg>/engines \
  --minImageTokens 4 --maxImageTokens 1024 --maxImageTokensPerImage 1024
```
`--maxKVCacheCapacity 2048` mirrors llama.cpp's `--ctx-size 2048`. If an engine build runs out of
memory on 8 GB, enable swap on NVMe for the build only (not during timed runs).

**e. Checks before real runs**
1. Smoke test: `python3 tensorrt/run_trt_experiment.py --config T8 --datasets blink_depth --limit 20`
   then look at `results_trt/smoke/` — answers should be single letters, no errors.
2. Image tokens: run the 100 TextVQA records from `results_vllm/tokcheck_textvqa.csv` and compare
   prompt token counts with `llama_prompt_tokens` (Edge-LLM's Python API reports 0 prompt tokens, so
   this needs the C++ `llm_inference` output or a debug log; method TBD on the board).
3. Prompt text: compare the rendered prompt with llama-server's `/apply-template` for the same record.
4. Board sanity check (agreement): run llama.cpp Config 2 once on this board; accuracy should match
   Aeon's exactly and tok/s should be within noise.

**f. Full runs**, one config at a time, inside `tmux` so a dropped SSH session does not stop them:
```bash
python3 tensorrt/run_trt_experiment.py --config T8 --resume
python3 tensorrt/run_trt_experiment.py --config T4 --resume
```
About 9,900 records per config (BLINK 267 + CV-Bench 2,638 + ScienceQA 2,017 + TextVQA 5,000).
Needs Aeon's `benchmarks/` folders (not in git) copied to the repo root.

## Log

| Date | Step | Result |
|---|---|---|
| 2026-10-18 | Edge-LLM v0.9.1 export env installed on x86 host | OK |
| 2026-10-18 | Download Cosmos-Reason2-2B @ 9ce19a1 | OK |
| 2026-10-18 | `int8_sq` quantize, first try | Failed: legacy `cnn_dailymail` dataset id; fixed with local calibration file |
| 2026-10-18 | `int8_sq` quantize | OK, 12 min, peak RAM 10.7 GB. W8A8_SQ_PER_CHANNEL; `lm_head` and `model.visual*` excluded. All language layers calibrated (the 105 "not calibrated" warnings are all visual or lm_head) |
| 2026-10-18 | `int4_awq` quantize | OK, 34 min, peak RAM 10.7 GB. W4A16_AWQ, group size 128; same exclusions. ModelOpt's CUDA extension could not be compiled (no CUDA toolkit / ninja on PATH on the host) and it fell back to its CPU path: slower, same algorithm |
| 2026-10-18 | ONNX export, both configs | OK, about 2 min each. LLM graph: 196 quantized linear layers (28 layers x 7 projections) + FP16 lm_head; INT8 as QuantizeLinear/DequantizeLinear around MatMul, INT4 as `Int4GroupwiseGemmPlugin`. Visual ONNX is FP16 and byte-identical across the two configs (779 MB, vs 782 MB F16 mmproj on the llama.cpp side). File hashes in `manifests/` |
