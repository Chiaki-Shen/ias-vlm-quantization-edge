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
(4,096 pixels; vLLM matched with 3,136). **Verified (2026-11-11):** applying Edge-LLM's
`getResizedImageSize` (`cpp/multimodal/qwenViTRunner.cpp`) to the 100 frozen TextVQA latency
records gives 576 to 1,024 image tokens (median 768), none capped. llama.cpp's `prompt_n` for the
same records is 27 to 38 tokens higher on every record (text + template), so image tokens match.

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
Status 2026-11-12: 1 and 2 done, 3 and 4 open (see Next steps).
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

## Results (2026-11-12)

### Accuracy (full datasets, `results_trt/`)

Same protocol (p2.1), prompts, parsers, scorers and `max_tokens` as llama.cpp. MCQ = mean of
`correct`, TextVQA = mean of `score`. Both TensorRT runs: 0 errors on all 9,922 records.

| Config | BLINK depth (124) | BLINK spatial (143) | CV-Bench (2,638) | ScienceQA (2,017) | TextVQA (5,000) |
|---|---:|---:|---:|---:|---:|
| **T8** TRT FP16v + INT8 SQ | 67.7 | 69.9 | 77.2 | 77.8 | 75.5 |
| Config 2 llama.cpp F16v + Q8_0 | 70.2 | 73.4 | 75.6 (118 errors) | 82.9 | 76.3 |
| **T4** TRT FP16v + INT4 AWQ | 77.4 | 77.6 | 73.8 | 79.0 | 72.0 |
| Config 3 llama.cpp F16v + Q4_K_M | 82.3 | 77.6 | 73.2 | not run | not run |

- BLINK is small (124/143 questions), so 3 to 5 point differences there are within noise.
- Config 2's CV-Bench has 118 failed requests counted as wrong; the TensorRT runs had none.
- **Open:** both TensorRT configs are 4 to 5 points below Config 2 on ScienceQA (T8 vs Config 2:
  1,474 both right, 199 only Config 2, 95 only T8). Settings are identical (p2.1, `context` prompt,
  parser v2, 16 tokens), so the next check is the rendered prompt on each side (checklist 2e.3).

### Latency and memory (`results_latency_trt/full/`)

`tensorrt/bench_latency_trt.py`, the TensorRT twin of `scripts/bench_latency.py`: same 200 frozen
records, same prompts and `max_tokens` (ScienceQA 16, TextVQA 32), 3 warm-ups per engine start,
3 rounds with rotated order, same cool-down rule, one request at a time, a fresh engine process per
slot. Desktop stopped (`systemctl stop display-manager`), 15W, zram only, tegrastats at 200 ms on
the host. Same log schema, so `scripts/bench_latency.py summarize` produced `summary.txt` unchanged.
Status: 1,200/1,200 valid, 0 worker deaths. Idle RAM 1,868 MB (Aeon: 1,636 MB).

| Config | ScienceQA median ms | TextVQA median ms | TextVQA p95 ms | Peak RAM MB (timed) |
|---|---:|---:|---:|---:|
| **T8** TRT INT8 | 218 | 951 | 1,327 | 5,934 |
| **T4** TRT INT4 | 269 | 1,102 | 1,558 | 4,929 |
| Config 2 llama.cpp Q8_0 (`results_latency/overnight`) | 637 | 2,637 | 3,761 | 4,928 |

- T8 is about 2.8x faster than Config 2 on both datasets and uses about 1 GB more RAM.
- T4 is slower than T8: answers are 2 to 3 tokens, so time is mostly vision encode + prefill,
  where W4A16 dequantization costs more than it saves.
- Per-round medians agree within 3 ms; memory drift per engine lifetime is -6 to -26 MB.
- `wall_ms` here is one in-process `LLM.generate()` call (image load + preprocess + encode + prefill +
  decode); llama.cpp's is an HTTP round trip to llama-server. Edge-LLM's Python API gives no
  prompt/decode split, so `prompt_ms`, `prompt_n`, `cache_n` are null (no prompt cache across calls).

## Next steps

1. **ScienceQA gap:** compare the exact prompt string Edge-LLM tokenizes with llama-server's
   `/apply-template` output for the same records (2e.3).
2. **Board sanity check (2e.4):** run llama.cpp Config 2 on this board (needs the GGUFs from Aeon);
   accuracy should match Aeon's exactly and latency should be within noise.
3. **Pair T4 properly:** Config 3 latency run and ScienceQA/TextVQA accuracy (Aeon).
4. Optional bonus: T16 (FP16 language) engine, if it fits in memory.
5. Paper caveats: TensorRT 10.7.0.23 from the course container (same image as llama.cpp's
   `jetson-dev`), the three Edge-LLM patches in `jetson/`, and matching by bit width, not method
   (SmoothQuant/AWQ vs Q8_0/Q4_K_M).

## Needed from Aeon

- [ ] `Cosmos-Reason2-2B-Q8_0.gguf` and `mmproj-Cosmos-Reason2-2B-F16.gguf` (or where the Q8_0
      came from), for the sanity check on Max's board. Hashes are in his Phase 0 manifest.
- [ ] Config 3 (F16v + Q4_K_M) latency run with `bench_latency.py`, plus ScienceQA and TextVQA
      accuracy, to pair with T4.
- [ ] His caveats list says two Edge-LLM patches; there are three (the third works around the
      ~2.5 GB `cudaMallocAsync` cap so T8's visual engine can load).
- [ ] His latency spec says `max_tokens` 16; `run_experiment.py` uses 32 for TextVQA. The TensorRT
      bench follows the script. Please confirm and fix the spec.
- [x] `benchmarks/` folders (received 2026-11-11, `SHA256SUMS` and README hashes OK).
- [x] Desktop state: on for accuracy runs, stopped for latency (spec received 2026-11-11).

## Log

| Date | Step | Result |
|---|---|---|
| 2026-10-18 | Edge-LLM v0.9.1 export env installed on x86 host | OK |
| 2026-10-18 | Download Cosmos-Reason2-2B @ 9ce19a1 | OK |
| 2026-10-18 | `int8_sq` quantize, first try | Failed: legacy `cnn_dailymail` dataset id; fixed with local calibration file |
| 2026-10-18 | `int8_sq` quantize | OK, 12 min, peak RAM 10.7 GB. W8A8_SQ_PER_CHANNEL; `lm_head` and `model.visual*` excluded. All language layers calibrated (the 105 "not calibrated" warnings are all visual or lm_head) |
| 2026-10-18 | `int4_awq` quantize | OK, 34 min, peak RAM 10.7 GB. W4A16_AWQ, group size 128; same exclusions. ModelOpt's CUDA extension could not be compiled (no CUDA toolkit / ninja on PATH on the host) and it fell back to its CPU path: slower, same algorithm |
| 2026-10-18 | ONNX export, both configs | OK, about 2 min each. LLM graph: 196 quantized linear layers (28 layers x 7 projections) + FP16 lm_head; INT8 as QuantizeLinear/DequantizeLinear around MatMul, INT4 as `Int4GroupwiseGemmPlugin`. Visual ONNX is FP16 and byte-identical across the two configs (779 MB, vs 782 MB F16 mmproj on the llama.cpp side). File hashes in `manifests/` |
| 2026-11-10 | 2a: board vs agreement | sjsujetson-45, Orin Nano 8GB Super devkit. L4T R36.4.3, CUDA 12.6.68, cuDNN 9.3.0.75, TensorRT 10.3.0.30 on the host: all match Aeon. Power set to 15W (mode 0, was MAXN_SUPER); jetson_clocks not forced (GPU idles at 306 MHz, max 612 MHz). Root and models on NVMe. Fan dynamic, profile `quiet`. `tensorrt-dev` 10.3.0.30 added for headers (9 new packages, nothing upgraded) |
| 2026-11-10 | 2c: ONNX copied to board | Both configs, all 22 files match `manifests/*.sha256` |
| 2026-11-10 | 2b: build v0.9.1 on host | **Failed**: v0.9.1 uses `nvinfer1::IStreamReaderV2`, which TensorRT 10.3 (JetPack 6.2) does not have, despite the docs listing JetPack 6.2 |
| 2026-11-10 | 2b: build in course container | Built inside `cmpelkk/jetson-llm:latest` (image `ccb3bdd0f7ad`): TensorRT 10.7.0.23, CUDA 12.6.85, cuDNN 9.6.0.74. **TensorRT column for the paper = 10.7.0.23**. Deviations, all in `jetson/build_edgellm.sh`: CuTe DSL groups `ffpa;gdn;gemm;int4_fp16_gemm;ssd` (no `f16_moe`: needs `cudaLibrary_t` from CUDA >= 12.8; MoE only); patch `jetson/edgellm-v0.9.1-jp62-fp4-plugin-guard.patch` (NVFP4 multi-GPU plugin needs TensorRT >= 10.8, unused on Orin); CUDA 12.0-12.6 cudart shim + `--wrap=_cudaLaunchKernelEx` passed to executables via `CMAKE_CXX_STANDARD_LIBRARIES` / `CMAKE_EXE_LINKER_FLAGS` (upstream only attaches them to shared libs) |
| 2026-11-10 | 2b: Python bindings | `setup_pybind.py` hard-codes `EMBEDDED_TARGET=jetson-thor`, so `_edgellm_runtime` is built from the main tree instead (`jetson/build_pybind.sh`) -> `build/pybind/`, cpython-312 (container Python). The runner must run inside the container |
| 2026-11-11 | 2d: llm_build segfault | Crash in libcuda inside `FMHAKernelLoader`. Cause: 2 of 11 sm87 FMHA cubins (`*_custom_mask_sm87`, new in v0.9.1, removed upstream in v0.10.0) are not valid images on L4T R36.4.3 (`cuModuleLoadData` -> 200; `cuobjdump` cannot parse them). The other 9 load fine (built with CUDA 12.8, toolkit=128). Patch `jetson/edgellm-v0.9.1-jp62-drop-sm87-custom-mask-fmha.patch` comments out the 2 table entries (custom-mask attention = tree/spec decoding, unused) |
| 2026-11-11 | 2d: engines | Needed a 16 GB NVMe swap file (`/swapfile_build`, prio 1, build only, not in fstab). INT8: LLM 2.6 GB in 5:23 (peak RSS 5.4 GB, 6.7 GB swap used), visual 789 MB in 1:25. INT4: LLM 1.9 GB in 1:52, visual 789 MB in 1:24 (3.2 GB swap). Both with `--maxBatchSize 1 --maxInputLen 2048 --maxKVCacheCapacity 2048`, visual `--minImageTokens 4 --maxImageTokens 1024 --maxImageTokensPerImage 1024` |
| 2026-11-11 | 2e: functional check (no datasets) | Needs `EDGELLM_PLUGIN_PATH=$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so` (the auto-search only looks in `build/core`, `build/lib`). **INT4 works**: loads in 9.8 s, about 1.5 s per short answer; correct on solid red/blue/black-square images and a seal photo; text-only OK (`jetson/engine_vary.py`). INT8 first failed to load: the visual engine (811 MB) gave OOM after the INT8 LLM engine (see next rows) |
| 2026-11-11 | 2e: INT8 OOM, cause | Not free memory: still fails with the desktop stopped (`systemctl isolate multi-user.target`, idle 560 MB) and 2.65 GB free. Measured on this driver (540.4.0, CUDA 12.6): `cudaMallocAsync` tops out at about 2.5 GB per process in total (largest single block 2,554 MB) while `cudaMalloc` is not capped. TensorRT's default allocator uses the async pool |
| 2026-11-11 | 2e: allocator patch | `jetson/edgellm-v0.9.1-jp62-sync-gpu-allocator.patch`: with `EDGELLM_SYNC_GPU_ALLOCATOR=1`, `deserializeCudaEngineFromFile` sets a `cudaMalloc`/`cudaFree` `IGpuAsyncAllocator` via `IRuntime::setGpuAllocator` (opt-in; compute unchanged). **Use it for both configs.** With it: INT8 and INT4 both load and pass `jetson/engine_vary.py` (red/blue/black-square/seal photo, text-only). INT4 output is identical with and without the patch. Peak system RAM during load + 5 requests: INT8 5,587 MB, INT4 4,582 MB (desktop off, NVMe swap file still on) |
| 2026-11-11 | How to run | `bash tensorrt/jetson/run_in_container.sh python3 tensorrt/run_trt_experiment.py --config T8 ...` sets `EDGE_LLM_PATH`, `EDGELLM_PLUGIN_PATH` and `EDGELLM_SYNC_GPU_ALLOCATOR=1` inside `jetson-llm:v1`. Before timed runs: remove `/swapfile_build`, and decide with Aeon whether the desktop is on or off (it changes idle RAM by about 1.5 GB) |
| 2026-11-11 | Aeon's environment manifest | His llama.cpp also ran in `cmpelkk/jetson-llm:latest` (`ccb3bdd0f7ad`, container CUDA 12.6.85), so both runtimes use the same container stack. Fan `quiet`, zram 6 x 635 MB only, 15W: Max's board matches after the reboot (build swap file gone) |
| 2026-11-11 | Datasets | Aeon's frozen `benchmarks/` unzipped to `/Developer/ias-vlm-quantization-edge/benchmarks` (symlink to the clone, since BLINK / CV-Bench store absolute image paths). `SHA256SUMS` OK, all 5 `data.jsonl` match the README. `--dry-run` OK for T8 and T4 |
| 2026-11-11 | 2e.1 smoke test | T8, blink_depth, 20 records: 15/20 (Config 2: 14/20 on the same 20), same answer on 17/20, 0 errors, median 235 ms. Desktop on: peak 6.8 GB, 608 MB available at the low point |
| 2026-11-11 | 2e.2 image tokens | Match llama.cpp (see "Image resolution") |
| 2026-11-11 | Latency bench | `jetson/bench_latency_trt.sh full`, headless, 21:47 to 22:11. 1,200/1,200 valid. Results above |
| 2026-11-11/12 | 2f full runs | Desktop stopped (accuracy is unaffected; more memory headroom for T8). `jetson/run_accuracy_after_bench.sh` ran T8 22:12 to 00:10, T4 00:10 to 02:28. 0 errors. Results above |
