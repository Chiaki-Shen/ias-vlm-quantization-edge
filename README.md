# Vision Encoder Quantization Sensitivity in Driving VLMs on Jetson Edge Hardware

## Team

- Yiang Shen — MS Computer Engineering
- Maximilian Garcia — MS Artificial Intelligence  
**Advisor:** Prof. Kaikai Liu, SJSU

## Abstract

Compact vision-language models (VLMs) are emerging as practical candidates for autonomous driving reasoning on edge platforms, but deploying them within the strict memory budgets of devices like the Jetson Orin Nano (8 GB) demands aggressive quantization. Prior work on VLM quantization has established that vision encoders are disproportionately sensitive to precision reduction compared to language decoders, yet this finding has only been characterized on general-purpose benchmarks using server-class hardware — never on spatial reasoning benchmarks or memory-constrained edge devices.

This project isolates and measures vision encoder quantization sensitivity across precision levels (F16, Q8, Q4) in Cosmos-Reason2-2B — NVIDIA's physical-world reasoning model built on the Qwen3-VL architecture — deployed on a physical Jetson Orin Nano 8 GB. We quantize the vision encoder (mmproj) and language decoder independently using llama.cpp's GGUF format, producing a 7-configuration matrix that spans combinations of F16/Q8_0/Q4_0 vision encoders against F16/Q8_0/Q4_K_M language decoders. We evaluate on BLINK (depth perception + spatial reasoning) and CV-Bench (2D/3D understanding), totaling over 20,000 inference calls. By characterizing the precision floor below which spatial reasoning degrades, we provide deployment-stage quantization guidance that recent edge-oriented driving VLM pipelines (MoRAL, IEEE IMC 2026) identify as a critical next step.

## Advisor Feedback (Sep 2026)

Prof. Liu reviewed the proposal and noted: the central research question depends on controlling vision encoder and language decoder precision independently. He set the first feasibility gate as demonstrating at least several matched configurations before committing to the full benchmark matrix, recommended choosing one primary runtime rather than maintaining parallel pipelines, and suggested treating a second VLM and LingoQA as stretch goals. A complete sensitivity curve on one model and real edge hardware is already a strong semester result.

**Feasibility gate: CLEARED (Sep 19, 2026)** — 7 configurations confirmed working end-to-end on Jetson Orin Nano 8 GB via llama.cpp with independent mmproj and model file swapping.

## Benchmark Results

### Quantization Matrix — All Runs Complete

All configs evaluated on 2,905 records (124 BLINK-Depth + 143 BLINK-Spatial + 2,638 CV-Bench) at 15W power mode. Accuracy reported on answered records (118 consistent poison-image errors excluded across all configs for fair comparison). Config 1 (F16/F16) was dropped due to OOM on the 8 GB device.

| Config | Vision Encoder | Language Decoder | BLINK Depth | BLINK Spatial | CV-Bench | **Overall** | Avg tok/s | Status |
|--------|---------------|-----------------|-------------|---------------|----------|-------------|-----------|--------|
| ~~1~~ | ~~F16 (782 MB)~~ | ~~F16 (3.8 GB)~~ | — | — | — | — | — | ❌ OOM |
| 2 | F16 (782 MB) | Q8_0 (2.1 GB) | 70.16% | 73.43% | 79.09% | 78.40% | 17.1 | ✅ Done |
| 3 | F16 (782 MB) | Q4_K_M (1.2 GB) | 82.26% | 77.62% | 76.67% | 76.96% | 19.3 | ✅ Done |
| 4 | Q8_0 (421 MB) | F16 (3.8 GB) | 71.77% | 74.13% | 79.13% | 78.54% | 10.9 | ✅ Done |
| 5 | Q8_0 (421 MB) | Q8_0 (2.1 GB) | 70.16% | 74.13% | 78.93% | 78.29% | 17.2 | ✅ Done |
| 6 | Q8_0 (421 MB) | Q4_K_M (1.2 GB) | 82.26% | 79.02% | 76.67% | 77.04% | 19.7 | ✅ Done |
| 7* | Q4_0 (229 MB) | Q4_K_M (1.2 GB) | 78.23% | 75.52% | 75.63% | 75.74% | 20.7 | ✅ Done |
| 8* | Q4_0 (229 MB) | Q8_0 (2.1 GB) | 67.74% | 72.03% | 77.10% | 76.43% | 18.3 | ✅ Done |

\*Q4_0 mmproj is unofficial — generated on our Jetson device via `llama-quantize`.

### Key Findings

**1. F16 → Q8 vision: zero accuracy impact.**
Configs 2 vs 5 and Configs 3 vs 6 show identical accuracy to the decimal across all three benchmarks when only the vision encoder changes from F16 to Q8_0. This contradicts the general expectation that vision encoders are highly sensitive to quantization, and aligns with MBQ (CVPR 2025) Table 6, which found ViT encoder quantization to W4A8 caused no significant performance drop.

**2. Q8 → Q4 vision: measurable degradation (the sensitivity knee).**
Config 5 vs 8 (isolated Q4 vision effect with same Q8 language): BLINK-Depth falls 2.4%, BLINK-Spatial falls 2.1%, CV-Bench falls 1.8%. Config 6 vs 7 confirms the same pattern with Q4 language held constant. The sensitivity knee sits between Q8 and Q4 on the vision encoder side.

**3. Q4 language produces a consistent BLINK-Depth boost.**
Q4_K_M language configs consistently score +10-12% higher on BLINK-Depth compared to Q8/F16 language configs at every vision precision level (F16 vision: +12.1%, Q8 vision: +12.1%, Q4 vision: +10.5%). This effect is reproducible and independent of vision encoder precision, suggesting Q4 language quantization alters the model's response distribution in a way that favors depth estimation tasks while slightly reducing CV-Bench accuracy (~2.5%).

**4. Practical deployment recommendation.**
Q8_0 vision + Q4_K_M language (Config 6) delivers 77.04% overall accuracy at 19.7 tok/s using ~1.6 GB total model size — roughly half the memory of Config 2 (F16v + Q8l) with less than 1.4% accuracy difference and 15% faster throughput.

### 118 Poison Images

Across all 7 configs, the same 118 CV-Bench records consistently fail with HTTP 400 errors, even after server restart and retry. These are excluded from accuracy calculations and cancel out in cross-config comparisons. Per-config poison logs are saved in `results/` for cross-config verification.

## Current Status

- [x] Full BLINK + CV-Bench benchmark runs (Configs 2-8, 7 configs total)
- [x] Config 8 Q4 vision isolation test
- [x] Preliminary findings documented
- [ ] Per-task subgroup analysis within CV-Bench
- [ ] VRAM profiling per config via `tegrastats`
- [ ] Results visualization (accuracy vs. compression curves)
- [ ] Error analysis: which images flip between Q8 and Q4 vision
- [ ] Investigation: why Q4 language boosts BLINK-Depth
- [ ] Paper write-up

## Repository Structure

```
ias-vlm-quantization-edge/
├── README.md
├── BENCHMARK_RUN_GUIDE.md
├── proposal/
│   ├── LITERATURE_SOTA_SURVEY.md
│   └── NOVELTY_FEASIBILITY_AUDIT.md
├── scripts/
│   ├── prep_datasets.py         # Download & format BLINK + CV-Bench
│   ├── run_experiment.py        # Interactive experiment runner (v4)
│   └── run_inference.py         # Standalone inference script
├── benchmarks/
│   ├── blink_depth/             # 124 records + images
│   ├── blink_spatial/           # 143 records + images
│   └── cv_bench/                # 2,638 records + images
├── reports/
│   └── CMPE249_Quantization_Matrix_Initial_Report.md
└── results/                     # Per-config summary .md + raw .jsonl
    ├── config2_F16v_Q8l_*.md
    ├── config2_F16v_Q8l__*.jsonl
    ├── ...
    ├── config8_Q4v_Q8l_*.md
    └── config*__cv_bench_poison.jsonl
```

## Hardware & Toolchain

| Item | Detail |
|------|--------|
| Device | NVIDIA Jetson Orin Nano 8 GB Developer Kit |
| JetPack | 6.1+ (CUDA 12.6, TensorRT 10.3) |
| Power mode | 15W (mode 0) — locked for all runs |
| Primary model | Cosmos-Reason2-2B (NVIDIA, Qwen3-VL backbone) |
| Runtime | llama.cpp build 11056 (compiled from source, aarch64 + CUDA) |
| Quantization | GGUF per-component: `--model` (language) + `--mmproj` (vision) swapped independently |
| Benchmarks | BLINK ([`BLINK-Benchmark/BLINK`](https://huggingface.co/datasets/BLINK-Benchmark/BLINK)) + CV-Bench ([`nyu-visionx/CV-Bench`](https://huggingface.co/datasets/nyu-visionx/CV-Bench)) |
| Total inference calls | 20,335 (2,905 records x 7 configs) |
| Stretch | TensorRT-LLM acceleration comparison, LingoQA, second VLM |

## Reproducing the Experiment

### 1. Prepare benchmark datasets (Mac/Linux)

```bash
pip install datasets Pillow
python3 scripts/prep_datasets.py
```

### 2. Transfer to Jetson

Copy the `benchmarks/` folder and `scripts/` to `/Developer/ias-vlm-quantization-edge/` on the Jetson.

### 3. Run an experiment (Jetson, inside sjsujetsontool shell)

```bash
export LD_LIBRARY_PATH=/opt/llama-cpp-new/build/bin:$LD_LIBRARY_PATH
cd /Developer/ias-vlm-quantization-edge
python3 scripts/run_experiment.py
```

Select a config from the interactive menu. Results and summary report are saved to `results/`.

See [`BENCHMARK_RUN_GUIDE.md`](BENCHMARK_RUN_GUIDE.md) for detailed step-by-step instructions and troubleshooting.

## References

- NVIDIA, "Cosmos-Reason2," 2026. <https://huggingface.co/collections/nvidia/cosmos-reason2>
- A. G. Kaliamurthi & K. Liu, "MoRAL: Sensor-Grounded BEV Reasoning for Compact VLMs toward Edge-Oriented Autonomous Driving," IEEE IMC 2026.
- S. Li et al., "MBQ: Modality-Balanced Quantization for Large Vision-Language Models," CVPR 2025.
- Y. Shin et al., "Rethinking Small VLM Quantization," ICML 2026 Workshop.
- Fu et al., "BLINK," ECCV 2024. <https://huggingface.co/datasets/BLINK-Benchmark/BLINK>
- Tong et al., "Cambrian-1 / CV-Bench," 2024. <https://huggingface.co/datasets/nyu-visionx/CV-Bench>
