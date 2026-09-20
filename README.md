# Vision Encoder Quantization Sensitivity in Driving VLMs on Jetson Edge Hardware

## Team
- Yiang Shen — MS Computer Engineering
- Maximilian Garcia — MS Artificial Intelligence  
**Advisor:** Prof. Kaikai Liu, SJSU

## Abstract

Compact vision-language models (VLMs) are emerging as practical candidates for autonomous driving reasoning on edge platforms, but deploying them within the strict memory budgets of devices like the Jetson Orin Nano (8 GB) demands aggressive quantization. Prior work on VLM quantization has established that vision encoders are disproportionately sensitive to precision reduction compared to language decoders, yet this finding has only been characterized on general-purpose benchmarks using server-class hardware — never on spatial reasoning benchmarks or memory-constrained edge devices.

This project isolates and measures vision encoder quantization sensitivity across precision levels (F16, Q8, Q4) in Cosmos-Reason2-2B — NVIDIA's physical-world reasoning model built on the Qwen3-VL architecture — deployed on a physical Jetson Orin Nano 8 GB. We quantize the vision encoder (mmproj) and language decoder independently using llama.cpp's GGUF format, producing a 7-configuration matrix that spans all combinations of F16/Q8_0/Q4_0 vision encoders against F16/Q8_0/Q4_K_M language decoders. We evaluate on BLINK (depth perception + spatial reasoning) and CV-Bench (2D/3D understanding), comparing per-config accuracy against the model's official published scores. By measuring accuracy, VRAM, and inference throughput at each configuration, we characterize the precision floor below which spatial reasoning degrades — providing the deployment-stage quantization guidance that recent edge-oriented driving VLM pipelines (MoRAL, IEEE IMC 2026) identify as a critical next step.

## Advisor Feedback (Sep 2026)

Prof. Liu reviewed the proposal and noted: the central research question depends on controlling vision encoder and language decoder precision independently. He set the first feasibility gate as demonstrating at least several matched configurations before committing to the full benchmark matrix, recommended choosing one primary runtime rather than maintaining parallel pipelines, and suggested treating a second VLM and LingoQA as stretch goals. A complete sensitivity curve on one model and real edge hardware is already a strong semester result.

**Feasibility gate: CLEARED (Sep 19, 2026)** — 7 configurations confirmed working end-to-end on Jetson Orin Nano 8 GB via llama.cpp with independent mmproj and model file swapping.

## Current Status

### Quantization Matrix — Smoke Test Complete

| Config | Vision Encoder | Language Decoder | Gen tok/s | Status |
|--------|---------------|-----------------|-----------|--------|
| 1 (baseline) | F16 (782 MB) | F16 (3.8 GB) | 11.2 | ✅ Smoke tested |
| 2 | F16 (782 MB) | Q8_0 (2.1 GB) | 17.5 | ✅ Smoke tested |
| 3 | F16 (782 MB) | Q4_K_M (1.2 GB) | 20.5 | ✅ Smoke tested |
| 4 | Q8_0 (421 MB) | F16 (3.8 GB) | 11.3 | ✅ Smoke tested |
| 5 | Q8_0 (421 MB) | Q8_0 (2.1 GB) | 19.7 | ✅ Smoke tested |
| 6 | Q8_0 (421 MB) | Q4_K_M (1.2 GB) | 22.4 | ✅ Smoke tested |
| 7 (stretch) | Q4_0 (229 MB)* | Q4_K_M (1.2 GB) | 21.2 | ✅ Smoke tested |

*Q4_0 mmproj is unofficial — generated on-device via `llama-quantize`. Accuracy TBD.

### Next Steps
- [ ] Full BLINK + CV-Bench benchmark runs across all 7 configs
- [ ] VRAM profiling per config via `tegrastats`
- [ ] Results visualization (accuracy vs. compression curves)
- [ ] Paper write-up

## Repository Structure

    ias-vlm-quantization-edge/
    ├── README.md
    ├── proposal/
    │   ├── LITERATURE_SOTA_SURVEY.md
    │   └── NOVELTY_FEASIBILITY_AUDIT.md
    ├── scripts/
    │   ├── prep_datasets.py
    │   ├── run_experiment.py
    │   └── run_inference.py
    ├── reports/
    │   └── CMPE249_Quantization_Matrix_Initial_Report.md
    └── results/

## Hardware & Toolchain

| Item | Detail |
|------|--------|
| Device | NVIDIA Jetson Orin Nano 8 GB Developer Kit |
| JetPack | 6.1+ (CUDA 12.6, TensorRT 10.3) |
| Primary model | Cosmos-Reason2-2B (NVIDIA, Qwen3-VL backbone) |
| Runtime | llama.cpp build 11056 (compiled from source, aarch64 + CUDA) |
| Quantization | GGUF per-component: `--model` (language) + `--mmproj` (vision) swapped independently |
| Benchmarks | BLINK (`BLINK-Benchmark/BLINK`) + CV-Bench (`nyu-visionx/CV-Bench`) |
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

## References

- NVIDIA, "Cosmos-Reason2," 2026. https://huggingface.co/collections/nvidia/cosmos-reason2
- A. G. Kaliamurthi & K. Liu, "MoRAL," IEEE IMC 2026.
- Y. Tang et al., "MBQ: Modality-Balanced Quantization for Large Vision-Language Models," CVPR 2025.
- Y. Shin et al., "Rethinking Small VLM Quantization," ICML 2026 Workshop.
- Fu et al., "BLINK," ECCV 2024. https://huggingface.co/datasets/BLINK-Benchmark/BLINK
- Tong et al., "Cambrian-1 / CV-Bench," 2024. https://huggingface.co/datasets/nyu-visionx/CV-Bench
