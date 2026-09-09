# Vision Encoder Quantization Sensitivity in Driving VLMs on Jetson Edge Hardware

## Team
- Yiang Shen — MS Computer Engineering
- Maximilian Garcia — MS Artificial Intelligence

## Abstract

Compact vision-language models (VLMs) are emerging as practical candidates for autonomous driving reasoning on edge platforms, but deploying them within the strict memory budgets of devices like the Jetson Orin Nano (8 GB) demands aggressive quantization. Prior work on VLM quantization has established that vision encoders are disproportionately sensitive to precision reduction compared to language decoders, yet this finding has only been characterized on general-purpose benchmarks (MME, MMMU) using server-class hardware — never on spatial reasoning benchmarks or memory-constrained edge devices.

This project isolates and measures vision encoder quantization sensitivity across precision levels (FP16, FP8, INT8, INT4) in Cosmos-Reason2-2B — NVIDIA's physical-world reasoning model built on the Qwen3-VL architecture — deployed on a physical Jetson Orin Nano 8 GB. We evaluate on BLINK (depth perception + spatial reasoning) and CV-Bench (2D/3D understanding), comparing quantized on-device accuracy against the model's official published scores. By quantizing the vision encoder and language decoder independently, we characterize the precision floor below which spatial reasoning performance degrades, providing the deployment-stage quantization guidance that recent edge-oriented driving VLM pipelines (MoRAL, IEEE IMC 2026) identify as a critical next step. TensorRT acceleration is measured at each quantization level per advisor request. LingoQA (driving-specific video QA) is a stretch goal benchmark.

## Track

Intelligent Autonomous Systems (IAS) — Deployment & Optimization Track

## Repository Structure

```
ias-vlm-quantization-edge/
├── README.md                              # this file (Deliverable A)
├── proposal/
│   ├── LITERATURE_SOTA_SURVEY.md          # Deliverable B: 10-paper SOTA survey
│   └── NOVELTY_FEASIBILITY_AUDIT.md       # Deliverable D: AI novelty & feasibility audit
├── src/                                    # (coming soon)
│   ├── inference/                          # llama.cpp / TensorRT-LLM inference runners
│   ├── quantization/                       # component-wise quantization scripts (VE vs. LLM)
│   ├── eval/                               # BLINK, CV-Bench, LingoQA scoring
│   └── utils/                              # logging, hardware metrics, data loading
├── configs/                                # model + quantization matrix configurations
├── results/                                # score tables, sensitivity curves, TensorRT comparisons
└── notebooks/                              # EDA + result analysis
```

## Hardware

- **Device:** NVIDIA Jetson Orin Nano 8 GB Developer Kit 
- **Primary model:** Cosmos-Reason2-2B 
- **Toolchain:** llama.cpp / GGUF (co-primary) + TensorRT-LLM (co-primary)
