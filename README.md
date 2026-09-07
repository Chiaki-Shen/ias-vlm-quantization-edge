# Vision Encoder Quantization Sensitivity in Driving VLMs on Jetson Edge Hardware

## Team

- **Yiang Shen** — MS Computer Engineering
- **Maximilian Garcia** — MS Artificial Intelligence

## Abstract

Compact vision-language models (VLMs) are emerging as practical candidates for autonomous driving reasoning on edge platforms, but deploying them within the strict memory budgets of devices like the Jetson Orin Nano (8 GB) demands aggressive quantization. Prior work on VLM quantization has established that vision encoders are disproportionately sensitive to precision reduction compared to language model backbones, yet this finding has only been characterized on general-purpose benchmarks (MME, MMMU) using server-class hardware — never on driving-domain tasks or memory-constrained edge devices. This project isolates and measures vision encoder quantization sensitivity across precision levels (FP16, INT8, INT4) in modern sub-4B driving VLMs deployed on a physical Jetson Orin Nano 8 GB, evaluated on the DriveLM driving visual question answering benchmark. By comparing component-wise quantization configurations — where the vision encoder and language backbone are quantized independently — we characterize the precision floor below which driving task performance degrades, providing the deployment-stage quantization guidance that recent edge-oriented driving VLM pipelines identify as a critical next step.

## Track

Intelligent Autonomous Systems (IAS) — Deployment & Optimization Track

## Repository Structure

- `/proposal` — Project proposal, novelty audit, and literature survey
- `/src` — Quantization scripts, inference pipeline, and evaluation framework
- `/configs` — Model and quantization configuration files
- `/results` — Benchmarking outputs, score tables, and visualizations
- `/docs` — Meeting notes, progress logs, and reference materials
