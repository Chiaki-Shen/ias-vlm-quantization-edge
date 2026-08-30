# Edge Deployment & Failure Mode Evaluation of Lightweight Safety-Critical Perception Models

## Team
- Yiang Shen — MS Computer Engineering
- Maximilian Garcia — MS Artificial Intelligence

## Abstract
Safety-critical perception systems deployed on edge devices face a well-known accuracy tradeoff under model compression (quantization, pruning). However, average accuracy metrics obscure a more dangerous pattern: compressed models may retain overall performance while disproportionately failing on hard subsets, such as rainy conditions, occlusion, low-light scenes. This project deploys a lightweight object detection model (YOLOv8-nano or MobileNet-V3) on a resource-constrained edge device and introduces a structured failure mode evaluation framework to measure how compression shifts failure distributions across normal and adverse conditions, not just average accuracy.

## Track
Intelligent Autonomous Systems (IAS) — Deployment & Optimization Track

## Repository Structure
- `/proposal` — Project proposal documents and novelty audit
- `/survey` — Literature & SOTA survey
- `/src` — Source code (model pipeline, benchmarking scripts, evaluation framework)
- `/results` — Benchmarking outputs and visualizations
