# Literature & SOTA Survey
## Project: Benchmarking Vision Encoder Quantization Sensitivity in Driving VLMs on Edge Hardware

**Team:** Yiang (Aeon) Shen, Maximilian (Max) Garcia
**Course:** CMPE 249 — Intelligent Autonomous Systems (Fall 2026)
**Last updated:** September 2026

---

## Survey Scope

This survey covers three intersecting research areas relevant to our project:

1. **Spatial reasoning and driving benchmarks for VLMs** — how Vision-Language Models are evaluated on depth perception, spatial understanding, and driving tasks (papers 1–5)
2. **VLM quantization and component-wise analysis** — how VLMs behave under quantization, especially vision encoder vs. language decoder sensitivity (papers 6–8)
3. **Compact VLMs for edge deployment** — physical-world reasoning models designed for resource-constrained platforms (papers 9–10)

No existing work covers the intersection: component-wise VLM quantization evaluated on spatial reasoning benchmarks deployed on edge hardware. That gap is our project's contribution.

---

## Paper Summaries

### 1. BLINK: Multimodal Large Language Models Can See but Not Perceive
**Fu et al. | ECCV 2024 | arXiv:2404.12390**

BLINK is a benchmark designed to expose a fundamental weakness in multimodal LLMs: they can identify objects but fail at core visual perception tasks that humans find trivial. The benchmark includes multiple subtasks — we focus on BlinkDepth (248 examples testing relative depth perception) and BlinkSpatial (286 examples testing spatial relation understanding). All tasks use a multiple-choice format, enabling simple accuracy scoring with no external judge model. BLINK revealed that even GPT-4V performs only slightly above random chance on depth and spatial tasks, while task-specific models do much better.

**Relevance to our project:** BLINK is one of our primary benchmarks. BlinkDepth and BlinkSpatial directly test the vision encoder's spatial perception capability — the exact capability we hypothesize will degrade under quantization. Official Cosmos-Reason2-2B scores (BlinkDepth 82.26, BlinkSpatial 75.52) provide direct comparison baselines. The MCQ format means evaluation is fully automated with no judge model overhead — critical for running many quantization configurations on-device.

---

### 2. CV-Bench: Vision-Centric 2D/3D Understanding Benchmark
**Tong et al. | 2024 (from Cambrian-1)**

CV-Bench is a vision-centric benchmark extracted from the Cambrian-1 project that evaluates 2D and 3D understanding in VLMs. It covers spatial relationships, depth ordering, and geometric reasoning across 2,638 multiple-choice examples. The benchmark specifically tests whether VLMs genuinely understand visual-spatial structure rather than relying on language priors — a concern echoed by DriveBench's finding that many driving VLMs generate plausible answers even with visual input removed.

**Relevance to our project:** CV-Bench is our second primary benchmark. Its larger example count (2,638 vs. BLINK's ~534) provides more statistical power for sensitivity curves. The official Cosmos-Reason2-2B score (78.74) is our comparison baseline. Together with BLINK, it enables cross-benchmark analysis: we can test whether depth perception (BlinkDepth) degrades faster than spatial reasoning (BlinkSpatial, CV-Bench) under vision encoder quantization.

---

### 3. LingoQA: Visual Question Answering for Autonomous Driving
**Marcu et al. | ECCV 2024 | arXiv:2312.14115**

LingoQA is a video-based driving VQA benchmark that tests understanding of real driving scenarios. It uses a lightweight text classifier (Lingo-Judge) rather than a full LLM for scoring, making evaluation cheaper than GPT Score approaches. Cosmos-Reason2-2B achieves 59.00 on LingoQA. However, LingoQA requires video input processing, which demands significantly more memory than still-image benchmarks.

**Relevance to our project:** LingoQA is our stretch goal benchmark — the only driving-specific benchmark in our evaluation suite. It adds domain relevance by testing actual driving scene understanding. However, video processing on the Jetson Orin Nano's 8 GB memory is high-risk, which is why it's a stretch goal rather than a primary benchmark. If achievable, LingoQA results would show whether driving-specific spatial reasoning degrades differently from general spatial benchmarks under quantization.

---

### 4. MPDrive: Improving Spatial Understanding with Marker-Based Prompt Learning for Autonomous Driving
**Zhang et al. | CVPR 2025 | arXiv:2504.00379**

MPDrive achieves state-of-the-art on the DriveLM leaderboard (85.18% accuracy) by overlaying visual markers (numbered circles, directional arrows) onto driving scene images before feeding them to the VLM. Evaluated at FP16 on datacenter GPUs. The visual marker approach depends heavily on the vision encoder correctly interpreting fine-grained spatial overlays.

**Relevance to our project:** MPDrive demonstrates that driving VLM performance depends critically on vision encoder fidelity — spatial markers are exactly the kind of fine-grained visual feature likely to be destroyed by aggressive quantization. MPDrive also freezes the vision encoder during fine-tuning, treating it as the more fragile component. This pattern supports our core hypothesis. While we don't evaluate on DriveLM directly, MPDrive's findings about vision encoder sensitivity to spatial features are relevant to our BLINK/CV-Bench spatial reasoning analysis.

---

### 5. ReasonDrive: Efficient VQA for Autonomous Vehicles with Reasoning-Enhanced Small VLMs
**Chahe & Zhou | arXiv:2504.10757, 2025**

ReasonDrive shows that small VLMs (3B–11B) can perform competitively on driving VQA when fine-tuned with explicit reasoning chains. Qwen2.5-VL-3B achieves 0.45 and Qwen2.5-VL-7B reaches 0.54 on DriveLM. Per-category analysis reveals perception accuracy can reach 0.68 while planning is harder. All results at FP16 on datacenter GPUs. Like MPDrive and VTS, ReasonDrive freezes vision encoder weights during fine-tuning.

**Relevance to our project:** ReasonDrive validates that small VLMs in the Qwen VL family (the backbone of Cosmos-Reason2-2B) can handle spatial reasoning tasks. Their per-category difficulty breakdown provides a reference for our cross-benchmark analysis — we test an analogous question: does depth perception (BlinkDepth) degrade faster than spatial reasoning (BlinkSpatial) under quantization? The vision encoder freeze pattern adds further supporting evidence for the asymmetry hypothesis.

---

### 6. MBQ: Modality-Balanced Quantization for Large Vision-Language Models
**CVPR 2025**

MBQ establishes the core finding that motivates our project: vision encoders are significantly more sensitive to quantization than language decoders in VLMs. They propose a mixed-precision scheme assigning higher precision to vision-sensitive layers based on modality-aware sensitivity analysis. Evaluated on general-domain benchmarks (MMBench, TextVQA, ScienceQA) with 7B+ models on datacenter GPUs.

**Relevance to our project:** MBQ is our most important prior work citation. They discovered the vision-vs-language quantization asymmetry; we characterize it in a regime they didn't cover — spatial reasoning benchmarks (BLINK, CV-Bench), a physical-world reasoning model (Cosmos-Reason2-2B), edge hardware (Jetson Orin Nano), and precision levels including FP8 and INT4. We frame our contribution as extending, not rediscovering, their finding.

---

### 7. Rethinking Small VLM Quantization
**Shin et al. | ICML 2026 Workshop**

Shin et al. perform component-wise quantization ablations (vision encoder / projector / LLM decoder quantized independently) on small VLMs (sub-3B) deployed on Jetson Orin hardware. They confirm the vision encoder sensitivity finding from MBQ in the small-model regime. Key limitations: they never push vision encoder precision below INT8, evaluate only on MME (general-domain), use BitsAndBytes as the quantization backend (which their own appendix flags for anomalous overhead), and test only on Orin NX and AGX Orin — explicitly excluding the Orin Nano due to OOM issues.

**Relevance to our project:** Shin et al. is the closest prior work to our experimental design. Our project extends their framework in four specific directions: (1) vision encoder precision below INT8 — we test FP8 and INT4, (2) spatial reasoning benchmarks with published baselines (BLINK, CV-Bench vs. MME), (3) Orin Nano hardware (which they excluded), and (4) llama.cpp/GGUF and TensorRT-LLM backends (vs. BitsAndBytes). Their methodology defines the baseline we build upon.

---

### 8. Video Token Sparsification for Efficient Multimodal LLMs in Autonomous Driving
**Ma et al. | arXiv:2409.11182, 2024**

VTS addresses VLM efficiency through visual token pruning rather than quantization, reducing visual token count by 40% via temporal sparsification across video frames. Achieves 33% throughput improvement on driving video QA (LingoQA). Even with token sparsification, the pipeline requires 42+ GB GPU memory. VTS freezes the vision encoder during adaptation.

**Relevance to our project:** VTS demonstrates that token-level efficiency (pruning) and weight-level efficiency (quantization) are orthogonal techniques. Their memory numbers (42+ GB for video VQA) validate our decision to treat LingoQA as a stretch goal rather than primary benchmark. The vision encoder freeze pattern continues the cross-paper evidence for vision encoder fragility.

---

### 9. MoRAL: Sensor-Grounded BEV Reasoning for Compact VLMs toward Edge-Oriented Autonomous Driving
**Govindarajulu & Liu | IEEE IMC 2026**

MoRAL deploys Cosmos-Reason2-2B (our primary model) on nuScenes data with a two-stage fine-tuning pipeline. The model fits within 4.61 GB peak VRAM on a consumer RTX 4070 at 42 tok/s in BF16 without quantization. The paper uses Gemma 4 31B as an offline LLM judge. The Future Work section explicitly identifies quantization characterization on edge hardware as a critical next step. MoRAL demonstrates that the Cosmos-Reason2-2B architecture is viable for compact deployment — but the quantization question it raises remains unanswered.

**Relevance to our project:** MoRAL is the direct upstream work our project extends. It provides three concrete anchors: (1) same primary model (Cosmos-Reason2-2B), (2) demonstrates edge-class deployment viability without quantization, and (3) its Future Work section is the explicit research gap we fill. This alignment was confirmed through Prof. Liu's in-person guidance during September 8 office hours.

---

### 10. Cosmos-Reason2: Physical AI Reasoning Models
**NVIDIA | 2026 | HuggingFace: nvidia/Cosmos-Reason2-2B**

The Cosmos-Reason2 family achieves state-of-the-art on physical AI benchmarks spanning spatial understanding (BlinkDepth 82.26, BlinkSpatial 75.52, CV-Bench 78.74), driving tasks (AV Collision 74.33, LingoQA 59.00), and general reasoning. Built on the Qwen3-VL architecture with physical-world reasoning post-training, the 2B variant is designed for deployment on resource-constrained platforms. Jetson AI Lab provides verified deployment instructions for the Orin Nano using llama.cpp/GGUF.

**Relevance to our project:** Cosmos-Reason2-2B is our primary evaluation target. Its published benchmark scores on BLINK and CV-Bench are our comparison baselines — we measure how much of this performance survives component-wise quantization on actual edge hardware. The availability of verified Jetson Orin Nano deployment via GGUF reduces setup risk significantly.

---

## Cross-Paper Patterns

Three patterns emerge consistently across the surveyed papers:

**1. Vision encoder fragility consensus.** MPDrive, ReasonDrive, and VTS all freeze vision encoder weights during fine-tuning while adapting language layers. MBQ and Shin et al. confirm this quantitatively — vision encoders degrade faster under reduced precision. This convergence from both the training and compression literatures strengthens the hypothesis our project tests.

**2. No edge deployment with quantization analysis.** Every VLM benchmark evaluation (BLINK, CV-Bench, LingoQA, DriveLM) runs exclusively on datacenter GPUs. Shin et al. deploy on Jetson but use general-domain benchmarks and exclude the Orin Nano. MoRAL deploys on a consumer GPU but without quantization. No published work combines quantization analysis with spatial reasoning benchmarks on Jetson Orin Nano.

**3. Published baselines enable direct comparison.** Cosmos-Reason2-2B's model card publishes scores on BLINK, CV-Bench, and LingoQA at full precision. This is a methodological advantage: we compare our quantized on-device results against these official numbers rather than needing to reproduce FP16 baselines ourselves on datacenter hardware we don't have.

---

## Gap Summary

| Axis | Covered by prior work | Not covered (our project) |
|---|---|---|
| Vision encoder quantization sensitivity | MBQ (general VLMs, datacenter), Shin et al. (small VLMs, Orin NX/AGX) | Cosmos-Reason2-2B on Orin Nano, FP8/INT4 vision encoder |
| Spatial reasoning benchmarks | BLINK, CV-Bench (all FP16, datacenter) | Any quantized configuration on edge hardware |
| Edge VLM deployment | Shin et al. (general benchmarks, Orin NX/AGX), MoRAL (no quantization) | Quantized spatial reasoning on Orin Nano |
| TensorRT acceleration on edge | Documented for object detection (YOLO), not for VLMs | VLM inference acceleration on Jetson with quantization |
| Driving-specific quantization | None | LingoQA under quantization (stretch goal) |
