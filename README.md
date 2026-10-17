# Vision Encoder Quantization Sensitivity in Driving VLMs on Jetson Edge Hardware

## Team

- Yiang Shen, MS Computer Engineering
- Maximilian Garcia, MS Artificial Intelligence

**Advisor:** Prof. Kaikai Liu, SJSU

## Abstract

Compact vision-language models (VLMs) are emerging as practical candidates for autonomous driving reasoning on edge platforms, but deploying them within the strict memory budgets of devices like the Jetson Orin Nano (8 GB) demands aggressive quantization. Prior work on VLM quantization has reported that vision encoders can be more sensitive to precision reduction than language decoders, yet this has mostly been characterized on general-purpose benchmarks using server-class hardware, not on spatial reasoning benchmarks or memory-constrained edge devices.

This project isolates and measures vision encoder quantization sensitivity in Cosmos-Reason2-2B (NVIDIA's physical-world reasoning model built on the Qwen3-VL architecture) deployed on a physical Jetson Orin Nano 8 GB. Using llama.cpp's GGUF format, we quantize the vision encoder (mmproj) and language decoder independently. A pilot matrix of 7 configurations (F16/Q8_0/Q4_0 vision against F16/Q8_0/Q4_K_M language) established feasibility. The primary experiment then holds the language decoder fixed at Q8_0 and sweeps the vision encoder across six precisions (F16, Q8_0, Q6_K, Q5_K_M, Q4_0, Q3_K_M) on five benchmarks: BLINK depth, BLINK spatial, CV-Bench, TextVQA and ScienceQA, totaling about 71,000 inference calls. Every comparison is paired against the F16 baseline with bootstrap confidence intervals. A controlled benchmark adds latency and peak memory for each encoder, and a feasibility test checks whether vLLM can serve the same model on the same device. By locating the precision floor below which visual reasoning degrades, we provide deployment-stage quantization guidance that recent edge-oriented driving VLM pipelines (MoRAL, IEEE IMC 2026) identify as a critical next step.

## Advisor Feedback (Sep 2026)

Prof. Liu reviewed the proposal and noted that the central research question depends on controlling vision encoder and language decoder precision independently. He set the first feasibility gate as demonstrating several matched configurations before committing to the full benchmark matrix, recommended choosing one primary runtime rather than maintaining parallel pipelines, and suggested treating a second VLM and LingoQA as stretch goals. A complete sensitivity curve on one model and real edge hardware is already a strong semester result. He also suggested evaluating other inference frameworks (vLLM, TensorRT-LLM) as a follow-up phase.

**Feasibility gate: CLEARED (Sep 19, 2026).** 7 configurations confirmed working end-to-end on Jetson Orin Nano 8 GB via llama.cpp with independent mmproj and model file swapping.

## Benchmark Results

### Table B: Primary vision-sensitivity matrix (language decoder fixed at Q8_0)

All runs at 15W power mode, temperature 0.0, `--ctx-size 2048`. Accuracy in %. Records that fail with a request error are excluded from accuracy, identically in every config (see [Excluded records](#excluded-records)). Columns use different metrics and sizes, so they are not averaged.

| Config | Vision encoder (mmproj size) | BLINK depth (n=124) | BLINK spatial (n=143) | CV-Bench (n=2,520) | TextVQA (n=4,995) | ScienceQA (n=2,017) |
|---:|---|---:|---:|---:|---:|---:|
| 2 | F16, 782 MB (baseline) | 70.16 | 73.43 | 79.09 | 76.41 | 82.94 |
| 5 | Q8_0, 421 MB | 70.16 | 74.13 | 78.93 | 76.32 | 82.50 |
| 9 | Q6_K*, 329 MB | 71.77 | 74.13 | 78.61 | 76.37 | 82.20 |
| 10 | Q5_K_M*, 295 MB | 72.58 | 73.43 | 79.25 | 76.48 | 82.35 |
| 8 | Q4_0*, 229 MB | 67.74 | 72.03 | 77.10 | 75.72 | 81.85 |
| 11 | Q3_K_M*, 202 MB | 69.35 | 71.33 | 77.94 | 73.76 | 82.20 |

\*Unofficial mmproj, generated from the F16 mmproj with `llama-quantize`.

TextVQA uses the standard VQA accuracy (min(#matching human answers / 3, 1)). ScienceQA is the test split restricted to questions with an image.

**Change from F16 (paired, points, 95% bootstrap CI).** Bold = interval excludes 0.

| Config | Vision | BLINK depth | BLINK spatial | CV-Bench | TextVQA | ScienceQA |
|---:|---|---|---|---|---|---|
| 5 | Q8_0 | +0.00 [+0.00, +0.00] | +0.70 [-1.40, +2.80] | -0.16 [-0.56, +0.24] | -0.10 [-0.41, +0.20] | -0.45 [-0.94, +0.05] |
| 9 | Q6_K | +1.61 [+0.00, +4.03] | +0.70 [-1.40, +3.50] | **-0.48 [-0.95, -0.04]** | -0.04 [-0.42, +0.33] | **-0.74 [-1.34, -0.15]** |
| 10 | Q5_K_M | +2.42 [+0.00, +5.65] | +0.00 [-2.80, +2.80] | +0.16 [-0.44, +0.79] | +0.07 [-0.38, +0.53] | -0.59 [-1.34, +0.15] |
| 8 | Q4_0 | -2.42 [-6.45, +1.61] | -1.40 [-4.90, +2.10] | **-1.98 [-2.86, -1.15]** | **-0.70 [-1.36, -0.01]** | **-1.09 [-1.88, -0.30]** |
| 11 | Q3_K_M | -0.81 [-5.65, +4.03] | -2.10 [-5.59, +0.70] | **-1.15 [-2.06, -0.28]** | **-2.66 [-3.41, -1.93]** | -0.74 [-1.54, +0.05] |

Paired percentile bootstrap (10,000 resamples, seed 249, an independent random stream per config and dataset), not adjusted for multiple comparisons. Exact McNemar p-values and discordant-pair counts for the binary-scored datasets are in [`results/table_b.md`](results/table_b.md); per-cell numbers are in [`results/table_b_deltas.csv`](results/table_b_deltas.csv) and input file hashes in [`results/table_b_provenance.json`](results/table_b_provenance.json).

### Latency and peak memory

Controlled benchmark on the same 200 frozen records for every config (100 ScienceQA + 100 TextVQA), 3 rotated rounds, 15W, one request at a time (`--parallel 1 --cache-ram 0`), prompt caching off. 3,600 of 3,600 timed requests valid. Peak RAM is the maximum system RAM sampled by `tegrastats` every 200 ms during the timed requests (idle baseline 1,636 MB of 7,620 MB). Details in [`results_latency/NOTES.md`](results_latency/NOTES.md).

| Config | Vision encoder | TextVQA median ms | vs F16 | ScienceQA median ms | Peak RAM MB | vs F16 |
|---:|---|---:|---:|---:|---:|---:|
| 2 | F16 | 2,637 | 0 | 637 | 4,928 | 0 |
| 5 | Q8_0 | 2,634 | -3 | 629 | 4,575 | -353 |
| 9 | Q6_K | 2,785 | +148 | 650 | 4,541 | -387 |
| 10 | Q5_K_M | 2,746 | +109 | 642 | 4,516 | -412 |
| 8 | Q4_0 | 2,640 | +3 | 632 | 4,455 | -473 |
| 11 | Q3_K_M | 2,749 | +112 | 646 | 4,456 | -472 |

Quantizing the vision encoder does not make requests faster. F16, Q8_0 and Q4_0 are within about 10 ms of each other, while the K-quant encoders (Q6_K, Q5_K_M, Q3_K_M) are 4 to 6% slower on TextVQA. The slowdown is entirely in prompt processing (image encoding and prefill); image token counts and generation time are identical across configs, which points to slower K-quant dequantization on this GPU (not profiled). Most of the memory saving comes from F16 to Q8_0 (353 MB, matching the 361 MB smaller file); going below Q8_0 saves at most another 120 MB.

### Key Findings

**1. Q8_0, Q6_K and Q5_K_M vision encoders stay close to F16.**
On the three large benchmarks (CV-Bench, TextVQA, ScienceQA), these encoders are within about 0.75 points of F16 on every dataset. On CV-Bench and TextVQA they are within 0.5 points, and Q8_0 and Q5_K_M are statistically indistinguishable from F16 there. ScienceQA shows a small consistent dip of 0.45 to 0.74 points for all three; only Q6_K's interval excludes 0 (McNemar p = 0.020). So "no impact" is too strong, but the cost of going from F16 down to 5-bit vision is under one point. This is in line with MBQ (CVPR 2025), which also reports limited degradation from ViT quantization.

**2. The knee is between 5-bit and 4-bit vision.**
Q4_0 (Config 8) is the only vision precision whose interval excludes 0 on all three large benchmarks: CV-Bench -1.98, ScienceQA -1.09, TextVQA -0.70 points. Q3_K_M (Config 11) is also significantly below F16 on CV-Bench (-1.15) and TextVQA (-2.66), but not on ScienceQA (-0.74, interval includes 0). The pilot's Q4 language pair (Config 6 vs 7) shows the same direction on CV-Bench.

**3. Reading text in images is the most sensitive task at 3 bits, and bit-width alone does not order the low precisions.**
Q3_K_M's TextVQA drop (-2.66 [-3.41, -1.93]) is the largest in the table, about four times the Q4_0 drop on the same set. A spot check of the records F16 gets right and Q3_K_M misses shows many character-level misreads of the right text (for example `1565-1898` read as `1565-1808`, `wanita` as `wantia`) while the answer format is unchanged, which points to lost fine visual detail rather than a broken model. On CV-Bench the order reverses: Q4_0 (-1.98) loses more than Q3_K_M (-1.15). K-quants use a different block layout and scaling than the legacy Q4_0 format, so quantization format matters, not just the nominal number of bits. ScienceQA changes little at every precision (all within 1.1 points), possibly because many of its questions can be answered from the text and hint; this has not been tested.

**4. BLINK is too small to resolve vision effects.**
With 124 and 143 questions, every BLINK interval in Table B is several points wide and includes 0. BLINK results are reported for completeness, but conclusions rest on the larger datasets.

**5. Language decoder precision matters more than vision precision down to 5 bits (pilot observation).**
In Table A, switching the language decoder from Q8_0 to Q4_K_M changes BLINK depth by about +12 points (Configs 2 vs 3, 5 vs 6) and CV-Bench by about -2.3 to -2.4 points, both larger than any vision effect above Q4_0. The BLINK depth gain is reproducible across vision precisions but rests on 124 questions and has not yet been tested with paired statistics; it is a lead for follow-up, not a conclusion.

**6. Deployment guidance: Q8_0 is the best vision encoder on this device.**
Q8_0 cuts peak RAM by 353 MB at no latency cost and under half a point of accuracy on every large benchmark. The lower precisions buy little on top of that: Q5_K_M saves only 59 MB more than Q8_0 and is 4% slower on TextVQA, Q4_0 saves 120 MB more but costs 0.7 to 2.0 points, and Q3_K_M saves no more memory than Q4_0, is slower, and loses 2.7 points on TextVQA, so it is not recommended for tasks that read text in images. For the language decoder, Q4_K_M (Config 6: about 1.6 GB of model files, 19.7 tok/s) is faster and smaller than Q8_0 but trades about 2.3 points of CV-Bench accuracy; it has not been evaluated on TextVQA or ScienceQA.

### Table A: Pilot and feasibility matrix (BLINK + CV-Bench)

The original 7-configuration matrix. It established that F16/F16 does not fit on the device, so Q8_0 is the highest language precision that can be held fixed across all vision precisions including F16. This is why Table B fixes the language decoder at Q8_0. 2,905 records per config (124 BLINK depth + 143 BLINK spatial + 2,638 CV-Bench, 118 CV-Bench records excluded).

| Config | Vision encoder | Language decoder | BLINK Depth | BLINK Spatial | CV-Bench | Overall* | Avg tok/s | Status |
|--------|---------------|-----------------|-------------|---------------|----------|-------------|-----------|--------|
| ~~1~~ | ~~F16 (782 MB)~~ | ~~F16 (3.8 GB)~~ | - | - | - | - | - | OOM |
| 2 | F16 (782 MB) | Q8_0 (2.1 GB) | 70.16% | 73.43% | 79.09% | 78.40% | 17.1 | Done |
| 3 | F16 (782 MB) | Q4_K_M (1.2 GB) | 82.26% | 77.62% | 76.67% | 76.96% | 19.3 | Done |
| 4 | Q8_0 (421 MB) | F16 (3.8 GB) | 71.77% | 74.13% | 79.13% | 78.54% | 10.9 | Done |
| 5 | Q8_0 (421 MB) | Q8_0 (2.1 GB) | 70.16% | 74.13% | 78.93% | 78.29% | 17.2 | Done |
| 6 | Q8_0 (421 MB) | Q4_K_M (1.2 GB) | 82.26% | 79.02% | 76.67% | 77.04% | 19.7 | Done |
| 7 | Q4_0 (229 MB) | Q4_K_M (1.2 GB) | 78.23% | 75.52% | 75.63% | 75.74% | 20.7 | Done |
| 8 | Q4_0 (229 MB) | Q8_0 (2.1 GB) | 67.74% | 72.03% | 77.10% | 76.43% | 18.3 | Done |

\*Overall is a sample-weighted pool of the three columns and is dominated by CV-Bench. It is kept here for continuity with the initial report but is not used for conclusions; Table B reports each dataset separately.

### Excluded records

Some records fail with HTTP 400 in every configuration, even after a server restart and retry. They are excluded from accuracy, and because the excluded set is identical in every config, they cancel out in the comparisons. The analysis script aborts if the CV-Bench error set ever differs between configs.

- **CV-Bench:** 118 of 2,638 records (91 Count, 27 Relation).
- **TextVQA:** 5 of 5,000 records (idx 691, 1185, 1578, 2497, 4041). These are the 5 largest images in the set, which suggests a request-size limit, but the cause has not been confirmed.

Per-config poison logs (`config*_poison.jsonl`) are saved in `results/`.

### Server retries

During TextVQA, 157 to 189 of 5,000 requests per configuration failed on the first attempt. The cause is a llama.cpp default: the server runs 4 parallel slots that share one 2,048-token KV cache, and idle slots keep their previous request's tokens, so a large image (about 700 to 1,000 tokens) often does not fit. The runner restarted the server and retried these requests. This does not affect accuracy: retried items score the same as the same items in configurations where they were not retried, and excluding all 884 records retried in any configuration changes no delta versus F16 by more than 0.11 points. The isolation tests are in `results_latency/testA` to `testC`; the latency benchmark uses `--parallel 1 --cache-ram 0` to avoid the problem.

### Scoring audit and reproducibility

Configs 2, 5 and 8 were originally run with an earlier runner that did not save raw model replies. To confirm their BLINK and CV-Bench scores, a stratified 200-question sample was re-run with the current runner on Configs 2, 8 and 9 (control): the new answer matched the saved answer in 600 of 600 comparisons. All three configs were then re-run in full on BLINK and CV-Bench with the current runner ([`results_rerun_oct4/`](results_rerun_oct4/NOTES.md)): all 2,905 predictions per configuration match the original runs exactly, including the same 118 failing CV-Bench requests, even though the records that needed a restart and retry differed between runs. All new runs save the raw reply, both parser outputs, and the per-record score. The TextVQA scorer was checked against the reference MMF implementation on 20,000 cases with 0 mismatches.

## vLLM Feasibility (Oct 2026)

Following Prof. Liu's suggestion, we tested whether vLLM can serve Cosmos-Reason2-2B on the same Jetson Orin Nano 8 GB, using NVIDIA's Jetson vLLM container and the FP8 checkpoint from NGC. Full setup, the failures and fixes, and the working launch command are in [`results_vllm/NOTES.md`](results_vllm/NOTES.md).

**It works, but only headless and on a thin memory margin** (peak 7,287 of 7,620 MB). It needed four fixes to the official Orin Nano recipe: disabling video profiling, removing the checkpoint's FP8 KV-cache setting (not supported on this GPU, SM87), raising the GPU memory share to 0.56 for a 1,280-token context, and turning off the multimodal processor cache, which otherwise grew until the kernel killed the server after about 170 images. With the image cap raised to about 1 megapixel, vLLM produced exactly the same image token counts as llama.cpp on 100 of 100 TextVQA records, and two full passes gave identical output on 400 of 400 records.

Accuracy on the first 200 records of each dataset (same records as Table B, temperature 0):

| Dataset | vLLM (FP8 language) | Config 2 (F16 vision) | Config 5 (Q8_0 vision) | vLLM minus Config 2 |
|---|---:|---:|---:|---|
| ScienceQA | 85.00 | 86.00 | 85.50 | -1.00 [-4.00, +2.00] |
| TextVQA | 74.80 | 78.15 | 78.50 | -3.35 [-7.50, +0.80] |

Both intervals include 0, so with 200 records neither difference is significant; the TextVQA gap is borderline. The checkpoint quantizes only the language model and leaves the whole vision tower unquantized, and vLLM has no equivalent of llama.cpp's separate `--mmproj` file. **vLLM therefore cannot run this project's vision-precision sweep**, which is why llama.cpp remains the primary runtime. vLLM is useful as a cross-runtime comparison point at fixed vision precision.

## Current Status

- [x] Pilot matrix on BLINK + CV-Bench (Configs 2 to 8)
- [x] Config 8 Q4 vision isolation test
- [x] Q6_K, Q5_K_M and Q3_K_M mmproj generated and validated (Configs 9 to 11)
- [x] Configs 9 to 11 on BLINK + CV-Bench
- [x] ScienceQA (2,017) and TextVQA (5,000) prepared, frozen and hashed
- [x] Configs 2, 5, 8, 9, 10, 11 on ScienceQA and TextVQA
- [x] Scoring audit, diagnostic re-run and full reproducibility re-run of legacy results
- [x] Table B with paired bootstrap CIs and McNemar tests
- [x] Retry analysis: server retries do not affect accuracy
- [x] Direct peak-memory profiling per config via `tegrastats`
- [x] Controlled latency measurement
- [x] vLLM feasibility test (accuracy, memory, reproducibility on 400 records)
- [ ] Config 3 (F16 vision + Q4_K_M language) on ScienceQA and TextVQA, so the 4-bit language pair covers all datasets
- [ ] TensorRT-LLM cross-runtime comparison on a second Jetson (Max)
- [ ] Results visualization (accuracy vs. compression curves)
- [ ] Per-task subgroup analysis within CV-Bench
- [ ] Error analysis: which records flip between F16 and Q4_0 / Q3_K_M vision (TextVQA misreads, CV-Bench task breakdown)
- [ ] Investigation: Q4_K_M language effect on BLINK depth (paired test, larger depth set)
- [ ] Paper write-up

## Repository Structure

```
ias-vlm-quantization-edge/
├── README.md
├── proposal/
│   ├── LITERATURE_SOTA_SURVEY.md
│   └── NOVELTY_FEASIBILITY_AUDIT.md
├── scripts/
│   ├── prep_datasets.py         # Download & format BLINK + CV-Bench
│   ├── prep_scienceqa.py        # Build the ScienceQA test-image set
│   ├── prep_textvqa.py          # Build the TextVQA validation set
│   ├── run_experiment.py        # Experiment runner (protocol p2.1)
│   ├── run_inference.py         # Standalone inference script
│   ├── make_table_b.py          # Table B with paired bootstrap CIs + McNemar
│   ├── bench_latency.py         # Controlled latency + peak-memory benchmark
│   ├── crash_report.py          # First failed request per server lifetime
│   ├── vllm_eval.py             # Table B prompts/scorers against a vLLM server
│   ├── vllm_tokcheck.py         # vLLM vs llama.cpp image token counts
│   └── vllm_compare.py          # Paired vLLM vs Config 2/5 comparison
├── tests/
│   ├── test_run_experiment.py
│   └── check_textvqa_against_reference.py
├── results/                     # Table A + Table B: per-config summaries + raw .jsonl
│   ├── table_b.md               # Table B, McNemar tests, edge proxies
│   ├── table_b.csv
│   ├── table_b_deltas.csv
│   ├── table_b_provenance.json  # Input file hashes, script hash, settings
│   ├── config*_*.md / *.summary.json
│   ├── config*__<dataset>.jsonl
│   └── config*__<dataset>_poison.jsonl
├── results_rerun_oct4/          # Reproducibility re-run of Configs 2, 5, 8
├── results_latency/             # Latency + memory benchmark, retry root-cause tests
└── results_vllm/                # vLLM feasibility: results, server logs, notes
```

Benchmark data (`benchmarks/`) is not tracked in git. It is rebuilt locally with the prep scripts below.

## Hardware & Toolchain

| Item | Detail |
|------|--------|
| Device | NVIDIA Jetson Orin Nano 8 GB Developer Kit |
| JetPack | 6.1+ (CUDA 12.6, TensorRT 10.3) |
| Power mode | 15W (mode 0), locked for all runs |
| Primary model | Cosmos-Reason2-2B (NVIDIA, Qwen3-VL backbone) |
| Runtime | llama.cpp build 11056 (compiled from source, aarch64 + CUDA) |
| Quantization | GGUF per-component: `--model` (language) + `--mmproj` (vision) swapped independently |
| Server settings | `--ctx-size 2048`, `--n-gpu-layers 999`, temperature 0.0, server restart every 50 records (latency benchmark adds `--parallel 1 --cache-ram 0`) |
| Benchmarks | BLINK ([`BLINK-Benchmark/BLINK`](https://huggingface.co/datasets/BLINK-Benchmark/BLINK)), CV-Bench ([`nyu-visionx/CV-Bench`](https://huggingface.co/datasets/nyu-visionx/CV-Bench)), ScienceQA test-image questions ([`lupantech/ScienceQA`](https://github.com/lupantech/ScienceQA)), TextVQA validation ([`lmms-lab/textvqa`](https://huggingface.co/datasets/lmms-lab/textvqa)) |
| Runner revisions | `7663286e` (BLINK, CV-Bench for Configs 9 to 11), `4bfae0d4` (ScienceQA, TextVQA, Oct 4 re-runs) |
| Total inference calls | about 71,000 in the accuracy experiments: 20,335 pilot (2,905 x 7) + 8,715 (Configs 9 to 11 on BLINK + CV-Bench) + 42,102 (6 configs x 7,017 ScienceQA + TextVQA); plus 3,600 timed latency-benchmark requests and two 400-request vLLM passes |
| vLLM (feasibility only) | `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`, NGC checkpoint `cosmos-reason2-2b:1208-fp8-static-kv8` |
| Stretch | TensorRT-LLM comparison, LingoQA, second VLM |

## Reproducing the Experiment

### 1. Prepare benchmark datasets (Mac/Linux)

```bash
pip install datasets Pillow
python3 scripts/prep_datasets.py     # BLINK + CV-Bench
python3 scripts/prep_scienceqa.py    # ScienceQA
python3 scripts/prep_textvqa.py      # TextVQA validation
```

Each dataset folder gets a `manifest.json` with its source, filter, record count and the SHA-256 of `data.jsonl`. Do not rebuild a dataset folder once results exist for it; a rebuild changes the manifest and blocks `--resume`.

### 2. Transfer to Jetson

Copy the generated `benchmarks/` folder and `scripts/` to `/Developer/ias-vlm-quantization-edge/` on the Jetson.

### 3. Run an experiment (Jetson, inside sjsujetsontool shell)

```bash
export LD_LIBRARY_PATH=/opt/llama-cpp-new/build/bin:$LD_LIBRARY_PATH
cd /Developer/ias-vlm-quantization-edge
python3 scripts/run_experiment.py --config-num 9 --datasets scienceqa,textvqa --resume
```

Run this inside the container shell; from the host shell `llama-server` is not found. The runner refuses to overwrite existing results unless `--resume` or `--overwrite` is given. `--limit N` runs the first N records into `results/smoke/`. Run `python3 scripts/run_experiment.py` with no arguments for the interactive menu.

### 4. Build Table B (Mac or Linux, needs numpy)

```bash
shasum -a 256 results/config*__*.jsonl > results.sha   # or verify against the Jetson-side hash list
python scripts/make_table_b.py --results-dir results --summary-dir results \
  --sha-file results.sha --out-dir results
```

The script reads the per-record JSONL files read-only, pairs configs on `idx`, cross-checks against the runner summaries, and verifies inputs against a `sha256sum`-format file. Point `--results-dir` at `results/` only: the re-run and benchmark folders contain files with the same names.

### 5. Latency and memory benchmark (Jetson)

Start `tegrastats --interval 200 --logfile <absolute path>` on the host, then inside the container:

```bash
python3 scripts/bench_latency.py select --out results_latency/records_frozen.json
python3 scripts/bench_latency.py run --records results_latency/records_frozen.json \
  --out results_latency/<new folder> --restart-every 0 --server-extra="--parallel 1 --cache-ram 0"
python3 scripts/bench_latency.py summarize results_latency/<new folder> --tegrastats <tegrastats log>
```

`results_latency/records_frozen.json` is the record set used here; reuse it instead of running `select` to measure the same records.

### 6. vLLM feasibility test

See [`results_vllm/NOTES.md`](results_vllm/NOTES.md) for the container, checkpoint edits and launch command. With the server running, `python3 scripts/vllm_eval.py` (from the repo root) writes `results_vllm/`, and `python3 scripts/vllm_compare.py` compares it with Configs 2 and 5.

## References

- NVIDIA, "Cosmos-Reason2," 2026. <https://huggingface.co/collections/nvidia/cosmos-reason2>
- A. G. Kaliamurthi & K. Liu, "MoRAL: Sensor-Grounded BEV Reasoning for Compact VLMs toward Edge-Oriented Autonomous Driving," IEEE IMC 2026.
- S. Li et al., "MBQ: Modality-Balanced Quantization for Large Vision-Language Models," CVPR 2025.
- Y. Shin et al., "Rethinking Small VLM Quantization," ICML 2026 Workshop.
- Fu et al., "BLINK," ECCV 2024. <https://huggingface.co/datasets/BLINK-Benchmark/BLINK>
- Tong et al., "Cambrian-1 / CV-Bench," 2024. <https://huggingface.co/datasets/nyu-visionx/CV-Bench>
- Lu et al., "Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering (ScienceQA)," NeurIPS 2022.
- Singh et al., "Towards VQA Models That Can Read (TextVQA)," CVPR 2019.
- NVIDIA Jetson AI Lab, "Cosmos Reason2 Models on Jetson," 2026. <https://www.jetson-ai-lab.com/tutorials/cosmos-reason2-vlm>
