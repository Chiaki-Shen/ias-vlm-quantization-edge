# Controlled latency and peak memory benchmark (Oct 5, 2026)

Script: `scripts/bench_latency.py` (sha256 661e8d3d). Same 200 frozen records for every
config (`records_frozen.json`: 100 ScienceQA + 100 TextVQA), Configs 2, 5, 9, 10, 8, 11
(all with the Q8_0 language decoder), 3 rotated rounds, 15W power mode, 5-minute
cool-down between slots, `cache_prompt=false` (verified: cache_n = 0 on all 3,600 requests).

Command (`overnight/`):
`run --records results_latency/records_frozen.json --out results_latency/overnight --restart-every 0 --server-extra="--parallel 1 --cache-ram 0"`

Status: 3,600 / 3,600 valid requests, 0 invalid, 0 server deaths.

| Config | Vision encoder | TextVQA median ms | vs F16 | ScienceQA median ms | Peak RAM MB (max of 3 rounds) | vs F16 |
|---|---|---:|---:|---:|---:|---:|
| 2 | F16 | 2,637 | 0 | 637 | 4,928 | 0 |
| 5 | Q8_0 | 2,634 | -3 | 629 | 4,575 | -353 |
| 9 | Q6_K | 2,785 | +148 | 650 | 4,541 | -387 |
| 10 | Q5_K_M | 2,746 | +109 | 642 | 4,516 | -412 |
| 8 | Q4_0 | 2,640 | +3 | 632 | 4,455 | -473 |
| 11 | Q3_K_M | 2,749 | +112 | 646 | 4,456 | -472 |

Peak RAM = max sampled system RAM from tegrastats (200 ms interval) during timed requests,
ignoring samples within 1 s of a phase edge. Idle baseline 1,636 MB of 7,620 MB.

Findings:
1. Quantizing the vision encoder gives no latency gain: F16, Q8_0 and Q4_0 are equal within ~10 ms.
2. K-quant encoders (Q6_K, Q5_K_M, Q3_K_M) are 4 to 6% slower on TextVQA. The gap is entirely
   in prompt processing (image encode + prefill); image token counts and generation time are
   identical across configs. Likely slower K-quant dequantization on this GPU (not profiled).
3. F16 to Q8_0 saves 353 MB, matching the mmproj file size drop (782 to 422 MB). Below Q8_0 the
   extra savings are small.

## Root cause of the TextVQA retries in the main experiment (`testA` to `testC`)

This llama.cpp build defaults to 4 parallel slots sharing one 2,048-token KV cache. Idle slots
keep their previous request's tokens, so the next TextVQA image (~700 to 1,000 tokens) often
does not fit. With the default `--cache-ram`, idle slot state also grows a host prompt cache
until the kernel OOM-kills the server.

| Test | Extra server args | Result (Config 2, 100 ScienceQA + 100 TextVQA) |
|---|---|---|
| testA | (defaults) | 200/200 valid after retries, 4 server deaths |
| testB | `--cache-ram 0` | 0 deaths, 50 invalid (HTTP 500, no free KV slot) |
| testC | `--parallel 1 --cache-ram 0` | 200/200 valid, 0 deaths |

The main experiment restarted the server and retried failed requests. Retried items score the
same as the same items in configs where they were not retried, and excluding every retried
TextVQA record changes no delta vs F16 by more than 0.11 points.

`p1/` and `p4/` are short pilot runs. `scripts/crash_report.py <run dir>` lists the first
failed request per server lifetime.
