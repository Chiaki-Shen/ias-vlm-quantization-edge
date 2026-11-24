# Latency breakdown (llama.cpp, Q8_0 language)

Source: `bd`. Parsed 1200 of 1200 valid, non-retried timed requests.

encode = vision encoder; prefill = prompt_ms - encode (LLM, text + image tokens); decode = token generation; overhead = wall - prompt_ms - predicted_ms.

Means are used for the stacked components (means add up to the mean wall time; medians do not). Median wall time shown for comparison with the Oct 5 run.


## scienceqa

| Vision | n | image tok | encode | prefill | decode | overhead | mean wall | median wall | encode share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F16 | 150 | 96 | 306 | 309 | 55 | 44 | 714 | 556 | 43% |
| Q8_0 | 150 | 96 | 305 | 309 | 55 | 44 | 713 | 558 | 43% |
| Q4_0 | 150 | 96 | 306 | 310 | 55 | 44 | 715 | 556 | 43% |
| Q3_K_M | 150 | 96 | 329 | 310 | 55 | 44 | 738 | 569 | 45% |

Paired mean difference vs F16 (same record and round), ms [95% bootstrap CI, 10,000 resamples, seed 249]:

| Vision | encode | prefill | decode | wall |
|---|---|---|---|---|
| Q8_0 | -1 [-2, -0] | +0 [-0, +0] | +0 [+0, +0] | -0 [-1, +0] |
| Q4_0 | +1 [+0, +1] | +1 [+0, +1] | +0 [+0, +0] | +2 [+1, +2] |
| Q3_K_M | +24 [+21, +27] | +1 [+0, +1] | +0 [+0, +0] | +25 [+21, +28] |

## textvqa

| Vision | n | image tok | encode | prefill | decode | overhead | mean wall | median wall | encode share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F16 | 150 | 768 | 1627 | 842 | 147 | 101 | 2717 | 2658 | 60% |
| Q8_0 | 150 | 768 | 1603 | 843 | 147 | 101 | 2694 | 2630 | 60% |
| Q4_0 | 150 | 768 | 1615 | 843 | 149 | 100 | 2707 | 2643 | 60% |
| Q3_K_M | 150 | 768 | 1715 | 842 | 146 | 100 | 2803 | 2735 | 61% |

Paired mean difference vs F16 (same record and round), ms [95% bootstrap CI, 10,000 resamples, seed 249]:

| Vision | encode | prefill | decode | wall |
|---|---|---|---|---|
| Q8_0 | -24 [-31, -18] | +1 [+0, +2] | -0 [-1, +0] | -23 [-30, -17] |
| Q4_0 | -12 [-16, -9] | +1 [-0, +1] | +2 [-1, +5] | -10 [-15, -5] |
| Q3_K_M | +88 [+84, +91] | -0 [-1, +1] | -1 [-4, +1] | +86 [+81, +90] |
