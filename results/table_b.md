# Table B: vision-encoder sensitivity (language decoder fixed Q8_0)

Accuracy in %. Rows with request errors are excluded from accuracy (n shown when any were excluded). Do not average columns.

| Config | Vision | Language | BLINK depth | BLINK spatial | CV-Bench | TextVQA VQA acc. | ScienceQA IMG acc. |
|---|---|---|---|---|---|---|---|
| 2 | F16 | Q8_0 | 70.16 | 73.43 | 79.09 (n=2520) | 76.41 (n=4995) | 82.94 |
| 5 | Q8_0 | Q8_0 | 70.16 | 74.13 | 78.93 (n=2520) | 76.32 (n=4995) | 82.50 |
| 9 | Q6_K | Q8_0 | 71.77 | 74.13 | 78.61 (n=2520) | 76.37 (n=4995) | 82.20 |
| 10 | Q5_K_M | Q8_0 | 72.58 | 73.43 | 79.25 (n=2520) | 76.48 (n=4995) | 82.35 |
| 8 | Q4_0 | Q8_0 | 67.74 | 72.03 | 77.10 (n=2520) | 75.72 (n=4995) | 81.85 |
| 11 | Q3_K_M | Q8_0 (optional) | 69.35 | 71.33 | 77.94 (n=2520) | 73.76 (n=4995) | 82.20 |

## Change from Config 2 (paired, 95% bootstrap CI, points)

| Config | Vision | BLINK depth | BLINK spatial | CV-Bench | TextVQA VQA acc. | ScienceQA IMG acc. |
|---|---|---|---|---|---|---|
| 5 | Q8_0 | +0.00 [+0.00, +0.00] | +0.70 [-1.40, +2.80] | -0.16 [-0.56, +0.24] | -0.10 [-0.41, +0.20] | -0.45 [-0.94, +0.05] |
| 9 | Q6_K | +1.61 [+0.00, +4.03] | +0.70 [-1.40, +3.50] | -0.48 [-0.95, -0.04]* | -0.04 [-0.42, +0.33] | -0.74 [-1.34, -0.15]* |
| 10 | Q5_K_M | +2.42 [+0.00, +5.65] | +0.00 [-2.80, +2.80] | +0.16 [-0.44, +0.79] | +0.07 [-0.38, +0.53] | -0.59 [-1.34, +0.15] |
| 8 | Q4_0 | -2.42 [-6.45, +1.61] | -1.40 [-4.90, +2.10] | -1.98 [-2.86, -1.15]* | -0.70 [-1.36, -0.01]* | -1.09 [-1.88, -0.30]* |
| 11 | Q3_K_M | -0.81 [-5.65, +4.03] | -2.10 [-5.59, +0.70] | -1.15 [-2.06, -0.28]* | -2.66 [-3.41, -1.93]* | -0.74 [-1.54, +0.05] |

`*` = 95% CI excludes 0 (unadjusted for multiple comparisons). Deltas use only records valid in both configs.

## Discordant pairs and McNemar p (binary datasets)

| Config | Dataset | n paired | base right / other wrong | base wrong / other right | p |
|---|---|---:|---:|---:|---:|
| 5 | BLINK depth | 124 | 0 | 0 | 1 |
| 5 | BLINK spatial | 143 | 1 | 2 | 1 |
| 5 | CV-Bench | 2520 | 16 | 12 | 0.572 |
| 5 | ScienceQA IMG acc. | 2017 | 18 | 9 | 0.122 |
| 9 | BLINK depth | 124 | 0 | 2 | 0.5 |
| 9 | BLINK spatial | 143 | 1 | 2 | 1 |
| 9 | CV-Bench | 2520 | 24 | 12 | 0.0652 |
| 9 | ScienceQA IMG acc. | 2017 | 26 | 11 | 0.0201 |
| 10 | BLINK depth | 124 | 0 | 3 | 0.25 |
| 10 | BLINK spatial | 143 | 2 | 2 | 1 |
| 10 | CV-Bench | 2520 | 30 | 34 | 0.708 |
| 10 | ScienceQA IMG acc. | 2017 | 34 | 22 | 0.141 |
| 8 | BLINK depth | 124 | 5 | 2 | 0.453 |
| 8 | BLINK spatial | 143 | 4 | 2 | 0.688 |
| 8 | CV-Bench | 2520 | 84 | 34 | 4.72e-06 |
| 8 | ScienceQA IMG acc. | 2017 | 43 | 21 | 0.00815 |
| 11 | BLINK depth | 124 | 5 | 4 | 1 |
| 11 | BLINK spatial | 143 | 4 | 1 | 0.375 |
| 11 | CV-Bench | 2520 | 83 | 54 | 0.0164 |
| 11 | ScienceQA IMG acc. | 2017 | 41 | 26 | 0.0864 |

## Edge measurements from JSONL (proxies)

| Config | Mean tok/s | Lowest mem_avail_mb |
|---|---:|---:|
| 2 | 16.8 | 20 |
| 5 | 16.9 | 28 |
| 9 | 17.0 | 23 |
| 10 | 17.0 | 28 |
| 8 | 17.3 | 18 |
| 11 | 17.0 | 33 |

Lowest available memory is a proxy for peak usage, not a direct peak-memory measurement. Tok/s is decode speed, not end-to-end latency.
