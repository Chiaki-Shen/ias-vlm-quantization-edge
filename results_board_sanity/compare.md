# Board sanity check: llama.cpp Config 2 on Max's board vs Aeon's

Accuracy counts errors as wrong (MCQ: exact match; TextVQA: mean VQA score). 'both' = records answered on both sides; error sets compared over common records. tok/s = median over records answered on both sides.

| dataset | A | B | common | both answered | acc A (all) | acc B (all) | identical preds | correct flips | errors A / B, same set | tok/s A / B |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| blink_depth | results_board_sanity | results | 124 | 124 | 70.16% | 70.16% | 124 (100.00%) | 0 | 0 / 0, yes | 18.22 / 16.67 |
| blink_depth | results_board_sanity | results_rerun_oct4 | 124 | 124 | 70.16% | 70.16% | 124 (100.00%) | 0 | 0 / 0, yes | 18.22 / 16.89 |
| blink_depth | results | results_rerun_oct4 | 124 | 124 | 70.16% | 70.16% | 124 (100.00%) | 0 | 0 / 0, yes | 16.67 / 16.89 |
| blink_spatial | results_board_sanity | results | 143 | 143 | 73.43% | 73.43% | 143 (100.00%) | 0 | 0 / 0, yes | 18.13 / 16.45 |
| blink_spatial | results_board_sanity | results_rerun_oct4 | 143 | 143 | 73.43% | 73.43% | 143 (100.00%) | 0 | 0 / 0, yes | 18.13 / 16.62 |
| blink_spatial | results | results_rerun_oct4 | 143 | 143 | 73.43% | 73.43% | 143 (100.00%) | 0 | 0 / 0, yes | 16.45 / 16.62 |
| cv_bench | results_board_sanity | results | 2638 | 2520 | 75.55% | 75.55% | 2520 (100.00%) | 0 | 118 / 118, yes | 18.03 / 16.39 |
| cv_bench | results_board_sanity | results_rerun_oct4 | 2638 | 2520 | 75.55% | 75.55% | 2520 (100.00%) | 0 | 118 / 118, yes | 18.03 / 16.42 |
| cv_bench | results | results_rerun_oct4 | 2638 | 2520 | 75.55% | 75.55% | 2520 (100.00%) | 0 | 118 / 118, yes | 16.39 / 16.42 |
| scienceqa | results_board_sanity | results | 2017 | 2017 | 82.94% | 82.94% | 2017 (100.00%) | 0 | 0 / 0, yes | 18.21 / 16.49 |
| textvqa | results_board_sanity | results | 5000 | 4995 | 76.34% | 76.34% | 4995 (100.00%) | 0 | 5 / 5, yes | 18.00 / 17.02 |
