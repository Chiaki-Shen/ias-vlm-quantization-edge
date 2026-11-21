# Depth follow-up: answer-order swap (2026-11-19)

Config 2 (F16 vision + Q8_0 language) and Config 3 (F16 vision + Q4_K_M language) on BLINK depth (124) and
the CV-Bench Depth records (600), with the two choices reversed and the ground truth flipped.
Analysis: `python3 scripts/depth_swap_analysis.py` -> `results/analysis/depth_swap.md`.

- Data: `python3 scripts/prep_swap.py` -> `benchmarks_swap/` (not in git, like `benchmarks/`). SHA-256:
  - `blink_depth/data.jsonl` 73df4817677a465ffaf298f63999cdd916c41f9f44ed3bc0bbdb4305d8342e7f
  - `cv_bench/data.jsonl` 57dc2ed1c75319deaac5a6c1061ecbb235e25d78229a8dbecce438283e38e845
- Run: `run_experiment.py --config-num {2,3} --data-dir benchmarks_swap --results-dir results_depth_swap
  --datasets blink_depth,cv_bench`, unchanged runner, same legacy prompt and parser as the original runs.
- Board: Max's Jetson (sjsujetson-45), llama.cpp e613ef2c8 (build 11056) in `cmpelkk/jetson-llm:latest`, the
  setup that matched Aeon's board record by record (`tensorrt/NOTES.md`, "Board sanity check"). 15W,
  desktop stopped. Config 2 21:54 to 22:35, Config 3 22:35 to 23:15. 1,448/1,448 records, 0 errors.
- Caveat: in BLINK depth the choices are "A is closer" / "B is closer", where A and B are points drawn in
  the image. After the swap, option letter A carries the text "B is closer", so the swapped BLINK
  prompt is harder to read than the original; CV-Bench Depth choices are object names and have no such clash.
