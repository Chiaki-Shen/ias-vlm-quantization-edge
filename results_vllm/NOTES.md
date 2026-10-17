# vLLM on Jetson Orin Nano 8GB: Cosmos-Reason2-2B (feasibility, Oct 5 2026)

Verdict: it works, headless, on a very thin memory margin (peak 7,287 of 7,620 MB, ~400 MB free at idle).
Vision tower is NOT quantized in this checkpoint (ignore list covers all model.visual.*), so vLLM cannot do the vision-precision sweep.

## Paths
- Image: ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin (21.3 GB on disk)
- Checkpoint (NGC nim/nvidia/cosmos-reason2-2b:1208-fp8-static-kv8, 3.24 GB):
  /Developer/models/cosmos-reason2-2b-vllm/cosmos-reason2-2b_v1208-fp8-static-kv8
- Edits to the checkpoint (backups next to the files):
  - preprocessor_config.json: size.longest_edge 16777216 -> 1048576 (shortest_edge 3136, orig 65536). preprocessor_config.json.bak is the ORIGINAL.
  - config.json: removed quantization_config.kv_cache_scheme (FP8 e4m3 KV cache is unsupported on SM87). config.json.bak is the ORIGINAL.

## Working launch (stop desktop first: sudo systemctl stop display-manager; sudo sysctl -w vm.drop_caches=3)
docker run --rm -it --runtime nvidia --network host \
  -v /Developer/models/cosmos-reason2-2b-vllm/cosmos-reason2-2b_v1208-fp8-static-kv8:/models/cosmos-reason2:ro \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin bash
# inside: cd /opt && source venv/bin/activate
vllm serve /models/cosmos-reason2 --served-model-name local --host 0.0.0.0 --port 8000 \
  --trust-remote-code --enforce-eager --max-model-len 1280 --max-num-batched-tokens 1280 \
  --gpu-memory-utilization 0.56 --max-num-seqs 1 --enable-chunked-prefill \
  --limit-mm-per-prompt '{"image":1,"video":0}' --mm-processor-cache-gb 0
Startup ~2.5 min. Ready when the log says "Application startup complete."

## What went wrong along the way (attempts)
1. Official Nano recipe as-is: KV cache -1.4 GiB (profiling reserved memory for a video input). Fix: "video":0.
2. Loaded, but output was garbage (FlashInfer + checkpoint FP8 KV cache; also Triton crashes: fp8e4nv unsupported on SM87).
   --kv-cache-dtype does not override the checkpoint's kv_cache_scheme; had to edit config.json.
3. 1280 ctx / 1M px at util 0.52: KV cache -0.01 GiB. Fix: util 0.56.
4. Kernel OOM kill after ~170 unique TextVQA images (default mm processor cache is 4 GiB). Fix: --mm-processor-cache-gb 0.
Always use ABSOLUTE paths for tegrastats --logfile (it runs as a daemon; relative paths silently go elsewhere).

## Results (rec_no 0..199, same records as Table B, temperature 0)
- Token check: vLLM prompt_tokens == llama.cpp prompt_n on 100/100 TextVQA timed records.
- ScienceQA 85.00% (Config 2 86.00, Config 5 85.50). TextVQA 74.80% (Config 2 78.15, Config 5 78.50).
- Two full passes gave identical raw text on 400/400 records. 0 errors, 0 truncated, 0 think blocks.
- Details: results_vllm/compare_summary.txt

## Files (repo layout)
- scripts/vllm_eval.py: runs the Table B prompts/scorers against the vLLM server (run from the repo root)
- scripts/vllm_tokcheck.py: compares vLLM prompt_tokens with llama.cpp prompt_n (uses results_latency/)
- scripts/vllm_compare.py: paired comparison vs Configs 2 and 5 -> results_vllm/compare_summary.txt
- results_vllm/: pass 1 and pass 2 results, token check, run logs
- results_vllm/vLLM_FP8w16kv__textvqa_partial_before_oom.jsonl: the first TextVQA pass, which the kernel OOM-killed after 181 records (fix 4 above); pass 1 was resumed after the fix
- results_vllm/vllm_logs/: server output of attempts 1 to 6
- results_vllm/tegra_logs/: tegrastats logs per attempt (gzipped)
