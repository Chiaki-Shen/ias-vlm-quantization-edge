#!/usr/bin/env bash
# Run a command inside the course container with the Edge-LLM v0.9.1 settings this board needs.
# Usage: bash tensorrt/jetson/run_in_container.sh python3 tensorrt/run_trt_experiment.py --config T8 --datasets blink_depth --limit 20
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
exec docker run --rm --runtime=nvidia --network host --user "$(id -u):$(id -g)" \
  -e HOME="$HOME" -e EDGE_LLM_PATH="$HOME/TensorRT-Edge-LLM" \
  -e EDGELLM_PLUGIN_PATH="$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so" \
  -e EDGELLM_SYNC_GPU_ALLOCATOR=1 \
  -v "$HOME:$HOME" -v /Developer:/Developer -w "$REPO" jetson-llm:v1 "$@"
