#!/usr/bin/env bash
# Build only the LLM engine for one config and reuse the FP16 visual engine from cr2b-int8sq (the visual
# ONNX is identical in every config). For the audit configs cr2b-int8sq-mm / cr2b-int4awq-mm.
# Run inside jetson-llm:v1 (tensorrt/jetson/run_in_container.sh). INT8 needs the NVMe swap file (NOTES.md 2d).
set -euo pipefail
CFG=$1; W=$HOME/tensorrt-edgellm-workspace; E=$HOME/TensorRT-Edge-LLM
export EDGELLM_PLUGIN_PATH=$E/build/libNvInfer_edgellm_plugin.so
mkdir -p "$W/logs" "$W/$CFG/engines"
( while true; do
    echo "$(date +%T) $(free -m | awk '/Mem/{print "used="$3" avail="$7}') $(free -m | awk '/Swap/{print "swap_used="$3}')"
    sleep 5
  done ) > "$W/logs/mem_$CFG.log" &
MON=$!; trap 'kill $MON' EXIT
cd "$E"
/usr/bin/time -v ./build/examples/llm/llm_build --onnxDir "$W/$CFG/onnx/llm" --engineDir "$W/$CFG/engines/llm" \
  --maxBatchSize 1 --maxInputLen 2048 --maxKVCacheCapacity 2048
[ -e "$W/$CFG/engines/visual" ] || cp -r "$W/cr2b-int8sq/engines/visual" "$W/$CFG/engines/visual"
ls -la "$W/$CFG/engines" "$W/$CFG/engines/llm"
