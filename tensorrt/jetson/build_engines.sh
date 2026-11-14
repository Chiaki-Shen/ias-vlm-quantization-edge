#!/usr/bin/env bash
# NOTES.md step 2d for one config (cr2b-int8sq | cr2b-int4awq). Runs inside jetson-llm:v1 (TensorRT 10.7).
set -euo pipefail
CFG=$1; W=$HOME/tensorrt-edgellm-workspace; E=$HOME/TensorRT-Edge-LLM
export EDGELLM_PLUGIN_PATH=$E/build/libNvInfer_edgellm_plugin.so
( while true; do echo "$(date +%T) $(free -m | awk '/Mem/{print "used="$3" avail="$7}') $(free -m | awk '/Swap/{print "swap_used="$3}')"; sleep 5; done ) > $W/logs/mem_$CFG.log &
MON=$!; trap "kill $MON" EXIT
cd $E
/usr/bin/time -v ./build/examples/llm/llm_build --onnxDir $W/$CFG/onnx/llm --engineDir $W/$CFG/engines/llm \
  --maxBatchSize 1 --maxInputLen 2048 --maxKVCacheCapacity 2048
/usr/bin/time -v ./build/examples/multimodal/visual_build --onnxDir $W/$CFG/onnx/visual --engineDir $W/$CFG/engines \
  --minImageTokens 4 --maxImageTokens 1024 --maxImageTokensPerImage 1024
ls -la $W/$CFG/engines $W/$CFG/engines/*
