#!/usr/bin/env bash
# T16: LLM engine only (visual engine = same FP16 visual ONNX as T8/T4, copied from cr2b-int8sq). Inside jetson-llm:v1.
set -euo pipefail
W=$HOME/tensorrt-edgellm-workspace; E=$HOME/TensorRT-Edge-LLM
export EDGELLM_PLUGIN_PATH=$E/build/libNvInfer_edgellm_plugin.so
( while true; do echo "$(date +%T) $(free -m | awk "/Mem/{print \"used=\"\$3\" avail=\"\$7}") $(free -m | awk "/Swap/{print \"swap_used=\"\$3}")"; sleep 5; done ) > $W/logs/mem_cr2b-fp16.log &
MON=$!; trap "kill $MON" EXIT
cd $E
/usr/bin/time -v ./build/examples/llm/llm_build --onnxDir $W/cr2b-fp16/onnx/llm --engineDir $W/cr2b-fp16/engines/llm \
  --maxBatchSize 1 --maxInputLen 2048 --maxKVCacheCapacity 2048
ls -la $W/cr2b-fp16/engines/llm
